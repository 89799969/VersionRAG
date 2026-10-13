"""Budget-matched retrieval experiment, separate from serving and gold labels.

Each question retrieves each channel once. The fused pool and BM25 pool are
reranked separately with the same model and input budget. Shared channel timing
is explicitly diagnostic timing, not five independent end-to-end requests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.evidence_alignment import EvidenceAnchor
from app.hybrid import rrf_fuse
from app.methods import METHOD_NAMES


@dataclass(frozen=True)
class RetrievalBudget:
    candidate_k: int = 30
    final_k: int = 5
    rrf_constant: int = 60

    def __post_init__(self):
        if not 1 <= self.final_k <= self.candidate_k <= 100 or self.rrf_constant <= 0:
            raise ValueError("Invalid retrieval budget")


def _invoke(call):
    started = time.perf_counter()
    try:
        hits = call()
        if len({str(h['id']) for h in hits}) != len(hits):
            raise ValueError("Duplicate ranked IDs")
        return {'execution_status': 'ok', 'hits': hits,
                'elapsed_ms': (time.perf_counter() - started) * 1000, 'error_type': None}
    except Exception as error:
        # Exception messages/provider bodies can contain secrets or user text.
        return {'execution_status': 'error', 'hits': None,
                'elapsed_ms': (time.perf_counter() - started) * 1000,
                'error_type': type(error).__name__}


def _not_run(reason):
    return {'execution_status': 'not_run', 'hits': None, 'elapsed_ms': None,
            'error_type': None, 'reason': reason}


def retrieve_pipelines(question: str, bm25, *, dense=None, reranker=None,
                       budget=RetrievalBudget()):
    # No source/gold-derived filter: this routine has no access to labels.
    channels = {'bm25': _invoke(lambda: bm25.search(question, budget.candidate_k)),
                'dense': (_invoke(lambda: dense.search(question, budget.candidate_k))
                          if dense is not None else _not_run('dense_not_enabled'))}
    pools = dict(channels)
    if all(c['execution_status'] == 'ok' for c in channels.values()):
        pools['hybrid'] = _invoke(lambda: rrf_fuse(
            {name: c['hits'] for name, c in channels.items()}, budget.candidate_k, budget.rrf_constant))
    else:
        pools['hybrid'] = _not_run('channel_unavailable')
    results = {}
    for method in METHOD_NAMES:
        base = 'bm25' if method == 'bm25_rerank' else ('hybrid' if method == 'hybrid_rerank' else method)
        pool = pools[base]
        if method.endswith('_rerank'):
            if pool['execution_status'] != 'ok':
                final = _not_run('candidate_pool_unavailable')
            elif reranker is None:
                final = _not_run('reranker_not_enabled')
            else:
                def rank():
                    hits = reranker.rerank(question, pool['hits'], budget.final_k)
                    candidates = {str(h['id']): h['contents'] for h in pool['hits']}
                    if len(hits) > budget.final_k or any(
                        str(h['id']) not in candidates or h['contents'] != candidates[str(h['id'])] for h in hits
                    ):
                        raise ValueError('Reranker changed candidate identities/text')
                    return hits
                final = _invoke(rank)
        else:
            final = {**pool, 'hits': pool['hits'][:budget.final_k] if pool['hits'] is not None else None}
        results[method] = {'method': method, 'candidate_pool': pool, 'final': final,
                           'channel_counts': {name: len(c['hits']) if c['hits'] is not None else None
                                              for name, c in channels.items()},
                           'channel_elapsed_ms': {name: c['elapsed_ms'] for name, c in channels.items()},
                           'timing_scope': 'shared_channel_experiment; not independent service latency'}
    return results


def anchor_groups(label):
    fields = set(EvidenceAnchor.__dataclass_fields__)
    return [[EvidenceAnchor(**{key: value for key, value in anchor.items() if key in fields})
             for anchor in group] for group in label['required_evidence_groups']]


def score_pipeline(label, result, snapshot, budget=RetrievalBudget()):
    groups = anchor_groups(label)
    def score(stage, k):
        if stage['hits'] is None:
            return None
        return snapshot.score(groups, [str(h['id']) for h in stage['hits']], k,
                              retrieval_version=snapshot.version)
    candidate = score(result['candidate_pool'], budget.candidate_k)
    final = score(result['final'], budget.final_k)
    if final is not None and final['status'] != 'ok':
        stage = 'label_blocked'
    elif result['final']['execution_status'] != 'ok':
        stage = 'execution_' + result['final']['execution_status']
    elif final['all_evidence_hit']:
        stage = 'retrieval_hit_answer_unassessed'
    elif candidate['all_evidence_hit']:
        stage = 'ranking_or_final_budget'
    else:
        stage = 'candidate_recall'
    return {**result, 'case_id': label['id'], 'label_status': label.get('label_status', 'pending'),
            'question_type': label.get('question_type', 'legacy'),
            'retrieval_target': label.get('retrieval_target', 'answer_evidence'),
            'candidate_score': candidate, 'final_score': final, 'failure_stage': stage}


def summarize_retrieval(rows):
    methods = {}
    for method in METHOD_NAMES:
        subset = [r for r in rows if r['method'] == method]
        scorable = [r for r in subset if r['final_score'] is not None and r['final_score']['status'] == 'ok']
        hits = sum(r['final_score']['all_evidence_hit'] is True for r in scorable)
        methods[method] = {'planned': len(subset), 'executed': sum(r['final']['execution_status'] == 'ok' for r in subset),
                           'scorable': len(scorable), 'hits': hits,
                           'not_run': sum(r['final']['execution_status'] == 'not_run' for r in subset),
                           'errors': sum(r['final']['execution_status'] == 'error' for r in subset),
                           'blocked': sum(r['failure_stage'] == 'label_blocked' for r in subset),
                           'hit_rate_scorable': hits / len(scorable) if scorable else None,
                           'complete_planned_score': hits / len(subset) if subset and len(scorable) == len(subset) else None}
    by_method = {name: {r['case_id']: r for r in rows if r['method'] == name} for name in METHOD_NAMES}
    paired = {}
    for base, selected in [('bm25', 'hybrid'), ('hybrid', 'hybrid_rerank'), ('bm25_rerank', 'hybrid_rerank')]:
        ids = sorted(set(by_method[base]) | set(by_method[selected]))
        groups = {'gains': [], 'losses': [], 'ties': [], 'unpaired': []}
        for cid in ids:
            pair = [by_method[name].get(cid) for name in (base, selected)]
            if any(r is None or r['final_score'] is None or r['final_score']['status'] != 'ok' for r in pair):
                groups['unpaired'].append(cid)
                continue
            old, new = [r['final_score']['all_evidence_hit'] for r in pair]
            groups['gains' if new and not old else 'losses' if old and not new else 'ties'].append(cid)
        paired[base + '->' + selected] = groups
    return {'methods': methods, 'paired': paired, 'scope': 'retrieval evidence only; not answer accuracy'}
