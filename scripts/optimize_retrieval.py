"""Development-only tuning, frozen selection, then exposed-test replication.

All experimental files stay in a new directory. No production configuration,
corpus, indexes, historical labels or reports are modified.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dense import DenseRetriever, ranked_rows
from app.embedding_api import APIEncoder, EmbeddingSettings
from app.evaluation import evidence_metrics, split_for
from app.retrieval import CORPUS, ROOT, ChineseBM25, read_jsonl, sha256
from scripts.compare_qwen_instructions import fingerprint, parse_result, read, save, score

OUT = ROOT / 'artifacts/reports/optimization_2026-10-07'
PREVIOUS = ROOT / 'artifacts/reports/qwen_instruction_2026-10-07'
BROAD = (
    "Retrieve passages that directly answer the user's question about Renmin University of China. "
    "The question may concern admissions, college introductions, programs, tuition, history, "
    "partnerships, or graduate outcomes. Match any explicitly specified year, college, program, "
    "and category. Prioritize passages containing the requested fact; do not assume constraints "
    "that the question does not state."
)
RETRIEVAL = {
    'baseline30': {'channel_k': 30, 'pool_k': 30, 'bm25_weight': 1.0},
    'expand60': {'channel_k': 60, 'pool_k': 60, 'bm25_weight': 1.0},
    'expand100': {'channel_k': 100, 'pool_k': 100, 'bm25_weight': 1.0},
    'keyword60': {'channel_k': 60, 'pool_k': 60, 'bm25_weight': 2.0},
    'dense60': {'channel_k': 60, 'pool_k': 60, 'bm25_weight': 0.5},
    'keyword30': {'channel_k': 30, 'pool_k': 30, 'bm25_weight': 2.0},
}
ALPHAS = [0.0, 0.25, 0.5, 1.0]


def fuse(channels, config):
    merged = {}
    for name, hits in channels.items():
        weight = config['bm25_weight'] if name == 'bm25' else 1.0
        seen = set()
        for rank, hit in enumerate(hits[:config['channel_k']], 1):
            key = str(hit['id'])
            if key in seen:
                continue
            seen.add(key)
            item = merged.setdefault(key, {**hit, 'score': 0.0, 'channel_ranks': {}})
            if item['contents'] != hit['contents']:
                raise ValueError('Conflicting passage text')
            item['score'] += weight / (60 + rank)
            item['channel_ranks'][name] = rank
    ordered = sorted(merged.values(), key=lambda r: (-r['score'], str(r['id'])))[:config['pool_k']]
    return [{**r, 'rank': n} for n, r in enumerate(ordered, 1)]


def blend(ranking, original, alpha):
    if set(ranking) != set(original) or len(ranking) != len(set(ranking)):
        raise ValueError('Blending requires identical unique candidate sets')
    if alpha == 0:
        return list(ranking)
    rerank_positions = {key: i for i, key in enumerate(ranking, 1)}
    original_positions = {key: i for i, key in enumerate(original, 1)}
    return sorted(ranking, key=lambda key: (
        -(1 / (10 + rerank_positions[key]) + alpha / (10 + original_positions[key])), key))


def pool_from(question, hits):
    budget = 8000 - len(question.encode('utf-8'))
    if budget <= 0 or not question.strip():
        raise ValueError('Invalid query')
    value = {'question': question, 'candidate_ids': [str(h['id']) for h in hits],
             'documents': [h['contents'].encode('utf-8')[:budget].decode('utf-8', errors='ignore') for h in hits]}
    return {**value, 'fingerprint': fingerprint(value),
            'channel_ranks': [h['channel_ranks'] for h in hits],
            'truncated_documents': sum(t != h['contents'] for t, h in zip(value['documents'], hits))}


def payload(pool, variant):
    value = {'model': variant['model'], 'query': pool['question'], 'documents': pool['documents'],
             'top_n': len(pool['candidate_ids']), 'return_documents': False}
    if variant['instruction'] is not None:
        value['instruction'] = variant['instruction']
    if variant['model'] == 'BAAI/bge-reranker-v2-m3':
        value.update(max_chunks_per_doc=8, overlap_tokens=50)
    return value


class Service:
    def __init__(self):
        settings = EmbeddingSettings.load()
        self.key = settings.api_key
        self.client = httpx.Client(timeout=settings.timeout, trust_env=False)
        self.requests, self.failures = 0, []

    def rerank(self, pool, variant):
        body = payload(pool, variant)
        started = time.perf_counter()
        for attempt in range(3):
            self.requests += 1
            try:
                response = self.client.post('https://api.siliconflow.cn/v1/rerank', json=body,
                    headers={'Authorization': 'Bearer ' + self.key})
            except httpx.RequestError:
                self.failures.append({'kind': 'network', 'attempt': attempt + 1})
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError('Network failure; no fallback') from None
            if response.status_code != 200:
                self.failures.append({'status': response.status_code, 'attempt': attempt + 1})
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f'Rerank HTTP {response.status_code}; no fallback')
            break
        elapsed = (time.perf_counter() - started) * 1000
        data = response.json()
        ranking, scores = parse_result(data, pool['candidate_ids'])
        return {'ranking': ranking, 'scores': scores, 'rerank_latency_ms': elapsed,
                'request_id': data.get('id'), 'payload_fingerprint': fingerprint(body)}


def protect(path, value):
    if path.exists():
        assert read(path) == value, f'Frozen file differs: {path}'
    else:
        save(path, value)


def samples(split):
    return [s for s in read_jsonl(ROOT / 'data/basic_qa_mapped.jsonl') if split_for(s) == split]


def summarize(rows, variants, alpha=0.0):
    result = []
    for variant in variants:
        subset = [r for r in rows if r['variant'] == variant]
        assert subset and all(r['status'] == 'ok' for r in subset)
        scored = [score(r['sample'], blend(r['ranking'], r['candidate_ids'], alpha)) for r in subset]
        result.append({'variant': variant, 'alpha': alpha, 'measurements': len(subset),
            'questions': len({r['id'] for r in subset}),
            **{k: float(np.mean([s[k] for s in scored])) for k in scored[0]},
            'candidate_recall': float(np.mean([evidence_metrics(r['sample'], r['candidate_ids'], len(r['candidate_ids']))['evidence_group_recall'] for r in subset])),
            'candidate_count': float(np.mean([len(r['candidate_ids']) for r in subset])),
            'rerank_p50_ms': float(np.percentile([r['rerank_latency_ms'] for r in subset], 50)),
            'rerank_p95_ms': float(np.percentile([r['rerank_latency_ms'] for r in subset], 95))})
    return result


def diagnose():
    """Only old DEVELOPMENT labels select one post-ranking weight."""
    qa = {s['id']: s for s in samples('dev')}
    old = [json.loads(l) for l in (PREVIOUS / 'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = []
    for r in old:
        if r['id'] in qa and r['variant'] == 'bge_control':
            rows.append({**r, 'sample': qa[r['id']]})
    trials = [summarize(rows, ['bge_control'], a)[0] for a in ALPHAS]
    winner = sorted(trials, key=lambda r: (-r['all_evidence_hit_at_5'], -r['mrr_at_5'], r['alpha']))[0]
    missed = []
    for s in qa.values():
        pool = read(PREVIOUS / 'pools' / (s['id'] + '.json'))
        if not evidence_metrics(s, pool['candidate_ids'], 30)['all_evidence_hit']:
            missed.append({'id': s['id'], 'question': s['question']})
    protect(OUT / 'diagnosis.json', {'split': 'dev', 'questions': 35,
        'candidate_missing': missed, 'post_rank_trials': trials, 'selected_alpha': winner['alpha'],
        'selection': 'Max mean development Hit@5, then MRR@5, then smaller alpha; no holdout metrics accessed.'})
    print('Development diagnosis:', len(missed), 'missing candidates; post-rank alpha', winner['alpha'], flush=True)
    for r in trials:
        print('post-rank', r['alpha'], 'hit5', round(r['all_evidence_hit_at_5'], 4), 'MRR', round(r['mrr_at_5'], 4), flush=True)


def freeze_manifest():
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in (
        CORPUS, ROOT / 'data/basic_qa_mapped.jsonl', ROOT / 'artifacts/indexes/dense/vectors.npy',
        ROOT / 'artifacts/indexes/dense/manifest.json', ROOT / 'artifacts/indexes/bm25/manifest.json',
        ROOT / 'app/retrieval.py', ROOT / 'app/reranking.py', ROOT / 'app/hybrid.py',
        ROOT / 'app/dense.py', ROOT / 'app/evaluation.py')}
    protect(OUT / 'manifest.json', {'project': 'rag-domain-study/admissions', 'hashes': hashes,
        'script_sha256': sha256(Path(__file__)), 'retrieval_trials': RETRIEVAL,
        'broad_instruction': BROAD, 'post_rank_trials': ALPHAS, 'rounds': 3,
        'shortlist_selection': 'Development candidate ceiling desc, retrieval MRR@5 desc, smaller pool first; select two nonbaseline pools.',
        'final_selection': 'Development rerank Hit@5 desc, MRR@5 desc, smaller candidate count, smaller alpha, variant name.',
        'timing': 'Pure rerank request latency; offline cached retrieval is separate; no generation.',
        'scope': 'Original holdout already exposed. Follow-up exploratory tuning, not untouched generalization measurement.'})


def channels_for(split):
    qa = samples(split)
    paths = [OUT / 'channels' / (s['id'] + '.json') for s in qa]
    if all(p.exists() for p in paths):
        return
    bm25, dense = ChineseBM25(), DenseRetriever()
    bm25.load(); dense.load()
    encoder = APIEncoder()
    signature = {'encoder': encoder.metadata(), 'queries': [{'id': s['id'], 'question': s['question']} for s in qa]}
    protect(OUT / f'{split}_query_manifest.json', signature)
    vector_path = OUT / f'{split}_queries.npy'
    if vector_path.exists():
        queries = np.load(vector_path)
    else:
        batches = []
        for begin in range(0, len(qa), 8):
            vec, _ = encoder.encode([s['question'] for s in qa[begin:begin + 8]], query=True)
            batches.append(vec)
        queries = np.concatenate(batches)
        np.save(vector_path, queries)
    assert queries.shape == (len(qa), 1024)
    for i, (s, query, path) in enumerate(zip(qa, queries, paths), 1):
        if path.exists():
            continue
        start = time.perf_counter()
        keywords = bm25.search(s['question'], 100)
        semantic = [{**dense.documents[index], 'score': sim, 'rank': n} for n, (index, sim)
                    in enumerate(ranked_rows(dense.vectors, query, 100), 1)]
        save(path, {'question': s['question'], 'channels': {'bm25': keywords, 'dense': semantic},
                    'cached_retrieval_ms': (time.perf_counter() - start) * 1000,
                    'query_vector_sha256': fingerprint(query.tolist())})
        if i % 10 == 0:
            print(split, 'channels prepared', i, '/', len(qa), flush=True)


def prepare_dev():
    freeze_manifest(); diagnose(); channels_for('dev')
    qa = samples('dev')
    summary = []
    for name, config in RETRIEVAL.items():
        rows = []
        for s in qa:
            channel = read(OUT / 'channels' / (s['id'] + '.json'))
            hits = fuse(channel['channels'], config)
            pool = pool_from(s['question'], hits)
            protect(OUT / 'pools' / name / (s['id'] + '.json'), pool)
            rows.append({'id': s['id'], 'question': s['question'], 'ids': pool['candidate_ids'],
                         'candidate_recall': evidence_metrics(s, pool['candidate_ids'], len(pool['candidate_ids']))['evidence_group_recall'],
                         **score(s, pool['candidate_ids'])})
        save(OUT / 'retrieval_trials' / (name + '.json'), rows)
        value = {'name': name, **config,
                 'candidate_recall': float(np.mean([r['candidate_recall'] for r in rows])),
                 'hit5': float(np.mean([r['all_evidence_hit_at_5'] for r in rows])),
                 'mrr5': float(np.mean([r['mrr_at_5'] for r in rows]))}
        summary.append(value)
        print('Development retrieval', value, flush=True)
    shortlist = sorted([r for r in summary if r['name'] != 'baseline30'], key=lambda r:
        (-r['candidate_recall'], -r['mrr5'], r['pool_k'], r['name']))[:2]
    variants = {'bge_baseline': {'pool': 'baseline30', 'model': 'BAAI/bge-reranker-v2-m3', 'instruction': None}}
    for item in shortlist:
        variants['bge_' + item['name']] = {'pool': item['name'], 'model': 'BAAI/bge-reranker-v2-m3', 'instruction': None}
    variants['qwen_unspecified'] = {'pool': 'baseline30', 'model': 'Qwen/Qwen3-Reranker-0.6B', 'instruction': None}
    variants['qwen_broad'] = {'pool': 'baseline30', 'model': 'Qwen/Qwen3-Reranker-0.6B', 'instruction': BROAD}
    protect(OUT / 'dev_plan.json', {'retrieval_summary': summary, 'shortlist': shortlist, 'variants': variants,
        'selected_post_alpha': read(OUT / 'diagnosis.json')['selected_alpha'],
        'selection_data': 'Only 35 development labels. No holdout result calculated for this optimization.'})


def run_phase(phase):
    plan = read(OUT / ('dev_plan.json' if phase == 'dev' else 'selection.json'))
    variants = plan['variants']
    qa = samples('dev') if phase == 'dev' else samples('holdout') + samples('demo')
    if phase != 'dev':
        for split in ('holdout', 'demo'):
            channels_for(split)
        for s in qa:
            channel = read(OUT / 'channels' / (s['id'] + '.json'))
            for name in {v['pool'] for v in variants.values()}:
                hits = fuse(channel['channels'], RETRIEVAL[name])
                protect(OUT / 'pools' / name / (s['id'] + '.json'), pool_from(s['question'], hits))
    path = OUT / (phase + '_rows.jsonl')
    rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done = {(r['round'], r['id'], r['variant']) for r in rows}
    service = Service()
    warm = qa[0]
    for variant in variants.values():
        service.rerank(read(OUT / 'pools' / variant['pool'] / (warm['id'] + '.json')), variant)
    for n in range(3):
        order = list(qa)
        random.Random(20261007 + n).shuffle(order)
        for number, s in enumerate(order, 1):
            names = list(variants)
            random.Random(f'{phase}:{n}:{s["id"]}').shuffle(names)
            for name in names:
                key = (n, s['id'], name)
                if key in done:
                    continue
                variant = variants[name]
                pool = read(OUT / 'pools' / variant['pool'] / (s['id'] + '.json'))
                row = {'phase': phase, 'round': n, 'id': s['id'], 'sample': s, 'split': split_for(s),
                       'variant': name, 'candidate_ids': pool['candidate_ids'], 'pool_fingerprint': pool['fingerprint']}
                try:
                    row.update(service.rerank(pool, variant), status='ok')
                except (RuntimeError, ValueError, KeyError) as exc:
                    row.update(status='error', error_type=type(exc).__name__)
                with path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                rows.append(row); done.add(key)
            save(OUT / (phase + '_status.json'), {'state': 'running', 'completed': len(rows),
                'expected': 3 * len(qa) * len(variants), 'requests': service.requests, 'failures': service.failures})
            if number % 10 == 0:
                print(phase, 'round', n + 1, '/3 queries', number, '/', len(qa), 'rows', len(rows), flush=True)
    save(OUT / (phase + '_status.json'), {'state': 'finished', 'completed': len(rows),
        'expected': 3 * len(qa) * len(variants), 'requests': service.requests, 'failures': service.failures})
    assert all(r['status'] == 'ok' for r in rows), 'Failures saved; do not publish an incomplete comparison'
    if phase == 'dev':
        alpha = plan['selected_post_alpha']
        trials = summarize(rows, variants, 0.0)
        if alpha:
            trials += summarize(rows, variants, alpha)
        save(OUT / 'dev_summary.json', trials)
        ordering = lambda r: (-r['all_evidence_hit_at_5'], -r['mrr_at_5'], r['candidate_count'], r['alpha'], r['variant'])
        winner = sorted(trials, key=ordering)[0]
        best_qwen = sorted([r for r in trials if r['variant'].startswith('qwen_')], key=ordering)[0]
        selected = {k: variants[k] for k in dict.fromkeys(['bge_baseline', winner['variant'], best_qwen['variant']])}
        protect(OUT / 'selection.json', {'variants': selected, 'winner': winner, 'best_qwen': best_qwen,
            'selected_post_alpha': alpha, 'dev_rows_sha256': sha256(path),
            'selection_rule': read(OUT / 'manifest.json')['final_selection'],
            'selected_before_test_requests': True})
        print('Frozen development winner:', winner, flush=True)
    else:
        result = {}
        for split in ('holdout', 'demo'):
            subset = [r for r in rows if r['split'] == split]
            values = summarize(subset, variants, 0.0)
            if plan['selected_post_alpha']:
                values += summarize(subset, variants, plan['selected_post_alpha'])
            result[split] = values
        save(OUT / 'test_summary.json', result)
        print('Frozen test complete; no selection changed after test.', flush=True)
    service.client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'dev', 'test'])
    stage = parser.parse_args().stage
    if stage == 'prepare':
        prepare_dev()
    else:
        run_phase(stage)


if __name__ == '__main__':
    main()
