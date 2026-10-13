"""Experiment protocol checks with synthetic pools, not retrieval quality gold."""
from dataclasses import asdict

import pytest

from app.batch_retrieval import RetrievalBudget, retrieve_pipelines, score_pipeline, summarize_retrieval
from app.evidence_alignment import EvidenceSnapshot, make_anchor
from app.methods import METHOD_NAMES


class Channel:
    def __init__(self, prefix):
        self.calls = []
        self.hits = [{'id': f'{prefix}{i}', 'contents': f'fixture {prefix}{i}', 'rank': i + 1,
                      'score': 100 - i} for i in range(30)]

    def search(self, question, top_k):
        self.calls.append((question, top_k))
        return self.hits[:top_k]


class Ranker:
    def __init__(self):
        self.calls = []

    def rerank(self, question, hits, top_k):
        self.calls.append((question, hits, top_k))
        return hits[-top_k:][::-1]


def snapshot_and_label(hits, target):
    documents = {str(h['id']): h['contents'] for h in hits}
    chunks = [{**h, 'metadata': {'document_id': str(h['id']), 'start': 0, 'end': len(h['contents'])}}
              for h in hits]
    anchor = make_anchor('a', target, documents[target], 0, len(documents[target]))
    return EvidenceSnapshot('v1', documents, chunks), {
        'id': 'fixture', 'label_status': 'synthetic', 'required_evidence_groups': [[asdict(anchor)]]}


def test_five_methods_share_raw_channels_and_match_rerank_budgets():
    bm, dense, ranker = Channel('b'), Channel('d'), Ranker()
    result = retrieve_pipelines('question', bm, dense=dense, reranker=ranker)
    assert tuple(result) == METHOD_NAMES
    assert bm.calls == dense.calls == [('question', 30)]
    assert len(ranker.calls) == 2
    assert all(len(pool) == 30 and k == 5 for _, pool, k in ranker.calls)
    hybrid_ids = {h['id'] for h in result['hybrid']['candidate_pool']['hits']}
    rerank_pool = ranker.calls[0][1]
    assert hybrid_ids == {h['id'] for h in rerank_pool}
    assert all(r['channel_counts'] == {'bm25': 30, 'dense': 30} for r in result.values())
    assert all(len(r['final']['hits']) == 5 for r in result.values())


def test_offline_mode_is_partial_not_five_fake_scores():
    bm = Channel('b')
    result = retrieve_pipelines('question', bm)
    snapshot, label = snapshot_and_label(bm.hits, 'b0')
    rows = [score_pipeline(label, r, snapshot) for r in result.values()]
    summary = summarize_retrieval(rows)
    assert summary['methods']['bm25']['hits'] == 1
    assert summary['methods']['bm25_rerank']['complete_planned_score'] is None
    assert rows[-1]['candidate_score']['all_evidence_hit'] is True
    assert rows[-1]['final_score'] is None
    assert summary['paired']['bm25->hybrid']['unpaired'] == ['fixture']


def test_candidate_miss_and_final_ranking_miss_are_separated():
    bm, ranker = Channel('b'), Ranker()
    result = retrieve_pipelines('question', bm, reranker=ranker)
    snapshot, label = snapshot_and_label(bm.hits, 'b0')
    row = score_pipeline(label, result['bm25_rerank'], snapshot)
    assert row['candidate_score']['all_evidence_hit'] is True
    assert row['final_score']['all_evidence_hit'] is False
    assert row['failure_stage'] == 'ranking_or_final_budget'
    result['bm25']['candidate_pool'] = {**result['bm25']['candidate_pool'], 'hits': bm.hits[1:]}
    result['bm25']['final'] = {**result['bm25']['final'], 'hits': bm.hits[1:6]}
    assert score_pipeline(label, result['bm25'], snapshot)['failure_stage'] == 'candidate_recall'


def test_exception_text_is_not_exported_and_failures_are_kept():
    class Failing:
        def search(self, *args):
            raise RuntimeError('secret-key-must-not-be-logged')
    result = retrieve_pipelines('question', Failing())
    assert result['bm25']['final']['execution_status'] == 'error'
    assert result['bm25']['final']['error_type'] == 'RuntimeError'
    assert 'secret-key' not in str(result)


def test_reranker_cannot_inject_unretrieved_or_changed_text():
    class BadRanker:
        def rerank(self, question, pool, top_k):
            return [{**pool[0], 'contents': 'changed'}]
    result = retrieve_pipelines('question', Channel('b'), reranker=BadRanker())
    assert result['bm25_rerank']['final']['execution_status'] == 'error'


def test_paired_gains_and_losses_are_not_collapsed_to_net_gain():
    bm = Channel('b')
    snapshot, label = snapshot_and_label(bm.hits, 'b0')
    result = retrieve_pipelines('question', bm, reranker=Ranker())
    rows = [score_pipeline(label, result['bm25'], snapshot), score_pipeline(label, result['bm25_rerank'], snapshot)]
    # A deliberately missing scheme is unpaired, never a loss or a zero score.
    assert summarize_retrieval(rows)['paired']['bm25_rerank->hybrid_rerank']['unpaired'] == ['fixture']


@pytest.mark.parametrize('candidate,final,constant', [(4, 5, 60), (101, 5, 60), (30, 5, 0)])
def test_invalid_budget_cannot_silently_change_operating_point(candidate, final, constant):
    with pytest.raises(ValueError):
        RetrievalBudget(candidate, final, constant)
