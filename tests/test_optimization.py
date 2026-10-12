import pytest

from app.hybrid import rrf_fuse
from scripts.optimize_retrieval import blend, fuse, payload, pool_from


def hit(key, text=None):
    return {'id': key, 'contents': text or key, 'score': 2.0, 'rank': 1}


def test_weighted_fusion_agrees_with_production_at_unit_weights():
    channels = {'bm25': [hit('a'), hit('b')], 'dense': [hit('b'), hit('c')]}
    config = {'channel_k': 30, 'pool_k': 30, 'bm25_weight': 1.0}
    assert fuse(channels, config) == rrf_fuse(channels, 30, 60)
    config['bm25_weight'] = 2.0
    assert fuse({'bm25': [hit('a')], 'dense': [hit('b')]}, config)[0]['id'] == 'a'


def test_fusion_detects_inconsistent_documents_and_duplicate_votes():
    config = {'channel_k': 30, 'pool_k': 30, 'bm25_weight': 1.0}
    assert len(fuse({'bm25': [hit('a'), hit('a')]}, config)) == 1
    with pytest.raises(ValueError):
        fuse({'bm25': [hit('a', 'one')], 'dense': [hit('a', 'two')]}, config)


def test_postranking_blends_ranks_and_never_invents_candidates():
    ranking = ['a', 'b', 'c']
    assert blend(ranking, ['c', 'b', 'a'], 0) == ranking
    assert set(blend(ranking, ['c', 'b', 'a'], 1)) == set(ranking)
    with pytest.raises(ValueError):
        blend(ranking, ['a', 'b', 'd'], .5)


def test_instruction_does_not_rewrite_query_and_no_gold_in_payload():
    hits = [dict(hit('a', '招生正文'), channel_ranks={'bm25': 1})]
    pool = pool_from('原问题', hits)
    body = payload(pool, {'model': 'Qwen/Qwen3-Reranker-0.6B', 'instruction': 'task only'})
    assert body['query'] == '原问题'
    assert body['documents'] == ['招生正文']
    assert body['instruction'] == 'task only'
    assert 'max_chunks_per_doc' not in body
    assert set(body) == {'model', 'query', 'documents', 'top_n', 'return_documents', 'instruction'}
