import json

import httpx
import pytest

from app.embedding_api import EmbeddingSettings
from scripts.compare_qwen_instructions import ADMISSIONS, Service, VARIANTS, parse_result, payload_for, score


def fixture_pool():
    return {'question':'2018年甲类艺术团的降分政策？','candidate_ids':['a','b'],
            'documents':['2018年甲类政策','2019年另一类别政策']}


def test_instruction_is_separate_from_original_query_and_candidate_texts():
    captured=[]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200,json={'results':[{'index':1,'relevance_score':.2},{'index':0,'relevance_score':.8}]})
    service=Service(EmbeddingSettings('offline-test-key'),httpx.MockTransport(handler))
    pool=fixture_pool()
    for variant in VARIANTS:
        assert service.rerank(pool,variant)['ranking']==['a','b']
    assert all(body['query']==pool['question'] and body['documents']==pool['documents'] for body in captured)
    assert 'instruction' not in captured[0] and 'instruction' not in captured[1]
    assert captured[-1]['instruction']==ADMISSIONS
    assert 'max_chunks_per_doc' not in captured[-1]


def test_response_must_cover_exact_frozen_candidate_pool():
    with pytest.raises(ValueError):
        parse_result({'results':[{'index':0,'relevance_score':.8},{'index':0,'relevance_score':.1}]},['a','b'])
    with pytest.raises(ValueError):
        parse_result({'results':[{'index':0,'relevance_score':float('nan')}]},['a'])


def test_metrics_keep_equivalent_evidence_and_missing_evidence_distinct():
    sample={'required_evidence_groups':[{'acceptable_corpus_passage_ids':['a','equiv']}]}
    assert score(sample,['irrelevant','equiv'])['all_evidence_hit_at_5']
    assert score(sample,['irrelevant','equiv'])['mrr_at_5']==.5
    assert score(sample,['irrelevant'])['ndcg_at_5_binary_annotations']==0
