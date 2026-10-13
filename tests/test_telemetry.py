"""Mock service/accounting checks, never real provider usage or user trials."""
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.embedding_api import APIEncoder, EmbeddingError, EmbeddingSettings
from app.generation import GenerationError, Settings, generate
from app.knowledge import digest
from app.reranking import APIReranker
from app.telemetry import RequestTrace, safe_usage, sum_generation_usage, trace_context, tracked_post


def test_retry_failures_and_missing_usage_never_become_zero_total(monkeypatch):
    monkeypatch.setattr('app.reranking.time.sleep', lambda _: None)
    responses = iter([httpx.Response(503, json={'usage': {'total_tokens': 7}}),
                      httpx.Response(200, json={'results': [{'index': 0, 'relevance_score': .9}]})])
    model = APIReranker(EmbeddingSettings('secret-test-key'), httpx.MockTransport(lambda _: next(responses)))
    ledger = RequestTrace()
    with trace_context(ledger):
        model.rerank('private-query', [{'id': 'a', 'contents': 'private-source', 'rank': 1, 'score': 1}], 1)
    record = ledger.finish(200)
    assert record['provider_attempt_count'] == 2
    assert [a['http_status'] for a in record['provider_attempts']] == [503, 200]
    assert record['usage_totals']['total_tokens'] == {'known_subtotal': 7, 'unknown_attempts': 1, 'complete_total': None}
    assert not any(secret in json.dumps(record) for secret in ['secret-test-key', 'private-query', 'private-source', 'Authorization'])


def test_timeout_is_an_attempt_even_when_no_response_exists():
    def timeout(request):
        raise httpx.ReadTimeout('secret-server-message', request=request)
    encoder = APIEncoder(EmbeddingSettings('secret-key'), httpx.MockTransport(timeout))
    ledger = RequestTrace()
    with trace_context(ledger), pytest.raises(EmbeddingError, match='timed out'):
        encoder.encode(['example'])
    record = ledger.finish(504)
    assert record['provider_attempt_count'] == 1
    assert record['provider_attempts'][0]['transport_status'] == 'timeout'
    assert record['usage_totals']['total_tokens']['complete_total'] is None
    assert 'secret-server-message' not in json.dumps(record)


def test_usage_is_preserved_when_generation_returns_bad_citation():
    payload = {'choices': [{'message': {'content': json.dumps({'answer': '错误答案', 'citations': ['invented'],
                                                              'insufficient_evidence': False})}}],
               'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15, 'api_key': 'not-loggable'}}
    ledger = RequestTrace()
    with trace_context(ledger), pytest.raises(GenerationError, match='outside'):
        generate('问题', [{'id': 'a', 'title': '教学', 'contents': '资料', 'metadata': {'source_id': 'fixture'}}],
                 Settings('secret', 'https://fixture.invalid', 'mock-model', 2),
                 httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    record = ledger.finish(502)
    assert record['provider_attempts'][0]['transport_status'] == 'http_ok'
    assert record['usage_totals']['total_tokens']['complete_total'] == 15
    assert record['semantic_quality'] == 'unreviewed' and 'not-loggable' not in json.dumps(record)


def test_missing_comparison_usage_and_invalid_numbers_are_unknown():
    assert sum_generation_usage([{'usage': {'total_tokens': 5}}, {'usage': {}}])['total_tokens'] is None
    assert sum_generation_usage([{'usage': {'total_tokens': 5}}, {'usage': {'total_tokens': 7}}])['total_tokens'] == 12
    assert safe_usage({'usage': {'total_tokens': True, 'prompt_tokens': -1, 'completion_tokens': '3'}}) == {
        'input_tokens': None, 'output_tokens': None, 'total_tokens': None}


class Retriever:
    def search(self, question, top_k, source_id):
        return [{'id': 'a', 'title': '教学', 'contents': '教学资料', 'metadata': {'source_id': 'fixture'}, 'score': 1, 'rank': 1}]


def test_http_validation_and_model_failures_are_in_same_request_window():
    rows = []
    def timeout(question, hits):
        raise GenerationError('Model request timed out.')
    with TestClient(create_app({'bm25': Retriever()}, timeout, trace_sink=rows.append)) as client:
        bad = client.post('/ask', json={'question': ''})
        fail = client.post('/ask', json={'question': 'private-question'}, headers={'X-Task-ID': 'unsafe@example.com'})
    assert len(rows) == 2 and [r['status_code'] for r in rows] == [422, 504]
    assert rows[1]['task_id'] == fail.headers['X-Task-ID'] and rows[1]['task_id'].startswith('t-')
    assert rows[1]['knowledge_version'] == 'baseline-static'
    assert rows[1]['implementation_version'].startswith('code-') and rows[1]['final_top_k'] == 5
    assert rows[1]['phases'][-1]['stage'] == 'generation_and_validation'
    assert rows[1]['phases'][-1]['status'] == 'error'
    assert rows[0]['request_id'] == bad.headers['X-Request-ID']
    assert 'private-question' not in json.dumps(rows) and 'unsafe@example.com' not in json.dumps(rows)


def test_concurrent_request_contexts_do_not_share_attempts_or_task_ids():
    rows = []
    def local_generate(question, hits):
        payload = {'choices': [{'message': {'content': json.dumps({'answer': '教学答复', 'citations': ['a'], 'insufficient_evidence': False})}}],
                   'usage': {'total_tokens': 15}}
        return generate(question, hits, Settings('fixture-secret', 'https://fixture.invalid', 'mock-model', 2),
                        httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    with TestClient(create_app({'bm25': Retriever()}, local_generate, trace_sink=rows.append)) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda i: client.post('/ask', json={'question': f'问题{i}'},
                                 headers={'X-Task-ID': 't-' + f'{i:032x}'}), range(8)))
    assert all(r.status_code == 200 for r in replies)
    assert len(rows) == len({r['request_id'] for r in rows}) == len({r['task_id'] for r in rows}) == 8
    assert all(r['provider_attempt_count'] == 1 and r['usage_totals']['total_tokens']['complete_total'] == 15 for r in rows)
    assert all(r['prompt_sha256'] and r['outcome'] == 'answered_unreviewed' for r in rows)


def test_trace_sink_failure_is_reported_without_breaking_answer(caplog):
    def broken(_):
        raise OSError('private-path-secret')
    with TestClient(create_app({'bm25': Retriever()}, trace_sink=broken)) as client:
        assert client.post('/retrieve', json={'question': '教学'}).status_code == 200
    assert 'telemetry_write_failed' in caplog.text and 'private-path-secret' not in caplog.text


def test_actual_context_hash_tracks_clipping_not_the_full_retrieved_chunk():
    rows = []
    class LongRetriever(Retriever):
        def search(self, *args):
            return [{**super().search(*args)[0], 'contents': '教学文本' * 5000}]
    def model(question, hits):
        payload = {'choices': [{'message': {'content': json.dumps({'answer': '答复', 'citations': ['a'], 'insufficient_evidence': False})}}]}
        return generate(question, hits, Settings('fixture', 'https://fixture.invalid', 'mock-model', 2),
                        httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    with TestClient(create_app({'bm25': LongRetriever()}, model, trace_sink=rows.append)) as client:
        result = client.post('/ask', json={'question': '教学资料'}).json()
    actual = result['evidence'][0]['content']
    assert result['evidence'][0]['truncated']
    assert rows[0]['evidence_sha256'] == [digest(actual)]
    assert rows[0]['evidence_sha256'] != [digest('教学文本' * 5000)]


def test_metadata_rejects_unapproved_fields():
    with pytest.raises(ValueError, match='Unapproved'):
        RequestTrace().update(api_key='secret')
