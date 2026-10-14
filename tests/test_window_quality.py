"""Synthetic human declarations for joining tests; not actual gold or users."""
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.runtime_metrics import summarize_window
from app.telemetry import RequestTrace, answer_signature, prompt_signature


def fixture(rid='a', version='v1', status=200):
    original = {'request_id': rid, 'trial_id': 'fixture-' + rid, 'case_id': 'q-' + rid,
                'question': '历史资料中是否有2025年未知专业的学费？', 'method': 'bm25',
                'knowledge_version': version, 'execution_status': 'ok', 'evidence': [],
                'answer': '所给资料不足，无法确认。', 'citations': [], 'insufficient_evidence': True}
    rubric = {'case_id': original['case_id'], 'label_version': 'fixture', 'label_status': 'human_confirmed',
              'expected_action': 'refuse', 'required_facts': [], 'knowledge_version': version}
    review = {'reviewer_type': 'human', 'reviewer_id': 'synthetic_declared_human',
              'reviewed_at': '2026-10-14T01:00:00+08:00', 'actual_action': 'refuse',
              'action_correct': True, 'time_scope_correct': True, 'no_extra_errors': True,
              'claim_inventory_complete': True, 'required_fact_mapping_complete': True, 'claims': []}
    trace = RequestTrace(request_id=rid, endpoint='/ask')
    trace.update(knowledge_version=version, method='bm25', question_sha256=prompt_signature(original['question']),
                 answer_sha256=answer_signature(original))
    return trace.finish(status), {**original, 'rubric_record': rubric, 'review_record': review, 'main_decision': 'FAIL'}


def test_partial_human_subset_does_not_prove_whole_window_or_real_users():
    first, grade = fixture()
    second, _ = fixture('b', 'v2')
    report = summarize_window([first, second], scored_answers=[grade], sampling_rule='fixture first request')
    quality = report['human_quality']
    assert quality['denominator'] == 2 and quality['reviewed'] == 1
    assert quality['reviewed_subset_strict_pass']['value'] == 1
    assert quality['complete_window_strict_pass']['value'] is None
    assert quality['by_knowledge_version']['v1']['strict_pass']['value'] == 1
    assert quality['by_knowledge_version']['v2']['strict_pass']['value'] is None
    assert report['real_volunteer_trial']['users'] is None
    assert report['cost_per_strict_success']['tokens_per_success'] is None


@pytest.mark.parametrize('change', ['answer', 'citations', 'question', 'missing_answer_hash'])
def test_request_id_alone_cannot_bind_changed_or_unrecorded_output(change):
    request, grade = fixture()
    if change == 'missing_answer_hash':
        request.pop('answer_sha256')
    elif change == 'citations':
        grade['citations'] = ['invented']
    else:
        grade[change] = 'different'
    quality = summarize_window([request], scored_answers=[grade], sampling_rule='all fixture requests')['human_quality']
    assert quality['reviewed'] == 0 and quality['complete_window_strict_pass']['value'] is None
    assert len(quality['binding_issues']) == 1


def test_model_judgment_and_precomputed_pass_flags_are_not_human_quality():
    request, grade = fixture()
    grade['review_record'].update(reviewer_type='model', judge_model='fixture-model')
    grade['main_decision'] = 'PASS'
    quality = summarize_window([request], scored_answers=[grade], sampling_rule='all fixture requests')['human_quality']
    assert quality['reviewed'] == 0 and quality['results']['INCOMPLETE'] == 1


def test_foreign_duplicate_version_changed_and_undeclared_sample_are_rejected():
    request, grade = fixture()
    with pytest.raises(ValueError, match='sampling'):
        summarize_window([request], scored_answers=[grade])
    with pytest.raises(ValueError, match='Duplicate quality'):
        summarize_window([request], scored_answers=[grade, deepcopy(grade)], sampling_rule='all')
    changed = deepcopy(grade)
    changed['request_id'] = 'foreign'
    with pytest.raises(ValueError, match='outside'):
        summarize_window([request], scored_answers=[changed], sampling_rule='all')
    changed = deepcopy(grade)
    changed['knowledge_version'] = 'wrong-version'
    with pytest.raises(ValueError, match='version/method'):
        summarize_window([request], scored_answers=[changed], sampling_rule='all')


def test_http_error_cannot_be_relabelled_success_and_remains_in_denominator():
    request, grade = fixture(status=504)
    with pytest.raises(ValueError, match='contradicts'):
        summarize_window([request], scored_answers=[grade], sampling_rule='all')
    grade.update(execution_status='timeout', answer=None, citations=[], review_record=None)
    quality = summarize_window([request], scored_answers=[grade], sampling_rule='all')['human_quality']
    assert quality['denominator'] == 1 and quality['results']['FAIL'] == 1


def test_real_api_response_is_bound_by_question_and_original_answer_signature():
    class Retriever:
        def search(self, *args):
            return []
    def generator(question, hits):
        return {'answer': '所给资料不足，无法确认。', 'citations': [], 'insufficient_evidence': True,
                'evidence': [], 'model_called': False}
    ledgers = []
    _, template = fixture(version='baseline-static')
    with TestClient(create_app({'bm25': Retriever()}, generator=generator, trace_sink=ledgers.append)) as client:
        response = client.post('/ask', json={'question': template['question']})
    assert response.status_code == 200
    trial = {**template, **response.json()}
    assert ledgers[0]['answer_sha256'] == answer_signature(trial)
    report = summarize_window(ledgers, scored_answers=[trial], sampling_rule='one synthetic request')
    assert report['human_quality']['complete_window_strict_pass']['value'] == 1
    assert report['real_volunteer_trial']['quality'] is None


def test_usage_per_success_requires_complete_quality_and_known_usage():
    request, grade = fixture()
    request['provider_attempts'] = [{'stage': 'generation', 'transport_status': 'http_ok', 'elapsed_ms': 1,
                                   'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}]
    request['provider_attempt_count'] = 1
    report = summarize_window([request], scored_answers=[grade], sampling_rule='all fixture requests')
    assert report['cost_per_strict_success']['tokens_per_success'] == 15
    assert report['cost_per_strict_success']['verified_money_per_success'] is None
    request['provider_attempts'][0]['usage']['total_tokens'] = None
    report = summarize_window([request], scored_answers=[grade], sampling_rule='all fixture requests')
    assert report['cost_per_strict_success']['tokens_per_success'] is None
