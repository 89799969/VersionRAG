"""Synthetic protocol tests, not human quality or paid model evaluations."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest

from app.answer_batch import AnswerBudget, execute_trial, plan_trials, validate_saved_retrieval
from app.evidence_alignment import EvidenceSnapshot
from app.generation import generate, Settings
from app.methods import METHOD_NAMES
from app.retrieval import ROOT, sha256
from app.review_workflow import prepare_blind_review, apply_reviews
from app.telemetry import RequestTrace, trace_context


def fixture():
    text = '2025年软件工程专业学费为每年8000元。' * 25
    chunk = {'id': '1', 'title': '历史招生资料', 'contents': text,
             'metadata': {'source_id': 's', 'document_id': 'd', 'start': 0, 'end': len(text)}}
    snapshot = EvidenceSnapshot('v-fixture', {'d': text}, [chunk])
    tasks = [{'id': 'q', 'question': '2025年软件工程学费？', 'question_type': 'single_fact',
              'label_status': 'synthetic', 'split': 'dev', 'human_confirmed': False}]
    rows = [{'case_id': 'q', 'method': method, 'final': {'execution_status': 'ok', 'hits': [chunk]}}
            for method in METHOD_NAMES]
    indexed = validate_saved_retrieval(tasks, rows, ['bm25', 'bm25_rerank'], [chunk])
    budget = AnswerBudget(context_max_chars=200)
    plan = plan_trials(tasks, indexed, ['bm25', 'bm25_rerank'], budget)
    return tasks, rows, chunk, snapshot, indexed, budget, plan


def test_every_repeated_trial_is_planned_before_execution_and_blind_pending():
    tasks, rows, chunk, snapshot, indexed, budget, plan = fixture()
    assert len(plan) == 6 and len({p['trial_id'] for p in plan}) == 6
    rubric = [{'case_id': 'q', 'label_version': 'test', 'label_status': 'synthetic',
               'expected_action': 'refuse', 'required_facts': []}]
    private, _, templates = prepare_blind_review(plan, rubric)
    _, metrics = apply_reviews(plan, rubric, private, templates)
    assert metrics['main_results'] == {'PASS': 0, 'FAIL': 0, 'INCOMPLETE': 6}
    assert metrics['planned_trials'] == 6


def test_actual_clipped_input_and_explicit_sampling_body_are_saved_without_gold():
    tasks, rows, chunk, snapshot, indexed, budget, plan = fixture()
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
            'answer': '2025年每年8000元。', 'citations': ['1'], 'insufficient_evidence': False})}}],
            'usage': {'prompt_tokens': 30, 'completion_tokens': 10, 'total_tokens': 40}})
    ledger = RequestTrace()
    with trace_context(ledger):
        actual = execute_trial(plan[0], indexed[('q', 'bm25')], snapshot,
            lambda q, hits, **kw: generate(q, hits, Settings('test', 'https://example.com', 'mock', 1),
                                          httpx.MockTransport(handler), **kw), budget)
    assert actual['execution_status'] == 'ok' and actual['evidence'][0]['truncated'] is True
    assert actual['evidence'][0]['knowledge_version'] == snapshot.version
    assert actual['usage']['total_tokens'] == 40
    assert ledger.finish()['provider_attempt_count'] == 1
    assert bodies[0]['temperature'] == 0 and bodies[0]['max_tokens'] == 1200
    assert len(bodies[0]['messages']) == 2 and 'expected_answer' not in json.dumps(bodies)


def test_retrieval_failure_remains_trial_failure_without_generator_call():
    tasks, rows, chunk, snapshot, indexed, budget, plan = fixture()
    row = {'final': {'execution_status': 'error', 'hits': None}}
    def forbidden(*args, **kwargs):
        raise AssertionError('Should not call model after retrieval failure')
    result = execute_trial(plan[0], row, snapshot, forbidden, budget)
    assert result['execution_status'] == 'error' and result['model_called'] is False
    assert result['failure_stage'] == 'retrieval_unavailable'


def test_timeout_keeps_actual_input_unknown_usage_and_sanitizes_provider_detail():
    tasks, rows, chunk, snapshot, indexed, budget, plan = fixture()
    def fail(*args, **kwargs):
        raise httpx.ReadTimeout('SENSITIVE_PROVIDER_BODY_TOKEN')
    result = execute_trial(plan[0], indexed[('q', 'bm25')], snapshot, fail, budget)
    assert result['execution_status'] == 'timeout' and result['evidence']
    assert result['usage']['total_tokens'] is None
    assert 'SENSITIVE_PROVIDER_BODY_TOKEN' not in json.dumps(result)


@pytest.mark.parametrize('mutation', ['contents', 'title', 'source', 'missing', 'duplicate'])
def test_changed_or_missing_saved_retrieval_is_rejected_before_generation(mutation):
    tasks, rows, chunk, _, _, _, _ = fixture()
    bad = deepcopy(rows)
    if mutation in {'contents', 'title'}:
        bad[0]['final']['hits'][0][mutation] = 'changed'
    elif mutation == 'source':
        bad[0]['final']['hits'][0]['metadata']['source_id'] = 'wrong'
    elif mutation == 'missing':
        bad = bad[1:]
    else:
        bad.append(bad[0])
    with pytest.raises(ValueError):
        validate_saved_retrieval(tasks, bad, ['bm25'], [chunk])


def test_context_mismatch_cannot_be_reported_as_success():
    tasks, rows, chunk, snapshot, indexed, budget, plan = fixture()
    result = execute_trial(plan[0], indexed[('q', 'bm25')], snapshot,
        lambda *a, **kw: {'answer': '正确答案', 'evidence': [], 'citations': ['1'], 'insufficient_evidence': False}, budget)
    assert result['execution_status'] == 'invalid_output' and result['answer'] is None


def test_cli_preparation_and_mock_execution_preserve_all_denominators(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('answer_batch_cli', ROOT / 'scripts/run_task_answers.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tasks, rows, chunk, snapshot, _, _, _ = fixture()
    tasks[0]['label_status'] = 'pending'
    tasks_dir, align, run = [tmp_path / name for name in ['tasks', 'align', 'retrieval']]
    for p in (tasks_dir, align, run):
        p.mkdir()
    corpus = tmp_path / 'corpus'
    corpus.write_text(json.dumps(chunk, ensure_ascii=False) + '\n', encoding='utf-8')
    def write(path, value, lines=False):
        path.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in value) if lines
                        else json.dumps(value, ensure_ascii=False), encoding='utf-8')
    write(align / 'sources.json', {'documents': {'d': chunk['contents']}, 'chunks': [chunk],
          'version': snapshot.version, 'corpus_sha256': sha256(corpus)})
    write(tasks_dir / 'labels.jsonl', tasks, True)
    write(tasks_dir / 'rubrics.jsonl', [{'case_id': 'q', 'label_version': 'fixture', 'label_status': 'pending',
          'expected_action': 'refuse', 'required_facts': []}], True)
    write(tasks_dir / 'manifest.json', {'labels_sha256': sha256(tasks_dir / 'labels.jsonl'),
          'rubrics_sha256': sha256(tasks_dir / 'rubrics.jsonl'), 'source_bundle_sha256': sha256(align / 'sources.json'),
          'snapshot_sha256': snapshot.sha256})
    write(run / 'retrieval.jsonl', rows, True)
    write(run / 'manifest.json', {'status': 'complete', 'retrieval_sha256': sha256(run / 'retrieval.jsonl'),
          'labels_sha256': sha256(tasks_dir / 'labels.jsonl'), 'snapshot_sha256': snapshot.sha256,
          'knowledge_version': snapshot.version, 'split': 'dev', 'case_ids': ['q'], 'budget': {'final_k': 5}})
    monkeypatch.setattr(module, 'CORPUS', corpus)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    args = ['run', '--tasks-dir', str(tasks_dir), '--alignment-dir', str(align), '--retrieval-dir', str(run),
            '--allow-pending-dev', '--output-dir']
    monkeypatch.setattr(sys, 'argv', args + [str(tmp_path / 'dry')])
    monkeypatch.setattr(module.Settings, 'load', lambda: pytest.fail('Dry preparation must not load credentials'))
    module.main()
    assert json.loads((tmp_path / 'dry/metrics.json').read_text())['execution_status_counts'] == {'not_run': 6}
    assert not (tmp_path / 'dry/requests.jsonl').exists()
    monkeypatch.setattr(module.Settings, 'load', lambda: Settings('fixture', 'https://example.com', 'mock', 1))
    count = []
    def model(q, hits, **kw):
        count.append(q)
        if len(count) == 2:
            raise httpx.ReadTimeout('PRIVATE')
        return generate(q, hits, kw.pop('settings'), httpx.MockTransport(lambda req: httpx.Response(200,
            json={'choices': [{'message': {'content': json.dumps({'answer': '8000元', 'citations': ['1'],
                  'insufficient_evidence': False})}}]})), **kw)
    monkeypatch.setattr(module, 'generate', model)
    monkeypatch.setattr(sys, 'argv', args + [str(tmp_path / 'mock'), '--enable-remote-generation'])
    module.main()
    metrics = json.loads((tmp_path / 'mock/metrics.json').read_text())
    assert metrics['planned_trials'] == 6 and metrics['execution_status_counts'] == {'ok': 5, 'timeout': 1}
    assert metrics['actual_provider_attempts'] == 5 and metrics['provider_usage']['total_tokens']['complete_total'] is None
    assert metrics['strict_answer_accuracy'] is None
