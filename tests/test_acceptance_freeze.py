"""Synthetic freeze gates; artificial human flags are not actual human gold."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util

import pytest

from app.batch_retrieval import RetrievalBudget
from app.methods import METHOD_NAMES
from app.retrieval import ROOT

spec = importlib.util.spec_from_file_location('freeze_cli', ROOT / 'scripts/freeze_new_acceptance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    labels = [{'id': cid, 'split': split, 'human_confirmed': True, 'label_status': 'human_confirmed'}
              for cid, split in [('dev1', 'dev'), ('accept1', 'acceptance')]]
    manifest = {'acceptance_ready_for_later_freeze': True, 'labels_sha256': 'labels', 'source_bundle_sha256': 'source',
                'snapshot_sha256': 'snapshot', 'knowledge_version': 'v'}
    dev = {'status': 'complete', 'remote_enabled': True, 'split': 'dev', 'case_ids': ['dev1'],
           'labels_sha256': 'labels', 'snapshot_sha256': 'snapshot', 'budget': asdict(RetrievalBudget()), 'code_sha256': {'code': 'hash'},
           'retrieval_models_indexes': {'dense_manifest_sha256': 'dense', 'bm25_manifest_sha256': 'bm', 'reranker': 'fixture'}}
    rows = [{'case_id': 'dev1', 'method': m, 'label_status': 'human_confirmed', 'retrieval_target': 'answer_evidence',
             'final': {'execution_status': 'ok'}, 'final_score': {'status': 'ok', 'all_evidence_hit': True},
             'failure_stage': 'retrieval_hit_answer_unassessed'} for m in METHOD_NAMES]
    return labels, manifest, dev, rows


def build(labels, manifest, dev, rows):
    return module.build_freeze(labels, manifest, dev, rows, selected_method='bm25_rerank',
                    rationale='synthetic fixture choice', code={'code': 'hash'}, rubrics_hash='rubrics')


def test_freeze_records_distinct_question_and_trial_budgets():
    freeze = build(*fixture())
    assert freeze['new_acceptance_questions'] == 1 and freeze['planned_generation_trials'] == 6
    assert freeze['answer_methods'] == ['bm25', 'bm25_rerank']
    assert freeze['execution_status'] == 'not_run' and freeze['remote_calls'] == 0
    assert freeze['max_retrieval_physical_attempts'] == 9


@pytest.mark.parametrize('bad', ['human', 'exposure', 'offline', 'different_code', 'missing_scheme', 'failed_trial'])
def test_freeze_rejects_missing_provenance_or_incomparable_development(bad):
    labels, manifest, dev, rows = deepcopy(fixture())
    if bad == 'human':
        labels[1]['human_confirmed'] = False
    elif bad == 'exposure':
        labels[1]['prior_dev_exposure_in_component'] = True
    elif bad == 'offline':
        dev['remote_enabled'] = False
    elif bad == 'different_code':
        dev['code_sha256'] = {'different': 'code'}
    elif bad == 'missing_scheme':
        rows = rows[:-1]
    else:
        rows[0]['final']['execution_status'] = 'error'
    with pytest.raises(ValueError):
        build(labels, manifest, dev, rows)
