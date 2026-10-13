import copy

import pytest

from app.benchmark_gates import validate_task_run
from app.review_workflow import record_hash


def tasks():
    return [{'id': 'd', 'split': 'dev', 'label_status': 'pending', 'human_confirmed': False},
            {'id': 'a', 'split': 'acceptance', 'label_status': 'pending', 'human_confirmed': False}]


def test_pending_acceptance_is_rejected_even_with_draft_dev_flag():
    with pytest.raises(ValueError, match='human gold'):
        validate_task_run(tasks(), {}, 'acceptance', allow_pending_dev=True)


def test_pending_development_requires_explicit_diagnostic_scope():
    with pytest.raises(ValueError, match='draft diagnostics'):
        validate_task_run(tasks(), {}, 'dev')
    assert len(validate_task_run(tasks(), {}, 'dev', allow_pending_dev=True)) == 1


def test_freeze_cannot_be_reused_after_cases_change_or_with_unconfirmed_labels():
    labels = tasks()
    freeze = {'acceptance_frozen': True, 'case_ids': ['a'], 'task_records_sha256': record_hash(labels[1:])}
    manifest = {'acceptance_ready_for_later_freeze': True}
    with pytest.raises(ValueError, match='human confirmation'):
        validate_task_run(labels, manifest, 'acceptance', freeze=freeze)
    labels[1].update(label_status='human_confirmed', human_confirmed=True)
    freeze['task_records_sha256'] = record_hash(labels[1:])
    assert len(validate_task_run(labels, manifest, 'acceptance', freeze=freeze)) == 1
    changed = copy.deepcopy(labels)
    changed[1]['expected_answer'] = 'changed'
    with pytest.raises(ValueError, match='changed'):
        validate_task_run(changed, manifest, 'acceptance', freeze=freeze)
