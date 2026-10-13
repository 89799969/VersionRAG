import csv

import pytest

from scripts.review_task_gold import read_reviews
from app.review_workflow import record_hash


def fixture(tmp_path):
    task = {'id': 'fixture', 'question': '问题', 'question_type': 'single_fact', 'expected_action': 'answer',
            'expected_answer': '答案', 'temporal_scope': '教学', 'mapping_status': 'mapped', 'split': 'dev'}
    row = {k: task[k] for k in ('question', 'question_type', 'expected_action', 'expected_answer', 'temporal_scope', 'mapping_status')}
    row.update(case_id='fixture', task_sha256=record_hash(task), provisional_split='dev', status='pending')
    path = tmp_path / 'review.csv'
    def write(value):
        with path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(value))
            writer.writeheader()
            writer.writerow(value)
    return path, task, row, write


def test_empty_csv_stays_pending_and_read_only_data_cannot_change(tmp_path):
    path, task, row, write = fixture(tmp_path)
    write(row)
    assert read_reviews(path, {'fixture': task}) == [('fixture', None)]
    row['expected_answer'] = '审核者改成另一个答案'
    write(row)
    with pytest.raises(ValueError, match='Read-only'):
        read_reviews(path, {'fixture': task})


def test_false_judgment_and_missing_provenance_are_not_truthy_strings(tmp_path):
    path, task, row, write = fixture(tmp_path)
    row.update(status='approve', reviewer_type='assistant', reviewer_id='', reviewed_at='', notes='教学')
    row.update({name: 'false' for name in ('question_clear', 'facts_correct', 'evidence_sufficient',
                'requirements_complete', 'temporal_scope_correct', 'novelty_checked')})
    write(row)
    result = read_reviews(path, {'fixture': task})[0][1]
    assert result.facts_correct is False and result.reviewer_id is None
    row['facts_correct'] = 'yes maybe'
    write(row)
    with pytest.raises(ValueError, match='Judgments'):
        read_reviews(path, {'fixture': task})
