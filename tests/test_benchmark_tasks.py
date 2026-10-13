"""Gold-review protocol examples, not actual human labeling or new scores."""
import copy
from dataclasses import asdict

import pytest
from pydantic import ValidationError

from app.benchmark_tasks import GoldReview, TaskDraft, apply_gold_review, compile_task
from app.evidence_alignment import EvidenceAnchor
from app.review_workflow import record_hash
from app.source_coordinates import passage_snapshot


def fixture():
    raw = [{'id': 'a', 'title': '教学专业', 'contents': '教学专业学制4年。', 'metadata': {'source_id': 's'}},
           {'id': 'b', 'title': '教学专业', 'contents': '历史记录：教学 专业 学制 4年。', 'metadata': {'source_id': 's'}}]
    snapshot, documents, _, origins = passage_snapshot(raw, 'fixture-v1')
    draft = TaskDraft(id='fixture', question='教学专业学制几年？', question_type='single_fact',
                      expected_answer='4年', temporal_scope='教学构造', facts=[
                          {'id': 'years', 'entity': '教学专业', 'attribute': '学制', 'values': ['4年'],
                           'sources': [{'source_id': 's', 'quote': '教学专业学制4年'}]}])
    task = compile_task(draft, documents, origins, snapshot.version, 'fixture-label-v1')
    task.update(split='dev', exposure_issues=[])
    return snapshot, task


def review(task, **changes):
    return GoldReview.model_validate({'case_id': task['id'], 'task_sha256': record_hash(task), 'verdict': 'approve',
            'reviewer_type': 'human', 'reviewer_id': 'synthetic_schema_test', 'reviewed_at': '2026-10-13T10:00:00+08:00',
            **{name: True for name in ('question_clear', 'facts_correct', 'evidence_sufficient', 'requirements_complete',
                                      'temporal_scope_correct', 'novelty_checked')}, **changes})


def test_equivalent_passages_are_alternatives_not_two_required_facts():
    snapshot, task = fixture()
    fields = set(EvidenceAnchor.__dataclass_fields__)
    groups = [[EvidenceAnchor(**{k: v for k, v in a.items() if k in fields}) for a in g] for g in task['required_evidence_groups']]
    assert len(groups) == 1 and len(groups[0]) == 2
    assert snapshot.score(groups, ['b'], 5, retrieval_version='fixture-v1')['all_evidence_hit']
    assert task['label_status'] == 'pending' and not task['human_confirmed']


def test_pending_and_assistant_approval_never_become_human_gold():
    _, task = fixture()
    assert not apply_gold_review(task, None)['human_confirmed']
    result = apply_gold_review(task, review(task, reviewer_type='assistant'))
    assert not result['human_confirmed'] and result['gold_review_status'] == 'approval_incomplete'


@pytest.mark.parametrize('missing', ['question_clear', 'facts_correct', 'evidence_sufficient', 'requirements_complete',
                                     'temporal_scope_correct', 'novelty_checked', 'reviewer_id', 'reviewed_at'])
def test_incomplete_human_declaration_cannot_pass(missing):
    _, task = fixture()
    result = apply_gold_review(task, review(task, **{missing: None}))
    assert not result['human_confirmed']


def test_known_exposure_mapping_gap_and_changed_answer_are_not_approvable():
    _, task = fixture()
    original_review = review(task)
    changed = copy.deepcopy(task)
    changed['expected_answer'] = '5年'
    with pytest.raises(ValueError, match='exact task'):
        apply_gold_review(changed, original_review)
    for patch in [{'split': 'regression'}, {'exposure_issues': ['known-exposure']}, {'mapping_status': 'blocked'}]:
        changed = {**task, **patch}
        assert not apply_gold_review(changed, review(changed))['human_confirmed']


def test_review_booleans_are_strict_and_dates_need_timezone():
    _, task = fixture()
    with pytest.raises(ValidationError):
        review(task, facts_correct='true')
    with pytest.raises(ValidationError):
        review(task, reviewed_at='2026-10-13T10:00:00')


def test_a_valid_human_schema_updates_rubric_provenance_only_not_system_quality():
    _, task = fixture()
    result = apply_gold_review(task, review(task))
    assert result['human_confirmed'] and result['rubric']['label_status'] == 'human_confirmed'
    assert 'answer_accuracy' not in result and 'PASS' not in result
