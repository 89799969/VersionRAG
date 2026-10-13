"""Synthetic arithmetic/protocol checks, not real human labels or performance."""
from copy import deepcopy

import pytest

from app.answer_metrics import factual_rates, reconcile_plan, summarize_answer_cohort
from app.answer_evaluation import metric


def row(cid='q', method='bm25', repeat=1, *, decision='PASS', human=True, expected='answer',
        actual='answer', correct=True, facts=(1, 1)):
    return {'case_id': cid, 'trial_id': f'{cid}-{method}-{repeat}', 'method': method,
            'repeat': repeat, 'planned_repeats': 3, 'question_type': 'single_fact',
            'main_decision': decision, 'human_reviewed': human, 'expected_action': expected,
            'actual_action': actual, 'action_correct': correct,
            'metrics': {name: metric(*facts, 'fixture_units')
                        for name in ('fact_correctness', 'fact_completeness', 'citation_support', 'citation_coverage')}}


def test_repeat_metrics_keep_independent_questions_and_methods_separate():
    rows = [row(cid, method, repeat) for cid in ['q1', 'q2', 'q3']
            for method in ['bm25', 'bm25_rerank'] for repeat in [1, 2, 3]]
    next(r for r in rows if r['trial_id'] == 'q2-bm25-2')['main_decision'] = 'FAIL'
    next(r for r in rows if r['trial_id'] == 'q3-bm25-3')['main_decision'] = 'INCOMPLETE'
    summary = summarize_answer_cohort(rows)
    assert summary['planned_trials'] == 18 and summary['unique_questions'] == 3
    stable = summary['by_method']['bm25']['stability']
    assert stable['results'] == {'PASS': 1, 'FAIL': 1, 'INCOMPLETE': 1}
    assert stable['three_of_three_pass'] == metric(1, 3, 'unique_questions', unknown=1)
    assert summary['by_method']['bm25_rerank']['stability']['three_of_three_pass']['value'] == 1


def test_missing_repeat_is_not_a_two_of_two_success():
    summary = summarize_answer_cohort([row(repeat=1), row(repeat=2)])
    assert summary['by_method']['bm25']['stability']['three_of_three_pass']['value'] is None


def test_macro_and_micro_are_different_and_partial_human_pool_stays_unknown():
    rows = [row('q1', facts=(1, 1)), row('q2', facts=(0, 3))]
    rates = factual_rates(rows)['fact_completeness']
    assert rates['micro']['value'] == .25 and rates['macro']['value'] == .5
    rows[1]['human_reviewed'] = False
    rates = factual_rates(rows)['fact_completeness']
    assert rates['micro']['value'] is None and rates['macro']['value'] is None
    assert rates['human_known_subtotal'] == {'numerator': 1, 'denominator': 1}
    assert rates['denominator_complete'] is False


def test_disposition_needs_human_actual_action_not_model_flag():
    rows = [row('refuse', expected='refuse', actual='refuse'),
            row('clarify', expected='clarify', actual='clarify'),
            row('answer', actual='refuse', correct=False)]
    summary = summarize_answer_cohort(rows)
    assert summary['disposition_rates']['correct_refuse']['value'] == 1
    assert summary['disposition_rates']['correct_clarify']['value'] == 1
    assert summary['disposition_rates']['over_refusal']['value'] == 1
    rows[0]['actual_action'] = None
    assert summarize_answer_cohort(rows)['disposition_rates']['correct_refuse']['value'] is None


def test_interrupted_batch_keeps_full_plan_and_rejects_foreign_or_changed_outputs():
    plan = [dict(row(cid, repeat=r), question='fixture', label_status='synthetic')
            for cid in ['q1', 'q2'] for r in [1, 2, 3]]
    actual = deepcopy(plan[:1])
    actual[0]['execution_status'] = 'ok'
    reconciled = reconcile_plan(plan, actual)
    assert len(reconciled) == 6
    assert sum(r.get('failure_stage') == 'planned_trial_missing' for r in reconciled) == 5
    assert all(r['execution_status'] == 'not_run' for r in reconciled[1:])
    changed = deepcopy(actual)
    changed[0]['question'] = 'different'
    with pytest.raises(ValueError, match='differs'):
        reconcile_plan(plan, changed)
    changed[0]['trial_id'] = 'foreign'
    with pytest.raises(ValueError, match='outside'):
        reconcile_plan(plan, changed)
