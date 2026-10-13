"""Human main metrics by scheme/type, with fixed plans and repeat denominators.

The input is explicit rubric/review decisions, not an automatic semantic judge.
Micro rates pool factual units; macro rates average nonempty trial rates. Neither
produces a complete primary rate while applicable rows have missing judgments.
"""
from collections import defaultdict

from app.answer_evaluation import metric, summarize_answers


FACT_METRICS = ('fact_correctness', 'fact_completeness', 'citation_support', 'citation_coverage')


def reconcile_plan(planned, actual):
    def index(rows):
        by_id = {r['trial_id']: r for r in rows}
        if len(by_id) != len(rows) or any(not isinstance(k, str) or not k for k in by_id):
            raise ValueError('Missing or duplicate planned/actual trial IDs')
        return by_id
    plan, observed = index(planned), index(actual)
    if set(observed) - set(plan):
        raise ValueError('Actual results contain a trial outside the fixed plan')
    rows = []
    for trial_id, item in plan.items():
        result = observed.get(trial_id)
        if result is None:
            rows.append({**item, 'execution_status': 'not_run', 'answer': None, 'citations': [],
                         'evidence': [], 'failure_stage': 'planned_trial_missing', 'human_review_status': 'pending'})
        else:
            if any(result.get(k) != item.get(k) for k in (
                    'case_id', 'method', 'repeat', 'planned_repeats', 'question', 'question_type', 'label_status')):
                raise ValueError('Actual trial identity/question differs from the fixed plan')
            rows.append(result)
    return rows


def factual_rates(rows):
    result = {}
    for name in FACT_METRICS:
        metrics = [r['metrics'][name] for r in rows]
        # A model/assistant judgment never becomes a human metric by pooling.
        eligible = [r['human_reviewed'] for r in rows]
        unknown = sum(not ok or m['unknown'] > 0 for ok, m in zip(eligible, metrics))
        primary_micro = metric(sum(m['numerator'] for ok, m in zip(eligible, metrics) if ok), sum(m['denominator'] for m in metrics),
                               metrics[0]['unit'] if metrics else 'not_observed', unknown=unknown)
        nonempty = [(r, m) for r, m in zip(rows, metrics) if m['denominator'] > 0]
        known_rates = [m['value'] for r, m in nonempty if r['human_reviewed'] and m['value'] is not None]
        macro = metric(sum(known_rates), len(nonempty), 'nonempty_trial_rates',
                       unknown=unknown)
        result[name] = {'micro': primary_micro, 'macro': macro,
                        'denominator_complete': unknown == 0,
                        'human_known_subtotal': {'numerator': sum(m['numerator'] for r, m in zip(rows, metrics) if r['human_reviewed']),
                                                'denominator': sum(m['denominator'] for r, m in zip(rows, metrics) if r['human_reviewed'])},
                        'scope': 'human main; macro is per trial, not independent-question accuracy'}
    return result


def disposition_rates(rows):
    result = {}
    for kind in ('refuse', 'clarify'):
        subset = [r for r in rows if r['expected_action'] == kind]
        unknown = sum(not r['human_reviewed'] or r['action_correct'] is None or r['actual_action'] is None for r in subset)
        passes = sum(r['human_reviewed'] and r['actual_action'] == kind and r['action_correct'] is True for r in subset)
        result['correct_' + kind] = metric(passes, len(subset), 'planned_' + kind + '_trials', unknown=unknown)
    answerable = [r for r in rows if r['expected_action'] == 'answer']
    unknown = sum(not r['human_reviewed'] or r['actual_action'] is None for r in answerable)
    result['over_refusal'] = metric(sum(r['human_reviewed'] and r['actual_action'] == 'refuse' for r in answerable),
                                    len(answerable), 'planned_answerable_trials', unknown=unknown)
    result['scope'] = 'human disposition judgment; model insufficient_evidence flag is not semantic judgment'
    return result


def stability(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['case_id']].append(row)
    counts = {'PASS': 0, 'FAIL': 0, 'INCOMPLETE': 0}
    cases = []
    for cid, repeated in sorted(groups.items()):
        declares = {r.get('planned_repeats') for r in repeated}
        repeats = [r.get('repeat') for r in repeated]
        is_three = declares == {3} and sorted(x for x in repeats if type(x) is int) == [1, 2, 3] and len(repeated) == 3
        # A missing repeat is unproven, even if surviving rows are all PASS.
        decision = ('INCOMPLETE' if not is_three else 'FAIL' if any(r['main_decision'] == 'FAIL' for r in repeated)
                    else 'INCOMPLETE' if any(r['main_decision'] == 'INCOMPLETE' for r in repeated) else 'PASS')
        counts[decision] += 1
        cases.append({'case_id': cid, 'decision': decision, 'three_repeats_present': is_three})
    return {'planned_unique_questions': len(groups), 'results': counts, 'cases': cases,
            'three_of_three_pass': metric(counts['PASS'], len(groups), 'unique_questions', unknown=counts['INCOMPLETE']),
            'scope': 'fixed retrieved evidence; generation stability, not repeated end-to-end retrieval'}


def summarize_answer_cohort(rows):
    summary = summarize_answers(rows)
    summary['unique_questions'] = len({r['case_id'] for r in rows})
    summary['factual_rates'] = factual_rates(rows)
    summary['disposition_rates'] = disposition_rates(rows)
    def subset_summary(subset):
        return {**summarize_answers(subset), 'unique_questions': len({r['case_id'] for r in subset}),
                'factual_rates': factual_rates(subset), 'disposition_rates': disposition_rates(subset)}
    methods = sorted({r.get('method') or 'unspecified' for r in rows})
    types = sorted({r.get('question_type') or 'unspecified' for r in rows})
    summary['by_method'] = {m: {**subset_summary([r for r in rows if (r.get('method') or 'unspecified') == m]),
                              'stability': stability([r for r in rows if (r.get('method') or 'unspecified') == m])}
                            for m in methods}
    summary['by_question_type'] = {t: subset_summary([r for r in rows if (r.get('question_type') or 'unspecified') == t]) for t in types}
    summary['question_method_types'] = {m: {t: subset_summary([r for r in rows if (r.get('method') or 'unspecified') == m
                                                                       and (r.get('question_type') or 'unspecified') == t])
                                           for t in types} for m in methods}
    summary['repeat_note'] = 'Repeated outputs do not increase unique question count; pooled trial rates remain trial rates'
    return summary
