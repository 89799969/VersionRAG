"""Join independently supplied scoring records to exact observed requests.

Human provenance remains a declaration. Transport statistics never prove
semantic quality, and quality does not establish volunteer/user provenance.
"""
from collections import Counter, defaultdict

from app.answer_evaluation import AnswerReview, CaseRubric, evaluate_answer, metric
from app.telemetry import answer_signature, prompt_signature

ANSWER_ENDPOINTS = {'/ask', '/ask-comparison', 'benchmark-answer-generation'}


def request_quality(records, scored_answers=None, *, sampling_rule=None):
    requests = {r['request_id']: r for r in records if r['endpoint'] in ANSWER_ENDPOINTS}
    scores = [] if scored_answers is None else scored_answers
    if scores and (not isinstance(sampling_rule, str) or not sampling_rule.strip()):
        raise ValueError('Reviewed window needs an explicit sampling rule declaration')
    if len({r.get('request_id') for r in scores}) != len(scores):
        raise ValueError('Duplicate quality request IDs; supply one adjudicated primary record')
    decisions, reviewed, issues = {}, set(), []
    for row in scores:
        rid = row.get('request_id')
        request = requests.get(rid)
        if request is None:
            raise ValueError('Quality record is outside the answer-request window')
        if (row.get('knowledge_version') != request.get('knowledge_version')
                or row.get('method') != request.get('method')):
            raise ValueError('Quality record differs from the observed version/method')
        question = row.get('question')
        if not isinstance(question, str) or request.get('question_sha256') != prompt_signature(question):
            issues.append({'request_id': rid, 'reason': 'question_unbound_or_changed'})
            continue
        status = request.get('status_code')
        failed_generation = request.get('outcome') in {'timeout', 'error', 'invalid_output'}
        if (type(status) is int and status >= 400) or failed_generation:
            if row.get('execution_status') == 'ok':
                raise ValueError('Successful quality output contradicts the HTTP failure')
        elif request.get('answer_sha256') != answer_signature(row):
            issues.append({'request_id': rid, 'reason': 'answer_unbound_or_changed'})
            continue
        rubric = CaseRubric.model_validate(row.get('rubric_record'))
        review = AnswerReview.model_validate(row['review_record']) if row.get('review_record') is not None else None
        result = evaluate_answer(rubric, row, review)
        # Never trust a precomputed PASS flag from the supplied file alone.
        decisions[rid] = result['main_decision']
        if result['human_reviewed']:
            reviewed.add(rid)
    counts = Counter(decisions.get(rid, 'INCOMPLETE') for rid in requests)
    reviewed_counts = Counter(decisions[rid] for rid in reviewed)
    versions = defaultdict(list)
    for rid, request in requests.items():
        versions[request.get('knowledge_version')].append(rid)
    by_version = {}
    for version, ids in versions.items():
        group = Counter(decisions.get(rid, 'INCOMPLETE') for rid in ids)
        by_version[str(version)] = {'requests': len(ids), 'reviewed': sum(rid in reviewed for rid in ids),
                                    'results': {key: group[key] for key in ('PASS', 'FAIL', 'INCOMPLETE')},
                                    'strict_pass': metric(group['PASS'], len(ids), 'answer_requests', unknown=group['INCOMPLETE'])}
    return {'status': 'not_reviewed' if not reviewed else 'partially_reviewed' if counts['INCOMPLETE'] else 'reviewed',
            'denominator': len(requests), 'reviewed': len(reviewed),
            'strict_pass': counts['PASS'] if reviewed else None,
            'results': {key: counts[key] for key in ('PASS', 'FAIL', 'INCOMPLETE')},
            'complete_window_strict_pass': metric(counts['PASS'], len(requests), 'answer_requests', unknown=counts['INCOMPLETE']),
            'reviewed_subset_strict_pass': metric(reviewed_counts['PASS'], len(reviewed), 'human_reviewed_requests',
                                                  unknown=reviewed_counts['INCOMPLETE']),
            'sampling_rule': sampling_rule, 'by_knowledge_version': by_version, 'binding_issues': issues,
            'scope': 'explicit human gold/review declarations; partial subset cannot prove whole-window quality',
            'provenance_note': 'Sampling and human identities are declared, not independently authenticated; users remain unproven'}
