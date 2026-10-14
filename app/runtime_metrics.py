"""Window accounting from request ledgers; transport outcome is not quality."""
from collections import Counter, defaultdict
import math

from app.telemetry import usage_totals
from app.window_quality import ANSWER_ENDPOINTS, request_quality


def distribution(values):
    values = sorted(values)
    def quantile(q):
        return values[max(0, math.ceil(q * len(values)) - 1)] if values else None
    return {'n': len(values), 'p50_ms': quantile(.50), 'p95_ms': quantile(.95)}


def summarize_window(records, *, observation_scope='local-observation', scored_answers=None, sampling_rule=None):
    if observation_scope not in {'local-observation', 'synthetic-protocol'}:
        raise ValueError('Real-user provenance cannot be established by a scope flag')
    if len({r['request_id'] for r in records}) != len(records):
        raise ValueError('Duplicate request IDs; do not double-count the ledger')
    for r in records:
        if r['schema_version'] != 'request-ledger-v1' or r['provider_attempt_count'] != len(r['provider_attempts']):
            raise ValueError('Invalid request ledger version or attempt count')
    versions, phases, service_times = defaultdict(list), defaultdict(list), defaultdict(list)
    attempts = []
    for r in records:
        versions[r.get('knowledge_version')].append(r['elapsed_ms'])
        for phase in r['phases']:
            phases[phase['stage']].append(phase['elapsed_ms'])
        for attempt in r['provider_attempts']:
            attempts.append(attempt)
            service_times[attempt['stage']].append(attempt['elapsed_ms'])
    statuses = Counter(r['status_code'] for r in records)
    answer_requests = [r for r in records if r['endpoint'] in ANSWER_ENDPOINTS]
    quality = request_quality(records, scored_answers, sampling_rule=sampling_rule)
    usage = usage_totals(attempts)
    strict = quality['complete_window_strict_pass']
    tokens_per_success = (usage['total_tokens']['complete_total'] / strict['numerator']
                          if strict['unknown'] == 0 and strict['numerator'] > 0
                          and usage['total_tokens']['complete_total'] is not None else None)
    return {'schema_version': 'runtime-window-v2', 'observation_scope': observation_scope,
            'status': 'observed' if records else 'not_run',
            'window': {'start': min((r['started_at'] for r in records), default=None),
                       'end': max((r['ended_at'] for r in records), default=None)},
            'requests': len(records), 'tasks': len({r['task_id'] for r in records}),
            'answer_requests': len(answer_requests),
            'http_success_requests': sum(count for status, count in statuses.items() if status is not None and 200 <= status < 400),
            'http_failed_requests': sum(count for status, count in statuses.items() if status is not None and status >= 400),
            'http_unknown_requests': statuses[None],
            'status_counts': {str(status): count for status, count in statuses.items()},
            'insufficient_evidence_responses': sum(r.get('outcome') == 'insufficient_evidence' for r in answer_requests),
            'human_quality': quality,
            'real_volunteer_trial': {'status': 'not_run', 'users': None, 'tasks': None, 'quality': None},
            'latency': {'total': distribution([r['elapsed_ms'] for r in records]),
                        'by_knowledge_version': {str(v): distribution(values) for v, values in versions.items()},
                        'phases': {stage: distribution(values) for stage, values in phases.items()},
                        'provider_attempts': {stage: distribution(values) for stage, values in service_times.items()}},
            'provider_attempts': len(attempts),
            'provider_transport_counts': dict(Counter(a['transport_status'] for a in attempts)),
            'provider_usage': usage,
            'usage_by_stage': {stage: usage_totals([a for a in attempts if a['stage'] == stage]) for stage in service_times},
            'money_cost': None,
            'cost_per_strict_success': {'total_tokens': usage['total_tokens']['complete_total'],
                                       'known_strict_successes': strict['numerator'], 'unknown_outcomes': strict['unknown'],
                                       'tokens_per_success': tokens_per_success, 'verified_money_per_success': None,
                                       'scope': 'all captured window attempts, including errors; excludes unobserved/background work'},
            'notes': ['All HTTP requests, including errors and management polls, are retained.',
                      'Task IDs link requests; task counts are not real-user counts.',
                      'Token totals include failed/retried attempts; partial known usage is not a complete total.',
                      'Nested phase durations overlap. Provider HTTP success does not imply a correct answer.',
                      'Synthetic-protocol attempts use controlled transports and are not billable provider calls.']}
