"""Window denominators, attempt usage and synthetic/user provenance invariants."""
import copy

import pytest

from app.runtime_metrics import summarize_window
from app.telemetry import RequestTrace


def row(identity, status, total):
    trace = RequestTrace(request_id=identity, endpoint='/ask')
    trace.update(knowledge_version='v1')
    trace.attempts.append({'stage': 'generation', 'transport_status': 'http_ok' if status == 200 else 'timeout',
                           'elapsed_ms': 2, 'usage': {'input_tokens': None, 'output_tokens': None, 'total_tokens': total}})
    return trace.finish(status)


def test_errors_and_partial_usage_stay_in_total_request_window():
    report = summarize_window([row('a', 200, 15), row('b', 504, None)], observation_scope='synthetic-protocol')
    assert report['requests'] == report['answer_requests'] == report['human_quality']['denominator'] == 2
    assert report['http_success_requests'] == report['http_failed_requests'] == 1
    assert report['provider_usage']['total_tokens'] == {'known_subtotal': 15, 'unknown_attempts': 1, 'complete_total': None}
    assert report['human_quality']['strict_pass'] is None
    assert report['real_volunteer_trial']['users'] is None and report['money_cost'] is None


def test_duplicate_records_cannot_inflate_usage_or_sample_size():
    r = row('a', 200, 15)
    with pytest.raises(ValueError, match='Duplicate'):
        summarize_window([r, copy.deepcopy(r)])


def test_unknown_http_status_and_empty_latency_are_not_fake_success():
    r = row('unknown', None, None)
    report = summarize_window([r])
    assert report['http_unknown_requests'] == 1 and report['http_success_requests'] == 0
    empty = summarize_window([])
    assert empty['status'] == 'not_run' and empty['latency']['total']['p95_ms'] is None


def test_real_user_claim_requires_actual_provenance_not_a_flag():
    with pytest.raises(ValueError, match='provenance'):
        summarize_window([], observation_scope='real-users')
