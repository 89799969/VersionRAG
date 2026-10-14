"""Private request/physical-attempt accounting, without prompts or credentials.

HTTP success is transport success, never semantic answer success. Missing usage
keeps the complete token total unknown even when other calls report their usage.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

CURRENT: ContextVar = ContextVar('rag_request_trace', default=None)
TOKENS = ('input_tokens', 'output_tokens', 'total_tokens')


def safe_usage(payload):
    usage = payload.get('usage') if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    def number(*keys):
        for key in keys:
            if type(usage.get(key)) is int and usage[key] >= 0:
                return usage[key]
        return None
    return {'input_tokens': number('prompt_tokens', 'input_tokens'),
            'output_tokens': number('completion_tokens', 'output_tokens'),
            'total_tokens': number('total_tokens')}


def usage_totals(attempts):
    return {key: {'known_subtotal': sum(a['usage'][key] for a in attempts if a['usage'][key] is not None),
                  'unknown_attempts': sum(a['usage'][key] is None for a in attempts),
                  'complete_total': (sum(a['usage'][key] for a in attempts) if all(a['usage'][key] is not None for a in attempts) else None)}
            for key in TOKENS}


def sum_generation_usage(attempts):
    """Compatibility fields; unknown usage is not silently counted as zero."""
    result = {}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        values = [a.get('usage', {}).get(key) for a in attempts]
        result[key] = sum(values) if all(type(v) is int and v >= 0 for v in values) else None
    return result


class RequestTrace:
    def __init__(self, request_id=None, task_id=None, endpoint='experiment'):
        self.request_id = request_id or str(uuid4())
        self.task_id = task_id if isinstance(task_id, str) and re.fullmatch(r't-[a-f0-9]{32}', task_id) else 't-' + uuid4().hex
        self.endpoint = endpoint
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started = time.perf_counter()
        self.lock = threading.Lock()
        self.metadata, self.phases, self.attempts = {}, [], []

    def update(self, **values):
        allowed = {'method', 'knowledge_version', 'prompt_sha256', 'model', 'outcome',
                   'refusal_reason', 'evidence_ids', 'evidence_sha256', 'implementation_version',
                   'candidate_k', 'final_top_k', 'rrf_constant', 'question_sha256', 'answer_sha256'}
        if set(values) - allowed:
            raise ValueError('Unapproved telemetry field')
        with self.lock:
            self.metadata.update(values)

    def finish(self, status_code=None, error_type=None):
        with self.lock:
            attempts = list(self.attempts)
            return {'schema_version': 'request-ledger-v1', 'request_id': self.request_id, 'task_id': self.task_id,
                    'endpoint': self.endpoint, 'started_at': self.started_at,
                    'ended_at': datetime.now(timezone.utc).isoformat(),
                    'elapsed_ms': (time.perf_counter() - self.started) * 1000,
                    'status_code': status_code, 'error_type': error_type, **self.metadata,
                    'phases': list(self.phases), 'provider_attempts': attempts,
                    'provider_attempt_count': len(attempts), 'usage_totals': usage_totals(attempts),
                    'money_cost': None, 'semantic_quality': 'unreviewed',
                    'observation_scope': 'local_request_ledger; real-user provenance must be established separately',
                    'timing_scope': 'nested phases overlap; do not sum phase times'}


@contextmanager
def trace_context(trace):
    token = CURRENT.set(trace)
    try:
        yield trace
    finally:
        CURRENT.reset(token)


def annotate(**values):
    trace = CURRENT.get()
    if trace is not None:
        trace.update(**values)


def prompt_signature(prompt):
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def answer_signature(record):
    """Bind a review to the returned answer/citations/actual evidence, not its ID alone."""
    payload = {key: record.get(key) for key in ('answer', 'citations', 'insufficient_evidence',
                                               'evidence', 'knowledge_version')}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def code_signature(directory):
    """Snapshot application source at app creation; no .env/config values."""
    directory = Path(directory)
    rows = [(p.relative_to(directory).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(directory.rglob('*.py'))]
    return 'code-' + hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


@contextmanager
def phase(stage):
    start, status = time.perf_counter(), 'ok'
    try:
        yield
    except BaseException:
        status = 'error'
        raise
    finally:
        trace = CURRENT.get()
        if trace is not None:
            with trace.lock:
                trace.phases.append({'stage': stage, 'status': status, 'elapsed_ms': (time.perf_counter() - start) * 1000})


def tracked_post(client, url, *, stage, model, **kwargs):
    """One record for one HTTP attempt, including HTTP errors and retries.

    URL, headers, body, response body and exception message are never stored.
    Parsing/citation failure after a 200 response remains a request-phase error.
    """
    trace = CURRENT.get()
    if trace is None:
        return client.post(url, **kwargs)
    start = time.perf_counter()
    row = {'stage': stage, 'model': model, 'http_status': None, 'transport_status': 'error',
           'error_type': None, 'usage': {k: None for k in TOKENS}}
    try:
        response = client.post(url, **kwargs)
        row['http_status'] = response.status_code
        row['transport_status'] = 'http_error' if response.is_error else 'http_ok'
        try:
            row['usage'] = safe_usage(response.json())
        except (ValueError, TypeError):
            row['response_json_invalid'] = True
        return response
    except httpx.TimeoutException:
        row['transport_status'], row['error_type'] = 'timeout', 'TimeoutException'
        raise
    except httpx.RequestError:
        row['transport_status'], row['error_type'] = 'network_error', 'RequestError'
        raise
    except Exception as error:
        row['error_type'] = type(error).__name__
        raise
    finally:
        row['elapsed_ms'] = (time.perf_counter() - start) * 1000
        with trace.lock:
            row['attempt_ordinal'] = len(trace.attempts) + 1
            trace.attempts.append(row)


class JsonlTraceSink:
    """Single-process file append. Not a multi-node durable telemetry backend."""
    def __init__(self, path: Path):
        self.path, self.lock = Path(path), threading.Lock()

    def __call__(self, record):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def emit_safely(sink, record):
    try:
        sink(record)
    except Exception:
        # No user data, file path, credential or exception message in the log.
        logging.getLogger('rag.telemetry').error('telemetry_write_failed')
        return False
    return True
