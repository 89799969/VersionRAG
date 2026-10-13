"""Loopback socket deployment and fixed synthetic load, with local model doubles.

No real provider calls or volunteer data. A dedicated knowledge directory and
ephemeral port are created; the owned server is stopped and the port rechecked.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import socket
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api import create_app
from app.generation import Settings, generate
from app.knowledge import EmptyRetriever, KnowledgeService
from app.retrieval import ROOT, read_jsonl, sha256
from app.runtime_metrics import summarize_window
from app.telemetry import JsonlTraceSink
from scripts.check_version_fact_task import ControlledEncoder


def run_load(output, count=30, concurrency=5):
    if not 1 <= concurrency <= 20 or not 1 <= count <= 1000:
        raise ValueError('Invalid fixed load')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    service = KnowledgeService(output / 'knowledge', ControlledEncoder())
    service.upsert('synthetic-load', '教学软件工程2025学费', '教学构造，非真实政策。教学软件工程2025年学费为每年8000元。')
    job = service.submit()
    service.run_one()
    built = service.job(job['id'])
    if built['state'] != 'succeeded':
        raise RuntimeError('Controlled load knowledge failed to build')

    def model_reply(request):
        content = json.loads(request.content)['messages'][1]['content']
        if '模拟超时' in content:
            raise httpx.ReadTimeout('controlled timeout', request=request)
        pid = re.search(r'\[片段 ([^\]]+)\]', content).group(1)
        answer = {'answer': '教学软件工程2025年学费为每年8000元。', 'citations': [pid], 'insufficient_evidence': False}
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(answer, ensure_ascii=False)}}]})

    def local_generate(question, hits):
        return generate(question, hits, Settings('local-controlled-placeholder', 'https://fixture.invalid', 'synthetic-local-model', 2),
                        httpx.MockTransport(model_reply))

    app = create_app({'bm25': EmptyRetriever()}, local_generate, knowledge=service,
                     trace_sink=JsonlTraceSink(output / 'requests.jsonl'))
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    address, port = listener.getsockname()
    server = uvicorn.Server(uvicorn.Config(app, host=address, port=port, access_log=False, log_level='error'))
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    replies = []
    try:
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError('Owned loopback server failed to start')
            time.sleep(.02)
        url = f'http://{address}:{port}'
        def request(i):
            with httpx.Client(base_url=url, timeout=10, trust_env=False) as client:
                response = client.post('/ask', json={'question': '教学软件工程2025年的学费是多少？'},
                                       headers={'X-Task-ID': 't-' + uuid4().hex})
                body = response.json()
                checks = {'http_ok': response.status_code == 200,
                          'version': body.get('knowledge_version') == built['version'],
                          'fact': '每年8000元' in body.get('answer', ''),
                          'source_version': bool(body.get('evidence')) and all(e['knowledge_version'] == built['version'] for e in body['evidence']),
                          'source_fact': bool(body.get('evidence')) and all('每年8000元' in e['content'] for e in body['evidence']),
                          'citations': set(body.get('citations', [])) == {e['passage_id'] for e in body.get('evidence', [])}}
                return {'synthetic_trial': i, 'request_id': body.get('request_id'), 'checks': checks, 'passed': all(checks.values())}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            replies = list(pool.map(request, range(count)))
        with httpx.Client(base_url=url, timeout=10, trust_env=False) as client:
            validation = client.post('/ask', json={'question': ''})
            timeout = client.post('/ask', json={'question': '教学软件工程学费，模拟超时'})
        if validation.status_code != 422 or timeout.status_code != 504:
            raise AssertionError('Controlled error response changed')
        if not all(r['passed'] for r in replies):
            raise AssertionError('Synthetic socket request fact/version checks failed')
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        if thread.is_alive():
            raise RuntimeError('Owned server did not stop; cleanup requires attention')
    with socket.socket() as probe:
        probe.settimeout(.5)
        port_closed = probe.connect_ex((address, port)) != 0
    if not port_closed:
        raise RuntimeError('Owned loopback port still accepts connections')
    ledger = read_jsonl(output / 'requests.jsonl')
    runtime = summarize_window(ledger, observation_scope='synthetic-protocol')
    report = {'schema_version': 'local-synthetic-load-v1', 'status': 'passed',
              'fixed_concurrency': concurrency, 'planned_happy_requests': count,
              'happy_protocol_passed': sum(r['passed'] for r in replies),
              'controlled_http_failures': {'validation': 422, 'generation_timeout': 504},
              'total_http_requests': len(ledger), 'request_checks': replies,
              'bound_to': address, 'ephemeral_port': port, 'owned_server_stopped': not thread.is_alive(),
              'port_closed': port_closed, 'knowledge_version': built['version'],
              'remote_provider_calls': 0, 'real_user_trials': None, 'human_answer_quality': None,
              'scope': 'real loopback HTTP socket deployment; controlled small corpus/models; synthetic load only',
              'runtime': runtime, 'code_sha256': sha256(Path(__file__)),
              'code_snapshot': {p.relative_to(ROOT).as_posix(): sha256(p)
                   for folder in ['app', 'scripts'] for p in sorted((ROOT / folder).rglob('*.py'))}}
    (output / 'runtime.json').write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'load-task.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--concurrency', type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new private load directory')
    report = run_load(args.output_dir, args.count, args.concurrency)
    print(json.dumps({k: report[k] for k in ('status', 'fixed_concurrency', 'planned_happy_requests', 'happy_protocol_passed',
                     'total_http_requests', 'owned_server_stopped', 'port_closed', 'remote_provider_calls', 'scope')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
