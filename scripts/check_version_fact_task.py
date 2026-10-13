"""Private synthetic version task through the real ASGI/DB/index pipeline.

Models use controlled local doubles. This validates version/fact/quote plumbing,
not real-model answer accuracy, socket deployment, load, or volunteer usage.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api import create_app
from app.generation import Settings, generate
from app.knowledge import EmptyRetriever, KnowledgeService
from app.retrieval import ROOT, sha256
from app.telemetry import JsonlTraceSink


class ControlledEncoder:
    dimension = 3

    def __init__(self):
        self.fail = False

    def metadata(self):
        return {'dimension': 3, 'model': 'synthetic-local-encoder', 'revision': 'fixture-v1'}

    def encode(self, texts, query=False):
        if self.fail:
            raise RuntimeError('controlled build failure')
        return np.tile(np.array([1, 0, 0], dtype=np.float32), (len(texts), 1)), 0


def wait_job(client, job_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get('/knowledge/jobs/' + job_id)
        response.raise_for_status()
        result = response.json()
        if result['state'] in {'succeeded', 'failed'}:
            return result
        time.sleep(.05)
    raise TimeoutError('Version worker did not finish within the test window')


def run_task(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    encoder = ControlledEncoder()
    service = KnowledgeService(output / 'knowledge', encoder)
    pending_switch = []

    def fixture_model(request):
        body = json.loads(request.content)
        content = body['messages'][1]['content']
        pid = re.search(r'\[片段 ([^\]]+)\]', content).group(1)
        value = re.search(r'学费为每年(\d+)元', content).group(1)
        answer = {'answer': f'教学软件工程2025年学费为每年{value}元。',
                  'citations': [pid], 'insufficient_evidence': False}
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(answer, ensure_ascii=False)}}]})

    def local_generate(question, hits):
        if pending_switch:
            service.activate(pending_switch.pop())
        return generate(question, hits, Settings('synthetic-local-placeholder', 'https://fixture.invalid', 'synthetic-local-model', 2),
                        httpx.MockTransport(fixture_model))

    app = create_app({'bm25': EmptyRetriever()}, local_generate, knowledge=service,
                     trace_sink=JsonlTraceSink(output / 'requests.jsonl'))
    question = '教学软件工程2025年的学费是多少？'
    task_id = 't-' + uuid4().hex
    steps, jobs = [], []
    with TestClient(app, headers={'X-Task-ID': task_id}) as client:
        def publish(fee):
            doc = {'document_id': 'synthetic-fee', 'title': '教学软件工程2025收费说明',
                   'text': f'教学构造材料，非真实招生政策。教学软件工程2025年学费为每年{fee}元。'}
            response = client.put('/knowledge/documents', json=doc)
            response.raise_for_status()
            result = wait_job(client, client.post('/knowledge/jobs', json={}).json()['id'])
            jobs.append(result)
            return result

        def ask(name, version, fee):
            response = client.post('/ask', json={'question': question, 'method': 'bm25'})
            response.raise_for_status()
            answer = response.json()
            evidence = answer['evidence']
            checks = {'request_version': answer['knowledge_version'] == version,
                      'evidence_version': bool(evidence) and all(e['knowledge_version'] == version for e in evidence),
                      'fact_value': f'每年{fee}元' in answer['answer'],
                      'quote_value': bool(evidence) and all(f'每年{fee}元' in e['content'] for e in evidence),
                      'citation_identity': set(answer['citations']) == {e['passage_id'] for e in evidence}}
            steps.append({'name': name, 'expected_version': version, 'expected_value': fee,
                          'checks': checks, 'passed': all(checks.values()), 'answer': answer})
            if not all(checks.values()):
                raise AssertionError('Version/fact/citation task failed at ' + name)

        first = publish(8000)
        assert first['state'] == 'succeeded' and first['published']
        ask('initial-published', first['version'], 8000)
        second = publish(9000)
        assert second['state'] == 'succeeded' and second['published']
        ask('changed-fact-published', second['version'], 9000)
        # Switch inside generation after retrieval has pinned V2. This answer
        # must retain V2 even though the globally active version is now V1.
        pending_switch.append(first['version'])
        ask('in-flight-retains-v2-during-rollback', second['version'], 9000)
        assert client.get('/knowledge/status').json()['active_version'] == first['version']
        ask('request-after-rollback', first['version'], 8000)
        encoder.fail = True
        failed = publish(10000)
        assert failed['state'] == 'failed'
        assert client.get('/knowledge/status').json()['active_version'] == first['version']
        ask('failed-build-keeps-old-version', first['version'], 8000)
    report = {'schema_version': 'synthetic-version-task-v1', 'status': 'passed', 'task_id': task_id,
              'task_count': 1, 'steps_passed': len(steps), 'steps_total': len(steps), 'steps': steps,
              'build_jobs': jobs, 'remote_calls': 0, 'real_model_calls': 0, 'human_reviews': 0,
              'scope': 'synthetic local ASGI version/fact/citation plumbing; controlled encoder and model, not answer benchmark',
              'knowledge_store': 'isolated output directory; real local corpus/versions unchanged',
              'source_sha256': sha256(Path(__file__)),
              'code_snapshot': {p.relative_to(ROOT).as_posix(): sha256(p)
                   for folder in ['app', 'scripts'] for p in sorted((ROOT / folder).rglob('*.py'))}}
    (output / 'version-task.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new private task directory')
    report = run_task(args.output_dir)
    print(json.dumps({k: report[k] for k in ('status', 'task_count', 'steps_passed', 'steps_total', 'remote_calls', 'scope')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
