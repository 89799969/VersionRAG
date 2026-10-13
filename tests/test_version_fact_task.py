"""One synthetic workflow, not five independent quality tasks or real users."""
import json

import pytest

from scripts.check_version_fact_task import run_task


def test_publish_fact_change_rollback_pinned_answer_and_failed_build(tmp_path):
    output = tmp_path / 'version-task'
    report = run_task(output)
    assert report['steps_passed'] == report['steps_total'] == 5 and report['task_count'] == 1
    assert report['remote_calls'] == report['real_model_calls'] == 0
    assert report['build_jobs'][-1]['state'] == 'failed'
    rows = [json.loads(line) for line in (output / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    questions = [r for r in rows if r['endpoint'] == '/ask']
    assert len(questions) == 5
    assert {r['task_id'] for r in rows} == {report['task_id']}
    assert all(r['provider_attempt_count'] == 1 and r['model'] == 'synthetic-local-model' for r in questions)
    assert all(r['usage_totals']['total_tokens']['complete_total'] is None for r in questions)
    assert questions[1]['knowledge_version'] == questions[2]['knowledge_version'] != questions[3]['knowledge_version']
    with pytest.raises(FileExistsError):
        run_task(output)
