"""Synthetic protocol tests; these do not stand in for actual human review."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.review_workflow import prepare_blind_review, apply_reviews
from app.retrieval import sha256


def fixtures():
    rubrics = [{'case_id': 'a', 'label_version': 'synthetic-v1', 'label_status': 'synthetic',
                'expected_action': 'refuse', 'required_facts': []}]
    trials = [{'trial_id': 'a-bm25-1', 'case_id': 'a', 'method': 'bm25', 'question': '教学问题',
               'execution_status': 'ok', 'answer': '资料不足', 'evidence': []},
              {'trial_id': 'a-final-1', 'case_id': 'a', 'method': 'final', 'question': '教学问题',
               'execution_status': 'error', 'answer': None, 'evidence': []}]
    return trials, rubrics


def test_blinding_hides_scheme_and_is_reproducible():
    trials, rubrics = fixtures()
    trials[0].update(citations=['provided-id'], insufficient_evidence=True)
    private, packets, templates = prepare_blind_review(trials, rubrics, 47)
    assert (private, packets, templates) == prepare_blind_review(trials, rubrics, 47)
    assert all('method' not in p and 'trial_id' not in p for p in packets)
    assert all(p['status'] == 'pending' and p['review'] is None for p in templates)
    packet = next(p for p in packets if p['execution_status'] == 'ok')
    assert packet['citations'] == ['provided-id'] and packet['insufficient_evidence'] is True


def test_review_results_retain_original_output_and_metadata_for_grouped_reports():
    trials, rubrics = fixtures()
    trials[0].update(question_type='insufficient', repeat=1, planned_repeats=3,
                     usage={'total_tokens': None}, citations=[], insufficient_evidence=True)
    private, _, templates = prepare_blind_review(trials, rubrics)
    results, summary = apply_reviews(trials, rubrics, private, templates)
    original = next(r for r in results if r['trial_id'] == trials[0]['trial_id'])
    assert original['answer'] == trials[0]['answer'] and original['usage'] == trials[0]['usage']
    assert original['rubric_record'] == rubrics[0] and original['method'] == 'bm25'
    assert summary['unique_questions'] == 1 and 'bm25' in summary['by_method']


def test_missing_reviews_keep_all_planned_trials_incomplete():
    trials, rubrics = fixtures()
    private, _, _ = prepare_blind_review(trials, rubrics)
    results, summary = apply_reviews(trials, rubrics, private, [])
    assert len(results) == summary['planned_trials'] == 2
    assert summary['main_results'] == {'PASS': 0, 'FAIL': 0, 'INCOMPLETE': 2}
    assert summary['strict_main_pass']['value'] is None


def test_changed_trial_or_rubric_cannot_reuse_old_review():
    trials, rubrics = fixtures()
    private, _, submitted = prepare_blind_review(trials, rubrics)
    changed = copy.deepcopy(trials)
    changed[0]['answer'] = '偷偷改回答'
    with pytest.raises(ValueError, match='changed'):
        apply_reviews(changed, rubrics, private, submitted)
    rubrics[0]['expected_action'] = 'clarify'
    with pytest.raises(ValueError, match='changed'):
        apply_reviews(trials, rubrics, private, submitted)


def test_foreign_duplicate_and_incomplete_maps_are_rejected():
    trials, rubrics = fixtures()
    private, _, submitted = prepare_blind_review(trials, rubrics)
    with pytest.raises(ValueError, match='duplicate'):
        apply_reviews(trials, rubrics, private, submitted + [submitted[0]])
    with pytest.raises(ValueError, match='unknown'):
        apply_reviews(trials, rubrics, private, [{**submitted[0], 'blind_id': 'other'}])
    with pytest.raises(ValueError, match='every planned'):
        apply_reviews(trials, rubrics, private[:1], [])


def test_pending_cannot_be_used_to_smuggle_a_completed_review():
    trials, rubrics = fixtures()
    private, _, submitted = prepare_blind_review(trials, rubrics)
    submitted[0]['review'] = {'action_correct': True}
    with pytest.raises(ValueError, match='Pending'):
        apply_reviews(trials, rubrics, private, submitted)


def test_actual_prepare_and_score_commands_keep_pending_and_refuse_overwrite(tmp_path):
    trials, rubrics = fixtures()
    for name, rows in [('trials.jsonl', trials), ('rubrics.jsonl', rubrics)]:
        (tmp_path / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'scripts/review_answer_batch.py'
    common = ['--trials', str(tmp_path / 'trials.jsonl'), '--rubrics', str(tmp_path / 'rubrics.jsonl')]
    prepare = [sys.executable, '-X', 'utf8', str(script), 'prepare', *common,
               '--output-dir', str(tmp_path / 'review')]
    subprocess.run(prepare, check=True, capture_output=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(script), 'score', *common,
                    '--review-dir', str(tmp_path / 'review'), '--submitted', str(tmp_path / 'review/review-template.jsonl'),
                    '--output-dir', str(tmp_path / 'score')], check=True, capture_output=True)
    metrics = json.loads((tmp_path / 'score/metrics.json').read_text(encoding='utf-8'))
    assert metrics['main_results']['INCOMPLETE'] == 2 and metrics['strict_main_pass']['value'] is None
    result = subprocess.run(prepare, capture_output=True)
    assert result.returncode != 0 and b'Output already exists' in result.stderr


def test_interrupted_cli_uses_manifest_bound_full_plan_instead_of_surviving_rows(tmp_path):
    rubric = {'case_id': 'a', 'label_version': 'fixture', 'label_status': 'synthetic',
              'expected_action': 'refuse', 'required_facts': []}
    plan = [{'trial_id': f'a-bm25-{i}', 'case_id': 'a', 'method': 'bm25', 'repeat': i, 'planned_repeats': 3,
             'question': 'fixture', 'question_type': 'insufficient', 'label_status': 'synthetic',
             'execution_status': 'not_run', 'answer': None, 'evidence': [], 'citations': []} for i in (1, 2, 3)]
    def save(name, rows):
        (tmp_path / name).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    save('plan.jsonl', plan)
    save('partial.jsonl', plan[:1])
    save('rubrics.jsonl', [rubric])
    manifest = {'status': 'running', 'planned_trials': 3, 'planned_trials_sha256': sha256(tmp_path / 'plan.jsonl'),
                'rubrics_sha256': sha256(tmp_path / 'rubrics.jsonl')}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'scripts/review_answer_batch.py'
    common = ['--trials', str(tmp_path / 'partial.jsonl'), '--rubrics', str(tmp_path / 'rubrics.jsonl'),
              '--planned-trials', str(tmp_path / 'plan.jsonl'), '--batch-manifest', str(tmp_path / 'manifest.json')]
    subprocess.run([sys.executable, '-X', 'utf8', str(script), 'prepare', *common,
                    '--output-dir', str(tmp_path / 'review')], capture_output=True, check=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(script), 'score', *common, '--review-dir', str(tmp_path / 'review'),
                    '--submitted', str(tmp_path / 'review/review-template.jsonl'), '--output-dir', str(tmp_path / 'score')],
                    capture_output=True, check=True)
    metrics = json.loads((tmp_path / 'score/metrics.json').read_text(encoding='utf-8'))
    results = [json.loads(l) for l in (tmp_path / 'score/answers.jsonl').read_text(encoding='utf-8').splitlines()]
    assert metrics['planned_trials'] == 3 and metrics['unique_questions'] == 1
    assert metrics['strict_main_pass']['unknown'] == 3 and metrics['planned_cohort_verified'] is True
    assert sum(r.get('failure_stage') == 'planned_trial_missing' for r in results) == 2
    changed = [dict(r) for r in plan]
    changed[0]['question'] = 'changed'
    save('plan.jsonl', changed)
    result = subprocess.run([sys.executable, '-X', 'utf8', str(script), 'prepare', *common,
                             '--output-dir', str(tmp_path / 'should-not-create')], capture_output=True)
    assert result.returncode != 0 and b'differ from original batch' in result.stderr
    assert not (tmp_path / 'should-not-create').exists()
