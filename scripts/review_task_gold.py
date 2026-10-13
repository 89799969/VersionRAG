"""Import actual task-gold reviews; no inference/model can approve its own gold."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.benchmark_tasks import GoldReview, apply_gold_review
from app.retrieval import read_jsonl, sha256
from app.review_workflow import record_hash


def read_reviews(path, tasks):
    if path.suffix.lower() != '.csv':
        result = []
        for envelope in read_jsonl(path):
            cid = envelope['case_id']
            if cid not in tasks or envelope['task_sha256'] != record_hash(tasks[cid]):
                raise ValueError('Unknown task or changed draft in review envelope')
            if envelope['status'] == 'pending' and envelope.get('review') is None:
                result.append((cid, None))
            elif envelope['status'] == 'submitted':
                result.append((cid, GoldReview.model_validate(envelope['review'])))
            else:
                raise ValueError('Invalid review envelope status')
        return result
    result = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            cid = row['case_id']
            if cid not in tasks or row['task_sha256'] != record_hash(tasks[cid]):
                raise ValueError('Unknown task or changed draft in CSV')
            task = tasks[cid]
            for name in ('question', 'question_type', 'expected_action', 'expected_answer', 'temporal_scope', 'mapping_status'):
                if row[name] != str(task[name]):
                    raise ValueError('Read-only draft fields changed in CSV; revise the task instead')
            if row['provisional_split'] != task['split']:
                raise ValueError('Read-only split changed in CSV')
            verdict = row['status'].strip()
            if verdict == 'pending':
                result.append((cid, None))
                continue
            if verdict not in {'approve', 'revise', 'reject'}:
                raise ValueError('CSV status must be pending/approve/revise/reject')
            def boolean(value):
                if not value.strip():
                    return None
                if value.strip().lower() in {'true', '是'}:
                    return True
                if value.strip().lower() in {'false', '否'}:
                    return False
                raise ValueError('Judgments must be true/false/是/否 or blank')
            judgments = {name: boolean(row[name]) for name in ('question_clear', 'facts_correct', 'evidence_sufficient',
                          'requirements_complete', 'temporal_scope_correct', 'novelty_checked')}
            review = GoldReview(case_id=cid, task_sha256=row['task_sha256'], verdict=verdict,
                                reviewer_type=row['reviewer_type'], reviewer_id=row['reviewer_id'].strip() or None,
                                reviewed_at=row['reviewed_at'].strip() or None, notes=row['notes'], **judgments)
            result.append((cid, review))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft-dir', type=Path, required=True)
    parser.add_argument('--submitted', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new review revision')
    tasks_list = read_jsonl(args.draft_dir / 'labels.jsonl')
    tasks = {t['id']: t for t in tasks_list}
    manifest = json.loads((args.draft_dir / 'manifest.json').read_text(encoding='utf-8'))
    if len(tasks) != len(tasks_list) or manifest['labels_sha256'] != sha256(args.draft_dir / 'labels.jsonl'):
        raise ValueError('Duplicate tasks or changed frozen draft labels')
    submitted = read_reviews(args.submitted, tasks)
    if len({cid for cid, _ in submitted}) != len(submitted):
        raise ValueError('Duplicate review task IDs')
    by_case = dict(submitted)
    reviewed = [apply_gold_review(t, by_case.get(t['id'])) for t in tasks_list]
    acceptance = [t for t in reviewed if t['split'] == 'acceptance']
    acceptance_ready = bool(acceptance) and all(t['human_confirmed'] for t in acceptance) and manifest['unreadable_report_count'] == 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, rows in [('labels.jsonl', reviewed), ('rubrics.jsonl', [t['rubric'] for t in reviewed])]:
        with (args.output_dir / filename).open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {'schema_version': 'task-gold-review-v1', 'draft_labels_sha256': manifest['labels_sha256'],
              'submitted_sha256': sha256(args.submitted), 'planned_tasks': len(reviewed),
              'human_confirmed': sum(t['human_confirmed'] for t in reviewed),
              'review_status_counts': dict(Counter(t['gold_review_status'] for t in reviewed)),
              'acceptance_planned': len(acceptance), 'acceptance_ready_for_later_freeze': acceptance_ready,
              'acceptance_frozen': False, 'remote_calls': 0,
              'provenance_note': 'Reviewer source is declared, not independently authenticated; assistant/model approval never becomes human gold',
              'scope': 'Label confirmation only; not answer grading or measured system quality'}
    (args.output_dir / 'review-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    updated_manifest = {**manifest, 'status': 'gold_review_recorded', 'human_confirmed': report['human_confirmed'],
                        'labels_sha256': sha256(args.output_dir / 'labels.jsonl'),
                        'rubrics_sha256': sha256(args.output_dir / 'rubrics.jsonl'),
                        'acceptance_ready_for_later_freeze': acceptance_ready,
                        'review_summary_sha256': sha256(args.output_dir / 'review-summary.json'), 'acceptance_frozen': False}
    (args.output_dir / 'manifest.json').write_text(json.dumps(updated_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
