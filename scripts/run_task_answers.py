"""Prepare/execute answer trials from a completed fixed retrieval batch.

No model call without --enable-remote-generation. A dry preparation records
every planned trial as not_run and is not an acceptance execution reservation.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.answer_batch import AnswerBudget, execute_trial, plan_trials, validate_saved_retrieval
from app.benchmark_gates import validate_task_run
from app.evidence_alignment import EvidenceSnapshot
from app.generation import Settings, SYSTEM_PROMPT, generate
from app.methods import METHOD_NAMES
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256
from app.review_workflow import prepare_blind_review, record_hash
from app.telemetry import JsonlTraceSink, RequestTrace, prompt_signature, trace_context, usage_totals


def save_rows(path, rows):
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks-dir', type=Path, required=True)
    parser.add_argument('--retrieval-dir', type=Path, required=True)
    parser.add_argument('--alignment-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'acceptance'], default='dev')
    parser.add_argument('--methods', nargs='+', default=['bm25', 'bm25_rerank'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--allow-pending-dev', action='store_true')
    parser.add_argument('--acceptance-freeze', type=Path)
    parser.add_argument('--enable-remote-generation', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new immutable answer run')
    budget = AnswerBudget(repeats=args.repeats)
    manifest = json.loads((args.tasks_dir / 'manifest.json').read_text(encoding='utf-8'))
    labels_path, rubrics_path = args.tasks_dir / 'labels.jsonl', args.tasks_dir / 'rubrics.jsonl'
    if sha256(labels_path) != manifest['labels_sha256'] or sha256(rubrics_path) != manifest['rubrics_sha256']:
        raise ValueError('Task labels/rubrics differ from manifest')
    freeze = json.loads(args.acceptance_freeze.read_text(encoding='utf-8')) if args.acceptance_freeze else None
    selected = validate_task_run(read_jsonl(labels_path), manifest, args.split,
                                allow_pending_dev=args.allow_pending_dev, freeze=freeze)
    source_path = args.alignment_dir / 'sources.json'
    if sha256(source_path) != manifest['source_bundle_sha256']:
        raise ValueError('Source bundle differs from labels')
    bundle = json.loads(source_path.read_text(encoding='utf-8'))
    if bundle['corpus_sha256'] != sha256(CORPUS):
        raise ValueError('Original corpus changed')
    original = {str(c['id']): c for c in read_jsonl(CORPUS)}
    chunks = [{**c, 'title': original[str(c['id'])]['title'],
               'contents': bundle['documents'][c['metadata']['document_id']]
               [c['metadata']['start']:c['metadata']['end']]} for c in bundle['chunks']]
    snapshot = EvidenceSnapshot(bundle['version'], bundle['documents'], chunks)
    if snapshot.sha256 != manifest['snapshot_sha256']:
        raise ValueError('Source snapshot changed')
    rm = json.loads((args.retrieval_dir / 'manifest.json').read_text(encoding='utf-8'))
    if rm.get('status') != 'complete' or rm.get('retrieval_sha256') != sha256(args.retrieval_dir / 'retrieval.jsonl'):
        raise ValueError('Retrieval batch is incomplete or changed')
    if any(rm.get(k) != value for k, value in {'labels_sha256': sha256(labels_path),
               'snapshot_sha256': snapshot.sha256, 'knowledge_version': snapshot.version,
               'split': args.split, 'case_ids': [t['id'] for t in selected]}.items()):
        raise ValueError('Retrieval batch is from different tasks/split/snapshot')
    rows = read_jsonl(args.retrieval_dir / 'retrieval.jsonl')
    required_methods = list(METHOD_NAMES) if args.split == 'acceptance' else args.methods
    indexed = validate_saved_retrieval(selected, rows, required_methods, chunks, rm['budget']['final_k'])
    code = {p.relative_to(ROOT).as_posix(): sha256(p) for folder in ['app', 'scripts']
            for p in sorted((ROOT / folder).rglob('*.py'))}
    generation_config = {'budget': asdict(budget), 'prompt_sha256': prompt_signature(SYSTEM_PROMPT),
                         'sampling_note': 'temperature fixed; provider weights/randomness not guaranteed',
                         'retrieval_repetition': 'fixed saved hits; repeats only generation'}
    if args.split == 'acceptance':
        if (freeze.get('code_sha256') != code or rm['code_sha256'] != code
                or freeze.get('answer_methods') != args.methods
                or freeze.get('generation_config') != generation_config
                or freeze.get('rubrics_sha256') != sha256(rubrics_path)
                or freeze.get('retrieval_budget') != rm['budget'] or not rm.get('remote_enabled')
                or freeze.get('retrieval_models_indexes') != rm.get('retrieval_models_indexes')
                or any(freeze.get(k) != manifest.get(k) for k in (
                       'labels_sha256', 'source_bundle_sha256', 'snapshot_sha256', 'knowledge_version'))):
            raise ValueError('Acceptance answer configuration differs from freeze')
    # Credential load happens only after all gates, and never during dry preparation.
    settings = Settings.load() if args.enable_remote_generation else None
    if settings and args.split == 'acceptance':
        if freeze.get('generation_model') != settings.model or freeze.get('generation_base_url') != settings.base_url:
            raise ValueError('Generation model/endpoint changed after freezing')
    planned = plan_trials(selected, indexed, args.methods, budget)
    if args.split == 'acceptance' and args.enable_remote_generation:
        with (args.acceptance_freeze.parent / 'acceptance-generation-reservation.json').open('x', encoding='utf-8') as handle:
            json.dump({'status': 'reserved_not_proven_complete', 'output_dir': str(args.output_dir.resolve())}, handle)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_rows(args.output_dir / 'planned-trials.jsonl', planned)
    out_manifest = {'status': 'running' if settings else 'prepared_not_run',
                    'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    'split': args.split, 'case_ids': [t['id'] for t in selected],
                    'methods': args.methods, 'unique_questions': len(selected), 'planned_trials': len(planned),
                    'planned_trials_sha256': sha256(args.output_dir / 'planned-trials.jsonl'),
                    'max_generation_calls': len(planned), 'code_sha256': code,
                    'generation_config': generation_config, 'generation_model': settings.model if settings else None,
                    'labels_sha256': sha256(labels_path), 'rubrics_sha256': sha256(rubrics_path),
                    'retrieval_sha256': sha256(args.retrieval_dir / 'retrieval.jsonl'),
                    'source_bundle_sha256': sha256(source_path), 'snapshot_sha256': snapshot.sha256,
                    'remote_enabled': bool(settings), 'money_cost': None,
                    'scope': 'draft dev diagnostics' if args.allow_pending_dev else 'human-gold answer trials'}
    manifest_path = args.output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    sink, actual = JsonlTraceSink(args.output_dir / 'requests.jsonl'), []
    task_ids = {t['id']: 't-' + uuid4().hex for t in selected}
    if settings:
        with (args.output_dir / 'answers.jsonl').open('x', encoding='utf-8') as handle:
            for trial in planned:
                trace = RequestTrace(task_id=task_ids[trial['case_id']], endpoint='benchmark-answer-generation')
                trace.update(method=trial['method'], knowledge_version=snapshot.version,
                             prompt_sha256=generation_config['prompt_sha256'], model=settings.model)
                with trace_context(trace):
                    result = execute_trial(trial, indexed[(trial['case_id'], trial['method'])], snapshot,
                                 lambda question, hits, **kw: generate(question, hits, settings=settings, **kw), budget)
                trace.update(outcome=result['execution_status'], evidence_ids=[e['passage_id'] for e in result['evidence']],
                             evidence_sha256=record_hash(result['evidence']))
                record = trace.finish()
                sink(record)
                result.update(request_id=record['request_id'], task_id=record['task_id'],
                              provider_attempt_count=record['provider_attempt_count'], usage_totals=record['usage_totals'])
                actual.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
    else:
        actual = planned
        save_rows(args.output_dir / 'answers.jsonl', actual)
    private, packets, templates = prepare_blind_review(actual, read_jsonl(rubrics_path), seed=20261013)
    for name, data in [('private-map.jsonl', private), ('review-packets.jsonl', packets), ('review-template.jsonl', templates)]:
        save_rows(args.output_dir / name, data)
    ledgers = read_jsonl(args.output_dir / 'requests.jsonl') if settings else []
    metrics = {'planned_questions': len(selected), 'planned_trials': len(planned),
               'execution_status_counts': dict(Counter(t['execution_status'] for t in actual)),
               'human_answer_reviews': 0, 'strict_answer_accuracy': None,
               'actual_provider_attempts': sum(r['provider_attempt_count'] for r in ledgers),
               'provider_usage': usage_totals([a for r in ledgers for a in r['provider_attempts']]) if settings else None,
               'money_cost': None, 'scope': 'execution accounting only; unreviewed outputs are not correct answers'}
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    out_manifest.update(status='complete_unreviewed' if settings else 'prepared_not_run',
                        answers_sha256=sha256(args.output_dir / 'answers.jsonl'))
    manifest_path.write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'directory': str(args.output_dir), **metrics}, ensure_ascii=False))


if __name__ == '__main__':
    main()
