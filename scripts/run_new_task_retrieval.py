"""New-task retrieval diagnostics; pending acceptance is rejected before calls."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.batch_retrieval import RetrievalBudget, retrieve_pipelines, score_pipeline, summarize_retrieval
from app.benchmark_gates import validate_task_run
from app.evidence_alignment import EvidenceSnapshot
from app.retrieval import ROOT, CORPUS, ChineseBM25, read_jsonl, sha256
from app.telemetry import JsonlTraceSink, RequestTrace, trace_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks-dir', type=Path, required=True)
    parser.add_argument('--alignment-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'acceptance'], default='dev')
    parser.add_argument('--allow-pending-dev', action='store_true')
    parser.add_argument('--enable-remote-retrieval', action='store_true')
    parser.add_argument('--acceptance-freeze', type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new immutable dev run directory')
    manifest = json.loads((args.tasks_dir / 'manifest.json').read_text(encoding='utf-8'))
    labels_path = args.tasks_dir / 'labels.jsonl'
    if manifest['labels_sha256'] != sha256(labels_path):
        raise ValueError('Task labels differ from their manifest')
    freeze = json.loads(args.acceptance_freeze.read_text(encoding='utf-8')) if args.acceptance_freeze else None
    selected = validate_task_run(read_jsonl(labels_path), manifest, args.split,
                                allow_pending_dev=args.allow_pending_dev, freeze=freeze)
    source_path = args.alignment_dir / 'sources.json'
    if manifest['source_bundle_sha256'] != sha256(source_path):
        raise ValueError('Source bundle differs from task preparation')
    bundle = json.loads(source_path.read_text(encoding='utf-8'))
    if bundle['corpus_sha256'] != sha256(CORPUS):
        raise ValueError('Original corpus changed')
    chunks = [{**c, 'contents': bundle['documents'][c['metadata']['document_id']]
                [c['metadata']['start']:c['metadata']['end']]} for c in bundle['chunks']]
    snapshot = EvidenceSnapshot(bundle['version'], bundle['documents'], chunks)
    if snapshot.sha256 != manifest['snapshot_sha256']:
        raise ValueError('Source coordinates changed')
    budget = RetrievalBudget()
    code = {p.relative_to(ROOT).as_posix(): sha256(p) for folder in ['app', 'scripts'] for p in sorted((ROOT / folder).rglob('*.py'))}
    if freeze:
        if freeze.get('retrieval_budget') != asdict(budget) or freeze.get('code_sha256') != code:
            raise ValueError('Configuration/code differs from frozen acceptance')
        if any(freeze.get(key) != manifest.get(key) for key in (
                'labels_sha256', 'source_bundle_sha256', 'snapshot_sha256', 'knowledge_version')):
            raise ValueError('Sources/labels differ from frozen acceptance')
    bm25 = ChineseBM25()
    bm25.load()
    dense = reranker = None
    if args.enable_remote_retrieval:
        from app.dense import DenseRetriever
        from app.reranking import APIReranker
        dense, reranker = DenseRetriever(), APIReranker()
        dense.load()
    models_indexes = {
        'bm25_manifest_sha256': sha256(bm25.index_path / 'manifest.json'),
        'dense_manifest_sha256': sha256(dense.index_path / 'manifest.json') if dense else None,
        'reranker': reranker.model if reranker else None,
        'provider_weight_revision': 'not_exposed_by_provider' if dense else None}
    if freeze and freeze.get('retrieval_models_indexes') != models_indexes:
        raise ValueError('Retrieval models/indexes differ from acceptance freeze')
    if args.split == 'acceptance':
        if not args.enable_remote_retrieval:
            raise ValueError('Final acceptance requires all five schemes, not an offline subset')
        # Reserve only after all local validations. A lock is not a live process
        # handle and never means a crashed/interrupted evaluation is complete.
        with (args.acceptance_freeze.parent / 'acceptance-retrieval-reservation.json').open('x', encoding='utf-8') as handle:
            json.dump({'output_dir': str(args.output_dir.resolve()), 'status': 'reserved_not_proven_complete'}, handle)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_manifest = {'scope': 'draft dev diagnostics; not human accuracy' if args.allow_pending_dev else 'human-gold task retrieval',
                    'split': args.split, 'case_ids': [r['id'] for r in selected], 'labels_sha256': sha256(labels_path),
                    'snapshot_sha256': snapshot.sha256, 'knowledge_version': snapshot.version,
                    'code_sha256': code, 'budget': asdict(budget), 'remote_enabled': args.enable_remote_retrieval,
                    'retrieval_models_indexes': models_indexes,
                    'acceptance_frozen': args.split == 'acceptance', 'money_cost': None}
    (args.output_dir / 'manifest.json').write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    sink, rows = JsonlTraceSink(args.output_dir / 'requests.jsonl'), []
    with (args.output_dir / 'retrieval.jsonl').open('x', encoding='utf-8') as handle:
        for case in selected:
            ledger = RequestTrace(endpoint='new-task-retrieval-experiment')
            ledger.update(knowledge_version=snapshot.version, outcome='retrieval_experiment')
            with trace_context(ledger):
                schemes = retrieve_pipelines(case['question'], bm25, dense=dense, reranker=reranker, budget=budget)
            sink(ledger.finish())
            for scheme in schemes.values():
                row = score_pipeline(case, scheme, snapshot, budget)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
    targets = sorted({r['retrieval_target'] for r in rows})
    summary = {'scope': run_manifest['scope'], 'planned_cases': len(selected),
               'question_types': dict(Counter(r['question_type'] for r in selected)),
               'by_retrieval_target': {target: summarize_retrieval([r for r in rows if r['retrieval_target'] == target]) for target in targets},
               'note': 'Insufficiency-context retrieval is diagnostic, not recall of a missing answer; label review and answer grading are separate',
               'human_main_answer_accuracy': None, 'remote_enabled': args.enable_remote_retrieval}
    (args.output_dir / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    run_manifest.update(status='complete', retrieval_sha256=sha256(args.output_dir / 'retrieval.jsonl'),
                        metrics_sha256=sha256(args.output_dir / 'metrics.json'))
    (args.output_dir / 'manifest.json').write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output_dir), 'planned_cases': len(selected), 'split': args.split,
                      'scope': summary['scope'], 'targets': targets, 'remote_enabled': args.enable_remote_retrieval}, ensure_ascii=False))


if __name__ == '__main__':
    main()
