"""Run a new, immutable five-method record; default mode makes no API calls."""
import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.batch_retrieval import RetrievalBudget, retrieve_pipelines, score_pipeline, summarize_retrieval
from app.evidence_alignment import EvidenceSnapshot, METRIC_VERSION
from app.retrieval import CORPUS, ROOT, ChineseBM25, read_jsonl, sha256
from app.telemetry import JsonlTraceSink, RequestTrace, trace_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'holdout', 'demo'], default='dev')
    # Live execution deliberately requires a per-run explicit flag. Permission
    # to use this flag must come from the human budget, not the saved manifest.
    parser.add_argument('--enable-remote-retrieval', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output already exists; choose a new run directory')
    bundle_path = args.alignment_dir / 'sources.json'
    labels_path = args.alignment_dir / 'labels.jsonl'
    bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
    if bundle['corpus_sha256'] != sha256(CORPUS):
        raise ValueError('Corpus differs from the fixed source bundle')
    chunks = [{**c, 'contents': bundle['documents'][c['metadata']['document_id']]
               [c['metadata']['start']:c['metadata']['end']]} for c in bundle['chunks']]
    snapshot = EvidenceSnapshot(bundle['version'], bundle['documents'], chunks)
    if snapshot.sha256 != bundle['snapshot_sha256']:
        raise ValueError('Source coordinate bundle checksum differs')
    labels = [r for r in read_jsonl(labels_path) if r['split'] == args.split]
    if not labels or len({r['id'] for r in labels}) != len(labels):
        raise ValueError('Selected split is empty or has duplicate case IDs')
    bm25 = ChineseBM25()
    bm25.load()
    dense = reranker = None
    if args.enable_remote_retrieval:
        from app.dense import DenseRetriever
        from app.reranking import APIReranker
        dense, reranker = DenseRetriever(), APIReranker()
        dense.load()  # checksum/model check before any question call
    budget = RetrievalBudget()
    code = {p.relative_to(ROOT).as_posix(): sha256(p)
            for folder in ['app', 'scripts'] for p in sorted((ROOT / folder).rglob('*.py'))}
    manifest = {'schema_version': 'retrieval-batch-v1', 'metric_version': METRIC_VERSION,
                'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'head_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'code_sha256': code, 'python': platform.python_version(), 'source_scope': bundle['source_scope'],
                'knowledge_version': snapshot.version, 'snapshot_sha256': snapshot.sha256,
                'corpus_sha256': sha256(CORPUS), 'labels_sha256': sha256(labels_path),
                'budget': asdict(budget), 'split': args.split, 'case_ids': [r['id'] for r in labels],
                'scope': 'old exposed questions, historical regression; not new unseen acceptance',
                'remote_enabled': args.enable_remote_retrieval,
                'bm25_index_manifest': json.loads((bm25.index_path / 'manifest.json').read_text(encoding='utf-8')),
                'dense_index_manifest': (json.loads((dense.index_path / 'manifest.json').read_text(encoding='utf-8')) if dense else None),
                'reranker_model': reranker.model if reranker else None,
                'money_cost': None, 'provider_usage': None,
                'timing_scope': 'channels shared per question; not five independent service requests'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    attempt_sink = JsonlTraceSink(args.output_dir / 'requests.jsonl')
    # Append per question so a failed local process still leaves completed rows;
    # never rerun or overwrite an acceptance batch under the same directory.
    with (args.output_dir / 'retrieval.jsonl').open('x', encoding='utf-8') as handle:
        for label in labels:
            ledger = RequestTrace(endpoint='retrieval-experiment')
            ledger.update(knowledge_version=snapshot.version, outcome='retrieval_experiment')
            with trace_context(ledger):
                pipelines = retrieve_pipelines(label['question'], bm25, dense=dense, reranker=reranker, budget=budget)
            # One accounting record for shared physical calls, not five copies.
            attempt_sink(ledger.finish())
            for pipeline in pipelines.values():
                row = score_pipeline(label, pipeline, snapshot, budget)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
    with (args.output_dir / 'retrieval.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        fields = ['case_id', 'method', 'label_status', 'execution_status', 'candidate_count', 'final_count',
                  'candidate_all_hit', 'final_all_hit', 'failure_stage', 'candidate_ids', 'final_ids',
                  'channel_counts', 'channel_elapsed_ms', 'rerank_elapsed_ms']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            candidates, final = r['candidate_pool'], r['final']
            writer.writerow({'case_id': r['case_id'], 'method': r['method'], 'label_status': r['label_status'],
                'execution_status': final['execution_status'], 'failure_stage': r['failure_stage'],
                'candidate_count': len(candidates['hits']) if candidates['hits'] is not None else None,
                'final_count': len(final['hits']) if final['hits'] is not None else None,
                'candidate_all_hit': r['candidate_score']['all_evidence_hit'] if r['candidate_score'] else None,
                'final_all_hit': r['final_score']['all_evidence_hit'] if r['final_score'] else None,
                'candidate_ids': json.dumps([h['id'] for h in candidates['hits']]) if candidates['hits'] is not None else '',
                'final_ids': json.dumps([h['id'] for h in final['hits']]) if final['hits'] is not None else '',
                'channel_counts': json.dumps(r['channel_counts']), 'channel_elapsed_ms': json.dumps(r['channel_elapsed_ms']),
                'rerank_elapsed_ms': final['elapsed_ms'] if r['method'].endswith('_rerank') else None})
    summary = summarize_retrieval(rows)
    summary['remote_enabled'] = args.enable_remote_retrieval
    (args.output_dir / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output_dir), 'methods': summary['methods'],
                      'remote_enabled': args.enable_remote_retrieval}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
