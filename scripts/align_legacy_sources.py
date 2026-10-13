"""Map old labels to fixed passage coordinates and replay saved retrieval only."""
import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation import split_for
from app.evidence_alignment import EvidenceAnchor, METRIC_VERSION
from app.retrieval import ROOT, CORPUS, read_jsonl, sha256
from app.source_coordinates import SOURCE_SCOPE, align_legacy_case, passage_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.batch_dir / 'manifest.json').read_text(encoding='utf-8'))
    corpus_hash = sha256(CORPUS)
    if baseline['data_sha256']['domainrag_text_corpus.jsonl'] != corpus_hash:
        raise ValueError('Frozen corpus changed')
    output = args.batch_dir / 'legacy-alignment'
    output.mkdir(exist_ok=False)
    version = 'upstream-passages-' + corpus_hash[:16]
    snapshot, documents, chunks, origins = passage_snapshot(read_jsonl(CORPUS), version)
    samples = read_jsonl(ROOT / 'data/basic_qa_mapped.jsonl')
    aligned = [dict(align_legacy_case(sample, documents, origins), split=split_for(sample)) for sample in samples]
    by_id = {row['id']: row for row in aligned}
    with (output / 'labels.jsonl').open('w', encoding='utf-8') as handle:
        for row in aligned:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    # Store each immutable text only once. Loading restores chunk.contents from it.
    source_bundle = {'version': version, 'source_scope': SOURCE_SCOPE, 'corpus_sha256': corpus_hash,
                     'snapshot_sha256': snapshot.sha256, 'documents': documents, 'origins': origins,
                     'chunks': [{'id': c['id'], 'metadata': c['metadata']} for c in chunks]}
    (output / 'sources.json').write_text(json.dumps(source_bundle, ensure_ascii=False), encoding='utf-8')
    with (output / 'human-review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['case_id', 'group', 'original_passage_id', 'source_id', 'document_id', 'start', 'end',
                  'document_sha256', 'quote', 'status', 'reviewer_type', 'reviewer_id', 'reviewed_at']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in aligned:
            for i, group in enumerate(row['required_evidence_groups']):
                for anchor in group:
                    writer.writerow({**{k: anchor[k] for k in fields if k in anchor},
                                     'case_id': row['id'], 'group': i, 'status': 'pending'})
    old_snapshot = json.loads((ROOT / 'artifacts/reports/comparison_snapshot.json').read_text(encoding='utf-8'))
    if old_snapshot['corpus_sha256'] != corpus_hash:
        raise ValueError('Saved historical retrieval belongs to another corpus')
    replay, summaries = [], {}
    for method in ('bm25', 'dense', 'hybrid', 'hybrid_rerank'):
        saved = read_jsonl(ROOT / f'artifacts/reports/{method}_comparison_holdout_k5.jsonl')
        for trial in saved:
            label = by_id[trial['id']]
            if label['mapping_status'] == 'blocked':
                result = {'status': 'needs_review', 'all_evidence_hit': None, 'mapping_issues': label['mapping_issues']}
            else:
                fields = set(EvidenceAnchor.__dataclass_fields__)
                groups = [[EvidenceAnchor(**{k: v for k, v in a.items() if k in fields}) for a in g]
                          for g in label['required_evidence_groups']]
                result = snapshot.score(groups, trial['retrieved_ids'], 5, retrieval_version=version)
            replay.append({'id': trial['id'], 'method': method, 'historical_id_hit': trial['all_evidence_hit'], **result})
        subset = [r for r in replay if r['method'] == method]
        summaries[method] = {'planned': len(subset), 'scored': sum(r['status'] == 'ok' for r in subset),
                             'blocked': sum(r['status'] != 'ok' for r in subset),
                             'span_hits': sum(r['all_evidence_hit'] is True for r in subset),
                             'original_id_hits': sum(r['historical_id_hit'] for r in subset)}
    (output / 'historical-replay.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in replay), encoding='utf-8')
    report = {'scope': 'technical migration and saved historical retrieval replay, not independent human gold or new performance',
              'source_scope': SOURCE_SCOPE, 'metric_version': METRIC_VERSION, 'knowledge_version': version,
              'snapshot_sha256': snapshot.sha256, 'corpus_sha256': corpus_hash,
              'labels_sha256': sha256(output / 'labels.jsonl'), 'questions': len(aligned),
              'mapped': sum(r['mapping_status'] == 'mapped' for r in aligned),
              'blocked_ids': [r['id'] for r in aligned if r['mapping_status'] == 'blocked'],
              'human_confirmed': 0, 'remote_calls': 0, 'historical_replay': summaries}
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
