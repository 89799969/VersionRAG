"""Audit old splits without changing them; save grouping input for new tasks."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation import split_for
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256
from app.source_families import build_source_families, audit_questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new audit directory')
    families = build_source_families(read_jsonl(CORPUS))
    labels = read_jsonl(ROOT / 'data/basic_qa_mapped.jsonl')
    cases = [{'id': c['id'], 'question': c['question'], 'split': split_for(c), 'previously_exposed': True,
              'source_ids': sorted({str(g['source_id']) for g in c['required_evidence_groups']})} for c in labels]
    audit = audit_questions(cases, families)
    summary = {'corpus_sha256': sha256(CORPUS), 'labels_sha256': sha256(ROOT / 'data/basic_qa_mapped.jsonl'),
               'source_count': len(families['source_to_family']), 'family_count': len(families['families']),
               'cross_source_duplicate_text_groups': len(families['duplicate_links']),
               'question_count': len(cases), 'cross_split_risk_pairs': sum(l['cross_split'] for l in audit['links']),
               'near_question_warning_pairs': sum(l['near_question_warning'] for l in audit['links']),
               'unknown_sources': audit['unknown_sources'], 'remote_calls': 0,
               'scope': 'risk audit, not proven semantic leakage; old labels/splits remain unchanged',
               'new_split_policy': 'components join shared source families and near-question warnings; exposed components are regression'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in [('source-families.json', families), ('question-audit.json', audit), ('summary.json', summary)]:
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
