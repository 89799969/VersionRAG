"""Inspect frozen legacy evidence labels without changing or re-evaluating them."""
import argparse
import hashlib
import json
from pathlib import Path

from app.evaluation import split_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    corpus_path = root / "data/domainrag_text_corpus.jsonl"
    labels_path = root / "data/basic_qa_mapped.jsonl"
    corpus = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    labels = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    questions = [sample for sample in labels if split_for(sample) == "holdout"]
    groups = [group for sample in questions for group in sample["required_evidence_groups"]]
    report_path = root / "artifacts/reports/hybrid_rerank_comparison_holdout_k5.json"
    historical_report = json.loads(report_path.read_text(encoding="utf-8"))
    report = {
        "scope": "readiness audit only; no new retrieval score or label migration",
        "corpus_chunks": len(corpus),
        "chunks_with_source_start_end": sum(
            all(key in chunk.get("metadata", {}) for key in ("start", "end")) for chunk in corpus),
        "historical_question_count": len(questions),
        "historical_required_group_count": len(groups),
        "groups_with_reference_text": sum(bool(group.get("reference_text")) for group in groups),
        "all_questions_single_group_single_id": all(
            len(sample["required_evidence_groups"]) == 1
            and len(sample["required_evidence_groups"][0]["acceptable_corpus_passage_ids"]) == 1
            for sample in questions),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "historical_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "historical_hit_rate": historical_report["all_evidence_hit_rate_at_k"],
        "remaining_work": ["verify canonical source text and coordinate system for legacy corpus",
                           "project existing quotes; review ambiguous and broad labels",
                           "freeze label version; rescore old and new retrieval with same span metric"],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
