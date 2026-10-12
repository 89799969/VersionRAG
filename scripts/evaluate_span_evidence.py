"""Offline evidence migration/scoring. No retrieval or paid model calls.

Run as: python -m scripts.evaluate_span_evidence --demo --output-dir PATH
Or replace --demo with --input a frozen JSON bundle in the demonstrated schema.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from app.evidence_alignment import (
    METRIC_VERSION, EvidenceAnchor, EvidenceSnapshot, anchor_quote, text_hash,
)


def evaluate_bundle(bundle: dict, k: int = 5) -> dict:
    if not bundle.get("label_version"):
        raise ValueError("A fixed label version is required.")
    snapshot = EvidenceSnapshot(bundle["knowledge_version"], bundle["documents"], bundle["chunks"])
    questions = bundle["questions"]
    if not questions or len({str(q["id"]) for q in questions}) != len(questions):
        raise ValueError("Questions must have unique IDs and a nonempty fixed denominator.")
    retrieval = bundle["retrieval"]
    if (len({str(row["id"]) for row in retrieval}) != len(retrieval)
            or {str(q["id"]) for q in questions} != {str(row["id"]) for row in retrieval}):
        raise ValueError("Retrieval must contain exactly one result per evaluation question.")
    by_id = {str(row["id"]): row for row in retrieval}
    rows = []
    for question in questions:
        groups = [[EvidenceAnchor(**anchor) for anchor in group]
                  for group in question["required_evidence_groups"]]
        result = by_id[str(question["id"])]
        rows.append({"id": str(question["id"]), **snapshot.score(
            groups, result["retrieved_ids"], k, retrieval_version=result["knowledge_version"])})
    blocked = [row["id"] for row in rows if row["status"] != "ok"]
    hit_count = sum(row["all_evidence_hit"] is True for row in rows)
    return {"metric_version": METRIC_VERSION, "label_version": bundle["label_version"],
            "knowledge_version": snapshot.version, "snapshot_sha256": snapshot.sha256,
            "k": k, "question_count": len(rows), "blocked_question_ids": blocked,
            "status": "needs_review" if blocked else "ok",
            "hit_count_among_scored": hit_count,
            "all_evidence_hit_rate": None if blocked else hit_count / len(rows),
            "rows": rows}


def demo_bundles() -> dict[str, dict]:
    # Teaching fixture; these are not factual claims about an actual programme.
    text = "示例资料：学费六千元。学制四年。"
    labels = [[asdict(anchor_quote("fee", "d", text, "学费六千元。"))],
              [asdict(anchor_quote("years", "d", text, "学制四年。"))]]
    boundary = text.index("学制")

    def make(version, contents, spans, selected):
        return {"label_version": "teaching-facts-v1", "knowledge_version": version,
                "documents": {"d": contents},
                "chunks": [{"id": name, "contents": contents[start:end],
                            "metadata": {"document_id": "d", "start": start, "end": end}}
                           for name, start, end in spans],
                "questions": [{"id": "demo", "question": "示例专业的学费和学制？",
                               "required_evidence_groups": labels}],
                "retrieval": [{"id": "demo", "knowledge_version": version,
                               "retrieved_ids": selected}]}

    spans = [("801", 0, boundary), ("802", boundary, len(text))]
    return {
        "old_complete": make("old", text, [("125", 0, len(text))], ["125"]),
        "new_partial": make("new", text, spans, ["801"]),
        "new_complete": make("new", text, spans, ["801", "802"]),
        "changed_text": make("changed", text.replace("六千", "八千"), spans, ["801", "802"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    bundles = demo_bundles() if args.demo else {
        "evaluation": json.loads(args.input.read_text(encoding="utf-8"))}
    reports = {name: evaluate_bundle(bundle, args.k) for name, bundle in bundles.items()}
    # Never overwrite another evaluation or any original benchmark artifact.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, bundle in bundles.items():
        encoded = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
        reports[name]["input_sha256"] = text_hash(encoded)
        (args.output_dir / f"{name}.input.json").write_text(encoded, encoding="utf-8")
        (args.output_dir / f"{name}.result.json").write_text(
            json.dumps(reports[name], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: report[key] for key in
                            ("status", "question_count", "all_evidence_hit_rate")}
                      for name, report in reports.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
