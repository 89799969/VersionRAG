"""Build, inspect, or evaluate the real corpus BM25 baseline."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation import evidence_metrics, split_for
from app.retrieval import ROOT, ChineseBM25, read_jsonl, tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "query", "evaluate"])
    parser.add_argument("--question", default="中国人民大学中法学院用哪三个语言进行教学？")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", choices=["dev", "holdout", "demo"], default="dev")
    args = parser.parse_args()
    retriever = ChineseBM25()
    if args.command == "build":
        print(json.dumps(retriever.build(), ensure_ascii=False, indent=2))
        return
    retriever.load()
    if args.command == "query":
        hits = retriever.search(args.question, args.top_k)
        print(json.dumps({"question": args.question, "tokens": tokenize(args.question),
                          "hits": [{"id": hit["id"], "source_id": hit["metadata"]["source_id"],
                                    "title": hit["title"], "score": hit["score"],
                                    "excerpt": hit["contents"][:350]} for hit in hits]},
                         ensure_ascii=False, indent=2))
        return
    samples = read_jsonl(ROOT / "data/basic_qa_mapped.jsonl")
    assignments = {sample["id"]: split_for(sample) for sample in samples}
    report_dir = ROOT / "artifacts/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "split_manifest.json").write_text(json.dumps(
        {"rule": "source-scenario-sha256-v2-reserve-exposed-source159",
         "demo_exclusion_reason": "Source159 inspected during selection; not untouched holdout",
         "assignments": assignments}, indent=2), encoding="utf-8")
    records = []
    for sample in samples:
        if assignments[sample["id"]] != args.split:
            continue
        start = time.perf_counter()
        hits = retriever.search(sample["question"], args.top_k)
        elapsed = (time.perf_counter() - start) * 1000
        ids = [str(hit["id"]) for hit in hits]
        records.append({"id": sample["id"], "question": sample["question"],
                        "retrieved_ids": ids, "required_evidence_groups": sample["required_evidence_groups"],
                        "latency_ms": elapsed, **evidence_metrics(sample, ids, args.top_k)})
    if not records:
        raise ValueError("Selected split is empty")
    summary = {
        "method": "bm25-jieba", "split": args.split, "questions": len(records),
        "top_k": args.top_k, "split_counts": {s: list(assignments.values()).count(s) for s in ("dev", "holdout", "demo")},
        "evidence_group_recall_at_k": statistics.mean(row["evidence_group_recall"] for row in records),
        "all_evidence_hit_rate_at_k": statistics.mean(row["all_evidence_hit"] for row in records),
        "mrr_at_k": statistics.mean(row["reciprocal_rank"] for row in records),
        "latency_median_ms": statistics.median(row["latency_ms"] for row in records),
        "scope": "annotated passage evidence only; not answer accuracy; alternative evidence may be unannotated",
    }
    stem = report_dir / f"bm25_{args.split}_k{args.top_k}"
    stem.with_suffix(".jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                             for row in records), encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
