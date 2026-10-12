"""Compare retrieval methods on the same fixed questions without generating answers."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dense import DenseRetriever
from app.evaluation import evidence_metrics, split_for
from app.hybrid import HybridRetriever
from app.reranking import RerankRetriever
from app.retrieval import ROOT, ChineseBM25, read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "holdout", "demo"], default="dev")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    bm25, dense = ChineseBM25(), DenseRetriever()
    bm25.load()
    dense.load()
    hybrid = HybridRetriever(bm25, dense)
    engines = {"bm25": bm25, "dense": dense, "hybrid": hybrid, "hybrid_rerank": RerankRetriever(hybrid)}
    qa_path = ROOT / "data/basic_qa_mapped.jsonl"
    samples = [sample for sample in read_jsonl(qa_path) if split_for(sample) == args.split]
    summaries = []
    for name, engine in engines.items():
        rows = []
        for sample in samples:
            start = time.perf_counter()
            hits = engine.search(sample["question"], args.top_k)
            elapsed = (time.perf_counter() - start) * 1000
            ids = [str(hit["id"]) for hit in hits]
            rows.append({"id": sample["id"], "question": sample["question"],
                         "retrieved_ids": ids, "channel_ranks": [hit.get("channel_ranks") for hit in hits],
                         "original_ranks": [hit.get("original_rank") for hit in hits],
                         "latency_ms": elapsed, **evidence_metrics(sample, ids, args.top_k)})
        summary = {"method": name, "split": args.split, "questions": len(rows), "top_k": args.top_k,
                   "question_ids": [sample["id"] for sample in samples], "qa_sha256": sha256(qa_path),
                   "evidence_group_recall_at_k": statistics.mean(row["evidence_group_recall"] for row in rows),
                   "all_evidence_hit_rate_at_k": statistics.mean(row["all_evidence_hit"] for row in rows),
                   "mrr_at_k": statistics.mean(row["reciprocal_rank"] for row in rows),
                   "latency_median_ms": statistics.median(row["latency_ms"] for row in rows),
                   "timing_scope": "retrieval includes query encoding API for dense/hybrid; excludes index build/startup",
                   "scope": "annotated evidence metrics, not answer accuracy; equivalent evidence may be unannotated"}
        stem = ROOT / "artifacts/reports" / f"{name}_comparison_{args.split}_k{args.top_k}"
        stem.with_suffix(".jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                                              encoding="utf-8")
        stem.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "question_ids"},
                         ensure_ascii=False, indent=2), flush=True)
    (ROOT / "artifacts/reports" / f"retrieval_comparison_{args.split}_k{args.top_k}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
