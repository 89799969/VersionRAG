"""Create a paired development comparison and select real improvements/regressions."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import ROOT, read_jsonl, sha256

REPORTS = ROOT / "artifacts/reports"
METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def main():
    summaries, detailed = {}, {}
    for method in METHODS:
        summaries[method] = json.loads((REPORTS / f"{method}_comparison_dev_k5.json").read_text(encoding="utf-8"))
        detailed[method] = {row["id"]: row for row in read_jsonl(REPORTS / f"{method}_comparison_dev_k5.jsonl")}
    expected = summaries["bm25"]["question_ids"]
    assert all(summaries[method]["question_ids"] == expected for method in METHODS)
    samples = {row["id"]: row for row in read_jsonl(ROOT / "data/basic_qa_mapped.jsonl")}
    paired = []
    for sample_id in expected:
        baseline = detailed["bm25"][sample_id]
        reranked = detailed["hybrid_rerank"][sample_id]
        paired.append({"id": sample_id, "question": baseline["question"],
                       "bm25_hit": baseline["all_evidence_hit"], "rerank_hit": reranked["all_evidence_hit"],
                       "mrr_delta": reranked["reciprocal_rank"] - baseline["reciprocal_rank"],
                       "bm25_ids": baseline["retrieved_ids"], "rerank_ids": reranked["retrieved_ids"],
                       "rerank_original_ranks": reranked["original_ranks"],
                       "acceptable_evidence_ids": [group["acceptable_corpus_passage_ids"]
                                                   for group in samples[sample_id]["required_evidence_groups"]]})
    gains = sorted([row for row in paired if not row["bm25_hit"] and row["rerank_hit"]],
                   key=lambda row: (-row["mrr_delta"], row["id"]))
    losses = sorted([row for row in paired if row["bm25_hit"] and not row["rerank_hit"]],
                    key=lambda row: (row["mrr_delta"], row["id"]))
    rerank_gains = []
    for sample_id in expected:
        before, after = detailed["hybrid"][sample_id], detailed["hybrid_rerank"][sample_id]
        if not before["all_evidence_hit"] and after["all_evidence_hit"]:
            rerank_gains.append({"id": sample_id, "question": after["question"],
                                 "hybrid_ids": before["retrieved_ids"], "rerank_ids": after["retrieved_ids"],
                                 "original_ranks": after["original_ranks"],
                                 "acceptable_evidence_ids": [group["acceptable_corpus_passage_ids"]
                                                             for group in samples[sample_id]["required_evidence_groups"]]})
    report = {"split": "dev", "questions": len(expected), "top_k": 5,
              "summaries": list(summaries.values()), "gained_annotated_hits": len(gains),
              "lost_annotated_hits": len(losses), "paired_cases": paired,
              "selected_gain": gains[0] if gains else None, "selected_loss": losses[0] if losses else None,
              "rerank_stage_gains": rerank_gains,
              "scope": "single development run; not answer accuracy, not untouched holdout, not significant-effect proof"}
    (REPORTS / "development_comparison_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot = {"corpus_sha256": sha256(ROOT / "data/domainrag_text_corpus.jsonl"),
                "qa_sha256": sha256(ROOT / "data/basic_qa_mapped.jsonl"),
                "dense_index": json.loads((ROOT / "artifacts/indexes/dense/manifest.json").read_text(encoding="utf-8")),
                "bm25_index": json.loads((ROOT / "artifacts/indexes/bm25/manifest.json").read_text(encoding="utf-8")),
                "parameters": {"top_k": 5, "candidate_k_per_channel": 30, "rrf_constant": 60,
                               "rerank_candidate_k": 30, "rerank_model": "BAAI/bge-reranker-v2-m3"},
                "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                                [ROOT / "app" / name for name in ["retrieval.py", "dense.py", "embedding_api.py",
                                                                  "hybrid.py", "reranking.py", "evaluation.py"]]},
                "note": "Captured after this development run, before any parameter tuning; provider weight revisions are not exposed."}
    (REPORTS / "comparison_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["questions", "gained_annotated_hits", "lost_annotated_hits",
                                                 "selected_gain", "selected_loss", "rerank_stage_gains"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
