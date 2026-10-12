"""Fixed 8-question dev generation comparison on captured retrieval contexts.

This is an optional real-model evaluation, never part of default pytest.
Literal keypoint coverage is separate from later semantic source review.
"""
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_mapping import compact
from app.evaluation import split_for
from app.generation import GenerationError, Settings, generate
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256

SAMPLE_IDS = [f"domainrag_extractive_{number:03d}" for number in (3, 6, 15, 19, 25, 33, 34, 49)]
METHODS = ("bm25", "hybrid_rerank")


def normalize(text):
    value = compact(text).lower()
    return re.sub(r"(\d{4})-(\d{2})-(\d{2})", lambda m: f"{m[1]}年{int(m[2])}月{int(m[3])}日", value)


def literal_coverage(answer, groups):
    # A group is one accepted answer: AND within, alternative groups are OR.
    text = normalize(answer)
    return max(sum(normalize(point) in text for point in group) / len(group) for group in groups)


def main():
    reports = ROOT / "artifacts/reports"
    snapshot = json.loads((reports / "comparison_snapshot.json").read_text(encoding="utf-8"))
    assert sha256(CORPUS) == snapshot["corpus_sha256"]
    assert sha256(ROOT / "data/basic_qa_mapped.jsonl") == snapshot["qa_sha256"]
    assert all(sha256(ROOT / path) == digest for path, digest in snapshot["code_sha256"].items())
    samples = {row["id"]: row for row in read_jsonl(ROOT / "data/basic_qa_mapped.jsonl")}
    corpus = {str(row["id"]): row for row in read_jsonl(CORPUS)}
    captured = {method: {row["id"]: row for row in read_jsonl(reports / f"{method}_comparison_dev_k5.jsonl")}
                for method in METHODS}
    settings, rows = Settings.load(), []
    report = {"sample_ids_fixed_before_calls": SAMPLE_IDS, "split": "dev", "top_k": 5,
              "model": settings.model, "generation_code_sha256": sha256(ROOT / "app/generation.py"),
              "qa_sha256": snapshot["qa_sha256"], "summaries": [], "cases": rows,
              "scope": "8 selected dev questions, one generation round replaying saved actual Top5; no new retrieval timing, no holdout answer accuracy, not isolated rerank benefit",
              "metric": "literal keypoint coverage: all points within one accepted group; max across alternative groups; only whitespace/case and ISO date normalization",
              "limitation": "literal misses paraphrases, ID validation does not establish semantic support; semantic review reported separately"}
    for sample_id in SAMPLE_IDS:
        sample = samples[sample_id]
        assert split_for(sample) == "dev"
        for method in METHODS:
            ids = captured[method][sample_id]["retrieved_ids"]
            hits = [{**corpus[pid], "rank": rank, "score": 0.0} for rank, pid in enumerate(ids, 1)]
            try:
                result = generate(sample["question"], hits, settings=settings)
            except GenerationError as exc:
                result = {"status": "generation_rejected", "error": str(exc), "answer": "",
                          "citations": [], "insufficient_evidence": None, "evidence": []}
            # Labels enter only the evaluator AFTER the normal generation call.
            coverage = literal_coverage(result["answer"], sample["expected_answer_groups"])
            row = {"id": sample_id, "method": method, "question": sample["question"],
                   "retrieved_ids": ids, "expected_answer_groups": sample["expected_answer_groups"],
                   "literal_keypoint_coverage": coverage,
                   "literal_complete_nonrefusal": coverage == 1.0 and result["insufficient_evidence"] is False,
                   **result}
            rows.append(row)
            (reports / "answer_replay_dev8.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: row.get(key) for key in ("id", "method", "answer", "citations",
                                                           "insufficient_evidence", "literal_keypoint_coverage", "error")}, ensure_ascii=False), flush=True)
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        report["summaries"].append({"method": method, "questions": len(selected),
                                   "literal_complete_nonrefusal_count": sum(row["literal_complete_nonrefusal"] for row in selected),
                                   "mean_literal_keypoint_coverage": statistics.mean(row["literal_keypoint_coverage"] for row in selected),
                                   "refusal_count": sum(row["insufficient_evidence"] is True for row in selected),
                                   "generation_rejection_count": sum(row.get("status") == "generation_rejected" for row in selected)})
    (reports / "answer_replay_dev8.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
