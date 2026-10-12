"""Replay captured dev Top5 contexts to separate pipeline gain from rerank gain.

Does not retrieve using labels or inject gold evidence. One matched generation
round uses the existing three captured rankings, question and same adapter.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_mapping import compact
from app.generation import GenerationError, Settings, generate
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256

reports = ROOT / "artifacts/reports"
snapshot = json.loads((reports / "comparison_snapshot.json").read_text(encoding="utf-8"))
assert sha256(CORPUS) == snapshot["corpus_sha256"]
assert all(sha256(ROOT / path) == digest for path, digest in snapshot["code_sha256"].items())
corpus = {str(row["id"]): row for row in read_jsonl(CORPUS)}
settings = Settings.load()
results = []
input_hashes = {}
for method in ("bm25", "hybrid", "hybrid_rerank"):
    input_path = reports / f"{method}_comparison_dev_k5.jsonl"
    input_hashes[method] = sha256(input_path)
    sample = next(row for row in read_jsonl(input_path) if row["id"] == "domainrag_extractive_033")
    hits = [{**corpus[pid], "rank": rank, "score": 0.0}
            for rank, pid in enumerate(sample["retrieved_ids"], 1)]
    try:
        answer = generate(sample["question"], hits, settings=settings)
    except GenerationError as exc:
        answer = {"status": "generation_rejected", "error": str(exc), "evidence": []}
    # Phrase audit comes after generation; not an instruction or input to model.
    support_ids = [item["passage_id"] for item in answer["evidence"]
                   if "即日起至2月5日" in compact(item["content"])]
    row = {"method": method, "retrieved_ids": sample["retrieved_ids"],
           "context_supporting_deadline_ids": support_ids, **answer}
    results.append(row)
    report = {"question_id": sample["id"], "question": sample["question"], "split": "dev",
              "retrieval_replay": "same previously captured Top5 IDs, unchanged corpus, no gold injection",
              "model": settings.model, "generation_code_sha256": sha256(ROOT / "app/generation.py"),
              "input_report_sha256": input_hashes, "results": results,
              "scope": "one matched generation round, historical dev case; not batch accuracy, deterministic proof or independent rerank causal estimate"}
    (reports / "rerank_attribution_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: row.get(key) for key in ("method", "answer", "citations", "insufficient_evidence",
                                                  "context_supporting_deadline_ids", "error")}, ensure_ascii=False), flush=True)
