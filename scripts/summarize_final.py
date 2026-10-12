"""Audit frozen retrieval hashes and combine genuine dev/holdout results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import ROOT, sha256

reports = ROOT / "artifacts/reports"
snapshot = json.loads((reports / "comparison_snapshot.json").read_text(encoding="utf-8"))
assert all(sha256(ROOT / path) == digest for path, digest in snapshot["code_sha256"].items())
assert sha256(ROOT / "data/basic_qa_mapped.jsonl") == snapshot["qa_sha256"]
assert sha256(ROOT / "data/domainrag_text_corpus.jsonl") == snapshot["corpus_sha256"]
dev = json.loads((reports / "retrieval_comparison_dev_k5.json").read_text(encoding="utf-8"))
holdout = json.loads((reports / "retrieval_comparison_holdout_k5.json").read_text(encoding="utf-8"))
for rows in (dev, holdout):
    assert all(row["question_ids"] == rows[0]["question_ids"] for row in rows)
assert not set(dev[0]["question_ids"]) & set(holdout[0]["question_ids"])
report = {"retrieval_code_and_data_unchanged_since_dev_snapshot": True, "parameters": snapshot["parameters"],
          "dev": dev, "holdout": holdout,
          "timing_limit": "single runs, unpaired timings; extended diagnostics ran briefly alongside the holdout run; not load or production latency testing",
          "scope": "35 dev and 53 previously untouched basic QA holdout; 2 demo excluded; annotated evidence coverage, not answer accuracy or significance proof"}
(reports / "final_retrieval_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"frozen_hashes_match": True, "holdout": [{k:r[k] for k in ["method", "questions", "evidence_group_recall_at_k", "mrr_at_k"]} for r in holdout]}, ensure_ascii=False, indent=2))
