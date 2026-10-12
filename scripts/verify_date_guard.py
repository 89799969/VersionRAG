"""Real regression probe after an observed publication/event-date confusion."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.generation import GenerationError, generate
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256

reports = ROOT / "artifacts/reports"
before = json.loads((reports / "answer_replay_dev8.json").read_text(encoding="utf-8"))
corpus = {str(row["id"]): row for row in read_jsonl(CORPUS)}
cases = []
for previous in before["cases"]:
    if previous["id"] != "domainrag_extractive_015":
        continue
    hits = [{**corpus[pid], "rank": rank, "score": 0.0} for rank, pid in enumerate(previous["retrieved_ids"], 1)]
    try:
        result = generate(previous["question"], hits)
    except GenerationError as exc:
        result = {"status": "generation_rejected", "error": str(exc), "insufficient_evidence": None}
    row = {"id": previous["id"], "method": previous["method"], "before_answer": previous["answer"],
           "before_insufficient_evidence": previous["insufficient_evidence"], **result}
    cases.append(row)
    report = {"generation_code_sha256": sha256(ROOT / "app/generation.py"), "cases": cases,
              "input_source": "unchanged captured Top5 from answer_replay_dev8.json; no external article or gold injected",
              "scope": "one development question, two real model requests after prompt clarification; not a deterministic prevention guarantee or full task accuracy"}
    (reports / "date_guard_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: row.get(k) for k in ["method", "answer", "insufficient_evidence", "error"]}, ensure_ascii=False), flush=True)
assert all(row["insufficient_evidence"] is True for row in cases), "Unclear event date should be marked insufficient"
