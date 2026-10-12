"""Exploratory extra control: same-context regeneration, no retrieval changes."""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.comparison import TitlePlanner
from app.comparison_generation import generate_comparison, finish_draft, Draft, validate_draft
from app.generation import Settings
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256
from scripts import experiment_gap_recovery as trial
from scripts.resume_comparison import judge_payload

MAIN = trial.OUT
CONTROL = ROOT / "artifacts/reports/gap_recovery_2026-10-08/same_context_control"


def main():
    CONTROL.mkdir(exist_ok=True)
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in
        [Path(__file__), ROOT / "app/comparison_generation.py", CORPUS, trial.QA, MAIN / "answers_labeled.jsonl"]}
    policy = CONTROL / "policy.json"
    if policy.exists():
        assert json.loads(policy.read_text(encoding="utf-8"))["immutable_sha256"] == hashes
    else:
        policy.write_text(json.dumps({"immutable_sha256": hashes,
            "reason": "017 changed generation while relevant evidence existed initially; distinguish retrieval from regenerate effect",
            "scope": "exploratory control added after main results, not preplanned confirmation",
            "policy": "same frozen initial hits/prompt/model, one new generation on nine original insufficiency triggers; replay others, retain failure",
            "selection": "all 31 eligible natural questions, not selected by final successes"}, ensure_ascii=False, indent=2), encoding="utf-8")
    firsts = [r for r in read_jsonl(MAIN / "answers_labeled.jsonl")
              if r["cohort"] == "natural_diagnostic" and r["arm"] == "no_recovery"]
    planner = TitlePlanner(read_jsonl(CORPUS))
    trial.OUT = CONTROL
    done = {r["id"] for r in trial.rows("answers.jsonl")}
    settings, transport = Settings.load(), trial.Transport()
    for first in firsts:
        if first["id"] in done:
            continue
        row = copy.deepcopy(first)
        row["arm"] = "same_context_regenerate"
        row["incremental_usage"] = {}
        if first["status"] == "ok" and first["insufficient_evidence"]:
            transport.role = "same_context:" + first["id"]
            start = time.perf_counter()
            try:
                final = generate_comparison(planner.plan(first["question"]), first["hits"], settings=settings, transport=transport)
                row.update(finish_draft(final))
                row["revision_called"] = True
                row["incremental_usage"] = final["usage"]
                row["total_usage"] = {k: first["usage"].get(k, 0) + final["usage"].get(k, 0)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            except Exception as exc:
                row.update({"status": "error", "error_type": type(exc).__name__})
            row["incremental_elapsed_ms"] = (time.perf_counter() - start) * 1000
        trial.append("answers.jsonl", row)
        print("same context", row["id"], row["status"], row.get("revision_called", False), flush=True)
    trial.judge()
    answers, reviews = trial.rows("answers.jsonl"), trial.rows("reviews.jsonl")
    samples = {r["id"]: r for r in read_jsonl(trial.QA)}
    rm = {r["id"]: r for r in reviews}
    assert len(answers) == len(rm) == 31
    for row in answers:
        first = next(r for r in firsts if r["id"] == row["id"])
        assert row["hits"] == first["hits"]
        if row["status"] == "ok":
            plan = planner.plan(row["question"])
            validate_draft(Draft.model_validate({"aspects": row["aspects"], "comparisons": row["comparisons"]}), plan, row["evidence"])
            assert rm[row["id"]]["input_sha256"] == judge_payload(row, samples[row["id"]])[0]
            assert row["evidence"] == first["evidence"]
    summary = []
    for split in ("dev", "holdout", "diagnostic"):
        selected = [r for r in answers if r["split"] == split]
        summary.append({"split": split, "questions": len(selected),
            "success_count": sum(rm[r["id"]]["answer_success"] for r in selected),
            "success_no_partial": sum(rm[r["id"]]["answer_success"] and not r.get("insufficient_evidence", True) for r in selected),
            "revision_calls": sum(r.get("revision_called", False) for r in selected),
            "success_ids": [r["id"] for r in selected if rm[r["id"]]["answer_success"]]})
    trial.save("summary.json", summary)
    trial.save("audit.json", {"status": "passed", "records": len(answers), "checks": ["Frozen initial hits/evidence unchanged", "Judge full input hashes", "31 questions and original failure retained"], "scope": "post hoc same-context control, single stochastic generation"})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
