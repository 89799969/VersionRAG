"""Finish control audit including original failed rows without saved hits."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.comparison import TitlePlanner
from app.comparison_generation import Draft, validate_draft
from app.retrieval import ROOT, CORPUS, read_jsonl, sha256
from scripts.resume_comparison import judge_payload

OUT = ROOT / "artifacts/reports/gap_recovery_2026-10-08"
C = OUT / "same_context_control"


def main():
    policy = json.loads((C / "policy.json").read_text(encoding="utf-8"))
    # The main view later gained derived anchor-retention counters. Recreate
    # the exact earlier label-only view for this control's frozen input hash.
    snapshot = []
    arms = ("no_recovery", "broad_priority", "targeted_priority", "broad_joint", "targeted_joint")
    for index, row in enumerate(read_jsonl(OUT / "answers.jsonl")):
        row["arm"] = arms[index % 5]
        snapshot.append(row)
    snapshot_path = C / "input_answers_labeled.jsonl"
    snapshot_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in snapshot), encoding="utf-8")
    for name, expected in policy["immutable_sha256"].items():
        path = snapshot_path if name.endswith("gap_recovery_2026-10-08/answers_labeled.jsonl") else ROOT / name
        assert sha256(path) == expected
    firsts = {r["id"]: r for r in snapshot
              if r["cohort"] == "natural_diagnostic" and r["arm"] == "no_recovery"}
    answers, reviews = read_jsonl(C / "answers.jsonl"), read_jsonl(C / "reviews.jsonl")
    rm = {r["id"]: r for r in reviews}
    planner = TitlePlanner(read_jsonl(CORPUS))
    samples = {r["id"]: r for r in read_jsonl(ROOT / "data/extended_qa_mapped.jsonl")}
    assert len(answers) == len({r["id"] for r in answers}) == len(rm) == 31
    for row in answers:
        first = firsts[row["id"]]
        assert row.get("hits") == first.get("hits")
        if row["status"] == "ok":
            validate_draft(Draft.model_validate({"aspects": row["aspects"], "comparisons": row["comparisons"]}),
                           planner.plan(row["question"]), row["evidence"])
            assert row["evidence"] == first["evidence"]
            assert rm[row["id"]]["input_sha256"] == judge_payload(row, samples[row["id"]])[0]
            review = rm[row["id"]]
            assert review["answer_success"] == (bool(review["claims"]) and review["facts_correct"]
                and review["request_complete"] and all(c["supported"] for c in review["claims"]))
        else:
            assert rm[row["id"]]["status"] == "not_judged"
    baseline_reviews = {r["id"]: r for r in read_jsonl(OUT / "reviews_labeled.jsonl")
                        if r["cohort"] == "natural_diagnostic" and r["arm"] == "no_recovery"}
    summary = []
    for split in ("dev", "holdout", "diagnostic"):
        selected = [r for r in answers if r["split"] == split]
        summary.append({"split": split, "questions": len(selected),
            "success_count": sum(rm[r["id"]]["answer_success"] for r in selected),
            "success_no_partial": sum(rm[r["id"]]["answer_success"] and not r.get("insufficient_evidence", True) for r in selected),
            "errors": sum(r["status"] != "ok" for r in selected),
            "gains": [r["id"] for r in selected if rm[r["id"]]["answer_success"] and not baseline_reviews[r["id"]]["answer_success"]],
            "regressions": [r["id"] for r in selected if not rm[r["id"]]["answer_success"] and baseline_reviews[r["id"]]["answer_success"]]})
    (C / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (C / "audit.json").write_text(json.dumps({"status": "passed", "records": 31,
        "checks": ["Frozen experiment script/input hashes unchanged", "Identical initial hits and actual evidence",
                   "Judge payload hashes and success recomputed", "Original failure without hits retained", "Original label-only input snapshot hash recovered"],
        "audit_note": "Original control audit assumed every row has hits; separate audit accepts explicitly retained baseline failure",
        "scope": "post hoc same-context control; stochastic single generation, not independent confirmation"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
