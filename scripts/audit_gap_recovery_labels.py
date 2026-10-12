"""Repair ONLY replay metadata in separate views, preserving measured records.

finish_draft(saved_first) preserved the saved arm name in skipped revisions.
The serial append order and recorded merge trace determine intended arms.
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import read_jsonl, sha256
from scripts import experiment_gap_recovery as trial
from scripts.resume_comparison import judge_payload


def main():
    out = trial.OUT
    original = read_jsonl(out / "answers.jsonl")
    reviews = read_jsonl(out / "reviews.jsonl")
    samples = {r["id"]: r for r in read_jsonl(trial.QA)}
    assert len(original) == len(reviews) == 180
    corrected, corrected_reviews, changes = [], [], []
    for index, (row, review) in enumerate(zip(original, reviews)):
        expected = trial.ARMS[index % len(trial.ARMS)]
        assert (row["cohort"], row["id"], row["arm"]) == (review["cohort"], review["id"], review["arm"])
        if index % 5:
            assert (row["cohort"], row["id"]) == (original[index - index % 5]["cohort"], original[index - index % 5]["id"])
        fixed, fixed_review = copy.deepcopy(row), copy.deepcopy(review)
        if row["arm"] != expected:
            assert row["arm"] == "split_balanced" and row["revision_called"] is False
            assert row["cohort"] == "natural_diagnostic"
            assert bool(row.get("merge_trace")) == expected.endswith("joint")
            changes.append({"source_line": index + 1, "id": row["id"], "old_arm": row["arm"], "correct_arm": expected})
            fixed["arm"] = fixed_review["arm"] = expected
        if row["status"] == "ok":
            before, _ = judge_payload(row, samples[row["id"]])
            after, _ = judge_payload(fixed, samples[row["id"]])
            assert before == after == review["input_sha256"]
            baseline = original[index - index % 5]
            anchors = trial.anchored_hashes(baseline)
            actual = {trial.digest(e["content"]) for e in row["evidence"]}
            fixed["old_cited_total"] = len(anchors)
            fixed["old_cited_retained"] = len(anchors & actual)
        corrected.append(fixed)
        corrected_reviews.append(fixed_review)
    for name, values in (("answers_labeled.jsonl", corrected), ("reviews_labeled.jsonl", corrected_reviews)):
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")
    trial.save("label_repair.json", {"status": "passed", "changes": changes,
        "original_answers_sha256": sha256(out / "answers.jsonl"),
        "original_reviews_sha256": sha256(out / "reviews.jsonl"),
        "checks": ["Only replayed arm labels changed", "Serial five-arm append order checked against merge traces",
                   "Every semantic judge input hash unchanged", "Original measurement files preserved"],
        "note": "Joint skipped-revision selected_ids can differ in order/aliases from actual context; evidence array is authoritative"})
    original_rows = trial.rows
    trial.rows = lambda name: read_jsonl(out / {"answers.jsonl": "answers_labeled.jsonl", "reviews.jsonl": "reviews_labeled.jsonl"}[name]) if name in ("answers.jsonl", "reviews.jsonl") else original_rows(name)
    trial.audit()
    print("Replay label repairs:", len(changes))


if __name__ == "__main__":
    main()
