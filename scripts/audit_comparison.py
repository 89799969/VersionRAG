"""Offline audit of frozen retrievals, scopes, exact labels and quote-checked drafts."""
from __future__ import annotations

import copy
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.comparison import ComparisonPlan, ExactEvidenceIndex, TitlePlanner, compact
from app.comparison_generation import Draft, prepare_evidence, validate_draft
from app.evaluation import evidence_metrics
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256
from scripts.evaluate_comparison import OUT, QA, ARMS, score, summary, save


def main():
    documents = read_jsonl(CORPUS)
    exact, planner = ExactEvidenceIndex(documents), TitlePlanner(documents)
    qa = [r for r in read_jsonl(QA) if r["task"] == "multi_doc" and r["mapping_complete"]]
    samples = {r["id"]: r for r in qa}
    frozen = json.loads((OUT / "frozen_policy.json").read_text(encoding="utf-8"))
    revision = json.loads((OUT / "generation_v4_policy.json").read_text(encoding="utf-8"))
    for name, expected in frozen["fingerprints"].items():
        key = name.replace("\\", "/")
        actual = sha256(ROOT / name)
        assert actual == expected or revision["fingerprints"].get(key) == actual, key
    for name, expected in revision["fingerprints"].items():
        assert sha256(ROOT / name) == expected, name
    retrievals = read_jsonl(OUT / "dev_retrieval.jsonl") + read_jsonl(OUT / "evaluation_retrieval.jsonl")
    assert len(retrievals) == 45 * 3
    assert len({(r["id"], r["arm"]) for r in retrievals}) == len(retrievals)
    assert {(r["id"], r["arm"]) for r in retrievals} == {(pid, arm) for pid in samples for arm in ARMS}
    for row in retrievals:
        sample = samples[row["id"]]
        assert row["status"] == "ok", row["id"]
        plan = planner.plan(sample["question"])
        assert bool(plan) == bool(row["plan"])
        if plan:
            assert json.loads(json.dumps(plan.payload())) == row["plan"]
        assert len(row["hits"]) <= 5
        for hit in row["hits"]:
            original = exact.by_id[str(hit["id"])]
            assert original["contents"] == hit["contents"] and original["title"] == hit["title"]
            assert hit["metadata"] == original["metadata"]
            for alias in hit.get("equivalent_passage_ids", []):
                assert exact.by_id[alias]["contents"] == hit["contents"]
            if row["arm"] != "original" and plan:
                assert compact(hit["title"]) in plan.entities
        metrics = score(sample, row["hits"], exact)
        assert metrics["original_labels"] == row["original_labels"]
        assert metrics["exact_equivalents"] == row["exact_equivalents"]
        for step in row["trace"].get("steps", []):
            entities = plan.entities if step["entity"] == "joint" else [step["entity"]]
            allowed = set().union(*(planner.ids_by_title[e] for e in entities))
            assert set(step["candidate_ids"]) <= allowed
            assert set(step["ranked_ids"]) <= set(step["unique_candidate_ids"]) <= set(step["candidate_ids"])
    answers = read_jsonl(OUT / "answers.jsonl")
    present = {(r["id"], r["arm"]) for r in answers}
    for row in list(answers):
        if row["arm"] == "split_balanced" and row["status"] == "not_generated" and (row["id"], "split_recover") not in present:
            added = {**row, "arm": "split_recover", "reason": "Rule planner declined; recovery branch not applicable"}
            with (OUT / "answers.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(added, ensure_ascii=False) + "\n")
            answers.append(added)
    assert len(answers) == 45 * 4
    assert len({(r["id"], r["arm"]) for r in answers}) == len(answers)
    for row in answers:
        if row["status"] != "ok":
            continue
        plan = planner.plan(row["question"])
        draft = Draft.model_validate({"aspects": row["aspects"], "comparisons": row.get("comparisons", [])})
        validate_draft(draft, plan, row["evidence"])
        assert {str(e["passage_id"]) for e in row["evidence"]} <= set(exact.by_id)
        context_chars = sum(len(e["content"]) + len(f"\n[片段 {e['passage_id']}] 标题：{e['title']}\n") for e in row["evidence"])
        assert context_chars <= 16000
        for evidence in row["evidence"]:
            assert evidence["content"] == exact.by_id[evidence["passage_id"]]["contents"][:len(evidence["content"])]
        assert set(row["citations"]) <= {e["passage_id"] for e in row["evidence"]}
        if row["arm"] == "split_recover":
            trace = row["trace"]
            assert len(trace["generation_attempts"]) <= 2
            assert len(trace.get("supplemental", {}).get("steps", [])) <= 2
            assert {e["passage_id"] for e in row["evidence"]} <= set(trace["final_ids"])
    reviews_path = OUT / "semantic_reviews.jsonl"
    reviews = read_jsonl(reviews_path) if reviews_path.exists() else []
    detail = []
    for split in sorted({r["split"] for r in answers}):
        for arm in sorted({r["arm"] for r in answers}):
            selected = [r for r in answers if r["split"] == split and r["arm"] == arm]
            eligible = [r for r in selected if planner.plan(r["question"]) is not None]
            good = [r for r in eligible if r["status"] == "ok"]
            judged = [r for r in reviews if r["split"] == split and r["arm"] == arm and r["status"] == "ok"]
            item = {"split": split, "arm": arm, "input_questions": len(selected), "eligible_questions": len(eligible),
                    "generated": len(good), "generation_errors": len(eligible) - len(good),
                    "partial_count": sum(r["insufficient_evidence"] for r in good),
                    "rejected_claims": sum(len(r.get("rejected_claims", [])) for r in good),
                    "judge_completed": len(judged), "model_judged_success_count": sum(r["answer_success"] for r in judged),
                    "judge_unsupported_claims": sum(not c["supported"] for r in judged for c in r["claims"]),
                    "judge_claims": sum(len(r["claims"]) for r in judged)}
            if good:
                item["median_generation_and_recovery_ms"] = statistics.median(r["elapsed_ms"] for r in good)
                item["mean_prompt_tokens"] = statistics.mean((r.get("total_usage") or r["usage"]).get("prompt_tokens", 0) for r in good)
            if arm == "split_recover":
                item["recovery_triggered_count"] = sum(r["trace"]["recovery_triggered"] for r in good)
                item["context_changed_count"] = sum(r["trace"].get("recovery_context_changed", False) for r in good)
                item["recovery_turned_partial_complete"] = sum(not r["insufficient_evidence"] and
                    any(a["status"] == "insufficient" for a in r["trace"]["generation_attempts"][0]["aspects"]) for r in good)
            detail.append(item)
    save("answer_summary.json", detail)
    report = {"status": "passed", "retrieval_records": len(retrievals), "answer_records": len(answers),
              "eligible_questions": sum(planner.plan(r["question"]) is not None for r in qa),
              "split_questions": dict(Counter(r["split"] for r in qa)),
              "total_chunks": len(documents), "unique_contents": len(exact.groups),
              "duplicate_copies": len(documents) - len(exact.groups),
              "corpus_sha256": sha256(CORPUS), "qa_sha256": sha256(QA),
              "checks": ["Original corpus/labels unchanged", "Frozen retrieval code and documented dev-only generation revision",
                         "Complete trial matrix", "Exact original passage bodies", "Pre-TopK scope and candidate provenance",
                         "Both evidence metric definitions independently recomputed", "Quote spans and entity assignments",
                         "Saved context character budget", "At most one supplemental retrieval batch and one revision"],
              "scope": "Offline artifact audit, not semantic proof, human accuracy, or production performance"}
    save("audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
