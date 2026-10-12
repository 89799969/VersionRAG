"""Preserve real failures and a clearly labeled constructed omission example."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_mapping import compact
from app.retrieval import ROOT

reports = ROOT / "artifacts/reports"
batch = json.loads((reports / "answer_replay_dev8.json").read_text(encoding="utf-8"))
row = next(r for r in batch["cases"] if r["id"] == "domainrag_extractive_019" and r["method"] == "hybrid_rerank")
points = row["expected_answer_groups"][0]
support = next(e for e in row["evidence"] if all(point in compact(e["content"]) for point in points))
constructed_answer = "中法学院的办学理念是高水平、国际性。"
missing = [point for point in points if point not in constructed_answer]
demo = {"kind": "controlled_constructed_answer_not_real_model_failure", "id": row["id"],
        "actual_model_answer": row["answer"], "constructed_answer": constructed_answer,
        "required_points": points, "missing_points": missing,
        "complete_evidence_id": support["passage_id"], "evidence_content": support["content"],
        "evidence_truncated": support["truncated"],
        "literal_coverage": (len(points) - len(missing)) / len(points),
        "scope": "actual model answered all points; intentionally constructed omission shows the diagnosis only and is excluded from real metrics"}
assert missing == ["有特色"] and not support["truncated"]
(reports / "generation_omission_demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2), encoding="utf-8")
review = {"batch_generation_code_sha256": batch["generation_code_sha256"],
          "literal_scores_before_date_guard": batch["summaries"],
          "refusal_cases": {"bm25": ["domainrag_extractive_019", "domainrag_extractive_033"], "hybrid_rerank": []},
          "supported_core_keypoint_cases": {"bm25": ["003", "006", "025", "034", "049"],
                                           "hybrid_rerank": ["003", "006", "019", "025", "033", "034", "049"]},
          "review_scope": "assistant reviewed source excerpts for requested core keypoints; not all extra claims or every citation, no learner mastery or independent human annotator claim",
          "date_case_015": {"official_answer": "2014-07-17", "listing_date_adjacent_to_title": "2014-07-16",
                            "problem": "retrieved reference is an article listing, not a statement of the first event date",
                            "external_contemporaneous_report": "https://edu.people.com.cn/n/2014/0715/c367001-25285439.html",
                            "external_report_first_event": "2014-07-15", "external_report_later_public_event": "2014-07-17",
                            "conclusion": "original label and source need correction review; treating the listing date as event date is also unsupported",
                            "policy": "original QA labels and retrieval results unchanged; external article used for audit only, not injected into generation"},
          "scope": "8 selected dev questions; one replay round before date policy fix; literal 5/8 and 7/8 are not general answer accuracy"}
(reports / "answer_replay_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"constructed_omitted_point": missing, "literal_coverage": demo["literal_coverage"],
                  "real_model_omission_claim": False}, ensure_ascii=False))
