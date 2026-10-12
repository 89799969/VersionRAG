"""Reproducible phrase audit of two manually reviewed development cases.

This deliberately checks known claims, not a general semantic answer judge.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_mapping import compact
from app.retrieval import ROOT, read_jsonl, sha256

path = ROOT / "artifacts/reports/development_answer_cases.json"
rows = json.loads(path.read_text(encoding="utf-8"))
claims = {"domainrag_extractive_033": "即日起至2月5日",
          "domainrag_extractive_034": "原则上应为县（县级市）及以下中学"}
reviews = []
for row in rows:
    phrase = claims[row["id"]]
    evidence = row["evidence"]
    supporting = [item["passage_id"] for item in evidence if compact(phrase) in compact(item["content"])]
    citations = row["citations"]
    review = {"id": row["id"], "method": row["method"], "claim_phrase": phrase,
              "supporting_context_ids": supporting,
              "supporting_citation_ids": [pid for pid in citations if pid in supporting],
              "other_citation_ids": [pid for pid in citations if pid not in supporting],
              "answer": row["answer"], "insufficient_evidence": row["insufficient_evidence"]}
    if row["id"].endswith("033") and row["method"] == "bm25":
        assert not supporting and row["insufficient_evidence"]
        review["conclusion"] = "No deadline in supplied context; refusal is appropriate."
    else:
        assert review["supporting_citation_ids"] and not row["insufficient_evidence"]
        assert ("2月5日" if row["id"].endswith("033") else phrase) in row["answer"]
        review["conclusion"] = "Answer claim supported; additional citations need separate per-claim review."
    reviews.append(review)
report = {"input_sha256": sha256(path), "cases": reviews,
          "scope": "two fixed development cases, manually selected claims plus phrase assertions; not batch answer accuracy",
          "annotation_limit": "034 rerank misses annotated 258 but retrieved 13453 supports the same condition",
          "citation_limit": "All citations in these answerable cases contain the checked phrase; ID membership alone still cannot establish semantic support in other cases"}
(ROOT / "artifacts/reports/answer_case_review.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
