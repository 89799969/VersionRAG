"""Inspect development labels and corpus evidence, never holdout examples."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation import split_for
from app.retrieval import CORPUS, ROOT, read_jsonl

cases = json.loads((ROOT / "artifacts/reports/development_comparison_review.json").read_text(encoding="utf-8"))
qa = {row["id"]: row for row in read_jsonl(ROOT / "data/basic_qa_mapped.jsonl")}
corpus = {str(doc["id"]): doc for doc in read_jsonl(CORPUS)}
selected = [cases["selected_loss"]] + cases["rerank_stage_gains"][:1]
output = []
for item in selected:
    sample = qa[item["id"]]
    assert split_for(sample) == "dev"
    ids = [group["acceptable_corpus_passage_ids"][0] for group in sample["required_evidence_groups"]]
    ids += item["rerank_ids"][:2]
    output.append({"id": sample["id"], "question": sample["question"],
                   "expected_answer_groups": sample["expected_answer_groups"],
                   "evidence": [{"id": passage_id, "title": corpus[passage_id]["title"],
                                 "contents": corpus[passage_id]["contents"]} for passage_id in dict.fromkeys(ids)]})
path = ROOT / "artifacts/reports/development_case_evidence.json"
path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(output, ensure_ascii=False, indent=2))
