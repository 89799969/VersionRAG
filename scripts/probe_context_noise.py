"""One official development question; controlled oracle-context ablation.

Uses gold explicitly for diagnosis, not for normal retrieval or score claims.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.generation import GenerationError, assemble_context, generate
from app.retrieval import CORPUS, ROOT, read_jsonl

sample = next(r for r in read_jsonl(ROOT / "data/extended_qa_mapped.jsonl") if r["id"] == "domainrag_noise_033")
assert sample["split"] == "dev"
corpus = {str(r["id"]): r for r in read_jsonl(CORPUS)}
positive_id = sample["required_evidence_groups"][0]["acceptable_corpus_passage_ids"][0]
negative_ids = [g["acceptable_corpus_passage_ids"][0] for g in sample["noise_evidence_groups"][:3]]
noisy_ids = negative_ids + [positive_id]
noisy_hits = [{**corpus[pid], "score": 1.0, "rank": rank} for rank, pid in enumerate(noisy_ids, 1)]
_, clipped = assemble_context(noisy_hits, max_chars=400)
clipped_hits = [{**corpus[item["passage_id"]], "contents": item["content"], "score": 1.0, "rank": rank}
                for rank, item in enumerate(clipped, 1)]
assert positive_id not in {item["passage_id"] for item in clipped}
settings = {"oracle_clean": [corpus[positive_id]], "oracle_plus_3_official_distractors": noisy_hits,
            "controlled_budget_400_positive_removed": clipped_hits}
results = []
for name, hits in settings.items():
    try:
        result = generate(sample["question"], hits)
    except GenerationError as exc:
        result = {"status": "generation_rejected", "error": str(exc)}
    row = {"variant": name, "input_ids": [str(hit["id"]) for hit in hits],
           "positive_in_final_context": positive_id in {str(hit["id"]) for hit in hits}, **result}
    results.append(row)
    report = {"sample_id": sample["id"], "question": sample["question"], "positive_id": positive_id,
              "scope": "one dev case; gold oracle and first 3 official noise references; third case is a controlled context-budget failure, not official benchmark score",
              "results": results}
    (ROOT / "artifacts/reports/context_noise_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: row.get(k) for k in ["variant", "input_ids", "answer", "insufficient_evidence", "error"]}, ensure_ascii=False), flush=True)
