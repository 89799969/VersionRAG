"""Fixed development diagnostics for multi-evidence and dated questions."""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dense import DenseRetriever
from app.evaluation import evidence_metrics
from app.hybrid import HybridRetriever
from app.reranking import RerankRetriever
from app.retrieval import ROOT, ChineseBM25, read_jsonl, sha256

bm25, dense = ChineseBM25(), DenseRetriever()
bm25.load()
dense.load()
engines = {"bm25": bm25, "hybrid_rerank": RerankRetriever(HybridRetriever(bm25, dense))}
path = ROOT / "data/extended_qa_mapped.jsonl"
samples = [r for r in read_jsonl(path) if r["split"] == "dev" and r["mapping_complete"]
           and r["task"] in {"multi_doc", "time_sensitive"}]
rows = []
for sample in samples:
    # The date is part of the official query, not a hidden gold metadata filter.
    question = (f"截至{sample['as_of']}年，{sample['question']}" if sample["as_of"] else sample["question"])
    for name, engine in engines.items():
        start = time.perf_counter()
        hits = engine.search(question, 5)
        ids = [str(hit["id"]) for hit in hits]
        rows.append({"id": sample["id"], "task": sample["task"], "method": name,
                     "question_sent": question, "retrieved_ids": ids,
                     "required_groups": len(sample["required_evidence_groups"]),
                     "latency_ms": (time.perf_counter() - start) * 1000,
                     **evidence_metrics(sample, ids, 5)})
summaries = []
for task in ("multi_doc", "time_sensitive"):
    for name in engines:
        selected = [r for r in rows if r["task"] == task and r["method"] == name]
        summaries.append({"task": task, "method": name, "questions": len(selected),
                          "evidence_group_recall_at_5": statistics.mean(r["evidence_group_recall"] for r in selected),
                          "all_evidence_hit_count": sum(r["all_evidence_hit"] for r in selected),
                          "mrr_at_5": statistics.mean(r["reciprocal_rank"] for r in selected)})
report = {"split": "dev", "qa_sha256": sha256(path), "summaries": summaries, "cases": rows,
          "scope": "small development diagnostics; not independent answer accuracy or exhaustive time reasoning",
          "date_rule": "prefix official as_of year to question; no date metadata inferred from gold for retrieval"}
(ROOT / "artifacts/reports/extended_retrieval_dev.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summaries, ensure_ascii=False, indent=2))
