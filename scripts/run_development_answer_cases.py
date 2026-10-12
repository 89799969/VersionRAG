"""Real answer comparison for fixed development examples, using only retrieved context."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dense import DenseRetriever
from app.evaluation import split_for
from app.generation import GenerationError, generate
from app.hybrid import HybridRetriever
from app.reranking import RerankRetriever
from app.retrieval import ROOT, ChineseBM25, read_jsonl

qa = {row["id"]: row for row in read_jsonl(ROOT / "data/basic_qa_mapped.jsonl")}
bm25, dense = ChineseBM25(), DenseRetriever()
bm25.load()
dense.load()
reranked = RerankRetriever(HybridRetriever(bm25, dense))
results = []
for sample_id in ("domainrag_extractive_033", "domainrag_extractive_034"):
    sample = qa[sample_id]
    assert split_for(sample) == "dev"
    for name, engine in {"bm25": bm25, "hybrid_rerank": reranked}.items():
        start = time.perf_counter()
        hits = engine.search(sample["question"], 5)
        try:
            answer = generate(sample["question"], hits)
        except GenerationError as exc:
            answer = {"status": "generation_rejected", "error": str(exc), "answer": None,
                      "citations": [], "insufficient_evidence": None}
        row = {"id": sample_id, "method": name, "question": sample["question"],
               "retrieved_ids": [str(hit["id"]) for hit in hits],
               "total_ms": (time.perf_counter() - start) * 1000, **answer}
        results.append(row)
        (ROOT / "artifacts/reports/development_answer_cases.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        # Gold remains outside the retrieval/generation call above.
        print(json.dumps({"id": sample_id, "method": name, "question": sample["question"],
                          "answer": row["answer"], "citations": row["citations"],
                          "insufficient_evidence": row["insufficient_evidence"], "error": row.get("error")},
                         ensure_ascii=False), flush=True)
