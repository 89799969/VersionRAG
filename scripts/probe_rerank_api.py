"""Real free-service probe using two public demo candidates, not a QA quality claim."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.reranking import APIReranker
from app.retrieval import ROOT

start = time.perf_counter()
hits = [{"id": "demo-language", "contents": "中法学院采用汉语、法语、英语三语教学。", "rank": 2, "score": 0.02},
        {"id": "demo-subject", "contents": "物理与化学专业招生信息。", "rank": 1, "score": 0.03}]
result = APIReranker().rerank("中法学院使用哪些语言教学？", hits, 2)
report = {"real_api_called": True, "model": APIReranker.model, "results": result,
          "latency_ms": (time.perf_counter() - start) * 1000,
          "scope": "integration probe using constructed candidates, not full retrieval evaluation"}
(ROOT / "artifacts/reports/rerank_api_probe.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
