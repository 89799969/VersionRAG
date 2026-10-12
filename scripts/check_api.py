"""Verify the live local HTTP service without calling the paid generator."""
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
with httpx.Client(base_url="http://127.0.0.1:8018", timeout=10, trust_env=False) as client:
    health = client.get("/health")
    health.raise_for_status()
    response = client.post("/retrieve", json={"question": "中国人民大学中法学院用哪三个语言进行教学？"})
    response.raise_for_status()
    retrieval = response.json()
    assert len(retrieval["hits"]) == 5
    assert retrieval["request_id"] == response.headers["X-Request-ID"]
    invalid = client.post("/ask", json={"question": "语言？", "top_k": 0})
    assert invalid.status_code == 422
    report = {"health": health.json(), "retrieval": retrieval, "invalid_top_k_status": invalid.status_code,
              "model_called": False}
    path = ROOT / "artifacts/reports/api_smoke.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"health": report["health"], "retrieved_ids": [item["id"] for item in retrieval["hits"]],
                      "trace_id_matches": True, "invalid_top_k_status": invalid.status_code,
                      "model_called": False}, ensure_ascii=False, indent=2))
