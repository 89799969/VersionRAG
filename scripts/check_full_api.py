"""Real HTTP smoke, all retrieval methods and one paid generation request."""
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
question = "中国人民大学2016年高水平艺术团招生报名截止日期是什么时候？"
with httpx.Client(base_url="http://127.0.0.1:8018", timeout=90, trust_env=False) as client:
    health = client.get("/health")
    health.raise_for_status()
    home = client.get("/")
    home.raise_for_status()
    assert "<textarea" in home.text
    rows = []
    for method in health.json()["retrieval_methods"]:
        response = client.post("/retrieve", json={"question": question, "method": method})
        response.raise_for_status()
        payload = response.json()
        assert payload["request_id"] == response.headers["X-Request-ID"]
        rows.append({"method": method, "ids": [hit["id"] for hit in payload["hits"]], "status": response.status_code})
    response = client.post("/ask", json={"question": question, "method": "hybrid_rerank"})
    response.raise_for_status()
    answer = response.json()
    assert answer["request_id"] == response.headers["X-Request-ID"]
    assert not answer["insufficient_evidence"] and "2月5日" in answer["answer"]
    assert set(answer["citations"]) <= {hit["passage_id"] for hit in answer["evidence"]}
report = {"health": health.json(), "home_status": home.status_code, "retrieval": rows, "answer": answer,
          "scope": "real local HTTP calls; one dated development question, not batch generation accuracy"}
(ROOT / "artifacts/reports/api_full_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"health": report["health"], "home_status": home.status_code, "retrieval": rows,
                  "answer": answer["answer"], "citations": answer["citations"]}, ensure_ascii=False, indent=2))
