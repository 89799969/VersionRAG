"""Real local HTTP + configured providers; isolated teaching document, no benchmark tuning.

Restores the initial active version and removes the teaching draft on exit. The
report is a lifecycle check, not answer accuracy or a performance benchmark.
"""
import argparse
import json
import time
from uuid import uuid4

import httpx

from app.retrieval import ROOT, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    args = parser.parse_args()
    document_id = "teaching-run-" + uuid4().hex[:12]
    report = {"kind": "real HTTP lifecycle / constructed teaching document",
              "document_id": document_id, "events": [], "checks": {},
              "code_sha256": {path: sha256(ROOT / path) for path in
                              ("app/knowledge.py", "app/knowledge_routes.py", "app/api.py")}}
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        def request(method, path, body=None, expected=200):
            response = client.request(method, path, json=body) if body is not None else client.request(method, path)
            assert response.status_code == expected, f"{path}: HTTP {response.status_code}"
            return response.json()

        def build():
            job = request("POST", "/knowledge/jobs", {}, 202)
            print(f"Submitted job {job['id']}", flush=True)
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                job = request("GET", "/knowledge/jobs/" + job["id"])
                if job["state"] in {"succeeded", "failed"}:
                    report["events"].append({"job": job})
                    assert job["state"] == "succeeded" and job["published"], "Build did not publish."
                    print(f"Published {job['version']}: {job['stats']}", flush=True)
                    return job
                time.sleep(0.5)
            raise AssertionError("Specific build job did not finish before deadline.")

        def ask(expected_date, version):
            answer = request("POST", "/ask", {"question": "这份教学项目申请通知规定的报名截止日期是什么时候？",
                             "source_id": document_id, "method": "hybrid_rerank", "top_k": 5})
            report["events"].append({"answer": answer})
            normalized = "".join(answer["answer"].split())
            assert expected_date in normalized and not answer["insufficient_evidence"]
            assert answer["knowledge_version"] == version and answer["model_called"]
            assert answer["citations"] and set(answer["citations"]) <= {e["passage_id"] for e in answer["evidence"]}
            assert all(e["knowledge_version"] == version and e["source_id"] == document_id for e in answer["evidence"])
            print(f"Answer verified: {expected_date}, version {version}", flush=True)
            return answer

        initial = request("GET", "/knowledge/status")["active_version"]
        assert initial is not None, "Run seed_knowledge before checking lifecycle."
        report["initial_version"] = initial
        try:
            payload = {"document_id": document_id, "title": "教学构造：项目申请通知",
                       "text": "教学构造资料：本次项目申请的报名截止日期为2月5日。"}
            request("PUT", "/knowledge/documents", payload)
            v1 = build()
            ask("2月5", v1["version"])
            payload["text"] = "教学构造资料：本次项目申请的报名截止日期为2月8日。"
            request("PUT", "/knowledge/documents", payload)
            v2 = build()
            ask("2月8", v2["version"])
            report["checks"]["one_changed_text_encoded"] = v2["stats"]["encoded_unique_texts"] == 1
            report["checks"]["original_chunks_reused"] = v2["stats"]["cache_hit_chunks"] == v2["stats"]["total_chunks"] - 1
            request("POST", "/knowledge/activate", {"version": "not-a-ready-version"}, 409)
            report["checks"]["rejected_activation_keeps_active"] = request("GET", "/knowledge/status")["active_version"] == v2["version"]
            request("POST", "/knowledge/activate", {"version": v1["version"]})
            ask("2月5", v1["version"])
            report["checks"]["rollback_answer_and_citations"] = True
            request("DELETE", "/knowledge/documents/" + document_id)
            deleted = build()
            report["checks"]["delete_requires_no_new_embedding"] = deleted["stats"]["encoded_unique_texts"] == 0
            for method in ("bm25", "dense", "hybrid", "hybrid_rerank"):
                hits = request("POST", "/retrieve", {"question": "报名截止日期", "source_id": document_id,
                               "method": method})
                report["checks"]["deleted_absent_" + method] = hits["hits"] == [] and hits["knowledge_version"] == deleted["version"]
            refusal = request("POST", "/ask", {"question": "报名截止日期", "source_id": document_id,
                              "method": "hybrid_rerank"})
            report["events"].append({"deleted_answer": refusal})
            report["checks"]["deleted_refuses_without_generation"] = (
                refusal["insufficient_evidence"] and not refusal["model_called"] and not refusal["citations"])
            assert all(report["checks"].values()), "Lifecycle verification failed."
            report["passed"] = True
        finally:
            request("POST", "/knowledge/activate", {"version": initial})
            request("DELETE", "/knowledge/documents/" + document_id)
            report["final_status"] = request("GET", "/knowledge/status")
            output = ROOT / "artifacts/reports/knowledge_lifecycle_http.json"
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Lifecycle checks passed; initial version restored.", flush=True)


if __name__ == "__main__":
    main()
