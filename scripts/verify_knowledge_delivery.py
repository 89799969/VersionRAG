"""Final no-model-call audit: tests, frozen baseline, persistent state and HTTP."""
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

import httpx

from app.retrieval import CORPUS, ROOT, read_jsonl, sha256


def main():
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "checks": {}, "remote_model_calls": 0}
    tests = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8")
    if tests.returncode:
        print(tests.stdout)
        print(tests.stderr)
        raise AssertionError("Project tests failed.")
    summary = re.search(r"(\d+) passed.*", tests.stdout)
    assert summary
    report["test_summary"] = summary.group(0)
    snapshot = json.loads((ROOT / "artifacts/reports/comparison_snapshot.json").read_text(encoding="utf-8"))
    baseline = {path: sha256(ROOT / path) == expected for path, expected in snapshot["code_sha256"].items()}
    baseline["corpus"] = sha256(CORPUS) == snapshot["corpus_sha256"]
    baseline["qa"] = sha256(ROOT / "data/basic_qa_mapped.jsonl") == snapshot["qa_sha256"]
    baseline["vectors"] = sha256(ROOT / "artifacts/indexes/dense/vectors.npy") == snapshot["dense_index"]["vectors_sha256"]
    report["frozen_baseline_unchanged"] = baseline
    assert all(baseline.values())
    lifecycle = json.loads((ROOT / "artifacts/reports/knowledge_lifecycle_http.json").read_text(encoding="utf-8"))
    assert lifecycle["passed"] and all(lifecycle["checks"].values())
    report["checks"]["real_lifecycle_report_passed"] = True
    report["lifecycle_code_note"] = "Recorded real-run hashes retained; later document-ID validation tightening covered by tests and HTTP below."
    report["code_sha256"] = {path: sha256(ROOT / path) for path in
                            ("app/knowledge.py", "app/knowledge_routes.py", "app/api.py", "tests/test_knowledge.py")}
    with sqlite3.connect(f"file:{(ROOT / 'artifacts/knowledge/knowledge.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        report["checks"]["sqlite_integrity"] = db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        report["draft_chunks"] = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        report["source_records"] = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        report["checks"]["teaching_draft_removed"] = db.execute("SELECT COUNT(*) FROM documents WHERE id=?", (lifecycle["document_id"],)).fetchone()[0] == 0
    assert report["draft_chunks"] == len(read_jsonl(CORPUS)) == 14406
    with httpx.Client(base_url="http://127.0.0.1:8018", timeout=30, trust_env=False) as client:
        health = client.get("/health")
        assert health.status_code == 200
        report["health"] = health.json()
        report["checks"]["initial_version_restored_after_restart"] = health.json()["knowledge_version"] == lifecycle["initial_version"]
        report["checks"]["worker_alive"] = health.json()["knowledge"]["worker_alive"]
        report["checks"]["management_page"] = client.get("/manage").status_code == 200
        invalid = client.put("/knowledge/documents", json={"document_id": "cannot/delete", "title": "check", "text": "check"})
        report["checks"]["invalid_document_id_422"] = invalid.status_code == 422
        response = client.post("/retrieve", json={"question": "中法学院教学语言", "method": "bm25"})
        result = response.json()
        report["checks"]["baseline_retrieval_version_and_trace"] = (
            response.status_code == 200 and bool(result["hits"]) and
            result["request_id"] == response.headers["X-Request-ID"] and
            result["knowledge_version"] == lifecycle["initial_version"] and
            all(hit["metadata"]["knowledge_version"] == result["knowledge_version"] for hit in result["hits"]))
    assert all(report["checks"].values())
    report["passed"] = True
    output = ROOT / "artifacts/reports/knowledge_delivery_check.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "tests": report["test_summary"], "checks": report["checks"],
                      "baseline_unchanged": True, "remote_model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
