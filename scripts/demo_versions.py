"""Controlled engineering fixture, not official dataset policy history."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import ROOT
from app.versions import VersionStore


def document(date, deadline):
    return {"id": "demo_deadline", "title": "教学演示：材料提交截止日期",
            "contents": f"教学演示资料，非真实招生政策。材料提交截止日期为{deadline}。",
            "metadata": {"source_id": "demo_policy", "effective_year": date}}


with tempfile.TemporaryDirectory(prefix="version-demo-", dir=ROOT / "artifacts") as temporary:
    store = VersionStore(Path(temporary))
    store.prepare("v1", [document(2025, "2月5日")])
    store.activate("v1")
    before = store.search("材料提交截止日期？")
    store.prepare("v2", [document(2026, "2月8日")])
    store.activate("v2")
    after = store.search("材料提交截止日期？")
    assert "2月8日" in after["hits"][0]["contents"] and "2月5日" not in after["hits"][0]["contents"]
    store.activate("v1")
    rollback = store.search("材料提交截止日期？")
    assert before == rollback
    report = {"before": before, "after": after, "rollback": rollback,
              "scope": "single-writer isolated BM25 snapshot demo; no official corpus changes; no dense hot reload or multi-process concurrency claim"}
(ROOT / "artifacts/reports/version_demo.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"before": before["version"], "after": after["version"], "rollback": rollback["version"],
                  "old_deadline_removed_from_active_search": True}, ensure_ascii=False))
