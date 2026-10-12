import json

import pytest

from app.versions import VersionStore


def doc(text):
    return {"id": "policy", "title": "policy", "contents": text, "metadata": {"source_id": "fixture"}}


def test_replacement_rollback_and_failed_activation(tmp_path):
    store = VersionStore(tmp_path)
    store.prepare("v1", [doc("申请截止日期为2月5日")])
    store.activate("v1")
    store.prepare("v2", [doc("申请截止日期为2月8日")])
    store.activate("v2")
    assert "2月8日" in store.search("截止日期")["hits"][0]["contents"]
    (tmp_path / "v2/corpus.jsonl").write_text(json.dumps(doc("篡改文本")), encoding="utf-8")
    store.activate("v1")
    with pytest.raises(ValueError, match="changed"):
        store.activate("v2")
    assert store.search("截止日期")["version"] == "v1"
    assert "2月5日" in store.search("截止日期")["hits"][0]["contents"]
    with pytest.raises(FileNotFoundError):
        store.activate("unfinished")
    assert store.search("截止日期")["version"] == "v1"
