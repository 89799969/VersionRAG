import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.generation import assemble_context
from app.knowledge import EmptyRetriever, KnowledgeError, KnowledgeService, split_document


class Encoder:
    dimension = 3

    def __init__(self, revision="test-v1"):
        self.calls = []
        self.revision = revision
        self.fail = False
        self.entered = self.release = None

    def metadata(self):
        return {"dimension": self.dimension, "model": "offline-test", "revision": self.revision}

    def encode(self, texts, query=False):
        self.calls.append((list(texts), query))
        if not query and self.entered:
            self.entered.set()
            assert self.release.wait(10), "test build was not released"
        if self.fail:
            raise KnowledgeError("Controlled encoder failure.")
        return np.tile(np.array([1, 0, 0], dtype=np.float32), (len(texts), 1)), 0


def build(service, auto_publish=True):
    job = service.submit(auto_publish)
    assert service.run_one()
    result = service.job(job["id"])
    assert result["state"] == "succeeded", result
    return result


def wait_job(client, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get(f"/knowledge/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["state"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.03)
    raise AssertionError("background job did not finish")


def test_incremental_cache_metadata_change_and_corrupt_cache(tmp_path):
    encoder = Encoder()
    service = KnowledgeService(tmp_path, encoder)
    service.upsert("a", "申请", "教学测试：报名截止日期为2月5日。")
    first = build(service)
    assert first["stats"]["encoded_unique_texts"] == 1
    assert not service.upsert("a", "申请", "教学测试：报名截止日期为2月5日。")["changed"]
    second = build(service)
    assert second["stats"]["encoded_unique_texts"] == 0
    assert second["stats"]["cache_hit_chunks"] == 1
    service.upsert("a", "修改标题", "教学测试：报名截止日期为2月5日。")
    assert build(service)["stats"]["encoded_unique_texts"] == 0
    service.upsert("a", "修改标题", "教学测试：报名截止日期为2月8日。")
    assert build(service)["stats"]["encoded_unique_texts"] == 1
    encoder.revision = "test-v2"
    assert build(service)["stats"]["encoded_unique_texts"] == 1
    with service.connection() as db:
        db.execute("UPDATE embedding_cache SET vector=?", (b"corrupt",))
    assert build(service)["stats"]["encoded_unique_texts"] == 1


def test_immutable_job_and_bounded_queue_do_not_publish_stale_draft(tmp_path):
    service = KnowledgeService(tmp_path, Encoder(), max_pending=1)
    service.upsert("a", "申请", "报名截止日期为2月5日。")
    queued = service.submit()
    with pytest.raises(KnowledgeError, match="queue is full"):
        service.submit()
    service.upsert("a", "申请", "报名截止日期为2月8日。")
    assert service.run_one()
    job = service.job(queued["id"])
    assert job["state"] == "succeeded" and not job["published"]
    assert service.status()["active_version"] is None
    service.activate(job["version"])
    assert "2月5日" in service.pin().engines["bm25"].search("报名")[0]["contents"]
    assert service.status()["draft_revision"] == 2


def test_failed_build_and_corrupt_activation_keep_both_previous_indexes(tmp_path):
    encoder = Encoder()
    service = KnowledgeService(tmp_path, encoder)
    service.upsert("a", "申请", "报名截止日期为2月5日。")
    old = build(service)
    service.upsert("a", "申请", "报名截止日期为2月8日。")
    encoder.fail = True
    job = service.submit()
    service.run_one()
    failed = service.job(job["id"])
    assert failed["state"] == "failed" and not failed["published"]
    assert service.status()["active_version"] == old["version"]
    encoder.fail = False
    ready = build(service, auto_publish=False)
    bm25_file = next((service.directory(ready["version"]) / "bm25").glob("*.npy"))
    with bm25_file.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(KnowledgeError, match="files changed"):
        service.activate(ready["version"])
    snapshot = service.pin()
    assert snapshot.version == old["version"]
    for method in ("bm25", "dense", "hybrid"):
        assert "2月5日" in snapshot.engines[method].search("报名")[0]["contents"]


def test_restart_recovers_queued_and_fails_interrupted_without_auto_replay(tmp_path):
    service = KnowledgeService(tmp_path, Encoder())
    service.upsert("a", "申请", "报名截止日期为2月5日。")
    queued = service.submit()
    interrupted = service.submit()
    with service.connection() as db:
        db.execute("UPDATE jobs SET state='running' WHERE id=?", (interrupted["id"],))
    restarted = KnowledgeService(tmp_path, Encoder())
    restarted.start()
    try:
        deadline = time.monotonic() + 10
        while restarted.job(queued["id"])["state"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.03)
        while restarted.job(queued["id"])["state"] == "running" and time.monotonic() < deadline:
            time.sleep(0.03)
        assert restarted.job(queued["id"])["state"] == "succeeded"
        assert restarted.job(interrupted["id"])["state"] == "failed"
        assert "interrupted" in restarted.job(interrupted["id"])["error"]
        second = KnowledgeService(tmp_path, Encoder())
        with pytest.raises(KnowledgeError, match="already has a writer"):
            second.start()
    finally:
        restarted.close()


def test_manual_rollback_during_build_wins_over_auto_publish(tmp_path):
    encoder = Encoder()
    service = KnowledgeService(tmp_path, encoder)
    service.upsert("a", "申请", "报名截止日期为2月5日。")
    old = build(service)
    service.upsert("a", "申请", "报名截止日期为2月8日。")
    encoder.entered, encoder.release = threading.Event(), threading.Event()
    job = service.submit()
    service.start()
    try:
        assert encoder.entered.wait(10)
        service.activate(old["version"])
        encoder.release.set()
        deadline = time.monotonic() + 10
        while service.job(job["id"])["state"] == "running" and time.monotonic() < deadline:
            time.sleep(0.03)
        result = service.job(job["id"])
        assert result["state"] == "succeeded" and not result["published"]
        assert service.pin().version == old["version"]
    finally:
        encoder.release.set()
        service.close()


def test_http_update_delete_rollback_and_request_pin_through_generation(tmp_path):
    service = KnowledgeService(tmp_path, Encoder())
    rollback_to = []

    def generator(question, hits):
        if rollback_to:
            service.activate(rollback_to.pop())  # publish happens after retrieval, before answer return
        _, evidence = assemble_context(hits)
        return {"answer": hits[0]["contents"] if hits else "证据不足",
                "citations": [hits[0]["id"]] if hits else [], "insufficient_evidence": not hits,
                "evidence": evidence, "model_called": False, "model": "offline-test"}

    with TestClient(create_app({"bm25": EmptyRetriever()}, generator, service)) as client:
        payload = {"document_id": "teaching-demo", "title": "教学构造申请通知",
                   "text": "教学测试：报名截止日期为2月5日。"}
        assert client.put("/knowledge/documents", json=payload).status_code == 200
        first = wait_job(client, client.post("/knowledge/jobs", json={}).json()["id"])
        payload["text"] = "教学测试：报名截止日期为2月8日。"
        client.put("/knowledge/documents", json=payload)
        second = wait_job(client, client.post("/knowledge/jobs", json={}).json()["id"])
        for method in ("bm25", "dense", "hybrid"):
            result = client.post("/retrieve", json={"question": "报名截止日期", "method": method}).json()
            assert result["knowledge_version"] == second["version"]
            assert "2月8日" in result["hits"][0]["contents"]
            assert result["hits"][0]["metadata"]["knowledge_version"] == second["version"]
        rollback_to.append(first["version"])
        answer = client.post("/ask", json={"question": "报名截止日期"}).json()
        assert "2月8日" in answer["answer"]
        assert answer["knowledge_version"] == answer["evidence"][0]["knowledge_version"] == second["version"]
        assert answer["citations"] == [answer["evidence"][0]["passage_id"]]
        assert client.get("/knowledge/status").json()["active_version"] == first["version"]
        assert "2月5日" in client.post("/ask", json={"question": "报名截止日期"}).json()["answer"]
        client.delete("/knowledge/documents/teaching-demo")
        deletion = wait_job(client, client.post("/knowledge/jobs", json={}).json()["id"])
        for method in ("bm25", "dense", "hybrid", "hybrid_rerank"):
            empty = client.post("/retrieve", json={"question": "报名截止日期", "method": method}).json()
            assert empty["hits"] == [] and empty["knowledge_version"] == deletion["version"]
        assert client.post("/ask", json={"question": "报名截止日期"}).json()["insufficient_evidence"]
        assert client.put("/knowledge/documents", json={**payload, "text": " "}).status_code == 422
        assert client.put("/knowledge/documents", json={**payload, "document_id": "cannot/delete"}).status_code == 422
        assert client.get("/knowledge/jobs/unknown").status_code == 404


def test_chunks_preserve_offsets_overlap_and_document_identity():
    text = "第一段申请资料。\n" * 180
    chunks = split_document("a", "申请", text)
    assert len(chunks) > 1
    for chunk in chunks:
        start, end = chunk["metadata"]["start"], chunk["metadata"]["end"]
        assert chunk["contents"] == text[start:end] and end - start <= 900
    assert chunks[1]["metadata"]["start"] == chunks[0]["metadata"]["end"] - 100
    assert {c["id"] for c in chunks}.isdisjoint(c["id"] for c in split_document("b", "申请", text))
