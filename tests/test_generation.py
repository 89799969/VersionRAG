import json

import httpx
import pytest

from app.generation import GenerationError, Settings, assemble_context, generate

HITS = [{"id": "1189", "title": "中法学院", "contents": "汉语、法语、英语三语教学",
         "metadata": {"source_id": "159", "url": "https://example.com/source"}}]
SETTINGS = Settings("test-placeholder", "https://example.com", "mock-model", 1)


def transport_for(citations):
    def respond(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "answer": "汉语、法语、英语", "citations": citations, "insufficient_evidence": False,
        })}}]})
    return httpx.MockTransport(respond)


def test_source_id_cannot_masquerade_as_passage_citation():
    with pytest.raises(GenerationError, match="outside"):
        generate("教学语言？", HITS, SETTINGS, transport_for(["159"]))
    assert generate("教学语言？", HITS, SETTINGS, transport_for(["1189"]))["citations"] == ["1189"]


def test_no_hits_does_not_call_model():
    assert generate("未知问题", [], SETTINGS)["model_called"] is False


def test_context_budget_tracks_truncation_and_duplicate_ids():
    long_hit = {**HITS[0], "contents": "甲" * 400}
    text, evidence = assemble_context([long_hit, long_hit], max_chars=200)
    assert len(text) <= 200
    assert len(evidence) == 1 and evidence[0]["truncated"] is True


def test_timeout_is_safe_and_not_silently_replaced_with_mock_answer():
    def fail(request):
        raise httpx.ReadTimeout("do-not-expose-provider-detail")
    with pytest.raises(GenerationError, match="timed out"):
        generate("教学语言？", HITS, SETTINGS, httpx.MockTransport(fail))
