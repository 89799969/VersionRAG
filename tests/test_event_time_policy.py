import json

import httpx

from app.generation import Settings, generate


def listing():
    return {"id": "listing", "title": "新闻列表", "contents": "首次举办开放日\n2014-07-16",
            "metadata": {"source_id": "fixture", "url": "https://rdzs.ruc.edu.cn/cms/list/ztlm/?cat=43"}}


def test_event_date_cannot_be_inferred_from_listing_without_calling_model():
    def unexpected_call(request):
        raise AssertionError("Listing-only event query must refuse before contacting provider")
    result = generate("何时首次举办开放日？", [listing()],
                      transport=httpx.MockTransport(unexpected_call))
    assert result["insufficient_evidence"] is True
    assert result["model_called"] is False
    assert result["excluded_listing_ids"] == ["listing"]
    assert result["evidence"] == []


def test_article_body_kept_and_explicit_publication_query_can_use_listing():
    def respond(request):
        prompt = request.read().decode("utf-8")
        if "发布日期" in json.loads(prompt)["messages"][1]["content"]:
            citation, answer = "listing", "该文章在列表中的日期为2014年7月16日。"
            assert "[片段 listing]" in json.loads(prompt)["messages"][1]["content"]
        else:
            citation, answer = "article", "2014年7月15日。"
            assert "[片段 listing]" not in json.loads(prompt)["messages"][1]["content"]
        payload = {"answer": answer, "citations": [citation], "insufficient_evidence": False}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
    settings = Settings("fake", "https://example.invalid", "mock", 10)
    article = {"id": "article", "title": "活动报道", "contents": "7月15日首次举办开放日。",
               "metadata": {"source_id": "fixture_body", "url": "https://rdzs.ruc.edu.cn/cms/item/1.html"}}
    result = generate("何时首次举办开放日？", [listing(), article], settings, httpx.MockTransport(respond))
    assert result["citations"] == ["article"]
    assert result["excluded_listing_ids"] == ["listing"]
    assert [e["passage_id"] for e in result["evidence"]] == ["article"]
    publication = generate("首次举办开放日这篇文章的发布日期是什么时候？", [listing()], settings, httpx.MockTransport(respond))
    assert publication["citations"] == ["listing"]
    assert publication["excluded_listing_ids"] == []
