import json

import httpx
import pytest

from app.comparison import ComparisonPlan, ComparisonRetriever, ExactEvidenceIndex, TitlePlanner, balanced_hits
from app.comparison_generation import Draft, bounded_answer, generate_comparison, prepare_evidence, validate_draft
from app.generation import GenerationError, Settings
from app.retrieval import ChineseBM25


def doc(pid, title, text, source=None):
    return {"id": pid, "title": title, "contents": text,
            "metadata": {"source_id": source or pid, "url": "https://example.com/professional/" + pid},
            "score": 1., "rank": 1}


DOCS = [doc("a", "政治学与行政学", "课程设置包括政治学理论和公共行政学"),
        doc("b", "外交学", "课程设置包括国际关系和外交谈判"),
        doc("c", "外交学", "课程设置包括国际关系和外交谈判")]
QUESTION = "2022年政治学与行政学专业和外交学专业在课程设置方面有哪些不同？"


def test_planner_uses_titles_preserves_year_and_declines_unknown_or_single_entity():
    planner = TitlePlanner(DOCS)
    plan = planner.plan(QUESTION)
    assert plan.entities == ("政治学与行政学", "外交学")
    assert all("2022年" in q for q in plan.queries)
    assert plan.facet == "课程设置方面"
    assert planner.plan("外交学有哪些课程？") is None
    assert planner.plan("政治学与行政学和不存在的专业有什么不同？") is None
    assert planner.allowed("外交学", "c", DOCS) == {"c"}
    school = doc("s", "中国人民大学", "学校介绍")
    school["metadata"]["url"] = "https://example.com/school/1"
    assert TitlePlanner(DOCS + [school]).plan("中国人民大学的" + QUESTION).entities == plan.entities


def test_exact_dedup_keeps_all_provenance_and_respects_scope():
    exact = ExactEvidenceIndex(DOCS)
    assert exact.equivalents(["b"]) == {"b", "c"}
    hits = exact.deduplicate([DOCS[1], DOCS[2]])
    assert len(hits) == 1 and hits[0]["equivalent_passage_ids"] == ["b", "c"]
    assert {r["source_id"] for r in hits[0]["equivalent_sources"]} == {"b", "c"}
    assert exact.deduplicate([DOCS[1]], {"b"})[0]["equivalent_passage_ids"] == ["b"]
    assert len(DOCS) == 3


def test_balancing_protects_second_entity_and_removes_exact_copies():
    groups = {"政治学与行政学": [DOCS[0], doc("x", DOCS[0]["title"], "另一个政治学片段")],
              "外交学": [DOCS[1], DOCS[2]]}
    hits = balanced_hits(groups, 3)
    assert [h["id"] for h in hits] == ["a", "b", "x"]
    assert hits[1]["evidence_entities"] == ["外交学"]
    assert len(balanced_hits({"甲": [DOCS[1]], "乙": [DOCS[2]]}, 2)) == 1
    with pytest.raises(ValueError):
        balanced_hits(groups, 1)


def test_allowed_ids_are_applied_before_keyword_topk(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in DOCS), encoding="utf-8")
    bm25 = ChineseBM25(corpus, tmp_path / "index")
    bm25.build()
    assert bm25.search("课程", 1, allowed_ids={"c"})[0]["id"] == "c"
    assert bm25.search("课程", 1, allowed_ids=set()) == []
    assert bm25.search("课程", 1, source_id="a", allowed_ids={"c"}) == []


class FakeHybrid:
    def search(self, question, top_k, source_id=None, *, allowed_ids=None):
        return [{**d, "rank": i + 1} for i, d in enumerate(DOCS) if d["id"] in allowed_ids][:top_k]


class FakeReranker:
    def rerank(self, question, hits, top_k):
        return hits[:top_k]


def test_scoped_retrieval_does_not_share_mutable_query_filters():
    retriever = ComparisonRetriever(FakeHybrid(), FakeReranker(), DOCS)
    plan = retriever.planner.plan(QUESTION)
    hits, trace = retriever.retrieve(plan)
    assert [h["id"] for h in hits] == ["a", "b"]
    assert len(trace["steps"]) == 2
    assert trace["steps"][1]["unique_candidate_ids"] == ["b"]
    hits, trace = retriever.retrieve(plan, source_id="a")
    assert [h["id"] for h in hits] == ["a"]
    with pytest.raises(ValueError):
        retriever.retrieve(plan, only_entities=["未请求的专业"])


def aspect(entity, hit=None):
    return {"entity": entity, "status": "supported" if hit else "insufficient",
            "claims": [{"text": hit["contents"], "quotes": [{"passage_id": hit["id"], "quote": hit["contents"]}]}] if hit else [],
            "missing_reason": "" if hit else "课程资料不足"}


def test_quote_guard_checks_context_span_entity_and_ids():
    plan = TitlePlanner(DOCS).plan(QUESTION)
    hits = balanced_hits({plan.entities[0]: [DOCS[0]], plan.entities[1]: [DOCS[1]]})
    _, evidence = prepare_evidence(hits)
    payload = {"aspects": [aspect(plan.entities[0], DOCS[0]), aspect(plan.entities[1], DOCS[1])]}
    validate_draft(Draft.model_validate(payload), plan, evidence)
    payload["aspects"][0]["claims"][0]["quotes"][0]["quote"] = "根本不存在的引文内容"
    with pytest.raises(GenerationError, match="absent"):
        validate_draft(Draft.model_validate(payload), plan, evidence)
    payload["aspects"][0]["claims"][0]["quotes"][0] = {"passage_id": "b", "quote": DOCS[1]["contents"]}
    with pytest.raises(GenerationError, match="different entity"):
        validate_draft(Draft.model_validate(payload), plan, evidence)
    payload["aspects"][0]["claims"][0]["quotes"][0]["passage_id"] = "unknown"
    with pytest.raises(GenerationError, match="outside"):
        validate_draft(Draft.model_validate(payload), plan, evidence)


def test_quote_anchoring_does_not_claim_semantic_entailment():
    # Even a wrong claim can quote real text. Semantic evaluation remains necessary.
    plan = TitlePlanner(DOCS).plan(QUESTION)
    _, evidence = prepare_evidence(DOCS[:2])
    payload = {"aspects": [aspect(plan.entities[0], DOCS[0]), aspect(plan.entities[1], DOCS[1])]}
    payload["aspects"][0]["claims"][0]["text"] = "这门专业只教授音乐"
    validate_draft(Draft.model_validate(payload), plan, evidence)


def test_comparison_conclusion_requires_quotes_for_both_entities():
    plan = TitlePlanner(DOCS).plan(QUESTION)
    _, evidence = prepare_evidence(DOCS[:2])
    payload = {"aspects": [aspect(plan.entities[0], DOCS[0]), aspect(plan.entities[1], DOCS[1])],
        "comparisons": [{"text": "课程设置不同", "quotes": [{"passage_id": "a", "quote": DOCS[0]["contents"]}]}]}
    with pytest.raises(GenerationError, match="both entities"):
        validate_draft(Draft.model_validate(payload), plan, evidence)
    payload["comparisons"][0]["quotes"].append({"passage_id": "b", "quote": DOCS[1]["contents"]})
    validate_draft(Draft.model_validate(payload), plan, evidence)


def test_context_budget_reserves_space_for_both_entities_and_tracks_truncation():
    hits = [doc("a", "甲专业", "甲" * 1000), doc("b", "乙专业", "乙" * 1000)]
    context, evidence = prepare_evidence(hits, 300)
    assert len(context) <= 300 and len(evidence) == 2
    assert all(e["truncated"] for e in evidence)


def test_real_adapter_can_be_tested_offline_and_rejects_bad_provider_structure():
    plan = TitlePlanner(DOCS).plan(QUESTION)
    payload = {"aspects": [aspect(plan.entities[0], DOCS[0]), aspect(plan.entities[1], DOCS[1])]}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {
        "content": json.dumps(payload, ensure_ascii=False)}}], "usage": {"prompt_tokens": 10}}))
    result = generate_comparison(plan, DOCS[:2], settings=Settings("fake", "https://example.com", "mock", 1), transport=transport)
    assert result["usage"]["prompt_tokens"] == 10
    bad = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(GenerationError, match="structure"):
        generate_comparison(plan, DOCS[:2], settings=Settings("fake", "https://example.com", "mock", 1), transport=bad)


def test_unanchored_claim_is_removed_without_changing_quote_or_other_valid_claims():
    plan = TitlePlanner(DOCS).plan(QUESTION)
    payload = {"aspects": [aspect(plan.entities[0], DOCS[0]), aspect(plan.entities[1], DOCS[1])]}
    payload["aspects"][1]["claims"].append({"text": "没有依据的结论", "quotes": [{"passage_id": "b", "quote": "编造不存在的原文"}]})
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {
        "content": json.dumps(payload, ensure_ascii=False)}}]}))
    result = generate_comparison(plan, DOCS[:2], settings=Settings("fake", "https://example.com", "mock", 1), transport=transport)
    assert len(result["aspects"][1]["claims"]) == 1
    assert result["aspects"][1]["status"] == "insufficient"
    assert len(result["rejected_claims"]) == 1


def test_recovery_has_one_round_and_preserves_partial_answers():
    plan = TitlePlanner(DOCS).plan(QUESTION)
    calls = []
    class Retriever:
        def retrieve(self, plan, **kwargs):
            calls.append(kwargs)
            hit = DOCS[1] if kwargs.get("supplemental") else DOCS[0]
            entity = plan.entities[1] if kwargs.get("supplemental") else plan.entities[0]
            return [{**hit, "evidence_entities": [entity]}], {"steps": []}
    def generator(plan, hits):
        # Still incomplete after recovery: preserve partial answer and stop.
        return {"aspects": [aspect(plan.entities[0], DOCS[0]), aspect(plan.entities[1])],
                "evidence": [], "usage": {"prompt_tokens": 10}, "model_called": False}
    result = bounded_answer(Retriever(), plan, recover=True, generator=generator)
    assert len(calls) == 2 and calls[1]["only_entities"] == [plan.entities[1]]
    assert len(result["trace"]["generation_attempts"]) == 2
    assert result["insufficient_evidence"] and "政治学理论" in result["answer"]
    assert result["total_usage"]["prompt_tokens"] == 20
    calls.clear()
    default = bounded_answer(Retriever(), plan, generator=generator)
    assert len(calls) == 1 and default['trace']['recovery_triggered'] is False


def test_unchanged_recovery_context_does_not_call_generator_again():
    retriever = ComparisonRetriever(FakeHybrid(), FakeReranker(), DOCS)
    plan = retriever.planner.plan(QUESTION)
    generations = []
    def generator(plan, hits):
        generations.append(hits)
        return {"aspects": [aspect(plan.entities[0]), aspect(plan.entities[1])], "evidence": [],
                "usage": {}, "model_called": False}
    result = bounded_answer(retriever, plan, recover=True, generator=generator)
    assert len(generations) == 1 and not result["trace"]["recovery_context_changed"]


def test_http_comparison_pins_one_version_and_returns_trace_without_model_network():
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from app.api import create_app, ComparisonRequest
    assert ComparisonRequest(question=QUESTION).recover is False
    hybrid = FakeHybrid()
    hybrid.bm25 = SimpleNamespace(documents=DOCS)
    engine = SimpleNamespace(retriever=hybrid, reranker=FakeReranker(), candidate_k=30)
    class Knowledge:
        pins = 0
        def start(self): pass
        def close(self): pass
        def pin(self):
            self.pins += 1
            return SimpleNamespace(version="v" + str(self.pins), engines={"hybrid_rerank": engine})
    knowledge = Knowledge()
    def mock(plan, hits):
        _, evidence = prepare_evidence(hits)
        return {"aspects": [aspect(plan.entities[0], DOCS[0]), aspect(plan.entities[1], DOCS[1])],
                "evidence": evidence, "usage": {}, "model_called": False}
    with TestClient(create_app({"hybrid_rerank": engine}, knowledge=knowledge,
                               comparison_generator=mock)) as client:
        response = client.post("/ask-comparison", json={"question": QUESTION})
        assert response.status_code == 200
        result = response.json()
        assert knowledge.pins == 1 and result["knowledge_version"] == "v1"
        assert result["trace"]["request_id"] == response.headers["X-Request-ID"]
        assert all(e["knowledge_version"] == "v1" for e in result["evidence"])
        assert not result["trace"]["recovery_triggered"]
        assert client.post("/ask-comparison", json={"question": QUESTION, "top_k": 1}).status_code == 422


def test_http_comparison_fallback_and_provider_error_are_explicit():
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from app.api import create_app
    hybrid = FakeHybrid()
    hybrid.bm25 = SimpleNamespace(documents=DOCS)
    class Engine:
        retriever, reranker, candidate_k = hybrid, FakeReranker(), 30
        def search(self, question, top_k, source_id): return DOCS[:2]
    def basic(question, hits): return {"answer": "普通回答", "evidence": []}
    def fail(plan, hits): raise GenerationError("Comparison model request timed out.")
    with TestClient(create_app({"hybrid_rerank": Engine()}, generator=basic,
                               comparison_generator=fail)) as client:
        assert client.post("/ask-comparison", json={"question": "外交学有哪些课程？"}).json()["trace"]["policy"] == "standard-fallback"
        assert client.post("/ask-comparison", json={"question": QUESTION}).status_code == 504
