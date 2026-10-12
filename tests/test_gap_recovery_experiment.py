import pytest

from app.comparison import ComparisonPlan
from scripts.experiment_gap_recovery import evidence_set, joint_merge, queries_for, validate_gaps


def hit(pid, entity, content):
    return {"id": pid, "title": entity, "contents": content, "metadata": {"source_id": pid},
            "evidence_entities": [entity], "score": 1., "rank": 1}


def test_gap_rejects_changed_entities_and_answer_number_in_search():
    with pytest.raises(ValueError):
        validate_gaps({"gaps": [{"entity": "丙", "missing_facet": "培养方式", "search_terms": ["实践教学"]}]}, ["乙"])
    with pytest.raises(ValueError):
        validate_gaps({"gaps": [{"entity": "乙", "missing_facet": "排名", "search_terms": ["全国第1"]}]}, ["乙"])


def test_targeted_query_keeps_original_year_facet_and_entity():
    plan = ComparisonPlan("2022年甲与乙培养方式有何不同", ("甲", "乙"), "培养方式", ("甲", "乙"))
    value = validate_gaps({"gaps": [{"entity": "乙", "missing_facet": "实践教学", "search_terms": ["实验课程"]}]}, ["乙"])
    query = queries_for(plan, value)["乙"]
    assert all(term in query for term in ("2022年", "乙专业", "培养方式", "实验课程"))


def test_content_set_detects_id_alias_and_reorder_without_new_information():
    a, b = hit("a", "甲", "旧正文甲"), hit("b", "乙", "旧正文乙")
    assert evidence_set([a, b]) == evidence_set([b, {**a, "id": "alias"}])
    assert evidence_set([a, b]) != evidence_set([a, {**b, "contents": "新正文乙"}])


def test_joint_selection_keeps_both_entities_and_one_cited_anchor_with_room_for_new():
    from app.comparison import ExactEvidenceIndex
    from types import SimpleNamespace
    from scripts.experiment_gap_recovery import digest
    a = hit("a", "甲", "甲专业已引用的培养机制")
    b = hit("b", "乙", "乙专业已引用的培养机制")
    new = [hit(str(i), "甲" if i % 2 else "乙", "新的课程描述" + str(i)) for i in range(6)]
    seen_queries = []
    class Ranker:
        def rerank(self, question, hits, top_k):
            seen_queries.append(question)
            return list(reversed(hits))
    engine = SimpleNamespace(exact=ExactEvidenceIndex([a, b] + new), reranker=Ranker())
    first = {"evidence": [{"passage_id": h["id"], "content": h["contents"]} for h in [a, b]],
             "aspects": [{"claims": [{"quotes": [{"passage_id": h["id"]}]}]} for h in [a, b]]}
    plan = ComparisonPlan("甲乙的培养方法有哪些差异", ("甲", "乙"), "培养方法", ("甲", "乙"))
    selected, trace = joint_merge(engine, plan, [a, b], new, first)
    assert seen_queries == [plan.question]
    assert {"a", "b"} <= {h["id"] for h in selected}
    assert len(selected) == 5 and len(trace["protected_ids"]) == 2
    assert {e for h in selected for e in h["evidence_entities"]} == {"甲", "乙"}
    assert digest(a["contents"]) in evidence_set(selected)
