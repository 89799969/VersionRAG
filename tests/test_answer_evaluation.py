"""Teaching counterexamples for the rubric; no real gold labels or model calls."""
import copy

import pytest
from pydantic import ValidationError

from app.answer_evaluation import AnswerReview, CaseRubric, evaluate_answer, summarize_answers


def fixture():
    rubric = {"case_id": "teaching", "label_version": "fixture-v1", "label_status": "synthetic",
              "expected_action": "answer", "knowledge_version": "v1", "required_facts": [
                  {"id": "fee", "entity": "示例专业A", "attribute": "学费", "values": ["8000元"], "year": "2025"}]}
    trial = {"case_id": "teaching", "execution_status": "ok", "answer": "示例专业A学费8000元", "citations": ["a"], "knowledge_version": "v1", "evidence": [
        {"passage_id": "a", "content": "教学构造：2025年示例专业A学费为每年8000元。", "knowledge_version": "v1"}]}
    review = {"reviewer_type": "synthetic", "action_correct": True, "time_scope_correct": True,
              "no_extra_errors": True, "claim_inventory_complete": True, "required_fact_mapping_complete": True,
              "claims": [{"id": "c1", "text": "示例专业A学费8000元", "fact_id": "fee", "entity": "示例专业A",
                          "attribute": "学费", "value": "8000元", "year": "2025", "conditions": [], "correct": True,
                          "citations": [{"passage_id": "a", "quote": "示例专业A学费为每年8000元。", "supported": True}]}]}
    return rubric, trial, review


def score(rubric, trial, review):
    return evaluate_answer(CaseRubric.model_validate(rubric), trial,
                           AnswerReview.model_validate(review) if review is not None else None)


def test_fixture_pass_never_becomes_independent_human_accuracy():
    result = score(*fixture())
    assert result["decision"] == "PASS"
    assert result["main_decision"] == "INCOMPLETE" and not result["human_reviewed"]
    summary = summarize_answers([result])
    assert summary["main_results"] == {"PASS": 0, "FAIL": 0, "INCOMPLETE": 1}
    assert summary["strict_main_pass"]["value"] is None


@pytest.mark.parametrize("field,value", [("year", "2024"), ("entity", "示例专业B"), ("value", "6000元")])
def test_correct_number_with_wrong_year_profession_or_value_is_a_failure(field, value):
    rubric, trial, review = fixture()
    review["claims"][0][field] = value
    result = score(rubric, trial, review)
    assert result["decision"] == "FAIL" and result["metrics"]["fact_completeness"]["numerator"] == 0


def test_existing_id_and_exact_quote_do_not_replace_semantic_support():
    rubric, trial, review = fixture()
    review["claims"][0]["citations"][0]["supported"] = False
    result = score(rubric, trial, review)
    pair = result["citation_pairs"][0]
    assert pair["id_exists"] and pair["quote_matches"]
    assert result["decision"] == "FAIL" and result["metrics"]["citation_support"]["numerator"] == 0


def test_actual_context_truncation_cannot_be_repaired_with_full_source():
    rubric, trial, review = fixture()
    trial["evidence"][0]["content"] = "教学构造：2025年示例专业A"
    result = score(rubric, trial, review)
    assert result["decision"] == "FAIL" and not result["citation_pairs"][0]["quote_matches"]


def test_missing_other_side_is_not_complete_and_wrong_comparison_fails():
    rubric, trial, review = fixture()
    rubric["required_facts"].append({"id": "fee-b", "entity": "示例专业B", "attribute": "学费", "values": ["6000元"]})
    result = score(rubric, trial, review)
    assert result["decision"] == "FAIL"
    assert result["metrics"]["fact_completeness"]["numerator"] == 1
    assert result["metrics"]["fact_completeness"]["denominator"] == 2
    rubric["required_facts"].pop()
    rubric["requires_comparison"] = True
    review["comparisons_correct"] = False
    assert score(rubric, trial, review)["decision"] == "FAIL"


@pytest.mark.parametrize("path", ["action_correct", "time_scope_correct", "claim_inventory_complete",
                                  "required_fact_mapping_complete", "correct", "supported", "entity"])
def test_missing_review_fields_remain_incomplete_not_false_success_or_false_failure(path):
    rubric, trial, review = fixture()
    if path in ("correct", "entity"):
        review["claims"][0].pop(path)
    elif path == "supported":
        review["claims"][0]["citations"][0].pop(path)
    else:
        review.pop(path)
    result = score(rubric, trial, review)
    assert result["decision"] == "INCOMPLETE"
    if path == "supported":
        assert result["metrics"]["citation_coverage"]["value"] is None


def test_refusal_without_citations_can_pass_but_zero_denominator_is_null():
    rubric, trial, review = fixture()
    rubric.update({"expected_action": "refuse", "required_facts": []})
    trial["evidence"] = []
    trial["citations"] = []
    review["claims"] = []
    result = score(rubric, trial, review)
    assert result["decision"] == "PASS" and result["metrics"]["citation_support"]["value"] is None
    review["action_correct"] = False
    assert score(rubric, trial, review)["decision"] == "FAIL"


def test_label_pending_and_model_judgment_do_not_become_main_pass():
    rubric, trial, review = fixture()
    rubric["label_status"] = "pending"
    review.update({"reviewer_type": "human", "reviewer_id": "reviewer_1", "reviewed_at": "2026-10-13T12:00:00+08:00"})
    assert score(rubric, trial, review)["main_decision"] == "INCOMPLETE"
    rubric["label_status"] = "human_confirmed"
    assert score(rubric, trial, review)["main_decision"] == "PASS"
    review.update({"reviewer_type": "model", "judge_model": "offline-fixture"})
    assert score(rubric, trial, review)["main_decision"] == "INCOMPLETE"


def test_known_failure_wins_over_unknown_and_model_boolean_strings_are_rejected():
    rubric, trial, review = fixture()
    review["no_extra_errors"] = False
    review.pop("action_correct")
    assert score(rubric, trial, review)["decision"] == "FAIL"
    review["no_extra_errors"] = "false"
    with pytest.raises(ValidationError):
        score(rubric, trial, review)


@pytest.mark.parametrize("status", ["timeout", "error", "invalid_output"])
def test_generation_failures_stay_in_planned_denominator(status):
    rubric, trial, _ = fixture()
    rubric["label_status"] = "human_confirmed"
    trial["execution_status"] = status
    result = score(rubric, trial, None)
    assert result["main_decision"] == "FAIL"
    assert summarize_answers([result])["strict_main_pass"]["denominator"] == 1


def test_fact_cannot_disable_its_required_citation_and_versions_must_match():
    rubric, trial, review = fixture()
    review["claims"][0].update({"requires_source": False, "citations": []})
    assert score(rubric, trial, review)["decision"] == "FAIL"
    rubric, trial, review = fixture()
    trial["evidence"][0]["knowledge_version"] = "v2"
    assert score(rubric, trial, review)["decision"] == "FAIL"


def test_missing_actual_output_cannot_pass_and_reviewer_cannot_rewrite_claim():
    rubric, trial, review = fixture()
    trial.pop('answer')
    assert score(rubric, trial, review)['decision'] == 'INCOMPLETE'
    rubric, trial, review = fixture()
    review['claims'][0]['text'] = '示例专业A学费6000元'
    with pytest.raises(ValueError, match='absent'):
        score(rubric, trial, review)


def test_reviewer_cannot_repair_original_wrong_citations_with_another_context():
    rubric, trial, review = fixture()
    trial['evidence'].append({'passage_id': 'b', 'content': '招生咨询电话12345', 'knowledge_version': 'v1'})
    trial['citations'] = ['b']
    result = score(rubric, trial, review)
    assert result['decision'] == 'FAIL'
    assert 'review_added_citation:c1:a' in result['failures']
    assert result['metrics']['citation_coverage']['numerator'] == 0


def test_missing_original_citation_list_and_unreviewed_extra_citation_are_unknown():
    rubric, trial, review = fixture()
    trial.pop('citations')
    assert score(rubric, trial, review)['decision'] == 'INCOMPLETE'
    rubric, trial, review = fixture()
    trial['evidence'].append({'passage_id': 'b', 'content': '咨询电话', 'knowledge_version': 'v1'})
    trial['citations'].append('b')
    result = score(rubric, trial, review)
    assert result['decision'] == 'INCOMPLETE' and 'output_citations_not_reviewed' in result['unknowns']


def test_actual_refusal_cannot_pass_a_human_claim_that_the_action_was_correct():
    rubric, trial, review = fixture()
    review['actual_action'] = 'refuse'
    result = score(rubric, trial, review)
    assert result['decision'] == 'FAIL' and 'wrong_actual_action' in result['failures']
