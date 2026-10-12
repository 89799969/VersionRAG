import copy

import pytest

from app.comparison import ComparisonPlan
from scripts.resume_comparison import judge_payload, merge_answers, recover_from_initial


def test_resume_preserves_original_and_replaces_only_failed_calls():
    original = [{"id": "1", "arm": "original", "question": "Q", "split": "holdout", "status": "error"},
                {"id": "2", "arm": "original", "question": "Q2", "split": "holdout", "status": "ok"}]
    snapshot = copy.deepcopy(original)
    resumed = [{**original[0], "status": "ok", "answer": "new"}]
    effective = merge_answers(original, resumed)
    assert original == snapshot
    assert effective[0]["answer"] == "new" and effective[1] == original[1]
    with pytest.raises(AssertionError):
        merge_answers(original, [{**original[1], "answer": "overwrite successful answer"}])


def test_paired_recovery_reuses_initial_generation_and_does_not_call_retrieval():
    plan = ComparisonPlan("Q", ("甲专业", "乙专业"), "课程", ("甲", "乙"))
    draft = {"aspects": [{"entity": e, "status": "supported", "claims": [], "missing_reason": ""} for e in plan.entities],
             "comparisons": [], "evidence": [], "usage": {"prompt_tokens": 12}}
    class Never:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("No supplemental retrieval expected")
    def never_generate(*args):
        raise AssertionError("Initial draft must be reused")
    result = recover_from_initial(Never(), plan, {"hits": [], "trace": {}}, draft, never_generate)
    assert result["total_usage"]["prompt_tokens"] == 12
    assert not result["trace"]["recovery_triggered"]
    assert len(result["trace"]["generation_attempts"]) == 1


def test_review_reuse_hash_includes_answer_and_actual_context():
    row = {"question": "Q", "answer": "A", "aspects": [], "comparisons": [],
           "evidence": [{"passage_id": "1", "title": "T", "content": "C"}]}
    first, _ = judge_payload(row, {"reference_answer": "R"})
    changed = copy.deepcopy(row)
    changed["evidence"][0]["content"] = "C2"
    assert judge_payload(changed, {"reference_answer": "R"})[0] != first
    changed = {**row, "answer": "A2"}
    assert judge_payload(changed, {"reference_answer": "R"})[0] != first
