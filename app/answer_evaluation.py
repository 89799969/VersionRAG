"""Strict answer accounting from explicit reviews, never from citation IDs alone.

Semantic correctness/support require human or separately identified model review.
This module validates that review, factual bindings and the actual model context.
Missing judgments stay unknown. Synthetic and model reviews cannot become human
main scores merely because their boolean fields all say true.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, model_validator

METRIC_VERSION = "fact-citation-strict-v3"
Decision = Literal["PASS", "FAIL", "INCOMPLETE"]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequiredFact(StrictRecord):
    id: str = Field(min_length=1)
    entity: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    values: list[str] = Field(min_length=1)
    year: str | None = None
    conditions: list[str] = Field(default_factory=list)


class CaseRubric(StrictRecord):
    case_id: str = Field(min_length=1)
    label_version: str = Field(min_length=1)
    label_status: Literal["pending", "human_confirmed", "upstream_unreviewed", "synthetic"]
    expected_action: Literal["answer", "refuse", "clarify"]
    required_facts: list[RequiredFact] = Field(default_factory=list)
    requires_comparison: bool = False
    knowledge_version: str | None = None

    @model_validator(mode="after")
    def valid_facts(self):
        if len({f.id for f in self.required_facts}) != len(self.required_facts):
            raise ValueError("Required fact IDs must be unique")
        if self.expected_action == "answer" and not self.required_facts:
            raise ValueError("Answerable cases require explicit facts")
        return self


class CitationReview(StrictRecord):
    passage_id: str = Field(min_length=1)
    quote: str | None = None
    supported: StrictBool | None = None


class ClaimReview(StrictRecord):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    fact_id: str | None = None
    entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    year: str | None = None
    conditions: list[str] | None = None
    correct: StrictBool | None = None
    requires_source: StrictBool = True
    citations: list[CitationReview] = Field(default_factory=list)


class AnswerReview(StrictRecord):
    reviewer_type: Literal["human", "model", "assistant", "synthetic"]
    reviewer_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    reviewed_at: AwareDatetime | None = None
    judge_model: str | None = None
    actual_action: Literal["answer", "refuse", "clarify"] | None = None
    action_correct: StrictBool | None = None
    time_scope_correct: StrictBool | None = None
    no_extra_errors: StrictBool | None = None
    claim_inventory_complete: StrictBool | None = None
    required_fact_mapping_complete: StrictBool | None = None
    comparisons_correct: StrictBool | None = None
    claims: list[ClaimReview] | None = None


def metric(numerator, denominator, unit, *, unknown=0):
    """Unknown or empty denominators never turn into a zero cost or 100% score."""
    return {"numerator": numerator, "denominator": denominator, "unit": unit,
            "unknown": unknown,
            "value": None if denominator == 0 or unknown or numerator is None else numerator / denominator}


def compact(text):
    return re.sub(r"\s+", "", text)


def evaluate_answer(rubric: CaseRubric, trial: dict, review: AnswerReview | None) -> dict:
    if trial.get("case_id") != rubric.case_id:
        raise ValueError("Trial and rubric case IDs differ")
    execution = trial.get("execution_status", "not_run")
    if execution not in {"ok", "not_run", "timeout", "error", "invalid_output"}:
        raise ValueError("Unknown execution status")
    evidence = {}
    for row in trial.get("evidence", []):
        pid = str(row["passage_id"])
        if pid in evidence:
            raise ValueError("Actual context passage IDs must be unique")
        evidence[pid] = row
    failures, unknowns = [], []
    if execution in {"timeout", "error", "invalid_output"}:
        failures.append("generation_" + execution)
    elif execution != "ok":
        unknowns.append("generation_not_run")
    answer = trial.get('answer')
    if execution == 'ok' and (not isinstance(answer, str) or not answer.strip()):
        unknowns.append('actual_answer_missing')
    actual_citations = trial.get('citations')
    if execution == 'ok':
        if actual_citations is None:
            unknowns.append('actual_citations_missing')
        elif (not isinstance(actual_citations, list)
              or any(not isinstance(c, str) for c in actual_citations)):
            raise ValueError('Actual citations must be the original list of passage IDs')
        elif any(c not in evidence for c in actual_citations):
            failures.append('actual_citation_outside_context')
    if rubric.knowledge_version is not None and execution == "ok":
        version = trial.get("knowledge_version")
        if version is None:
            unknowns.append("request_version_missing")
        elif version != rubric.knowledge_version:
            failures.append("request_version_wrong")
        for row in evidence.values():
            if row.get("knowledge_version") is None:
                unknowns.append("evidence_version_missing")
            elif row["knowledge_version"] != version:
                failures.append("mixed_evidence_version")

    claim_rows, pairs, covered, observed = [], [], set(), set()
    facts = {f.id: f for f in rubric.required_facts}
    if review is None:
        unknowns.append("review_missing")
    else:
        if review.reviewer_type == "human" and (not review.reviewer_id or not review.reviewed_at):
            unknowns.append("human_provenance_missing")
        if review.reviewer_type == "model" and not review.judge_model:
            unknowns.append("judge_model_missing")
        for name in ("action_correct", "time_scope_correct", "no_extra_errors"):
            value = getattr(review, name)
            if value is False:
                failures.append(name + "_false")
            elif value is None:
                unknowns.append(name + "_unknown")
        if review.actual_action is not None and review.actual_action != rubric.expected_action:
            failures.append('wrong_actual_action')
        if review.claim_inventory_complete is not True:
            unknowns.append("claim_inventory_incomplete")
        if rubric.expected_action == "answer" and review.required_fact_mapping_complete is not True:
            unknowns.append("required_fact_mapping_incomplete")
        if rubric.requires_comparison:
            if review.comparisons_correct is False:
                failures.append("comparison_wrong")
            elif review.comparisons_correct is None:
                unknowns.append("comparison_unreviewed")
        if review.claims is None:
            unknowns.append("claims_unreviewed")
        claims = review.claims or []
        if len({c.id for c in claims}) != len(claims):
            raise ValueError("Reviewed claim IDs must be unique")
        for claim in claims:
            # The review must inventory actual answer text, not a corrected
            # paraphrase supplied by the reviewer after seeing the gold facts.
            if execution == 'ok' and isinstance(answer, str) and answer.strip() and compact(claim.text) not in compact(answer):
                raise ValueError('Reviewed claim text is absent from the actual answer')
            correct = claim.correct
            binding = True
            fact = facts.get(claim.fact_id)
            if claim.fact_id is not None and fact is None:
                raise ValueError("Unknown required fact ID")
            if fact is not None:
                observed.add(fact.id)
                for name in ("entity", "attribute", "year", "conditions"):
                    expected, actual = getattr(fact, name), getattr(claim, name)
                    if expected is None:
                        continue
                    if actual is None:
                        binding = None
                        unknowns.append("fact_binding_missing:" + claim.id + ":" + name)
                    elif (set(actual) != set(expected) if name == "conditions" else actual != expected):
                        binding = False
                        failures.append("wrong_fact_binding:" + claim.id + ":" + name)
                if claim.value is None:
                    binding = None if binding is True else binding
                    unknowns.append("fact_value_missing:" + claim.id)
                elif compact(claim.value) not in {compact(v) for v in fact.values}:
                    binding = False
                    failures.append("wrong_fact_value:" + claim.id)
            if binding is False:
                correct = False
            elif binding is None and correct is True:
                correct = None
            if correct is False:
                failures.append("incorrect_claim:" + claim.id)
            elif correct is None:
                unknowns.append("claim_correctness_unknown:" + claim.id)
            if fact is not None and correct is True and binding is True:
                covered.add(fact.id)

            claim_pairs = []
            if len({c.passage_id for c in claim.citations}) != len(claim.citations):
                raise ValueError("Duplicate claim-citation pairs")
            for citation in claim.citations:
                context = evidence.get(citation.passage_id)
                exists = context is not None
                quote_matches = None if citation.quote is None else bool(
                    exists and citation.quote.strip() and compact(citation.quote) in compact(context["content"]))
                supported = citation.supported
                used_by_model = None if actual_citations is None else citation.passage_id in actual_citations
                if not exists or quote_matches is False:
                    supported = False
                elif used_by_model is False:
                    supported = False
                    failures.append('review_added_citation:' + claim.id + ':' + citation.passage_id)
                elif used_by_model is None and supported is not False:
                    supported = None
                if supported is False:
                    failures.append("unsupported_citation:" + claim.id + ":" + citation.passage_id)
                elif supported is None or quote_matches is None:
                    unknowns.append("citation_support_unknown:" + claim.id + ":" + citation.passage_id)
                pair = {"claim_id": claim.id, "passage_id": citation.passage_id,
                        "id_exists": exists, "used_by_model": used_by_model,
                        "quote_matches": quote_matches, "supported": supported}
                claim_pairs.append(pair)
                pairs.append(pair)
            if not claim_pairs or any(p["supported"] is False or p["quote_matches"] is False for p in claim_pairs):
                source_covered = False
            elif any(p["supported"] is None or p["quote_matches"] is None for p in claim_pairs):
                source_covered = None
            else:
                source_covered = True
            requires_source = claim.requires_source or fact is not None
            if requires_source and not claim_pairs:
                failures.append("uncited_claim:" + claim.id)
            claim_rows.append({"claim_id": claim.id, "correct": correct,
                               "requires_source": requires_source, "source_covered": source_covered})
        missing = sorted(set(facts) - observed)
        if actual_citations is not None and set(actual_citations) - {p['passage_id'] for p in pairs}:
            unknowns.append('output_citations_not_reviewed')
        if rubric.expected_action == "answer" and missing:
            if (review.claim_inventory_complete is True and review.required_fact_mapping_complete is True
                    and review.claims is not None):
                failures.append("missing_required_facts:" + ",".join(missing))
            else:
                unknowns.append("required_facts_unreviewed")

    decision = "FAIL" if failures else ("INCOMPLETE" if unknowns else "PASS")
    source_kind = review.reviewer_type if review else None
    human_ready = bool(review and source_kind == "human" and review.reviewer_id and review.reviewed_at
                       and rubric.label_status == "human_confirmed")
    objective_failure = execution in {"timeout", "error", "invalid_output"} and rubric.label_status == "human_confirmed"
    main_decision = decision if human_ready or objective_failure else "INCOMPLETE"
    required_sources = [c for c in claim_rows if c["requires_source"]]
    return {
        "metric_version": METRIC_VERSION, "case_id": rubric.case_id, "label_version": rubric.label_version,
        "label_status": rubric.label_status, "assessment_source": source_kind,
        "expected_action": rubric.expected_action,
        "actual_action": review.actual_action if review else None,
        "action_correct": review.action_correct if review else None,
        "execution_status": execution,
        "decision": decision, "main_decision": main_decision, "human_reviewed": human_ready,
        "failures": sorted(set(failures)), "unknowns": sorted(set(unknowns)),
        "claim_rows": claim_rows, "citation_pairs": pairs,
        "metrics": {
            "fact_correctness": metric(sum(c["correct"] is True for c in claim_rows), len(claim_rows), "output_facts",
                                       unknown=int(review is None or review.claim_inventory_complete is not True)
                                       + sum(c["correct"] is None for c in claim_rows)),
            "fact_completeness": metric(len(covered), len(facts), "required_facts",
                                        unknown=int(review is None or review.claim_inventory_complete is not True
                                                    or review.required_fact_mapping_complete is not True)
                                        + sum(c["correct"] is None for c in claim_rows)),
            "citation_support": metric(sum(p["supported"] is True and p["quote_matches"] is True for p in pairs),
                                       len(pairs), "claim_citation_pairs",
                                       unknown=int(review is None or review.claim_inventory_complete is not True)
                                       + int('output_citations_not_reviewed' in unknowns)
                                       + sum(p["supported"] is None or p["quote_matches"] is None for p in pairs)),
            "citation_coverage": metric(sum(c["source_covered"] is True for c in required_sources), len(required_sources),
                                        "facts_requiring_sources",
                                        unknown=int(review is None or review.claim_inventory_complete is not True)
                                        + sum(c["source_covered"] is None for c in required_sources)),
        },
    }


def summarize_answers(results: list[dict]) -> dict:
    counts = {status: sum(r["main_decision"] == status for r in results) for status in ("PASS", "FAIL", "INCOMPLETE")}
    return {"metric_version": METRIC_VERSION, "planned_trials": len(results), "main_results": counts,
            "strict_main_pass": metric(counts["PASS"], len(results), "planned_trials", unknown=counts["INCOMPLETE"]),
            "human_review_coverage": metric(sum(r["human_reviewed"] for r in results), len(results), "planned_trials"),
            "scope": "Human-confirmed labels and human reviews; unreviewed outputs remain INCOMPLETE"}
