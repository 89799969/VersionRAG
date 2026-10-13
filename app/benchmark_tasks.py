"""Curated task drafts and explicit gold review, independent of retrieval output.

Sources are fixed upstream passages. Position alignment is mechanical; semantic
gold, novelty and temporal validity remain pending until actual human review.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from pydantic import AwareDatetime, Field, StrictBool, model_validator

from app.answer_evaluation import CaseRubric, RequiredFact, StrictRecord
from app.evidence_alignment import make_anchor
from app.review_workflow import record_hash
from app.source_coordinates import locate_reference


class SourceQuote(StrictRecord):
    source_id: str
    quote: str = Field(min_length=6)


class FactDraft(RequiredFact):
    sources: list[SourceQuote] = Field(min_length=1)


class TaskDraft(StrictRecord):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    question_type: Literal['single_fact', 'multi_fact', 'comparison', 'temporal_fact', 'insufficient', 'temporal_insufficient', 'scope_conflict']
    expected_action: Literal['answer', 'refuse', 'clarify'] = 'answer'
    facts: list[FactDraft] = Field(default_factory=list)
    background: list[SourceQuote] = Field(default_factory=list)
    expected_answer: str
    temporal_scope: str
    review_notes: str = ''
    previously_exposed: StrictBool = False

    @model_validator(mode='after')
    def valid_task(self):
        if self.expected_action == 'answer' and not self.facts:
            raise ValueError('An answerable draft needs independent facts')
        if self.expected_action != 'answer' and not self.background:
            raise ValueError('Insufficiency needs a declared source scope and rationale')
        if len({f.id for f in self.facts}) != len(self.facts):
            raise ValueError('Duplicate fact IDs')
        return self


def align_quote(source: SourceQuote, documents, origins, evidence_id):
    alternatives, issues = [], []
    for pid, origin in origins.items():
        if origin['source_id'] != source.source_id:
            continue
        document_id = origin['document_id']
        located = locate_reference(documents[document_id], source.quote)
        if located['reason'] == 'reference_absent':
            continue
        if located['status'] != 'mapped':
            issues.append({'evidence_id': evidence_id, 'passage_id': pid, 'reason': located['reason']})
            continue
        start, end = located['ranges'][0]
        alternatives.append({**asdict(make_anchor(evidence_id + ':' + pid, document_id, documents[document_id], start, end)),
                             'source_id': source.source_id, 'original_passage_id': pid})
    if not alternatives:
        issues.append({'evidence_id': evidence_id, 'source_id': source.source_id, 'reason': 'no_unambiguous_source_quote'})
    return alternatives, issues


def compile_task(spec: TaskDraft, documents, origins, knowledge_version, label_version):
    groups, issues, group_fact_ids = [], [], []
    targets = [(fact.id, fact.sources) for fact in spec.facts] if spec.expected_action == 'answer' else [
        ('scope-background-' + str(i), [quote]) for i, quote in enumerate(spec.background)]
    for fact_id, sources in targets:
        alternatives = []
        for source_index, source in enumerate(sources):
            anchors, failures = align_quote(source, documents, origins, spec.id + ':' + fact_id + ':' + str(source_index))
            alternatives.extend(anchors)
            issues.extend(failures)
        groups.append(alternatives)
        group_fact_ids.append(fact_id)
    facts = [RequiredFact.model_validate({k: v for k, v in f.model_dump().items() if k != 'sources'}).model_dump() for f in spec.facts]
    rubric = CaseRubric(case_id=spec.id, label_version=label_version, label_status='pending',
                        expected_action=spec.expected_action, required_facts=facts,
                        requires_comparison=spec.question_type == 'comparison', knowledge_version=knowledge_version)
    sources = sorted({quote.source_id for fact in spec.facts for quote in fact.sources} | {quote.source_id for quote in spec.background})
    return {'id': spec.id, 'question': spec.question, 'question_type': spec.question_type,
            'source_ids': sources, 'temporal_scope': spec.temporal_scope, 'expected_answer': spec.expected_answer,
            'review_notes': spec.review_notes, 'expected_action': spec.expected_action,
            'retrieval_target': 'answer_evidence' if spec.expected_action == 'answer' else 'insufficiency_context_only',
            'required_evidence_groups': groups, 'group_fact_ids': group_fact_ids,
            'mapping_status': 'blocked' if any(not g for g in groups) else 'mapped', 'mapping_issues': issues,
            'rubric': rubric.model_dump(), 'label_status': 'pending', 'human_confirmed': False,
            'author_provenance': 'assistant_curated_from_raw_corpus_without_retrieval_results',
            'previously_exposed': spec.previously_exposed,
            'novelty_status': 'pending_manual_check; automated family/question checks are incomplete evidence'}


class GoldReview(StrictRecord):
    case_id: str
    task_sha256: str
    reviewer_type: Literal['human', 'assistant', 'model']
    reviewer_id: str | None = Field(default=None, pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    reviewed_at: AwareDatetime | None = None
    verdict: Literal['approve', 'revise', 'reject', 'pending'] = 'pending'
    question_clear: StrictBool | None = None
    facts_correct: StrictBool | None = None
    evidence_sufficient: StrictBool | None = None
    requirements_complete: StrictBool | None = None
    temporal_scope_correct: StrictBool | None = None
    novelty_checked: StrictBool | None = None
    notes: str = ''


def apply_gold_review(task, review: GoldReview | None):
    if review is None:
        return {**task, 'gold_review_status': 'pending', 'gold_review': None}
    if review.case_id != task['id'] or review.task_sha256 != record_hash(task):
        raise ValueError('Gold review is not bound to this exact task')
    checks = ('question_clear', 'facts_correct', 'evidence_sufficient', 'requirements_complete',
              'temporal_scope_correct', 'novelty_checked')
    if review.verdict == 'approve':
        ready = (review.reviewer_type == 'human' and review.reviewer_id and review.reviewed_at
                 and all(getattr(review, k) is True for k in checks) and task['mapping_status'] == 'mapped'
                 and not task.get('exposure_issues') and task.get('split') != 'regression')
        status = 'human_confirmed' if ready else 'approval_incomplete'
    else:
        status = review.verdict
    confirmed = status == 'human_confirmed'
    return {**task, 'label_status': 'human_confirmed' if confirmed else 'pending', 'human_confirmed': confirmed,
            'rubric': {**task['rubric'], 'label_status': 'human_confirmed' if confirmed else 'pending'},
            'gold_review_status': status, 'gold_review': review.model_dump(mode='json')}
