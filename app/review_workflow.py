"""Bind a blind review to one exact trial and rubric; missing review stays pending.

This module does not judge semantics or certify who filled in the review. The
reviewer's declared provenance is saved so an assistant/model review cannot be
mistaken for an independently performed human review.
"""
from __future__ import annotations

import hashlib
import json
import random

from app.answer_evaluation import AnswerReview, CaseRubric, evaluate_answer
from app.answer_metrics import summarize_answer_cohort


def record_hash(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def _index(records, key):
    result = {}
    for record in records:
        identity = record.get(key)
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError('Missing or duplicate ' + key)
        result[identity] = record
    return result


def prepare_blind_review(trials, rubrics, seed=0):
    by_trial = _index(trials, 'trial_id')
    by_case = _index(rubrics, 'case_id')
    order = sorted(by_trial)
    random.Random(seed).shuffle(order)
    private, packets, templates = [], [], []
    for i, trial_id in enumerate(order, 1):
        trial = by_trial[trial_id]
        rubric = by_case.get(trial['case_id'])
        if rubric is None:
            raise ValueError('Trial has no rubric')
        CaseRubric.model_validate(rubric)
        blind_id = 'review-' + str(i).zfill(5)
        binding = {'blind_id': blind_id, 'trial_sha256': record_hash(trial), 'rubric_sha256': record_hash(rubric)}
        private.append({**binding, 'trial_id': trial_id, 'case_id': trial['case_id']})
        # Keep actual input and output but hide method, trial ordinal and trial ID.
        # Knowledge/year must remain visible for source-validity decisions.
        packets.append({**binding, 'question': trial.get('question'), 'execution_status': trial.get('execution_status'),
                        'knowledge_version': trial.get('knowledge_version'), 'answer': trial.get('answer'),
                        'citations': trial.get('citations'), 'insufficient_evidence': trial.get('insufficient_evidence'),
                        'evidence': trial.get('evidence', []),
                        'rubric': {k: v for k, v in rubric.items() if k not in {'case_id', 'label_version'}}})
        templates.append({**binding, 'status': 'pending', 'review': None})
    return private, packets, templates


def apply_reviews(trials, rubrics, private_map, submitted):
    by_trial = _index(trials, 'trial_id')
    by_case = _index(rubrics, 'case_id')
    by_blind = _index(private_map, 'blind_id')
    by_review = _index(submitted, 'blind_id')
    if set(by_review) - set(by_blind):
        raise ValueError('Review contains an unknown blind ID')
    if {r['trial_id'] for r in private_map} != set(by_trial) or len(private_map) != len(by_trial):
        raise ValueError('Review map must cover every planned trial exactly once')
    results = []
    for binding in private_map:
        trial = by_trial[binding['trial_id']]
        rubric = by_case[trial['case_id']]
        if binding['case_id'] != trial['case_id'] or binding['trial_sha256'] != record_hash(trial) or binding['rubric_sha256'] != record_hash(rubric):
            raise ValueError('Trial or rubric changed after review packet creation')
        envelope = by_review.get(binding['blind_id'])
        review = None
        if envelope:
            if envelope['trial_sha256'] != binding['trial_sha256'] or envelope['rubric_sha256'] != binding['rubric_sha256']:
                raise ValueError('Review refers to a different trial or rubric')
            status = envelope.get('status')
            if status not in {'pending', 'submitted'}:
                raise ValueError('Unknown review status')
            if status == 'pending' and envelope.get('review') is not None:
                raise ValueError('Pending review cannot supply judgments')
            if status == 'submitted':
                review = AnswerReview.model_validate(envelope.get('review'))
        result = evaluate_answer(CaseRubric.model_validate(rubric), trial, review)
        results.append({**trial, 'rubric_record': rubric, 'blind_id': binding['blind_id'],
                        'review_record': review.model_dump(mode='json') if review else None, **result})
    summary = summarize_answer_cohort(results)
    summary['provenance_note'] = 'Reviewer declarations are retained; the program cannot independently authenticate the human reviewer'
    return results, summary
