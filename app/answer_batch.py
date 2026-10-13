"""Saved-retrieval answer trials with actual input and honest failure accounting.

Gold labels are used only for scoring/recording, never passed to the generator.
Repeated trials reuse the same saved retrieval pool. They estimate generation
variation conditional on that pool, not end-to-end reretrieval variation.
"""
from dataclasses import asdict, dataclass
import time

from app.generation import assemble_context, filter_event_time_evidence
from app.methods import METHOD_NAMES
from app.telemetry import safe_usage


@dataclass(frozen=True)
class AnswerBudget:
    repeats: int = 3
    context_max_chars: int = 16000
    max_output_tokens: int = 1200
    temperature: float = 0.0

    def __post_init__(self):
        if type(self.repeats) is not int or not 1 <= self.repeats <= 3:
            raise ValueError('Use one to three planned repeats')
        if not 200 <= self.context_max_chars <= 16000 or not 1 <= self.max_output_tokens <= 8192:
            raise ValueError('Invalid answer budget')
        if not 0 <= self.temperature <= 2:
            raise ValueError('Invalid temperature')


def validate_saved_retrieval(labels, rows, methods, chunks, final_k=5):
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHOD_NAMES):
        raise ValueError('Unknown or duplicate answer schemes')
    by_case = {t['id']: t for t in labels}
    if len(by_case) != len(labels):
        raise ValueError('Duplicate task IDs')
    indexed = {}
    chunks = {str(c['id']): c for c in chunks}
    for row in rows:
        if row['method'] not in METHOD_NAMES:
            raise ValueError('Unknown retrieval scheme')
        key = row['case_id'], row['method']
        if key in indexed:
            raise ValueError('Duplicate retrieval case/scheme')
        if row['case_id'] not in by_case:
            raise ValueError('Foreign retrieval task')
        indexed[key] = row
        final = row['final']
        hits = final.get('hits')
        if final['execution_status'] != 'ok':
            if hits is not None:
                raise ValueError('Failed retrieval cannot supply successful hits')
            continue
        if not isinstance(hits, list) or len(hits) > final_k or len({str(h['id']) for h in hits}) != len(hits):
            raise ValueError('Invalid final evidence budget/identities')
        for hit in hits:
            original = chunks.get(str(hit['id']))
            if original is None or hit['contents'] != original['contents'] or hit['title'] != original['title']:
                raise ValueError('Saved retrieval evidence differs from the fixed source snapshot')
            if str(hit['metadata']['source_id']) != str(original['metadata']['source_id']):
                raise ValueError('Saved evidence source differs from the fixed snapshot')
            if hit['metadata'].get('url') != original['metadata'].get('url'):
                raise ValueError('Saved source URL differs from the fixed snapshot')
    if any((cid, method) not in indexed for cid in by_case for method in methods):
        raise ValueError('Saved retrieval must contain every planned task/scheme, including failures')
    return indexed


def plan_trials(labels, indexed, methods, budget):
    return [dict(trial_id=f"{task['id']}--{method}--{repeat}", case_id=task['id'], method=method,
                 repeat=repeat, planned_repeats=budget.repeats,
                 question=task['question'], question_type=task['question_type'],
                 label_status=task['label_status'], execution_status='not_run',
                 answer=None, citations=[], evidence=[], knowledge_version=None,
                 retrieval_execution_status=indexed[(task['id'], method)]['final']['execution_status'])
            for task in labels for method in methods for repeat in range(1, budget.repeats + 1)]


def execute_trial(planned, retrieval, snapshot, generate_answer, budget=AnswerBudget()):
    """Call once at most. No hidden retry or retrieval fallback in this experiment."""
    started = time.perf_counter()
    trial = {**planned, 'knowledge_version': snapshot.version,
             'answer_budget': asdict(budget), 'usage': safe_usage({}), 'error_type': None,
             'model_called': False, 'human_review_status': 'pending'}
    if retrieval['final']['execution_status'] != 'ok':
        return {**trial, 'execution_status': 'error', 'failure_stage': 'retrieval_unavailable',
                'elapsed_ms': (time.perf_counter() - started) * 1000}
    hits = retrieval['final']['hits']
    usable, excluded = filter_event_time_evidence(trial['question'], hits)
    _, evidence = assemble_context(usable, budget.context_max_chars)
    evidence = [{**e, 'knowledge_version': snapshot.version} for e in evidence]
    trial.update(evidence=evidence, excluded_listing_ids=excluded)
    try:
        output = generate_answer(trial['question'], hits, context_max_chars=budget.context_max_chars,
                                 max_output_tokens=budget.max_output_tokens, temperature=budget.temperature)
        # Record what was actually submitted, including clipping and filtering.
        # A different adapter/context must not masquerade as this frozen trial.
        if output.get('evidence') != [{k: v for k, v in e.items() if k != 'knowledge_version'} for e in evidence]:
            raise ValueError('Generation context differs from its recorded input')
        if not isinstance(output.get('answer'), str) or not output['answer'].strip():
            raise ValueError('Missing actual answer')
        if type(output.get('insufficient_evidence')) is not bool or not isinstance(output.get('citations'), list):
            raise ValueError('Invalid actual answer structure')
        allowed = {e['passage_id'] for e in evidence}
        if any(not isinstance(c, str) or c not in allowed for c in output['citations']):
            raise ValueError('Unknown citation')
        if not output['insufficient_evidence'] and not output['citations']:
            raise ValueError('Answer without citations')
        trial.update(answer=output['answer'], citations=output['citations'],
                     insufficient_evidence=output['insufficient_evidence'],
                     model_called=output.get('model_called'), model=output.get('model'),
                     usage=safe_usage({'usage': output.get('usage')}), execution_status='ok',
                     failure_stage='answer_unreviewed')
    except Exception as error:
        # The per-attempt transport ledger supplies usage/time on failures.
        # Only class/category is retained here; no provider body or credentials.
        cause = type(error).__name__
        category = ('timeout' if 'Timeout' in cause or str(error) == 'Model request timed out.'
                    else 'invalid_output' if isinstance(error, ValueError)
                    or str(error) in {'Model returned an invalid answer structure.',
                                     'Model cited a passage outside the supplied context.',
                                     'Model answered without citing supplied evidence.'}
                    else 'error')
        trial.update(execution_status=category, error_type=cause, failure_stage='generation_' + category,
                     model_called=None if evidence else False)
    return {**trial, 'elapsed_ms': (time.perf_counter() - started) * 1000}
