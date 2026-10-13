"""Freeze human-gold acceptance only after a complete five-scheme dev comparison.

This command makes no model calls and supplies no human judgments. Gold must
come from the separate reviewed task revision. A freeze is a reproducibility
record, not permission to spend money or a completed evaluation.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.answer_batch import AnswerBudget
from app.batch_retrieval import RetrievalBudget, summarize_retrieval
from app.benchmark_gates import validate_task_run
from app.generation import SYSTEM_PROMPT
from app.methods import METHOD_NAMES
from app.retrieval import ROOT, read_jsonl, sha256
from app.review_workflow import record_hash
from app.telemetry import prompt_signature


def build_freeze(labels, manifest, dev_manifest, dev_rows, *, selected_method, rationale, code, rubrics_hash):
    dev = validate_task_run(labels, manifest, 'dev')
    acceptance = [t for t in labels if t['split'] == 'acceptance']
    if (not manifest.get('acceptance_ready_for_later_freeze') or not acceptance
            or any(t['label_status'] != 'human_confirmed' or t.get('human_confirmed') is not True for t in acceptance)):
        raise ValueError('Acceptance freeze requires human-confirmed gold')
    if any(t.get('previously_queried_dev') or t.get('prior_dev_exposure_in_component') for t in acceptance):
        raise ValueError('Previously queried development components cannot become acceptance')
    budget = asdict(RetrievalBudget())
    if (dev_manifest.get('status') != 'complete' or not dev_manifest.get('remote_enabled')
            or dev_manifest.get('split') != 'dev' or dev_manifest.get('case_ids') != [t['id'] for t in dev]
            or dev_manifest.get('labels_sha256') != manifest['labels_sha256']
            or dev_manifest.get('snapshot_sha256') != manifest['snapshot_sha256']
            or dev_manifest.get('budget') != budget or dev_manifest.get('code_sha256') != code):
        raise ValueError('Development comparison is incomplete or differs from final labels/code/budget')
    models = dev_manifest.get('retrieval_models_indexes')
    if not models or not models.get('dense_manifest_sha256') or not models.get('bm25_manifest_sha256') or not models.get('reranker'):
        raise ValueError('Development models/indexes must be recorded before freezing')
    expected = {(t['id'], method) for t in dev for method in METHOD_NAMES}
    if {(r['case_id'], r['method']) for r in dev_rows} != expected or len(dev_rows) != len(expected):
        raise ValueError('Development comparison must cover all five schemes and all planned dev questions')
    if any(r['final']['execution_status'] != 'ok' or r.get('label_status') != 'human_confirmed'
           or not r.get('final_score') or r['final_score']['status'] != 'ok' for r in dev_rows):
        raise ValueError('Unexecuted/failed/unscorable development results cannot select final acceptance')
    if selected_method not in METHOD_NAMES or not isinstance(rationale, str) or not rationale.strip():
        raise ValueError('Select a known scheme and record the development rationale')
    methods = list(dict.fromkeys(['bm25', selected_method]))
    answer_budget = AnswerBudget()
    config = {'budget': asdict(answer_budget), 'prompt_sha256': prompt_signature(SYSTEM_PROMPT),
              'sampling_note': 'temperature fixed; provider weights/randomness not guaranteed',
              'retrieval_repetition': 'fixed saved hits; repeats only generation'}
    return {'schema_version': 'new-acceptance-freeze-v1', 'acceptance_frozen': True,
            'case_ids': [t['id'] for t in acceptance], 'task_records_sha256': record_hash(acceptance),
            'rubrics_sha256': rubrics_hash, 'labels_sha256': manifest['labels_sha256'],
            'source_bundle_sha256': manifest['source_bundle_sha256'],
            'snapshot_sha256': manifest['snapshot_sha256'], 'knowledge_version': manifest['knowledge_version'],
            'code_sha256': code, 'retrieval_budget': budget, 'generation_config': config,
            'retrieval_models_indexes': models,
            'answer_methods': methods, 'selected_method': selected_method, 'selection_rationale': rationale,
            'dev_answer_evidence_metrics': summarize_retrieval([r for r in dev_rows if r['retrieval_target'] == 'answer_evidence']),
            'new_acceptance_questions': len(acceptance),
            'planned_generation_trials': len(acceptance) * len(methods) * answer_budget.repeats,
            'max_retrieval_logical_calls': {'embedding': len(acceptance), 'rerank': 2 * len(acceptance)},
            'max_retrieval_physical_attempts': 9 * len(acceptance),
            'selection_scope': 'developer choice under fixed evidence budget; not demonstrated answer improvement',
            'supplement_retrieval': 'disabled; no demonstrated dev strict-quality benefit',
            'money_cost': None, 'remote_calls': 0, 'execution_status': 'not_run'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks-dir', type=Path, required=True)
    parser.add_argument('--dev-run', type=Path, required=True)
    parser.add_argument('--selected-method', choices=METHOD_NAMES, required=True)
    parser.add_argument('--rationale', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; acceptance freezes are immutable')
    manifest = json.loads((args.tasks_dir / 'manifest.json').read_text(encoding='utf-8'))
    labels_path, rubric_path = args.tasks_dir / 'labels.jsonl', args.tasks_dir / 'rubrics.jsonl'
    if sha256(labels_path) != manifest['labels_sha256'] or sha256(rubric_path) != manifest['rubrics_sha256']:
        raise ValueError('Reviewed gold labels/rubrics changed')
    dev_manifest = json.loads((args.dev_run / 'manifest.json').read_text(encoding='utf-8'))
    dev_path = args.dev_run / 'retrieval.jsonl'
    if dev_manifest.get('retrieval_sha256') != sha256(dev_path):
        raise ValueError('Development results changed or batch incomplete')
    code = {p.relative_to(ROOT).as_posix(): sha256(p) for folder in ['app', 'scripts']
            for p in sorted((ROOT / folder).rglob('*.py'))}
    freeze = build_freeze(read_jsonl(labels_path), manifest, dev_manifest, read_jsonl(dev_path),
             selected_method=args.selected_method, rationale=args.rationale, code=code, rubrics_hash=sha256(rubric_path))
    # Whitelisted non-secret configuration only; never copy the dotenv map.
    config = dotenv_values(ROOT / '.env')
    model = os.getenv('DEEPSEEK_MODEL') or config.get('DEEPSEEK_MODEL') or 'deepseek-flash'
    endpoint = (os.getenv('DEEPSEEK_BASE_URL') or config.get('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com').rstrip('/')
    parsed = urlsplit(endpoint)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use a clean HTTPS generation endpoint without embedded secrets')
    freeze.update(generation_model=model, generation_base_url=endpoint,
                  generation_weight_revision='not_exposed_by_provider',
                  dev_manifest_sha256=sha256(args.dev_run / 'manifest.json'), dev_retrieval_sha256=sha256(dev_path),
                  created_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'freeze.json').write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'directory': str(args.output_dir), 'acceptance_questions': freeze['new_acceptance_questions'],
                     'planned_generation_trials': freeze['planned_generation_trials'], 'remote_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
