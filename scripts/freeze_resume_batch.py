"""Freeze a new evidence batch without credentials, model calls or old report writes."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-id', required=True)
    args = parser.parse_args()
    if not args.batch_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.batch_id):
        parser.error('Use an alphanumeric batch ID, with optional hyphens/underscores.')
    directory = ROOT / 'artifacts/resume-evidence' / args.batch_id
    config = dotenv_values(ROOT / '.env')
    endpoint = urlsplit(config.get('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com')
    if endpoint.username or endpoint.password:
        raise ValueError('Model URL contains credentials; use a clean URL before freezing.')
    safe_url = urlunsplit((endpoint.scheme, endpoint.netloc, endpoint.path, '', ''))
    source_files = [p for folder in ('app', 'scripts', 'tests') for p in (ROOT / folder).rglob('*.py')]
    code = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(source_files)}
    data = {name: digest(ROOT / 'data' / name) for name in (
        'domainrag_text_corpus.jsonl', 'basic_qa_mapped.jsonl', 'extended_qa_mapped.jsonl')}
    dense_manifest = ROOT / 'artifacts/indexes/dense/manifest.json'
    dense = json.loads(dense_manifest.read_text(encoding='utf-8')) if dense_manifest.exists() else {}
    encoder = dense.get('encoder', {})
    safe_encoder = {key: encoder.get(key) for key in (
        'model', 'model_revision', 'dimension', 'normalization', 'query_instruction',
        'max_input_utf8_bytes', 'document_preprocessing')}
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {
        'schema_version': 'resume-evidence-v1', 'batch_id': args.batch_id,
        'project': 'VersionRAG', 'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'baseline_commit': head, 'environment': {'python': platform.python_version(), 'platform': platform.system()},
        'code_sha256': code, 'data_sha256': data,
        'models': {'embedding': safe_encoder, 'reranker': 'BAAI/bge-reranker-v2-m3',
                   'generation': {'model': config.get('DEEPSEEK_MODEL') or 'deepseek-flash',
                                  'base_url': safe_url, 'weight_revision': 'not_exposed'}},
        'source_scope': 'immutable upstream preprocessed passages; not recovered full webpages',
        'budgets': {'candidate_per_channel': 30, 'rerank_input_per_single_query': 30, 'final_top_k': 5,
                    'comparison_budget': 'must be recorded per entity and per task', 'new_remote_call_budget': None},
        'new_cases': {'status': 'not_prepared', 'planned': 80, 'planned_dev': 40, 'planned_acceptance': 40,
                      'actual': 0, 'human_confirmed': 0, 'label_sha256': None, 'acceptance_frozen': False},
        'historical_scope': {'basic_dev': 35, 'original_holdout': 53, 'demo': 2,
                             'previously_exposed': True, 'use': 'historical regression, never new blind acceptance'},
        'runtime': {'status': 'not_run', 'real_users': None, 'observed_requests': None, 'money_cost': None},
    }
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'batch_id': args.batch_id, 'directory': str(directory), 'baseline_commit': head,
                      'remote_calls': 0, 'new_cases': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
