"""One-round, fixed development answer replay; not held-out answer accuracy."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation import split_for
from app.generation import GenerationError, Settings, assemble_context, filter_event_time_evidence, generate
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256
from scripts.compare_qwen_instructions import fingerprint, read, save
from scripts.evaluate_answer_replay import literal_coverage
from scripts.optimize_retrieval import OUT, blend, protect

IDS = [f'domainrag_extractive_{i:03d}' for i in (3, 4, 6, 15, 19, 25, 33, 34, 49)]


def main():
    selection = read(OUT / 'selection.json')
    settings = Settings.load()
    corpus = {str(r['id']): r for r in read_jsonl(CORPUS)}
    qa = {r['id']: r for r in read_jsonl(ROOT / 'data/basic_qa_mapped.jsonl')}
    measurements = [json.loads(l) for l in (OUT / 'dev_rows.jsonl').read_text(encoding='utf-8').splitlines()]
    plan = {'ids_fixed_before_generation': IDS, 'round': 0, 'model': settings.model,
        'selection_sha256': sha256(OUT / 'selection.json'), 'generation_sha256': sha256(ROOT / 'app/generation.py'),
        'methods': ['bge_baseline', 'frozen_winner'],
        'scope': 'Nine selected development questions; one round replay of captured Top5. Literal coverage is not semantic answer correctness or unseen test performance.'}
    protect(OUT / 'answer_probe_plan.json', plan)
    path = OUT / 'answer_probe.json'
    rows = read(path)['rows'] if path.exists() else []
    done = {(r['id'], r['method']) for r in rows}
    for key in IDS:
        sample = qa[key]
        assert split_for(sample) == 'dev'
        for method in plan['methods']:
            if (key, method) in done:
                continue
            variant = 'bge_baseline' if method == 'bge_baseline' else selection['winner']['variant']
            source = next(r for r in measurements if r['id'] == key and r['variant'] == variant and r['round'] == 0)
            ranking = source['ranking'] if method == 'bge_baseline' else blend(source['ranking'], source['candidate_ids'], selection['winner']['alpha'])
            hits = [{**corpus[id], 'rank': n, 'score': 0.0} for n, id in enumerate(ranking[:5], 1)]
            usable, excluded = filter_event_time_evidence(sample['question'], hits)
            context, _ = assemble_context(usable)
            start = time.perf_counter()
            try:
                result = generate(sample['question'], hits, settings=settings)
                status = 'ok'
            except GenerationError as exc:
                result = {'answer': '', 'citations': [], 'insufficient_evidence': None, 'error': str(exc)}
                status = 'error'
            coverage = literal_coverage(result['answer'], sample['expected_answer_groups'])
            row = {'id': key, 'method': method, 'question': sample['question'], 'top5': ranking[:5],
                'context_fingerprint': fingerprint({'query': sample['question'], 'context': context}),
                'excluded_listing_ids': excluded, 'generation_ms': (time.perf_counter() - start) * 1000,
                'status': status, 'literal_keypoint_coverage': coverage,
                'literal_complete_nonrefusal': status == 'ok' and coverage == 1 and result['insufficient_evidence'] is False,
                **result}
            rows.append(row)
            save(path, {'plan': plan, 'rows': rows, 'state': 'running'})
            print(json.dumps({k: row.get(k) for k in ('id', 'method', 'answer', 'citations', 'status', 'literal_keypoint_coverage')}, ensure_ascii=False), flush=True)
    summaries = []
    for method in plan['methods']:
        subset = [r for r in rows if r['method'] == method]
        summaries.append({'method': method, 'questions': len(subset),
            'literal_complete_nonrefusal': sum(r['literal_complete_nonrefusal'] for r in subset),
            'refusals': sum(r['insufficient_evidence'] is True for r in subset),
            'errors': sum(r['status'] != 'ok' for r in subset),
            'prompt_tokens': sum(r.get('usage', {}).get('prompt_tokens', 0) for r in subset),
            'completion_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in subset)})
    save(path, {'plan': plan, 'rows': rows, 'summaries': summaries, 'state': 'finished'})
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
