"""Read-only causal diagnostics. No model calls, no original label edits."""
import collections
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation import evidence_metrics, split_for
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256
from scripts.compare_qwen_instructions import save
from scripts.optimize_retrieval import OUT, blend


def main():
    qa_path = ROOT / 'data/basic_qa_mapped.jsonl'
    original_hashes = {str(p): sha256(p) for p in (CORPUS, qa_path)}
    corpus = {str(r['id']): r for r in read_jsonl(CORPUS)}
    qa = {r['id']: r for r in read_jsonl(qa_path)}
    groups = collections.defaultdict(list)
    for key, r in corpus.items():
        groups[r['contents']].append(key)
    # Only byte-for-byte identical Unicode contents, including titles/whitespace.
    # This is a diagnostic alternative, never a replacement for the old metric.
    duplicate_aware = copy.deepcopy(qa)
    for sample in duplicate_aware.values():
        for group in sample['required_evidence_groups']:
            group['acceptable_corpus_passage_ids'] = sorted({other
                for key in group['acceptable_corpus_passage_ids']
                for other in groups[corpus[key]['contents']]})
    rows = {phase: [json.loads(l) for l in (OUT / (phase + '_rows.jsonl')).read_text(encoding='utf-8').splitlines()]
            for phase in ('dev', 'test')}
    diagnostics, recovered, ranking_steps = [], {}, []
    for phase, measurements in rows.items():
        for variant in sorted({r['variant'] for r in measurements}):
            subset = [r for r in measurements if r['variant'] == variant and r['split'] != 'demo']
            for alpha in (0.0, 0.25):
                old, exact, unique = [], [], []
                for row in subset:
                    ranking = blend(row['ranking'], row['candidate_ids'], alpha)
                    a = evidence_metrics(qa[row['id']], ranking, 5)
                    b = evidence_metrics(duplicate_aware[row['id']], ranking, 5)
                    assert b['evidence_group_recall'] >= a['evidence_group_recall']
                    old.append(a['all_evidence_hit']); exact.append(b['all_evidence_hit'])
                    unique.append(len({corpus[key]['contents'] for key in ranking[:5]}))
                diagnostics.append({'phase': phase, 'variant': variant, 'alpha': alpha,
                    'measurements': len(subset), 'original_hit5': float(np.mean(old)),
                    'exact_duplicate_diagnostic_hit5': float(np.mean(exact)),
                    'mean_unique_texts_top5': float(np.mean(unique)),
                    'measurements_with_duplicate_text_top5': sum(n < 5 for n in unique)})
        recovered[phase] = []
        keys = {r['id'] for r in measurements if r['split'] != 'demo'}
        for key in sorted(keys):
            a = next(r for r in measurements if r['id'] == key and r['round'] == 0 and r['variant'] == 'bge_baseline')
            b = next(r for r in measurements if r['id'] == key and r['round'] == 0 and r['variant'] == 'bge_expand100')
            if evidence_metrics(qa[key], a['candidate_ids'], len(a['candidate_ids']))['all_evidence_hit']:
                continue
            if not evidence_metrics(qa[key], b['candidate_ids'], len(b['candidate_ids']))['all_evidence_hit']:
                continue
            relevant = {str(i) for g in qa[key]['required_evidence_groups'] for i in g['acceptable_corpus_passage_ids']}
            rank = lambda ids: next((i for i, k in enumerate(ids, 1) if k in relevant), None)
            pairs = []
            for n in range(3):
                row = next(r for r in measurements if r['id'] == key and r['round'] == n and r['variant'] == 'bge_expand100')
                pairs.append({'round': n + 1, 'rerank': rank(row['ranking']),
                              'after_blend': rank(blend(row['ranking'], row['candidate_ids'], 0.25))})
            recovered[phase].append({'id': key, 'question': qa[key]['question'], 'ranks': pairs})
    for key in ('domainrag_extractive_023', 'domainrag_extractive_037', 'domainrag_extractive_066'):
        relevant = {str(i) for g in qa[key]['required_evidence_groups'] for i in g['acceptable_corpus_passage_ids']}
        rank = lambda ids: next((i for i, k in enumerate(ids, 1) if k in relevant), None)
        for n in range(3):
            a = next(r for r in rows['test'] if r['id'] == key and r['round'] == n and r['variant'] == 'bge_baseline')
            b = next(r for r in rows['test'] if r['id'] == key and r['round'] == n and r['variant'] == 'bge_expand100')
            ranking_steps.append({'id': key, 'round': n + 1, 'baseline_rerank': rank(a['ranking']),
                'expanded_rerank': rank(b['ranking']), 'expanded_after_blend': rank(blend(b['ranking'], b['candidate_ids'], 0.25)),
                'original_fusion_rank': rank(b['candidate_ids'])})
    assert all(sha256(Path(p)) == value for p, value in original_hashes.items())
    counts = {split: {'questions': sum(split_for(s) == split for s in qa.values()),
        'sources': len({str(g['source_id']) for s in qa.values() if split_for(s) == split for g in s['required_evidence_groups']})}
        for split in ('dev', 'holdout', 'demo')}
    value = {'corpus_passages': len(corpus), 'unique_exact_contents': len(groups),
        'redundant_copies': len(corpus) - len(groups), 'redundant_copy_fraction': (len(corpus) - len(groups)) / len(corpus),
        'duplicate_groups': sum(len(ids) > 1 for ids in groups.values()), 'split_counts': counts,
        'case066_identical_body_ids': groups[corpus['686']['contents']],
        'diagnostic_scope': 'All original files and scores preserved. Duplicate-aware scores only expand annotation IDs by exact contents equality; not a model improvement or independently adjudicated semantic Gold.',
        'diagnostic_scores': diagnostics, 'newly_recovered_annotation_candidates': recovered,
        'regression_stage_ranks': ranking_steps, 'original_hashes': original_hashes}
    save(OUT / '原因分析.json', value)
    print('Exact unique contents:', len(groups), 'redundant copies:', len(corpus) - len(groups), flush=True)
    print('All original labels/corpus unchanged; diagnostic scores kept separate.', flush=True)


if __name__ == '__main__':
    main()
