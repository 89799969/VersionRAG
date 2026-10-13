"""Balance whole source/question components using task metadata, not outcomes.

This is a provisional preparation step. Once acceptance is frozen it must not
be rebalanced to chase retrieval/generation performance.
"""
from collections import Counter, defaultdict
import hashlib
import itertools


def balance_task_groups(tasks, seed='VersionRAG-draft-split-v1', *, locked_dev_ids=()):
    groups = defaultdict(list)
    for task in tasks:
        if task['split'] != 'regression':
            groups[task['group_id']].append(task)
    keys = sorted(groups, key=lambda k: hashlib.sha256((seed + '|' + k).encode()).hexdigest())
    locked_dev_ids = set(locked_dev_ids)
    if locked_dev_ids - {t['id'] for t in tasks}:
        raise ValueError('Previously queried dev tasks cannot silently disappear')
    locked_groups = {t['group_id'] for t in tasks if t['id'] in locked_dev_ids and t['split'] != 'regression'}
    locked_indices = {i for i, key in enumerate(keys) if key in locked_groups}
    vectors = [Counter(t['question_type'] for t in groups[k]) for k in keys]
    totals = sum(vectors, Counter())
    n = sum(totals.values())
    feasible_types = {kind for kind in totals if sum(v[kind] > 0 for v in vectors) >= 2}
    def objective(bits):
        dev = Counter()
        for bit, vector in zip(bits, vectors):
            if bit:
                dev.update(vector)
        missing = sum(dev[k] == 0 or dev[k] == totals[k] for k in feasible_types)
        # Lexicographic: cover types when feasible, then balance counts/types.
        imbalance = abs(2 * sum(dev.values()) - n)
        type_distance = sum(abs(2 * dev[k] - totals[k]) / totals[k] for k in totals)
        return missing, imbalance, type_distance
    if len(keys) - len(locked_indices) <= 18:
        free_indices = [i for i in range(len(keys)) if i not in locked_indices]
        def expand(free):
            mapping = dict(zip(free_indices, free))
            return tuple(True if i in locked_indices else mapping[i] for i in range(len(keys)))
        bits = min((expand(free) for free in itertools.product((False, True), repeat=len(free_indices))), key=objective)
        strategy = 'exhaustive whole-component metadata balance'
    else:
        # Deterministic local search for larger preparation batches. This is a
        # heuristic, not a guarantee that the globally best partition is found.
        bits = tuple(i in locked_indices or i % 2 == 0 for i in range(len(keys)))
        for _ in range(4 * len(keys)):
            proposals = [tuple(not b if j == i else b for j, b in enumerate(bits))
                         for i in range(len(keys)) if i not in locked_indices]
            if not proposals:
                break
            selected = min(proposals, key=objective)
            if objective(selected) >= objective(bits):
                break
            bits = selected
        strategy = 'deterministic local search over whole components'
    assignments = {key: 'dev' if bit else 'acceptance' for key, bit in zip(keys, bits)}
    result = [{**t, 'split': assignments[t['group_id']] if t['split'] != 'regression' else 'regression'} for t in tasks]
    counts = {split: dict(Counter(t['question_type'] for t in result if t['split'] == split))
              for split in ('dev', 'acceptance', 'regression')}
    audit = {'strategy': strategy, 'seed': seed, 'fresh_component_count': len(keys),
             'type_counts_by_split': counts, 'objective': objective(bits),
             'metadata_only': True, 'retrieval_results_used': False, 'acceptance_frozen': False,
             'locked_previously_queried_dev_ids': sorted(locked_dev_ids),
             'locked_dev_component_count': len(locked_groups),
             'types_not_splittable_across_both': sorted(set(totals) - feasible_types)}
    return result, audit
