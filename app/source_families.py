"""Conservative grouping before QA split assignment, without retrieval outputs.

Exact normalized duplicate passages join their entire source families. Similar
questions and URLs are audit warnings, not silently asserted equal facts. Exact
grouping cannot detect all paraphrases/reposts; semantic warnings need review.
"""
from __future__ import annotations

import hashlib
import re


def normalize(text):
    return re.sub(r'\s+', '', text).replace('\\n', '').replace('\\r', '').replace('\\t', '').lower()


class Families:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        current = key
        while self.parent[current] != current:
            current = self.parent[current]
        while key != current:
            previous = self.parent[key]
            self.parent[key] = current
            key = previous
        return current

    def join(self, left, right):
        roots = sorted({self.find(left), self.find(right)})
        self.parent[roots[-1]] = roots[0]


def build_source_families(corpus):
    families, seen, duplicate_groups = Families(), {}, {}
    for doc in corpus:
        sid, pid = str(doc['metadata']['source_id']), str(doc['id'])
        families.find(sid)
        text = normalize(doc['contents'])
        # Short boilerplate can connect otherwise unrelated pages. Joining is
        # conservative for split safety; record all links for human inspection.
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if digest in seen:
            other_source, other_pid = seen[digest]
            if other_source != sid:
                families.join(other_source, sid)
                duplicate_groups.setdefault(digest, {(other_source, other_pid)}).add((sid, pid))
        else:
            seen[digest] = sid, pid
    by_family = {}
    for sid in sorted(families.parent):
        root = families.find(sid)
        by_family.setdefault(root, []).append(sid)
    names = {root: 'family-' + hashlib.sha256('|'.join(sources).encode()).hexdigest()[:16]
             for root, sources in by_family.items()}
    return {'source_to_family': {sid: names[families.find(sid)] for sid in sorted(families.parent)},
            'families': {names[root]: sources for root, sources in by_family.items()},
            'duplicate_links': [{'normalized_sha256': digest, 'passages': [
                {'source_id': sid, 'passage_id': pid} for sid, pid in sorted(items)]}
                for digest, items in sorted(duplicate_groups.items())],
            'limitations': 'Exact normalized copies only; paraphrase, semantic equivalence and source authority are not certified'}


def grams(text, size=3):
    text = normalize(text)
    return {text[i:i + size] for i in range(max(len(text) - size + 1, 1))} if text else set()


def audit_questions(cases, source_families, near_threshold=0.65):
    if not 0 < near_threshold <= 1:
        raise ValueError('Invalid similarity threshold')
    mapped, links, unknown = [], [], []
    for case in cases:
        sources = sorted(set(map(str, case['source_ids'])))
        missing = [sid for sid in sources if sid not in source_families['source_to_family']]
        if missing:
            unknown.append({'case_id': case['id'], 'source_ids': missing})
        mapped.append({**case, 'family_ids': sorted({source_families['source_to_family'][sid] for sid in sources if sid not in missing})})
    features = [grams(c['question']) for c in mapped]
    for i, left in enumerate(mapped):
        for j in range(i + 1, len(mapped)):
            right = mapped[j]
            overlap = sorted(set(left['family_ids']) & set(right['family_ids']))
            union = features[i] | features[j]
            similarity = len(features[i] & features[j]) / len(union) if union else 0
            near = similarity >= near_threshold
            if overlap or near:
                links.append({'left': left['id'], 'right': right['id'], 'shared_families': overlap,
                              'question_jaccard_3gram': similarity, 'near_question_warning': near,
                              'cross_split': left.get('split') != right.get('split'),
                              'needs_human_review': near})
    return {'cases': mapped, 'links': links, 'unknown_sources': unknown,
            'near_threshold': near_threshold, 'similarity_scope': 'character 3-gram warnings, not semantic leakage proof'}


def assign_grouped_splits(cases, source_families, *, dev_fraction=0.5, seed='new-v1'):
    """Shared families and flagged similar questions stay in one component.

    Exposed historical/rephrased cases must already be marked regression by the
    curator. This grouping cannot determine novelty from a new ID.
    """
    if not 0 < dev_fraction < 1 or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Invalid split fraction or duplicate question IDs')
    if any(type(c.get('previously_exposed')) is not bool for c in cases):
        raise ValueError('Every case needs an explicit exposure declaration from the curator')
    audit = audit_questions(cases, source_families)
    if audit['unknown_sources']:
        raise ValueError('Cannot split cases with unknown source identities')
    components = Families()
    for c in cases:
        components.find(c['id'])
    for link in audit['links']:
        components.join(link['left'], link['right'])
    members = {}
    for c in cases:
        members.setdefault(components.find(c['id']), []).append(c['id'])
    exposed = {components.find(c['id']) for c in cases if c.get('previously_exposed') is True}
    assignments = {}
    for root, ids in members.items():
        digest = hashlib.sha256((seed + '|' + '|'.join(sorted(ids))).encode()).hexdigest()
        split = 'regression' if root in exposed else ('dev' if int(digest[:8], 16) / 2 ** 32 < dev_fraction else 'acceptance')
        assignments.update({cid: split for cid in ids})
    return [{**c, 'split': assignments[c['id']], 'group_id': components.find(c['id'])} for c in audit['cases']]
