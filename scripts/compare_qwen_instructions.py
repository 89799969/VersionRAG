"""Instruction ablation for THIS admissions RAG, using its corpus and 90 questions.

Production adapters/configs and historical results are left unchanged. Freeze one
30-candidate hybrid pool per query before running fresh paired rerank requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dense import DenseRetriever
from app.embedding_api import EmbeddingSettings
from app.evaluation import evidence_metrics, split_for
from app.hybrid import HybridRetriever
from app.retrieval import CORPUS, ROOT, ChineseBM25, read_jsonl, sha256

OUT = ROOT / 'artifacts/reports/qwen_instruction_2026-10-07'
QWEN = 'Qwen/Qwen3-Reranker-0.6B'
GENERAL = 'Given a web search query, retrieve relevant passages that answer the query'
ADMISSIONS = (
    'Retrieve passages from Chinese university admissions documents that directly answer the query. '
    'Match the specified year, admissions category, college or major, eligibility conditions, and deadlines. '
    'Prioritize exact applicable rules over merely similar topics; do not substitute rules from a different year or category.'
)
VARIANTS = {
    'bge_control': {'model': 'BAAI/bge-reranker-v2-m3', 'instruction': None},
    'qwen_unspecified': {'model': QWEN, 'instruction': None},
    'qwen_general': {'model': QWEN, 'instruction': GENERAL},
    'qwen_admissions': {'model': QWEN, 'instruction': ADMISSIONS},
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def pool_from(question, candidates, retrieval_ms):
    unique = {}
    for row in candidates:
        unique.setdefault(str(row['id']), row)
    hits = list(unique.values())
    # Exactly the production BGE adapter's pre-rerank clipping; all four variants
    # see these SAME strings, including whitespace. Query is never rewritten.
    budget = 8000 - len(question.encode('utf-8'))
    if budget <= 0 or not question.strip() or len(hits) != 30:
        raise ValueError('Expected a nonblank query and exactly 30 hybrid candidates')
    texts = [row['contents'].encode('utf-8')[:budget].decode('utf-8', errors='ignore') for row in hits]
    value = {'question': question, 'candidate_ids': [str(row['id']) for row in hits], 'documents': texts}
    return {**value, 'fingerprint': fingerprint(value), 'retrieval_ms': retrieval_ms,
        'channel_ranks': [row.get('channel_ranks') for row in hits],
        'truncated_documents': sum(a != row['contents'] for a, row in zip(texts, hits))}


def payload_for(pool, variant):
    config = VARIANTS[variant]
    body = {'model': config['model'], 'query': pool['question'], 'documents': pool['documents'],
            'top_n': len(pool['candidate_ids']), 'return_documents': False}
    if config['instruction'] is not None:
        body['instruction'] = config['instruction']
    if variant == 'bge_control':
        body.update(max_chunks_per_doc=8, overlap_tokens=50)
    return body


def parse_result(data, ids):
    results = data['results']
    if len(results) != len(ids):
        raise ValueError('Incomplete rerank response')
    seen, ranked = set(), []
    for row in results:
        index, score = row['index'], float(row['relevance_score'])
        if type(index) is not int or not 0 <= index < len(ids) or index in seen or not math.isfinite(score):
            raise ValueError('Invalid rerank response index or score')
        seen.add(index)
        ranked.append((ids[index], score))
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return [pair[0] for pair in ranked], [pair[1] for pair in ranked]


def score(sample, ranking):
    at5 = evidence_metrics(sample, ranking, 5)
    at10 = evidence_metrics(sample, ranking, 10)
    at30 = evidence_metrics(sample, ranking, 30)
    relevant = {str(key) for group in sample['required_evidence_groups'] for key in group['acceptable_corpus_passage_ids']}
    dcg = sum((1 if key in relevant else 0) / math.log2(i+2) for i, key in enumerate(ranking[:5]))
    idcg = sum(1 / math.log2(i+2) for i in range(min(5, len(relevant))))
    return {'evidence_recall_at_5': at5['evidence_group_recall'],
        'evidence_recall_at_10': at10['evidence_group_recall'],
        'all_evidence_hit_at_5': at5['all_evidence_hit'],
        'mrr_at_5': at5['reciprocal_rank'], 'mrr_at_30': at30['reciprocal_rank'],
        'ndcg_at_5_binary_annotations': dcg/idcg}


class Service:
    def __init__(self, settings=None, transport=None):
        self.settings = settings or EmbeddingSettings.load()
        self.client = httpx.Client(timeout=self.settings.timeout, trust_env=False, transport=transport)
        self.requests = 0
        self.failures = []

    def rerank(self, pool, variant):
        body = payload_for(pool, variant)
        started = time.perf_counter()
        response = None
        for attempt in range(3):
            self.requests += 1
            try:
                response = self.client.post('https://api.siliconflow.cn/v1/rerank', json=body,
                    headers={'Authorization': 'Bearer ' + self.settings.api_key})
            except httpx.RequestError:
                self.failures.append({'variant': variant, 'kind': 'network_error', 'attempt': attempt+1})
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError('Rerank network failure; details withheld') from None
            if response.status_code != 200:
                self.failures.append({'variant': variant, 'http_status': response.status_code, 'attempt': attempt+1})
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f'Rerank service HTTP {response.status_code}; no model fallback')
            break
        elapsed = (time.perf_counter()-started)*1000
        value = response.json()
        ranking, scores = parse_result(value, pool['candidate_ids'])
        return {'ranking': ranking, 'scores': scores, 'rerank_latency_ms': elapsed,
            'request_id': value.get('id'), 'trace_id': response.headers.get('x-siliconcloud-trace-id'),
            'payload_fingerprint': fingerprint(body)}


def prepare(rounds):
    samples = read_jsonl(ROOT/'data/basic_qa_mapped.jsonl')
    assert len(samples) == 90
    assert {split:sum(split_for(s)==split for s in samples) for split in ('dev','holdout','demo')} == {'dev':35,'holdout':53,'demo':2}
    manifest = {'project': 'rag-domain-study/admissions/DomainRAG', 'rounds': rounds,
        'variants': VARIANTS, 'channel_k': 30, 'fused_candidate_k': 30, 'rrf_constant':60,
        'evaluation_k':[5,10], 'samples':samples,
        'hashes':{str(path.relative_to(ROOT)):sha256(path) for path in (
            CORPUS, ROOT/'data/basic_qa_mapped.jsonl', ROOT/'artifacts/indexes/dense/manifest.json',
            ROOT/'artifacts/indexes/dense/vectors.npy', ROOT/'artifacts/indexes/bm25/manifest.json',
            ROOT/'app/reranking.py', ROOT/'app/hybrid.py', ROOT/'app/evaluation.py')},
        'script_sha256':sha256(Path(__file__)),
        'timing':'Pure rerank request latency, retry waiting included. Precomputed hybrid retrieval measured once per query and not added into repeated latency. Excludes generation.',
        'gold_scope':'Original annotation groups, not exhaustive equivalent evidence. Original holdout already exposed; this is follow-up exploration, not fresh untouched holdout.',
        'instruction_policy':'Fixed generic and admissions instructions before this run; no per-question answers, evidence IDs or gold hints in payload; no prompt tuning after reading results.',
        'server_instruction_default':'Unknown when instruction field omitted',
        'api_docs':'https://docs.siliconflow.cn/docs/api/rerank-post'}
    frozen=OUT/'manifest.json'
    if frozen.exists():
        if read(frozen)!=manifest:
            raise ValueError('Frozen experiment manifest differs; preserve it and choose a new output directory')
    else:
        save(frozen,manifest)
    bm25,dense=ChineseBM25(),DenseRetriever()
    bm25.load(); dense.load()
    hybrid=HybridRetriever(bm25,dense)
    for number,sample in enumerate(samples,1):
        path=OUT/'pools'/(sample['id']+'.json')
        if path.exists():
            assert read(path)['question']==sample['question']
            continue
        started=time.perf_counter()
        candidates=hybrid.search(sample['question'],30)
        pool=pool_from(sample['question'],candidates,(time.perf_counter()-started)*1000)
        save(path,pool)
        if number%10==0:
            print(f'admissions candidate pools {number}/90 frozen',flush=True)
    return manifest


def aggregate(rows, expected):
    summaries=[]
    for variant in VARIANTS:
        subset=[r for r in rows if r['variant']==variant and r['status']=='ok']
        elapsed=[r['rerank_latency_ms'] for r in subset]
        keys=('evidence_recall_at_5','evidence_recall_at_10','all_evidence_hit_at_5','mrr_at_5','mrr_at_30','ndcg_at_5_binary_annotations')
        summaries.append({'variant':variant,'expected_rows':expected,'successful_rows':len(subset),
            'failed_rows':sum(r['variant']==variant and r['status']!='ok' for r in rows),
            **{key:float(np.mean([r[key] for r in subset])) if subset else None for key in keys},
            'rerank_p50_ms':float(np.percentile(elapsed,50)) if elapsed else None,
            'rerank_p95_ms':float(np.percentile(elapsed,95)) if elapsed else None,
            'mean_candidate_count':float(np.mean([r['candidate_count'] for r in subset])) if subset else None})
    return summaries


def run(rounds):
    manifest=prepare(rounds)
    service=Service()
    path=OUT/'rows.jsonl'
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done={(r['round'],r['id'],r['variant']) for r in rows}
    # Warm each variant identically; these requests are not scored or timed in summary.
    warm=read(OUT/'pools'/(manifest['samples'][0]['id']+'.json'))
    for variant in VARIANTS:
        service.rerank(warm,variant)
    for n in range(rounds):
        cases=list(manifest['samples'])
        random.Random(20261007+n).shuffle(cases)
        for number,sample in enumerate(cases,1):
            pool=read(OUT/'pools'/(sample['id']+'.json'))
            names=list(VARIANTS)
            random.Random(f'{n}:{sample["id"]}').shuffle(names)
            for variant in names:
                key=(n,sample['id'],variant)
                if key in done:
                    continue
                row={'round':n,'id':sample['id'],'question':sample['question'],'split':split_for(sample),
                    'variant':variant,'instruction':VARIANTS[variant]['instruction'],
                    'pool_fingerprint':pool['fingerprint'],'candidate_ids':pool['candidate_ids'],
                    'candidate_count':len(pool['candidate_ids'])}
                started=time.perf_counter()
                try:
                    row.update(service.rerank(pool,variant),status='ok')
                    row.update(score(sample,row['ranking']))
                except (RuntimeError,ValueError,KeyError) as exc:
                    row.update(status='error',error_type=type(exc).__name__,elapsed_failure_ms=(time.perf_counter()-started)*1000)
                with path.open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                rows.append(row);done.add(key)
            save(OUT/'status.json',{'state':'running','completed_rows':len(rows),'expected_rows':rounds*90*len(VARIANTS),
                'model_requests_this_process':service.requests,'service_failures_this_process':service.failures})
            if number%10==0:
                print(f'admissions instruction round {n+1}/{rounds}, queries {number}/90, rows {len(rows)}',flush=True)
    save(OUT/'summary.json',{'all':aggregate(rows,rounds*90),
        'per_split':{split:aggregate([r for r in rows if r['split']==split],rounds*sum(split_for(s)==split for s in manifest['samples'])) for split in ('dev','holdout','demo')},
        'per_round':{str(n):{split:aggregate([r for r in rows if r['round']==n and r['split']==split],sum(split_for(s)==split for s in manifest['samples'])) for split in ('dev','holdout','demo')} for n in range(rounds)},
        'timing_scope':manifest['timing'],'generation_evaluated':False})
    save(OUT/'status.json',{'state':'finished','completed_rows':len(rows),'expected_rows':rounds*90*len(VARIANTS),
        'model_requests_this_process':service.requests,'service_failures_this_process':service.failures})
    print('Admissions RAG instruction experiment finished.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args()
    if args.rounds<1:
        parser.error('rounds must be positive')
    run(args.rounds)


if __name__=='__main__':
    main()
