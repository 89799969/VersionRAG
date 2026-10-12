"""Audit and report the complete admissions retrieval optimization experiment."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation import evidence_metrics, split_for
from app.dense import DenseRetriever, ranked_rows
from app.generation import assemble_context, filter_event_time_evidence
from app.retrieval import CORPUS, ROOT, ChineseBM25, read_jsonl, sha256
from scripts.compare_qwen_instructions import fingerprint, read, save, score
from scripts.optimize_retrieval import OUT, PREVIOUS, RETRIEVAL, blend, fuse, payload, pool_from, summarize
from scripts.evaluate_answer_replay import literal_coverage


def load_rows(phase):
    return [json.loads(l) for l in (OUT / (phase + '_rows.jsonl')).read_text(encoding='utf-8').splitlines()]


def verify(phase, plan, corpus, qa):
    rows = load_rows(phase)
    status = read(OUT / (phase + '_status.json'))
    assert status['state'] == 'finished'
    ids = {k for k, s in qa.items() if (split_for(s) == 'dev') == (phase == 'dev')}
    expected = {(n, k, v) for n in range(3) for k in ids for v in plan['variants']}
    assert len(rows) == len(expected)
    assert {(r['round'], r['id'], r['variant']) for r in rows} == expected
    assert all(r['status'] == 'ok' for r in rows)
    checked = set()
    for r in rows:
        assert r['sample'] == qa[r['id']] and r['split'] == split_for(qa[r['id']])
        config = plan['variants'][r['variant']]
        pool = read(OUT / 'pools' / config['pool'] / (r['id'] + '.json'))
        key = (config['pool'], r['id'])
        if key not in checked:
            channel = read(OUT / 'channels' / (r['id'] + '.json'))
            for hits in channel['channels'].values():
                assert len(set(str(h['id']) for h in hits)) == len(hits)
                for h in hits:
                    assert h['contents'] == corpus[str(h['id'])]['contents']
            recomputed = pool_from(qa[r['id']]['question'], fuse(channel['channels'], RETRIEVAL[config['pool']]))
            assert pool == recomputed
            checked.add(key)
        assert pool['fingerprint'] == r['pool_fingerprint']
        assert r['candidate_ids'] == pool['candidate_ids']
        assert fingerprint(payload(pool, config)) == r['payload_fingerprint']
        assert len(r['ranking']) == len(set(r['ranking'])) == len(pool['candidate_ids'])
        assert set(r['ranking']) == set(pool['candidate_ids'])
        assert len(r['scores']) == len(r['ranking']) and all(math.isfinite(s) for s in r['scores'])
        assert all(a >= b for a, b in zip(r['scores'], r['scores'][1:]))
    return rows, status


def compact(r):
    return {k: r[k] for k in ('variant', 'alpha', 'all_evidence_hit_at_5', 'mrr_at_5',
        'ndcg_at_5_binary_annotations', 'candidate_recall', 'candidate_count', 'rerank_p95_ms')}


def main():
    manifest, dev_plan, selection = [read(OUT / n) for n in ('manifest.json', 'dev_plan.json', 'selection.json')]
    assert manifest['script_sha256'] == sha256(ROOT / 'scripts/optimize_retrieval.py')
    for relative, value in manifest['hashes'].items():
        assert sha256(ROOT / relative) == value, relative
    qa = {s['id']: s for s in read_jsonl(ROOT / 'data/basic_qa_mapped.jsonl')}
    corpus = {str(s['id']): s for s in read_jsonl(CORPUS)}
    bm25, dense = ChineseBM25(), DenseRetriever()
    bm25.load(); dense.load()
    for split in ('dev', 'holdout', 'demo'):
        query_manifest = read(OUT / (split + '_query_manifest.json'))
        expected_queries = [{'id': s['id'], 'question': s['question']} for s in qa.values() if split_for(s) == split]
        assert query_manifest['queries'] == expected_queries
        assert query_manifest['encoder'] == dense.get_encoder().metadata()
        vectors = np.load(OUT / (split + '_queries.npy'))
        assert vectors.shape == (len(expected_queries), 1024) and np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        for sample, query in zip(expected_queries, vectors):
            channel = read(OUT / 'channels' / (sample['id'] + '.json'))
            assert channel['query_vector_sha256'] == fingerprint(query.tolist())
            keywords = bm25.search(sample['question'], 100)
            assert channel['channels']['bm25'] == keywords
            ranked = ranked_rows(dense.vectors, query, 100)
            recorded = channel['channels']['dense']
            assert [str(dense.documents[i]['id']) for i, _ in ranked] == [str(r['id']) for r in recorded]
            assert np.allclose([s for _, s in ranked], [r['score'] for r in recorded], atol=1e-7)
    dev, dev_status = verify('dev', dev_plan, corpus, qa)
    test, test_status = verify('test', selection, corpus, qa)
    assert selection['dev_rows_sha256'] == sha256(OUT / 'dev_rows.jsonl')
    runtime = read(OUT / 'test_runtime_policy.json')
    assert runtime['selection_sha256'] == sha256(OUT / 'selection.json')
    assert runtime['launcher_sha256'] == sha256(ROOT / 'scripts/run_optimization_paced.py')
    assert all(r['pacing_wait_ms'] >= 0 for r in test)
    assert (OUT / 'selection.json').stat().st_mtime <= (OUT / 'test_rows.jsonl').stat().st_mtime
    alpha = dev_plan['selected_post_alpha']
    dev_summary = summarize(dev, dev_plan['variants'], 0.0)
    if alpha:
        dev_summary += summarize(dev, dev_plan['variants'], alpha)
    assert dev_summary == read(OUT / 'dev_summary.json')
    order = lambda r: (-r['all_evidence_hit_at_5'], -r['mrr_at_5'], r['candidate_count'], r['alpha'], r['variant'])
    assert sorted(dev_summary, key=order)[0] == selection['winner']
    assert sorted([r for r in dev_summary if r['variant'].startswith('qwen_')], key=order)[0] == selection['best_qwen']
    test_summary = {}
    for split in ('holdout', 'demo'):
        rows = [r for r in test if r['split'] == split]
        result = summarize(rows, selection['variants'], 0.0)
        if alpha:
            result += summarize(rows, selection['variants'], alpha)
        test_summary[split] = result
    assert test_summary == read(OUT / 'test_summary.json')
    # Recompute every DEVELOPMENT shortlist score, including rejected attempts.
    for summary in dev_plan['retrieval_summary']:
        recorded = read(OUT / 'retrieval_trials' / (summary['name'] + '.json'))
        for r in recorded:
            channel = read(OUT / 'channels' / (r['id'] + '.json'))
            ids = [str(h['id']) for h in fuse(channel['channels'], RETRIEVAL[summary['name']])]
            assert ids == r['ids']
            for k, value in score(qa[r['id']], ids).items():
                assert math.isclose(r[k], value, abs_tol=1e-12)
            assert r['candidate_recall'] == evidence_metrics(qa[r['id']], ids, len(ids))['evidence_group_recall']
        assert summary['candidate_recall'] == float(np.mean([r['candidate_recall'] for r in recorded]))
        assert summary['mrr5'] == float(np.mean([r['mrr_at_5'] for r in recorded]))
    cases = []
    winner_name, winner_alpha = selection['winner']['variant'], selection['winner']['alpha']
    for key, s in qa.items():
        phase_rows = dev if split_for(s) == 'dev' else test
        baseline = sorted([r for r in phase_rows if r['id'] == key and r['variant'] == 'bge_baseline'], key=lambda r: r['round'])
        optimized = sorted([r for r in phase_rows if r['id'] == key and r['variant'] == winner_name], key=lambda r: r['round'])
        assert len(baseline) == len(optimized) == 3
        pairs = []
        for a, b in zip(baseline, optimized):
            ranking = blend(b['ranking'], b['candidate_ids'], winner_alpha)
            pairs.append({'round': a['round'], 'baseline_top5': a['ranking'][:5], 'optimized_top5': ranking[:5],
                'baseline_metrics': score(s, a['ranking']), 'optimized_metrics': score(s, ranking),
                'baseline_candidate_hit': evidence_metrics(s, a['candidate_ids'], len(a['candidate_ids']))['all_evidence_hit'],
                'optimized_candidate_hit': evidence_metrics(s, b['candidate_ids'], len(b['candidate_ids']))['all_evidence_hit']})
        delta = float(np.mean([p['optimized_metrics']['all_evidence_hit_at_5'] - p['baseline_metrics']['all_evidence_hit_at_5'] for p in pairs]))
        rr_delta = float(np.mean([p['optimized_metrics']['mrr_at_5'] - p['baseline_metrics']['mrr_at_5'] for p in pairs]))
        if delta != 0 or abs(rr_delta) > 1e-12:
            cases.append({'id': key, 'split': split_for(s), 'question': s['question'],
                'annotated_ids': [g['acceptable_corpus_passage_ids'] for g in s['required_evidence_groups']],
                'mean_hit_difference': delta, 'mean_mrr_difference': rr_delta, 'pairs': pairs})
    save(OUT / 'changed_cases.json', cases)
    h = test_summary['holdout']
    base = next(r for r in h if r['variant'] == 'bge_baseline' and r['alpha'] == 0)
    win = next(r for r in h if r['variant'] == winner_name and r['alpha'] == winner_alpha)
    holdout_cases = [c for c in cases if c['split'] == 'holdout']
    answer_probe = read(OUT / 'answer_probe.json')
    assert answer_probe['state'] == 'finished' and len(answer_probe['rows']) == 18
    assert answer_probe['plan'] == read(OUT / 'answer_probe_plan.json')
    assert answer_probe['plan']['generation_sha256'] == sha256(ROOT / 'app/generation.py')
    assert answer_probe['plan']['selection_sha256'] == sha256(OUT / 'selection.json')
    assert {(r['id'], r['method']) for r in answer_probe['rows']} == {
        (key, name) for key in answer_probe['plan']['ids_fixed_before_generation'] for name in answer_probe['plan']['methods']}
    for r in answer_probe['rows']:
        assert split_for(qa[r['id']]) == 'dev'
        variant = 'bge_baseline' if r['method'] == 'bge_baseline' else winner_name
        source = next(a for a in dev if a['id'] == r['id'] and a['round'] == 0 and a['variant'] == variant)
        ids = source['ranking'] if r['method'] == 'bge_baseline' else blend(source['ranking'], source['candidate_ids'], winner_alpha)
        assert ids[:5] == r['top5']
        hits = [corpus[key] for key in r['top5']]
        usable, _ = filter_event_time_evidence(r['question'], hits)
        context, evidence = assemble_context(usable)
        assert fingerprint({'query': r['question'], 'context': context}) == r['context_fingerprint']
        assert literal_coverage(r['answer'], qa[r['id']]['expected_answer_groups']) == r['literal_keypoint_coverage']
        assert set(r['citations']).issubset({a['passage_id'] for a in evidence})
    audit = {'passed': True, 'project': manifest['project'], 'original_files_unchanged': True,
        'verified_original_hash_paths': list(manifest['hashes']),
        'all_90_raw_channel_rankings_replayed_from_original_indexes_and_saved_query_vectors': True,
        'development_rows': len(dev), 'test_rows': len(test), 'all_combinations_complete': True,
        'pool_texts_and_fusion_recomputed_from_original_corpus': True,
        'payload_fingerprints_match_query_documents_instruction': True,
        'development_selection_recomputed_and_frozen': True, 'all_summary_metrics_recomputed': True,
        'holdout_generation_evaluated': False, 'development_answer_probe_requests': 18,
        'model_failures': dev_status['failures'] + test_status['failures'],
        'holdout_delta_points': (win['all_evidence_hit_at_5'] - base['all_evidence_hit_at_5']) * 100,
        'holdout_improved_questions': sum(c['mean_hit_difference'] > 0 for c in holdout_cases),
        'holdout_worse_questions': sum(c['mean_hit_difference'] < 0 for c in holdout_cases),
        'selection': selection}
    save(OUT / 'audit.json', audit)
    lines = ['# 招生资料RAG：优化思路、全部尝试与真实结果', '',
        '**本轮结论：没有验证出可替换原BGE的整体改进方案。** 开发选出的BGE100+轻量融合在53题复测的标注前5命中84.9%→81.1%，MRR略升；开发Qwen指令收益没有作为已验证的复测收益。保留原业务配置，不能把开发题提高写成已验收优化。', '',
        '本报告只针对D:/interview/rag-domain-study，语料14,406片段。35题用于开发选择，53题为已公开原留出复测，2题演示单独统计。没有改标签、答案或片段ID，也没有把电商结果带入。', '',
        '## 先诊断，再选择改动', '',
        '旧开发题中7/35的标注证据不在30候选内，28/35构成当时候选覆盖上限。还有第004题等标注证据已在候选内，却排到第6名。分别测试召回候选数量与排序处理；不把“没召回”和“排错”混为一种问题。', '',
        '三个衡量对象：候选覆盖是重排前是否已包含标注证据；前5命中是最终5条是否包含标注证据；Mean Reciprocal Rank（MRR，平均倒数排名）@5衡量标注证据在前5排得多靠前。normalized Discounted Cumulative Gain（nDCG，归一化折损累计增益）沿用二值原标注。95th Percentile（P95，第95百分位）延迟仅计重排服务请求。', '',
        '## 尝试一：补充原检索排名，开发题选权重', '',
        '先复用上轮保存的开发题BGE完整30排名，不重新调用模型。对每条候选计算 1/(10+重排名次) + α/(10+原融合名次)。α=0保持原重排顺序；候选集合完全不变。排名融合避免直接比较两个模型的原始分数。常数10固定，仅测试下列4个α。', '',
        '| α | 开发前5命中（三轮均值） | MRR@5 |', '|---:|---:|---:|']
    for r in read(OUT / 'diagnosis.json')['post_rank_trials']:
        lines.append(f'| {r["alpha"]} | {r["all_evidence_hit_at_5"]:.1%} | {r["mrr_at_5"]:.4f} |')
    lines += ['', f'按事先规定的“命中优先，其次MRR，再较小α”选出α={alpha}，后续不再改权重。该选择来自旧开发测量，下面用新三轮真实请求再检查；不能只靠缓存成绩宣称稳定提升。', '',
        '## 尝试二：候选数量与两路权重', '',
        '保持原分词、Best Matching 25（BM25）参数k1=1.5/b=0.75、BGE-M3文档向量和语料。每题一次生成查询向量，保存两路前100；各实验从同一两路结果截取。没有重建文档向量或更换编码模型。', '',
        'Reciprocal Rank Fusion（RRF，倒数排名融合）分数为 w/(60+关键词名次) + 1/(60+向量名次)，某路没有该候选则该项为0。w=2表示关键词排名贡献翻倍。两路并集按ID去重；这不是按文本语义去重。', '',
        '| 方案 | 每路候选 | 融合保留 | 关键词/向量权重 | 开发候选覆盖 | 融合前5命中 | 融合MRR@5 |',
        '|---|---:|---:|---|---:|---:|---:|']
    for r in dev_plan['retrieval_summary']:
        lines.append(f'| {r["name"]} | {r["channel_k"]} | {r["pool_k"]} | {r["bm25_weight"]}:1 | {r["candidate_recall"]:.1%} | {r["hit5"]:.1%} | {r["mrr5"]:.4f} |')
    lines += ['', '以上6项都真实计算，包括退步方案。按开发候选覆盖、融合MRR、较小池的固定规则，选两个非基线进入重排：' + '、'.join(r['name'] for r in dev_plan['shortlist']) + '。其余候选参数没有调用重排服务，不把它们写成重排后成绩。扩大候选提高覆盖上限，不能保证最终前5，也会增加模型处理量。', '',
        '## 尝试三：把Qwen任务指令改为资料问答', '',
        '上次招生指令强调条件/截止时间，但资料还包括学院介绍、专业培养、学校历史、合作和毕业去向。本次事先写下更宽的任务指令，并强调只匹配用户明确给出的限制。以下英文作为独立instruction字段发送，原问题、正文和标准答案不拼入指令。', '', '```text', manifest['broad_instruction'], '```', '',
        'Qwen未指定指令与宽领域指令共用baseline30候选；服务端省略instruction时的默认行为未知。BGE不发送instruction字段。未更换Qwen型号或调用其他收费重排型号。', '',
        '## 新三轮开发结果：包含成功与失败尝试', '',
        '| 方案 | 后融合α | 候选数 | 前5命中 | MRR@5 | 二值nDCG@5 | 重排P50毫秒 | 重排P95毫秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in dev_summary:
        lines.append(f'| {r["variant"]} | {r["alpha"]} | {r["candidate_count"]:.0f} | {r["all_evidence_hit_at_5"]:.1%} | {r["mrr_at_5"]:.4f} | {r["ndcg_at_5_binary_annotations"]:.4f} | {r["rerank_p50_ms"]:.1f} | {r["rerank_p95_ms"]:.1f} |')
    lines += ['', f'开发阶段选定：{winner_name}，后融合α={winner_alpha}；最佳Qwen对照为{selection["best_qwen"]["variant"]}、α={selection["best_qwen"]["alpha"]}。选择文件selection.json在复测前保存，规则为开发命中、MRR、较少候选、较小α、方案名。不在看到53题结果后更换最终推荐。', '',
        '5种真实服务配置×35题×3轮，模型顺序与问题顺序随机交错；后融合只离线重新排序同一结果，不增加模型调用。四种旧α与六种候选组合说明有多次比较，小样本开发成绩可能过拟合。', '',
        '## 原53题复测：同期基线与冻结方案', '',
        '| 方案 | 后融合α | 候选覆盖 | 前5命中 | MRR@5 | 二值nDCG@5 | 重排P50毫秒 | 重排P95毫秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in h:
        lines.append(f'| {r["variant"]} | {r["alpha"]} | {r["candidate_recall"]:.1%} | {r["all_evidence_hit_at_5"]:.1%} | {r["mrr_at_5"]:.4f} | {r["ndcg_at_5_binary_annotations"]:.4f} | {r["rerank_p50_ms"]:.1f} | {r["rerank_p95_ms"]:.1f} |')
    lines += ['', '本轮53题只复测开发最佳Qwen配置，没有同期复测Qwen未指定指令。因此76.7%是本轮宽领域指令方案成绩，不能量化该指令在53题上的增益；此前81.1%来自另一批调用，不直接拿来计算本轮指令因果效果。开发题40.0%→52.4%则来自同池交错对照。']
    lines += ['', f'冻结方案相对本轮同期原BGE：前5标注命中{base["all_evidence_hit_at_5"]:.1%}→{win["all_evidence_hit_at_5"]:.1%}，差{audit["holdout_delta_points"]:+.2f}个百分点；MRR {base["mrr_at_5"]:.4f}→{win["mrr_at_5"]:.4f}。按逐题三轮均值，{audit["holdout_improved_questions"]}题命中改善、{audit["holdout_worse_questions"]}题退步。', '',
        '以下每轮分母仍为53，不把三轮当159道独立题：', '',
        '| 轮次 | 同期原BGE命中 | 冻结方案命中 |', '|---|---:|---:|']
    for n in range(3):
        a = [r for r in test if r['round'] == n and r['split'] == 'holdout' and r['variant'] == 'bge_baseline']
        b = [r for r in test if r['round'] == n and r['split'] == 'holdout' and r['variant'] == winner_name]
        lines.append(f'| {n + 1} | {sum(score(r["sample"], r["ranking"])["all_evidence_hit_at_5"] for r in a)}/53 | {sum(score(r["sample"], blend(r["ranking"], r["candidate_ids"], winner_alpha))["all_evidence_hit_at_5"] for r in b)}/53 |')
    lines += ['', '候选增加时仍只最终保留5条，本实验没有增加生成片段数量；文本总长度可能变化，未评估全量生成费用或线上问答延迟。表中后融合共用同一次重排延迟，没有把额外计算伪装成独立请求；后融合的本机处理时间不在该列。', '',
        f'50th Percentile（P50，第50百分位，也就是中位数）重排耗时{base["rerank_p50_ms"]:.1f}→{win["rerank_p50_ms"]:.1f}毫秒，约{win["rerank_p50_ms"]/base["rerank_p50_ms"]:.2f}倍。原BGE的P95受限流重试拉高，不能因100候选的P95较低就断言它更快。', '',
        f'开发候选覆盖80.0%→97.1%，但53题原30候选覆盖已为{base["candidate_recall"]:.1%}，100候选仍为{win["candidate_recall"]:.1%}：复测没有新增可命中的标注问题。该覆盖差异解释了开发收益为何不能直接推广；具体排序变化仍需逐题复核。', '',
        '旧83.0%单轮基线保留，不拿它直接替代同期基线。53题已在历史对话中公开，只能称冻结方案的探索性复测，不能称新独立留出验收。现有原标注未穷尽等价证据，本次没有修改评分口径来抬高成绩。', '',
        '## 逐题变化，待人工复核', '',
        'changed_cases.json完整保留开发、复测和演示中的命中/排名变化，包含改善和退步。以下只列53题命中发生变化的问题，不把标注失败直接等同于语义失败：', '',
        '| 题号 | 问题 | 三轮平均命中差 |', '|---|---|---:|']
    for c in holdout_cases:
        if c['mean_hit_difference']:
            lines.append(f'| {c["id"]} | {c["question"]} | {c["mean_hit_difference"]:+.3f} |')
    lines += ['', '对这3个标注退步案例的原文抽查（助手复核，不是独立人工全量评测）：', '',
        '- 第023题：优化三轮第1均为13912，其中明确列出哲学、伦理学、宗教学和政治学/经济学/哲学复合本科专业；原标注8864退出前5，但13915等片段仍有相应专业说明。',
        '- 第037题：优化前5的7839、1640、1099等仍明确写“每年均从全校不同专业中选拔学生”；原标注7840未出现不能证明这一事实丢失。',
        '- 第066题：优化三轮第1均为10059，明确写“位于江苏省苏州市独墅湖科教创新区”；原标注686未进前5，仍有地点证据。',
        '因此本轮可以说“原标注指标未提高”，不能把-3.77个百分点直接解释成真实答案质量下降。反过来也不能凭3题抽查宣称系统已达到完整证据100%。这些重复/等价片段使标注命中对ID排名很敏感，是后续评测应先解决的问题。']
    lines += ['', '## 真实答案检查：9道固定开发题，18次生成', '',
        '采用本项目原生成函数、DeepSeek配置、前5正文、事件日期过滤和16,000字符预算。问题列表在调用前固定：003、004、006、015、019、025、033、034、049；复用开发第1轮捕获的上下文。标签只在生成后评价，不发送给模型。', '',
        '| 方案 | 字面要点完整且未拒答 | 拒答 | 调用失败 | 输入token总计 | 输出token总计 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in answer_probe['summaries']:
        lines.append(f'| {r["method"]} | {r["literal_complete_nonrefusal"]}/{r["questions"]} | {r["refusals"]} | {r["errors"]} | {r["prompt_tokens"]} | {r["completion_tokens"]} |')
    lines += ['', 'Token是模型分词单位。本表统计18次真实返回的用量；不是全量生成成本或可泛化的答案准确率。单轮、选定开发题、仅按字面要点与拒答标志统计，尚未全量人工判定引用支持程度。', '',
        '- 第004题：原方案前5已经有8490等等价片段明确包含“2010年”；优化增加原标注1099，但两组真实回答都是2010年。这是标注命中改善，不是原答案错误被修好。',
        '- 第006题：原方案这次以缺少更具体日期为由拒答，优化这次答2007年；两组原文均含“自2007年以来连续承办五届”。单次生成的解释差异，不能证明扩大候选导致答案改善，需固定上下文重复试验或人工语义评估。',
        '- 第015题：两组都因证据不足拒答，没有为抬高成功率关闭原日期过滤规则。', '',
        '完整答案、引用、实际上下文、拒答标志和用量见answer_probe.json。引用ID通过校验仍不等于每条引用都语义支持答案。', '',
        '## 演示与运行口径', '', '演示2题的原BGE/冻结方案平均前5命中：' +
        f'{next(r for r in test_summary["demo"] if r["variant"] == "bge_baseline" and r["alpha"] == 0)["all_evidence_hit_at_5"]:.1%}/' +
        f'{next(r for r in test_summary["demo"] if r["variant"] == winner_name and r["alpha"] == winner_alpha)["all_evidence_hit_at_5"]:.1%}。演示不参与选型。', '',
        '开发阶段记录88次HTTP 429限流事件；所有请求最终成功。复测前加入统一1.35秒最小请求起始间隔，配置与启动脚本哈希保存为test_runtime_policy.json，未改变已冻结的模型、候选或指标参数。主动等待在重排计时前发生，pacing_wait_ms逐次保存；表中纯重排延迟仍含服务重试等待。开发与复测调度节奏不同，不通过跨阶段P95推断速度改善。', '',
        '## 核验、限制与复现', '',
        f'- 开发保存{len(dev)}条、复测/演示保存{len(test)}条真实测量；组合齐全，最终请求失败0。重试事件数{len(audit["model_failures"])}，所有错误事件见阶段status.json。',
        '- 原文、向量文件、双索引清单、标签和生产适配器哈希保持；核验路径见audit.json。90题两路原始排名由原索引与保存的查询向量重新运行核对，逐条核验候选原文、两路融合、输入指纹和全部指标，重新推导开发选型。完整原始名次/分数保留。',
        '- 检索/重排进行了全量复测，生成仅9道开发题单轮检查；未测原53题生成答案准确率或全量人工等价证据。40项离线测试通过。未改业务配置，未部署开发题选出的方案。',
        '- 若继续优化，在开发集上进行；独立验收需要新评测集及人工复核，而不是继续用同53题挑最高分。', '',
        '```powershell', 'cd D:/interview/rag-domain-study',
        '.venv/Scripts/python.exe -X utf8 scripts/optimize_retrieval.py prepare',
        '.venv/Scripts/python.exe -X utf8 scripts/optimize_retrieval.py dev',
        '.venv/Scripts/python.exe -X utf8 scripts/run_optimization_paced.py',
        '.venv/Scripts/python.exe -X utf8 scripts/probe_optimization_answers.py',
        '.venv/Scripts/python.exe -X utf8 scripts/audit_optimization.py',
        '.venv/Scripts/python.exe -X utf8 -m pytest -q', '```', '',
        '现有目录用于续接，完成记录会跳过；独立复测应另建实验目录，不覆盖历史。manifest.json为预设方案；diagnosis.json为旧开发排序实验；dev_plan.json为候选短名单；selection.json为复测前冻结选择；channels/、pools/、*_rows.jsonl、*_summary.json与audit.json为完整证据。']
    (OUT / '完整优化过程与结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k not in ('selection', 'model_failures')}, ensure_ascii=False), flush=True)
    print('Retained service retry events:', len(audit['model_failures']), flush=True)
    print('baseline', compact(base), 'selected', compact(win), flush=True)


if __name__ == '__main__':
    main()
