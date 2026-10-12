"""Recompute every metric from THIS project's saved instruction experiment."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.evaluation import split_for
from app.retrieval import CORPUS,ROOT,read_jsonl,sha256
from scripts.compare_qwen_instructions import OUT,VARIANTS,aggregate,fingerprint,payload_for,read,save,score


def paired_change(rows,left,right,field):
    ids=sorted({row['id'] for row in rows})
    deltas=[]
    per_query=[]
    for key in ids:
        a=[row[field] for row in rows if row['id']==key and row['variant']==left]
        b=[row[field] for row in rows if row['id']==key and row['variant']==right]
        assert len(a)==len(b)>0
        delta=float(np.mean(a)-np.mean(b))
        deltas.append(delta)
        per_query.append({'id':key,'difference':delta})
    rng=np.random.default_rng(20261007)
    boot=np.mean(rng.choice(deltas,(10000,len(deltas)),replace=True),axis=1)
    return {'left':left,'right':right,'metric':field,'unique_questions':len(ids),
        'mean_difference':float(np.mean(deltas)),'query_paired_bootstrap_95_interval':np.percentile(boot,[2.5,97.5]).tolist(),
        'improved_questions':sum(x>1e-12 for x in deltas),'worse_questions':sum(x< -1e-12 for x in deltas),
        'unchanged_questions':sum(abs(x)<=1e-12 for x in deltas),'per_query':per_query,
        'scope':'Descriptive only. Three measurements per query are averaged first; questions are not independently human-adjudicated and original holdout was already exposed.'}


def main():
    manifest=read(OUT/'manifest.json')
    status=read(OUT/'status.json')
    assert status['state']=='finished'
    rows=[json.loads(line) for line in (OUT/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    samples={row['id']:row for row in manifest['samples']}
    expected={(n,key,variant) for n in range(manifest['rounds']) for key in samples for variant in VARIANTS}
    assert len(rows)==len(expected) and {(r['round'],r['id'],r['variant']) for r in rows}==expected
    assert all(r['status']=='ok' for r in rows),'Failed requests retained; complete comparison needs successful matched rows'
    assert manifest['script_sha256']==sha256(ROOT/'scripts/compare_qwen_instructions.py')
    for path,value in manifest['hashes'].items():
        assert sha256(ROOT/path)==value, f'Original project input changed: {path}'
    corpus={str(row['id']):row['contents'] for row in read_jsonl(CORPUS)}
    pools={key:read(OUT/'pools'/(key+'.json')) for key in samples}
    for key,pool in pools.items():
        budget=8000-len(samples[key]['question'].encode('utf-8'))
        assert pool['documents']==[corpus[passage_id].encode('utf-8')[:budget].decode('utf-8',errors='ignore') for passage_id in pool['candidate_ids']]
    for row in rows:
        pool=pools[row['id']]
        base={key:pool[key] for key in ('question','candidate_ids','documents')}
        assert fingerprint(base)==pool['fingerprint']==row['pool_fingerprint']
        assert row['candidate_ids']==pool['candidate_ids'] and row['candidate_count']==30
        assert row['question']==samples[row['id']]['question'] and row['split']==split_for(samples[row['id']])
        assert row['instruction']==VARIANTS[row['variant']]['instruction']
        assert fingerprint(payload_for(pool,row['variant']))==row['payload_fingerprint']
        assert len(row['ranking'])==len(set(row['ranking']))==30
        assert set(row['ranking'])==set(pool['candidate_ids'])
        for key,value in score(samples[row['id']],row['ranking']).items():
            assert math.isclose(row[key],value,abs_tol=1e-12),(row['id'],row['variant'],key)
    recalculated={'all':aggregate(rows,90*manifest['rounds']),
        'per_split':{split:aggregate([r for r in rows if r['split']==split],manifest['rounds']*sum(split_for(s)==split for s in samples.values())) for split in ('dev','holdout','demo')}}
    summary=read(OUT/'summary.json')
    assert recalculated['all']==summary['all'] and recalculated['per_split']==summary['per_split']
    holdout=[row for row in rows if row['split']=='holdout']
    comparisons=[paired_change(holdout,a,b,key) for a,b in (
        ('qwen_general','qwen_unspecified'),('qwen_admissions','qwen_unspecified'),
        ('qwen_admissions','qwen_general'),('qwen_admissions','bge_control'))
        for key in ('all_evidence_hit_at_5','mrr_at_5','ndcg_at_5_binary_annotations')]
    audit={'passed':True,'project':'rag-domain-study/admissions','verified_rows':len(rows),
        'verified_questions':len(samples),'split_counts':{split:sum(split_for(s)==split for s in samples.values()) for split in ('dev','holdout','demo')},
        'original_corpus_indexes_labels_and_production_adapters_unchanged':True,
        'identical_question_and_30_candidate_texts_for_all_variants':True,
        'candidate_texts_match_this_project_corpus_after_production_clipping':True,
        'custom_instruction_in_separate_field_and_payload_fingerprint_verified':True,
        'all_metrics_recomputed':True,'generation_evaluated':False,'paired_holdout_comparisons':comparisons}
    save(OUT/'audit.json',audit)
    differences=[]
    by_key={(r['round'],r['id'],r['variant']):r for r in rows}
    for key,sample in samples.items():
        if split_for(sample)!='holdout':
            continue
        for n in range(manifest['rounds']):
            old=by_key[(n,key,'qwen_unspecified')]
            new=by_key[(n,key,'qwen_admissions')]
            if old['ranking'][:5]!=new['ranking'][:5]:
                differences.append({'id':key,'round':n,'question':sample['question'],
                    'annotated_evidence_groups':sample['required_evidence_groups'],
                    'unspecified_top5':old['ranking'][:5],'admissions_top5':new['ranking'][:5],
                    'unspecified_hit':old['all_evidence_hit_at_5'],'admissions_hit':new['all_evidence_hit_at_5'],
                    'unspecified_mrr':old['mrr_at_5'],'admissions_mrr':new['mrr_at_5']})
    save(OUT/'changed_cases.json',differences)
    names={'bge_control':'BGE同期对照','qwen_unspecified':'Qwen未指定指令','qwen_general':'Qwen通用指令','qwen_admissions':'Qwen招生任务指令'}
    lines=['# 本项目：Qwen招生任务指令补测','',
        '**项目为rag-domain-study招生资料知识库。** 使用本项目14,406片段、关键词/向量索引和90题原评测，不使用电商政策、80题或其成绩。',
        '', '## 修改了什么','',
        '- 正常问答仍使用原BGE适配器。本次只新增实验脚本，没有切换业务配置。',
        '- 先使用项目原双路召回：每路30，倒数排名融合常数60，冻结融合后的30条候选及顺序。',
        '- 四组都使用相同原始问题、相同候选文本、相同返回数量；三种Qwen配置只改变独立的instruction字段。',
        '- 90题每组运行3轮，每题随机交错四组调用顺序，预热不计入。保存1,080条完整结果及请求指纹、延迟、排名、分数和错误记录。',
        '- 35开发、53原留出、2演示分别报告。原留出此前已公开，本次是后续探索，不是新独立测试集。',
        '', '## 两条事先固定的指令','',
        '**通用指令**（Qwen官方示例）：', '', '```text',VARIANTS['qwen_general']['instruction'],'```','',
        '**招生任务指令**：', '', '```text',VARIANTS['qwen_admissions']['instruction'],'```','',
        '中文含义：寻找能直接回答问题的招生资料；优先匹配年份、招生类别、学院专业、申请条件及截止时间，避免以其他年份或类别的相似规则替代。没有给模型标准答案、Gold片段编号或按题设置提示。',
        '', '未指定指令表示请求不传instruction；服务端是否自行加入默认指令未核实。官方接口支持Qwen0.6B的该字段，实际请求字段由离线Mock测试及保存的请求指纹核验；成功请求返回200。',
        '', '## 原53题留出的同期结果','',
        'normalized Discounted Cumulative Gain（nDCG，归一化折损累计增益）；Mean Reciprocal Rank（MRR，平均倒数排名）；50th/95th Percentile（P50/P95，第50/95百分位）。',
        '', '| 方案 | 前5证据命中率 | 前10证据覆盖 | MRR@5 | 二值标注nDCG@5 | 重排P50 ms | 重排P95 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['per_split']['holdout']:
        lines.append(f'| {names[row["variant"]]} | {row["all_evidence_hit_at_5"]:.1%} | {row["evidence_recall_at_10"]:.1%} | {row["mrr_at_5"]:.4f} | {row["ndcg_at_5_binary_annotations"]:.4f} | {row["rerank_p50_ms"]:.1f} | {row["rerank_p95_ms"]:.1f} |')
    lines.extend(['', '当前基础题每题一个标注证据组，因此前5证据命中率是“命中至少一个该组可接受片段的问题比例”。三轮重复相同53题，不当作159道独立题；均值包含三轮排序变化。不是回答准确率，也没有重写Gold补全等价证据。',
        '', '本次延迟只计重排请求，包括重试等待，不含查询编码、关键词/向量/融合、生成。不能与历史83.0%报告中的全检索延迟直接比较。候选准备时的单轮召回耗时另外保存。',
        '', '二值nDCG仅将现有标注片段视为相关（1），其他视为0；不是电商实验关键2/辅助1的分级指标，也不表示穷尽真实语义相关片段。',
        '', '### 同题三轮的命中数',''])
    for n in range(manifest['rounds']):
        counts={names[variant]:sum(r['all_evidence_hit_at_5'] for r in holdout if r['round']==n and r['variant']==variant) for variant in VARIANTS}
        lines.append(f'- 第{n+1}轮：'+ '；'.join(f'{name} {count}/53' for name,count in counts.items())+'。')
    comparison=next(c for c in comparisons if c['left']=='qwen_admissions' and c['right']=='qwen_unspecified' and c['metric']=='all_evidence_hit_at_5')
    interval=comparison['query_paired_bootstrap_95_interval']
    lines.extend(['',f'**招生指令相对未指定指令：** 命中率均值差{comparison["mean_difference"]*100:+.2f}个百分点；按每题三轮均值计，{comparison["improved_questions"]}题改善、{comparison["worse_questions"]}题变差、{comparison["unchanged_questions"]}题不变。成对重采样描述性95%区间为[{interval[0]*100:+.2f}, {interval[1]*100:+.2f}]个百分点，不据此断言模型普遍优劣。',
        '', '逐题排序变动、改善和退步都保留在changed_cases.json。不在看到成绩后继续调指令并把同一批题包装成新留出。',
        '', '## 排名变化与人工抽查边界','',
        '- 第026题“艺术学院的本科专业有哪些方向？”：标注片段4975在Qwen未指定指令时三轮均排第2；招生指令下排第10、10、11，退出前5。新前5出现2014/2015招生简章。此处只证明标注片段排序退步，不据此判定最终回答错误。',
        '- 第071题“政治学与行政学本科毕业生有多少比例的学生选择就业？”：标注片段1064从三轮第1降到15、15、14。人工抽查冻结文本发现，招生指令三轮第1的相邻片段1063同样明确包含“20-25%选择就业”。因此该题的标注命中失败不能解释为真实证据丢失；原标注未覆盖这个等价片段，本次不改标签。',
        '- 本次只抽查案例，不是53题全量人工语义复评。不能将-6.29个百分点写成实际回答准确率下降。指令强调招生规则，而评测也包含学院介绍、专业方向和毕业去向；领域约束偏窄是值得后续验证的解释，尚不是已证明的因果。',
        '- 选型：现有标注指标没有支持替换原BGE或上线这条招生指令，继续使用原业务配置。若继续调指令，先在开发题上改，再准备新独立评测题及等价证据复核。',
        '', '## 开发与演示结果','', '| 用途 | 方案 | 前5证据命中率 | MRR@5 |', '|---|---|---:|---:|'])
    for split in ('dev','demo'):
        for row in summary['per_split'][split]:
            lines.append(f'| {split} | {names[row["variant"]]} | {row["all_evidence_hit_at_5"]:.1%} | {row["mrr_at_5"]:.4f} |')
    lines.extend(['', '## 复现与证据','', '```powershell','cd D:/interview/rag-domain-study',
        '.venv/Scripts/python.exe -X utf8 -m pytest tests/test_instruction_ablation.py -q',
        '.venv/Scripts/python.exe -X utf8 scripts/compare_qwen_instructions.py --rounds 3',
        '.venv/Scripts/python.exe -X utf8 scripts/analyze_qwen_instructions.py','```','',
        '重跑会沿用冻结池并跳过已有测量，仍会进行预热；独立复测需新输出目录，不覆盖本轮记录。',
        '', '- manifest.json：项目、90题、分组、原文件哈希和预设指令。',
        '- pools/：每题冻结30候选及真正发给四组的文本。',
        '- rows.jsonl/summary.json：1,080条逐次记录和分组/分轮统计。',
        '- audit.json：原语料、索引、标签、生产适配器未变；请求指纹、候选一致与指标重算。',
        '- changed_cases.json：原53题上的逐题排序变化。','',
        '[硅基流动官方指令字段](https://docs.siliconflow.cn/docs/api/rerank-post)、[Qwen官方模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)。'])
    (OUT/'补测结果与解释.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Admissions-only audit passed: {len(rows)} rows, 90 questions, same 30 candidates.',flush=True)


if __name__=='__main__':
    main()
