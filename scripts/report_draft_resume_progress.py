"""Extend an interim delivery with actual draft/gate evidence, not gold scores."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import ROOT, read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-dir', type=Path, required=True)
    parser.add_argument('--draft-dir', type=Path, required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new interim delivery')
    prior, draft, run = args.prior_dir.resolve(), args.draft_dir.resolve(), args.diagnostic_dir.resolve()
    manifest = json.loads((prior / 'manifest.json').read_text(encoding='utf-8'))
    prep = json.loads((draft / 'manifest.json').read_text(encoding='utf-8'))
    diagnostic = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    if prep['human_confirmed'] != 0 or prep['acceptance_frozen'] or diagnostic['remote_enabled']:
        raise ValueError('This report only describes pending offline draft preparation')
    manifest.update({'schema_version': 'resume-evidence-draft-interim-v1', 'status': 'INCOMPLETE',
                     'new_task_drafts': prep, 'new_acceptance_executed': 0,
                     'current_code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p)
                          for folder in ['app', 'scripts', 'tests'] for p in sorted((ROOT / folder).rglob('*.py'))},
                     'phase_input_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in [draft / 'labels.jsonl',
                          draft / 'manifest.json', draft / 'human-review.csv', run / 'retrieval.jsonl', run / 'metrics.json']}})
    metrics = json.loads((prior / 'metrics.json').read_text(encoding='utf-8'))
    metrics['new_task_preparation'] = {'planned_total': 80, 'actual_drafts': prep['actual_drafts'], 'human_confirmed': 0,
                                     'provisional_split_counts': prep['split_counts'], 'question_type_counts': prep['type_counts'],
                                     'mapped_drafts': prep['mapped'], 'blocked_ids': prep['blocked_ids'],
                                     'scope': 'mechanical preparation only; not human gold or performance'}
    metrics['new_pending_dev_diagnostics'] = diagnostic
    runtime = json.loads((prior / 'runtime.json').read_text(encoding='utf-8'))
    runtime['new_task_evaluation'] = {'draft_local_dev_cases': diagnostic['planned_cases'], 'acceptance_cases_run': 0,
                                    'human_label_reviews': 0, 'real_generation_trials': 0, 'remote_calls': 0}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'retrieval.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        fields = ['scope', 'case_id', 'method', 'question_type', 'retrieval_target', 'label_status', 'execution_status',
                  'candidate_count', 'final_count', 'candidate_all_hit', 'final_all_hit', 'failure_stage']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with (prior / 'retrieval.csv').open(encoding='utf-8-sig', newline='') as old:
            for row in csv.DictReader(old):
                writer.writerow({**{k: row.get(k, '') for k in fields}, 'scope': 'old_exposed_dev', 'retrieval_target': 'answer_evidence'})
        for r in read_jsonl(run / 'retrieval.jsonl'):
            writer.writerow({'scope': 'new_pending_dev_diagnostic', **{k: r.get(k, '') for k in fields if k in r},
                             'execution_status': r['final']['execution_status'],
                             'candidate_count': len(r['candidate_pool']['hits']) if r['candidate_pool']['hits'] is not None else '',
                             'final_count': len(r['final']['hits']) if r['final']['hits'] is not None else '',
                             'candidate_all_hit': r['candidate_score']['all_evidence_hit'] if r['candidate_score'] else '',
                             'final_all_hit': r['final_score']['all_evidence_hit'] if r['final_score'] else ''})
    shutil.copyfile(draft / 'human-review.csv', args.output_dir / 'human-review.csv')
    (args.output_dir / 'answers.jsonl').write_text('', encoding='utf-8')
    answer_bm25 = diagnostic['by_retrieval_target']['answer_evidence']['methods']['bm25']
    changes = f'''# VersionRAG阶段回传：80题准备

状态INCOMPLETE，完整P0—P3目标继续。历史、合成工程、草稿诊断分别统计；没有新增真实模型或人工主成绩。

草稿{prep['actual_drafts']}/{prep['planned_total']}题，{prep['source_family_count']}来源家族，映射{prep['mapped']}，人工确认{prep['human_confirmed']}。旧题{prep['old_qa_count']}与公开查询{prep['public_query_count']}自动查重不能证明语义独立，仍需人审。

分组{json.dumps(prep['split_counts'], ensure_ascii=False)}；题型{json.dumps(prep['type_counts'], ensure_ascii=False)}。已查询{len(prep.get('previously_queried_dev_ids', []))}道开发题及来源组件锁在开发侧，分组不按检索成绩。前后文帮助核对年份、表头与例外，不自动批准。

只做{diagnostic['planned_cases']}道开发关键词草稿诊断：可回答前五完整证据命中{answer_bm25['hits']}/{answer_bm25['planned']}。不足/澄清题只检查指定背景，不计处置正确率。其余四方案not_run/null；验收未运行。开发集合变了，不能与之前16题7/14说成优化提升。

新增回答批次绑定已完成检索和固定原文/标签/版本；保存全计划、实际裁剪过滤证据、原始回答引用、失败和用量。不向生成器传金标。固定temperature=0、16000字符、1200输出tokens；三次重复固定检索证据，只测条件生成变化。开发240个占位全not_run，真实生成0，本包answers.jsonl为空。

新增正式冻结器要求人工金标、完整五方案开发对照、当前标签/代码/预算一致，记录选择理由、模型索引、提示和调用上限。冻结不是费用授权，预留不是完成证明。实际未确认冻结与验收生成被拒绝，无输出目录和模型调用。

评分升级fact-citation-strict-v2：盲审保留原始引用，审核者不能事后给模型补上另一段正确引用。缺原始引用/额外引用未审保持未知，答案正确配错引用的反例FAIL。181项完整回归通过，模拟服务/模型不是质量成绩。

八文件保留scope区分旧开发与新待确认开发，不合并。human-review.csv为{prep['actual_drafts']}行待审核模板。原数据/历史报告未修改，凭据/验收题/私有记录不推送。

未完成：人工金标、待回复的旧35题远端检索预算、新开发真实检索生成、最终选型和验收冻结、正式生成与人审、真实模型版本任务与自愿试用。目标继续。
'''
    (args.output_dir / 'changes.md').write_text(changes, encoding='utf-8')
    failures = (prior / 'failures.md').read_text(encoding='utf-8') + '\n## 新草稿诊断\n\n新标注仍pending，不将下面条目解释为正式质量。\n'
    for row in read_jsonl(run / 'retrieval.jsonl'):
        if row['method'] == 'bm25' and row['final_score']['all_evidence_hit'] is False:
            failures += '- ' + row['case_id'] + '：' + row['failure_stage'] + '；原文区间与候选见' + run.name + '/retrieval.jsonl。\n'
    (args.output_dir / 'failures.md').write_text(failures, encoding='utf-8')
    for name, data in [('manifest.json', manifest), ('metrics.json', metrics), ('runtime.json', runtime)]:
        (args.output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'directory': str(args.output_dir), 'files': len(list(args.output_dir.iterdir())), 'status': 'INCOMPLETE',
                      'new_drafts': prep['actual_drafts'], 'human_confirmed': 0, 'new_acceptance_executed': 0, 'remote_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
