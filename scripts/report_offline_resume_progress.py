"""Eight-file interim delivery from actual offline results; never acceptance scores."""
import argparse
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import ROOT, read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.batch_dir = args.batch_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new delivery directory')
    baseline_path = args.batch_dir / 'manifest.json'
    alignment = args.batch_dir / 'legacy-alignment'
    run = args.batch_dir / 'retrieval-dev-offline'
    families = args.batch_dir / 'source-family-audit'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    audit = json.loads((alignment / 'audit.json').read_text(encoding='utf-8'))
    retrieval_manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if retrieval_manifest['remote_enabled']:
        raise ValueError('This interim report only accepts the offline run')
    retrieval_metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    family_summary = json.loads((families / 'summary.json').read_text(encoding='utf-8'))
    rows = read_jsonl(run / 'retrieval.jsonl')
    bm = [r for r in rows if r['method'] == 'bm25']
    stages = Counter(r['failure_stage'] for r in bm)
    actual = {'candidate_hits': sum(r['candidate_score']['all_evidence_hit'] for r in bm),
              'final_hits': sum(r['final_score']['all_evidence_hit'] for r in bm), 'denominator': len(bm)}
    created = datetime.now(timezone(timedelta(hours=8))).isoformat()
    input_files = [baseline_path, alignment / 'audit.json', alignment / 'labels.jsonl',
                   run / 'manifest.json', run / 'retrieval.jsonl', run / 'retrieval.csv', run / 'metrics.json',
                   families / 'summary.json', families / 'source-families.json', families / 'question-audit.json']
    manifest = {'schema_version': 'resume-evidence-interim-v1', 'status': 'INCOMPLETE', 'created_at': created,
                'baseline_commit': baseline['baseline_commit'], 'baseline_manifest_sha256': sha256(baseline_path),
                'data_sha256': baseline['data_sha256'], 'source_scope': audit['source_scope'],
                'knowledge_version': audit['knowledge_version'], 'snapshot_sha256': audit['snapshot_sha256'],
                'label_status': 'upstream_unreviewed', 'new_acceptance': {'status': 'not_prepared', 'actual': 0, 'human_confirmed': 0},
                'inputs': {p.relative_to(ROOT).as_posix(): sha256(p) for p in input_files},
                'current_code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p)
                    for folder in ['app', 'scripts', 'tests'] for p in sorted((ROOT / folder).rglob('*.py'))},
                'retrieval_config': retrieval_manifest['budget'],
                'freeze_scope': 'interim code snapshot, not final acceptance freeze; baseline manifest is unchanged',
                'remote_calls': 0, 'generation_trials': 0, 'money_cost': None}
    def ratio(n, d, unit, scope):
        return {'numerator': n, 'denominator': d, 'unit': unit, 'scope': scope, 'value': n / d if d else None}
    metrics = {'status': 'INCOMPLETE',
               'legacy_mapping': {'mapped': audit['mapped'], 'planned': audit['questions'], 'human_confirmed': 0,
                                  'blocked_ids': audit['blocked_ids'], 'scope': 'coordinate migration only; not human gold'},
               'historical_saved_replay': audit['historical_replay'],
               'current_old_dev': {'methods': retrieval_metrics['methods'], 'paired': retrieval_metrics['paired'],
                  'candidate_complete_hit': ratio(actual['candidate_hits'], len(bm), 'questions', 'old exposed dev, upstream labels'),
                  'top5_complete_hit': ratio(actual['final_hits'], len(bm), 'questions', 'old exposed dev, upstream labels'),
                  'failure_stages': dict(stages)},
               'new_acceptance': {'status': 'not_run', 'strict_main_pass': ratio(None, 0, 'planned_trials', 'new human acceptance'),
                                  'human_reviewed': 0, 'planned_executed_trials': 0},
               'source_family_audit': family_summary,
               'warning': 'Historical replay, current old-dev diagnostics, and new acceptance must not be merged'}
    times = sorted(r['channel_elapsed_ms']['bm25'] for r in bm)
    percentile = lambda q: times[max(0, math.ceil(q * len(times)) - 1)] if times else None
    runtime = {'local_offline_run': {'status': 'completed', 'case_count': len(bm), 'method_records': len(rows),
                                    'remote_calls': 0, 'generation_calls': 0,
                                    'bm25_search_ms': {'p50': percentile(.50), 'p95': percentile(.95), 'n': len(times)},
                                    'scope': 'local shared-channel diagnostic; not service SLA or model-answer quality'},
               'real_volunteer_trial': {'status': 'not_run', 'window': None, 'real_users': None,
                                        'requests': None, 'quality': None},
               'version_answer_task': {'status': 'not_run', 'passed': None, 'total': None},
               'synthetic_service_load': {'status': 'not_run', 'requests': None, 'p95_ms': None},
               'money_cost': None, 'provider_usage': None, 'failed_and_retried_service_usage': None}
    failures = ['# 当前失败记录（离线阶段）', '', '这些是旧开发题的证据检索失败，未测生成，不能解释成回答错误率。', '',
                '| 题号 | 当前失败阶段 | 候选完整命中 | 前五编号 |', '|---|---|---|---|']
    for r in bm:
        if r['final_score']['all_evidence_hit'] is not True:
            failures.append(f"| {r['case_id']} | {r['failure_stage']} | {r['candidate_score']['all_evidence_hit']} | "
                            + ', '.join(r['final_score']['retrieved_ids']) + ' |')
    failures += ['', '逐题原文区间、候选及前五保存在上层 retrieval-dev-offline/retrieval.jsonl。',
                 '未执行的四方案为 not_run，分数 null；没有把它们算零或套用旧成绩。',
                 '新人工金标、实际生成审核、版本事实任务、服务遥测及真实试用仍待完成。']
    changes = f'''# VersionRAG改造：离线阶段回传

状态：INCOMPLETE。八份文件是阶段交付，完整P0—P3验收尚未完成。

已实施：第五个“关键词＋同模型精排”入口；区间评分区分原文/标签失效与索引漏证据；事实—引用三态评分；原回答/评分标准哈希绑定的随机审核；同候选30、前五5、融合常数60的五方案批次程序；来源家族及相似问题风险分组。

实际验证：旧90题全部映射到固定上游预切分片段的字符坐标，人工确认仍为0。重放旧53题的四方案命中数保持39、28、33、44。新离线运行旧35道开发题，关键词候选30完整命中{actual['candidate_hits']}/35，前五命中{actual['final_hits']}/35；{stages['ranking_or_final_budget']}题候选已命中而前五漏掉，{stages['candidate_recall']}题候选未命中。四个依赖远端的方案尚未运行，不能评价精排增益。

分组审计：{family_summary['source_count']}个来源归为{family_summary['family_count']}个精确重复家族，{family_summary['cross_split_risk_pairs']}对旧题有跨集合风险。精确重复不能覆盖所有转载/改写，相似问题只是人工复查警告。旧划分与旧报告未改。

审核入口：先用 scripts/review_answer_batch.py prepare 导出随机审核包，再用 score 汇总实际提交。未审核为INCOMPLETE；声明为模型或助手的评分不进入人工主成绩；修改回答或评分标准后旧审核失效。程序保存审核来源声明，无法独立认证填写者。语义支持仍需真实审核，不是自动字符串判断。

冻结：本文件夹manifest记录当前源文件哈希，原批次基线manifest不修改。还未完成最终验收冻结或选型，不能把当前工作区哈希当已发布提交。

下一步：准备新任务与人工事实/证据；预算确定后对旧开发题跑五方案，定位得失后选配置；冻结新验收任务并执行五方案检索、两方案生成及人工审核；补版本事实任务、失败重试用量和试用窗口。真实用户、收费和未测质量保持null/not_run。完整计划没有缩减。
'''
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(run / 'retrieval.csv', args.output_dir / 'retrieval.csv')
    # No new generation trial exists; a nonempty answer file would invent work.
    (args.output_dir / 'answers.jsonl').write_text('', encoding='utf-8')
    shutil.copyfile(alignment / 'human-review.csv', args.output_dir / 'human-review.csv')
    changes += '\nhuman-review.csv目前是90道旧题的证据标注待确认表，不是已生成回答的审核表。answers.jsonl为空，因为新批次尚无真实生成。\n'
    for name, content in [('changes.md', changes), ('failures.md', '\n'.join(failures) + '\n')]:
        (args.output_dir / name).write_text(content, encoding='utf-8')
    for name, content in [('manifest.json', manifest), ('metrics.json', metrics), ('runtime.json', runtime)]:
        (args.output_dir / name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'INCOMPLETE', 'directory': str(args.output_dir), 'files': sorted(p.name for p in args.output_dir.iterdir()),
                      'old_dev': actual, 'remote_calls': 0, 'new_acceptance_trials': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
