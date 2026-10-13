"""Prepare blind review packets or score submitted reviews without model calls."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.review_workflow import prepare_blind_review, apply_reviews
from app.answer_metrics import reconcile_plan
from app.retrieval import read_jsonl, sha256
from app.retrieval import ROOT


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'score'])
    parser.add_argument('--trials', type=Path, required=True)
    parser.add_argument('--rubrics', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--review-dir', type=Path)
    parser.add_argument('--submitted', type=Path)
    parser.add_argument('--planned-trials', type=Path,
                        help='Fixed full plan, including interrupted/unexecuted trials')
    parser.add_argument('--batch-manifest', type=Path,
                        help='Generated batch manifest binds the saved full plan and input cohort')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output already exists; choose a new directory')
    trials, rubrics = read_jsonl(args.trials), read_jsonl(args.rubrics)
    if args.planned_trials:
        if args.batch_manifest is None:
            parser.error('--planned-trials requires --batch-manifest to verify its origin')
        batch = json.loads(args.batch_manifest.read_text(encoding='utf-8'))
        plan = read_jsonl(args.planned_trials)
        if (batch.get('planned_trials_sha256') != sha256(args.planned_trials)
                or batch.get('planned_trials') != len(plan)
                or batch.get('rubrics_sha256') != sha256(args.rubrics)):
            raise ValueError('Plan/rubrics differ from original batch manifest')
        if batch.get('status') in {'complete_unreviewed', 'prepared_not_run'} and batch.get('answers_sha256') != sha256(args.trials):
            raise ValueError('Completed batch answers changed')
        trials = reconcile_plan(plan, trials)
    elif any(t.get('planned_repeats') is not None for t in trials):
        parser.error('New benchmark trials require --planned-trials to prove the complete denominator')
    if args.operation == 'prepare':
        private, packets, templates = prepare_blind_review(trials, rubrics, args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_jsonl(args.output_dir / 'private-map.jsonl', private)
        write_jsonl(args.output_dir / 'review-packets.jsonl', packets)
        write_jsonl(args.output_dir / 'review-template.jsonl', templates)
        (args.output_dir / 'HOW_TO_REVIEW.md').write_text(
            '# 人工审核说明\n\n只给审核者 review-packets.jsonl 和 review-template.jsonl；private-map.jsonl 留给汇总程序。'
            '审核顺序已打乱，方案名隐藏，但措辞仍可能暴露方案，不能保证完全盲审。\n\n'
            '逐题核对真实输入证据、问题的年份和对象、回答中每条关键事实及比较结论。'
            'review-template 的 review 初始为 null、status 为 pending；审核完才填 submitted。'
            ' review 按 app/answer_evaluation.py 中 AnswerReview 的字段填写。'
            '正确性、引用支持、事实盘点完整性分别判断；引用文字相同不代表支持结论。'
            'claims 中 text 要逐字摘录实际回答里的事实句，不能用审核者修正后的表述替代。'
            '没有确认的字段用 null，不填 true。reviewer_type 如实填 human/model/assistant/synthetic；'
            'human 需匿名 reviewer_id 和带时区 reviewed_at。\n\n'
            '缺审核保留 INCOMPLETE。标注未人工确认时，任何审核也不能产生人工主成绩。'
            '请勿改动已冻结 trial/rubric 后沿用旧审核。程序保存审核来源声明，但无法验证是否真的由人工填写。\n', encoding='utf-8')
        summary = {'planned_trials': len(trials), 'review_status': 'pending', 'remote_calls': 0}
    else:
        if args.review_dir is None or args.submitted is None:
            parser.error('score requires --review-dir and --submitted')
        results, summary = apply_reviews(trials, rubrics, read_jsonl(args.review_dir / 'private-map.jsonl'),
                                         read_jsonl(args.submitted))
        summary['planned_cohort_verified'] = args.planned_trials is not None
        summary['cohort_scope'] = 'fixed_full_plan' if args.planned_trials else 'provided_records_only; full cohort not proven'
        if args.planned_trials is None:
            summary['strict_main_pass']['value'] = None
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_jsonl(args.output_dir / 'answers.jsonl', results)
        (args.output_dir / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'trials_sha256': sha256(args.trials), 'rubrics_sha256': sha256(args.rubrics),
                'operation': args.operation, 'seed': args.seed if args.operation == 'prepare' else None,
                'submitted_sha256': sha256(args.submitted) if args.submitted else None,
                'planned_trials_sha256': sha256(args.planned_trials) if args.planned_trials else None,
                'batch_manifest_sha256': sha256(args.batch_manifest) if args.batch_manifest else None,
                'scoring_code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p)
                                       for p in sorted((ROOT / 'app').glob('*.py'))},
                'remote_calls': 0, 'scope': 'explicit review accounting; not automatic semantic grading'}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'directory': str(args.output_dir), 'planned_trials': summary['planned_trials'],
                      'unique_questions': summary.get('unique_questions'),
                      'main_results': summary.get('main_results'),
                      'strict_main_pass': summary.get('strict_main_pass'),
                      'planned_cohort_verified': summary.get('planned_cohort_verified'),
                      'remote_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
