"""Prepare private source-aligned drafts without running any retrieval/model.

The 80-task target is still a plan. This script reports actual drafts and all
known exposures; its provisional split is not a human-confirmed acceptance set.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.benchmark_tasks import TaskDraft, compile_task
from app.evidence_alignment import EvidenceSnapshot
from app.retrieval import ROOT, CORPUS, read_jsonl, sha256
from app.review_workflow import record_hash
from app.source_families import assign_grouped_splits, audit_questions, normalize
from app.task_splitting import balance_task_groups


def historical_questions():
    """Public QA labels plus query strings in public reports, never new outputs."""
    questions, issues, files = defaultdict(set), [], {}
    for path in sorted((ROOT / 'artifacts/reports').rglob('*')):
        if path.suffix not in {'.json', '.jsonl'}:
            continue
        files[path.relative_to(ROOT).as_posix()] = sha256(path)
        try:
            rows = read_jsonl(path) if path.suffix == '.jsonl' else [json.loads(path.read_text(encoding='utf-8-sig'))]
        except (ValueError, UnicodeError):
            issues.append(path.relative_to(ROOT).as_posix())
            continue
        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in {'question', 'query', 'original_question'} and isinstance(value, str) and 1 <= len(value) <= 1500:
                        questions[value].add(path.relative_to(ROOT).as_posix())
                    elif isinstance(value, (dict, list)):
                        walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        for row in rows:
            walk(row)
    return [{'id': 'public-query-' + str(i), 'question': q, 'source_ids': [], 'previously_exposed': True,
             'provenance': sorted(paths)} for i, (q, paths) in enumerate(sorted(questions.items()))], files, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specs', type=Path, required=True)
    parser.add_argument('--alignment-dir', type=Path, required=True)
    parser.add_argument('--family-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prior-draft-dir', type=Path)
    parser.add_argument('--prior-dev-run', type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new draft revision')
    bundle = json.loads((args.alignment_dir / 'sources.json').read_text(encoding='utf-8'))
    if sha256(CORPUS) != bundle['corpus_sha256']:
        raise ValueError('Frozen raw corpus changed')
    chunks = [{**c, 'contents': bundle['documents'][c['metadata']['document_id']]
                [c['metadata']['start']:c['metadata']['end']]} for c in bundle['chunks']]
    snapshot = EvidenceSnapshot(bundle['version'], bundle['documents'], chunks)
    if snapshot.sha256 != bundle['snapshot_sha256']:
        raise ValueError('Frozen source snapshot changed')
    families = json.loads((args.family_dir / 'source-families.json').read_text(encoding='utf-8'))
    old_paths = [ROOT / 'data/basic_qa_mapped.jsonl', ROOT / 'data/extended_qa_mapped.jsonl']
    old = [c for path in old_paths for c in read_jsonl(path)]
    old_cases = [{'id': c['id'], 'question': c['question'], 'previously_exposed': True,
                  'source_ids': sorted({str(g['source_id']) for g in c['required_evidence_groups']})} for c in old]
    public, report_hashes, report_errors = historical_questions()
    # Explicit source scope and quotes come from the curator specs, not a
    # retriever, candidate pool, generated answer or grader-produced truth.
    specs = [TaskDraft.model_validate(s) for s in json.loads(args.specs.read_text(encoding='utf-8'))]
    if len({s.id for s in specs}) != len(specs):
        raise ValueError('Duplicate draft IDs')
    tasks = [compile_task(s, bundle['documents'], bundle['origins'], snapshot.version, 'draft-gold-v1') for s in specs]
    combined = old_cases + public + tasks
    if len({c['id'] for c in combined}) != len(combined):
        raise ValueError('Draft/public/old case identities overlap')
    assigned = {c['id']: c for c in assign_grouped_splits(combined, families, seed='VersionRAG-new-gold-20261013-v1')}
    audit = audit_questions(combined, families)
    draft_ids = {c['id'] for c in tasks}
    draft_links = [link for link in audit['links'] if link['left'] in draft_ids or link['right'] in draft_ids]
    for task in tasks:
        task.update({key: assigned[task['id']][key] for key in ('split', 'group_id', 'family_ids')})
        exposure_links = [link for link in draft_links if task['id'] in (link['left'], link['right'])
                          and (link['left'] not in draft_ids or link['right'] not in draft_ids)]
        task['exposure_issues'] = exposure_links
        task['novelty_status'] = 'known_exposure_regression' if task['split'] == 'regression' else 'pending_human_novelty_review'
    locked_dev = []
    if args.prior_draft_dir or args.prior_dev_run:
        if not args.prior_draft_dir or not args.prior_dev_run:
            raise ValueError('Expansion requires both prior draft labels and the actual prior dev run')
        old_labels = args.prior_draft_dir / 'labels.jsonl'
        prior_run = json.loads((args.prior_dev_run / 'manifest.json').read_text(encoding='utf-8'))
        if prior_run['labels_sha256'] != sha256(old_labels) or prior_run['split'] != 'dev':
            raise ValueError('Prior dev run does not bind these old labels')
        old_by_id = {r['id']: r for r in read_jsonl(old_labels)}
        new_by_id = {r['id']: r for r in tasks}
        locked_dev = prior_run['case_ids']
        for cid in locked_dev:
            if cid not in new_by_id or cid not in old_by_id:
                raise ValueError('Previously queried dev case missing from expansion')
            if new_by_id[cid]['question'] != old_by_id[cid]['question'] or new_by_id[cid]['source_ids'] != old_by_id[cid]['source_ids']:
                raise ValueError('Previously queried dev identity changed; preserve the exposure before revision')
    tasks, balance_audit = balance_task_groups(tasks, locked_dev_ids=locked_dev)
    queried_groups = {t['group_id'] for t in tasks if t['id'] in locked_dev}
    for task in tasks:
        task['previously_queried_dev'] = task['id'] in locked_dev
        task['prior_dev_exposure_in_component'] = task['group_id'] in queried_groups
        if task['prior_dev_exposure_in_component'] and task['split'] != 'regression':
            task['novelty_status'] = 'development_exposed_component; never fresh acceptance'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    def write_lines(path, rows):
        with path.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_lines(args.output_dir / 'labels.jsonl', tasks)
    write_lines(args.output_dir / 'rubrics.jsonl', [t['rubric'] for t in tasks])
    templates = [{'case_id': t['id'], 'task_sha256': record_hash(t), 'status': 'pending', 'review': None} for t in tasks]
    write_lines(args.output_dir / 'gold-review-template.jsonl', templates)
    packets = ['# 新题标注审核包（草稿）', '', '80题仍是计划，本包是首批实际草稿。没有运行检索或模型；标签全部待人工确认。',
               '审核任务是否明确、事实是否正确且完整、证据是否充分、历史年份及不足处置是否成立、是否属于旧题改写。',
               '引文位置准确只说明文本定位成功，不等于事实真值或新颖性已经确认。集合事实的结构化value可规范化顺序，实际答复事实句必须原样摘录。',
               '只有真实人工填写GoldReview的来源、时区日期和六项明确判断后，程序才可标注人工确认。模型/助手批准不算人审。']
    with (args.output_dir / 'human-review.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        fields = ['case_id', 'task_sha256', 'question', 'question_type', 'provisional_split', 'expected_action',
                  'expected_answer', 'temporal_scope', 'mapping_status', 'status', 'reviewer_type', 'reviewer_id',
                  'reviewed_at', 'question_clear', 'facts_correct', 'evidence_sufficient', 'requirements_complete',
                  'temporal_scope_correct', 'novelty_checked', 'notes']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for t in tasks:
            writer.writerow({**{k: t.get(k) for k in fields if k in t}, 'case_id': t['id'], 'task_sha256': record_hash(t),
                             'provisional_split': t['split'], 'status': 'pending'})
            packets += ['', '## ' + t['id'] + ' · ' + t['question_type'] + ' · 临时' + t['split'],
                        '', t['question'], '', '预期处置：' + t['expected_action'] + '；草稿答案：' + t['expected_answer'],
                        '时间边界：' + t['temporal_scope'], '说明：' + t['review_notes'], '文本定位：' + t['mapping_status']]
            for fact in t['rubric']['required_facts']:
                packets.append('- 必需事实 ' + fact['id'] + '：' + fact['entity'] + ' / ' + fact['attribute'] + ' / ' + ' | '.join(fact['values']))
            for i, group in enumerate(t['required_evidence_groups']):
                packets.append('- 证据组 ' + str(i) + '（等价出处满足其一；不同组须全部满足）：')
                for a in group:
                    origin = bundle['origins'][a['original_passage_id']]
                    packets.append('  - 来源 ' + a['source_id'] + ' / 原片段 ' + a['original_passage_id'] + ' / ' + origin['title']
                                   + f" / 字符区间[{a['start']},{a['end']})：" + a['quote'].replace('\n', ' '))
                    text = bundle['documents'][a['document_id']]
                    context = text[max(0, a['start'] - 180):min(len(text), a['end'] + 180)]
                    packets.append('    上下文（用于核对对象/年份/限定语，不自动当作事实批准）：' + context.replace('\n', ' '))
            if t['mapping_issues']:
                packets.append('待查定位问题：' + json.dumps(t['mapping_issues'], ensure_ascii=False))
            if t['exposure_issues']:
                packets.append('已知暴露关联：' + json.dumps(t['exposure_issues'], ensure_ascii=False))
    (args.output_dir / '审核包.md').write_text('\n'.join(packets) + '\n', encoding='utf-8')
    registry = {'public_questions': public, 'report_sha256': report_hashes, 'unreadable_reports': report_errors,
                'old_qa_count': len(old), 'old_qa_sha256': {p.name: sha256(p) for p in old_paths},
                'exposure_scope': 'both old QA datasets and query fields in public reports; semantic paraphrases still need human checking'}
    (args.output_dir / 'exposure-registry.json').write_text(json.dumps(registry, ensure_ascii=False), encoding='utf-8')
    (args.output_dir / 'split-audit.json').write_text(json.dumps({'links': draft_links, 'unknown_sources': audit['unknown_sources'],
                  'metadata_balance': balance_audit}, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'schema_version': 'new-task-drafts-v1', 'status': 'pending_human_review', 'planned_total': 80,
                'previously_queried_dev_ids': locked_dev,
                'source_family_count': len({f for t in tasks for f in t['family_ids']}),
                'actual_drafts': len(tasks), 'human_confirmed': 0, 'acceptance_frozen': False,
                'split_counts': dict(Counter(t['split'] for t in tasks)), 'type_counts': dict(Counter(t['question_type'] for t in tasks)),
                'mapped': sum(t['mapping_status'] == 'mapped' for t in tasks), 'blocked_ids': [t['id'] for t in tasks if t['mapping_status'] != 'mapped'],
                'source_scope': bundle['source_scope'], 'knowledge_version': snapshot.version, 'snapshot_sha256': snapshot.sha256,
                'source_bundle_sha256': sha256(args.alignment_dir / 'sources.json'), 'specs_sha256': sha256(args.specs),
                'family_sha256': sha256(args.family_dir / 'source-families.json'), 'labels_sha256': sha256(args.output_dir / 'labels.jsonl'),
                'rubrics_sha256': sha256(args.output_dir / 'rubrics.jsonl'), 'exposure_registry_sha256': sha256(args.output_dir / 'exposure-registry.json'),
                'old_qa_count': len(old), 'public_query_count': len(public), 'unreadable_report_count': len(report_errors),
                'source_ids': sorted({sid for t in tasks for sid in t['source_ids']}),
                'remote_calls_this_preparation': 0, 'retrieval_runs_this_preparation': 0,
                'scope': 'assistant-curated draft; prior dev exposures retained, no human gold or acceptance/performance claim'}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
