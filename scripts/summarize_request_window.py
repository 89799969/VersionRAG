"""Summarize a private request ledger, without claiming real-user quality."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import read_jsonl, sha256
from app.runtime_metrics import summarize_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=['local-observation', 'synthetic-protocol'], default='local-observation')
    parser.add_argument('--scored-answers', type=Path, help='Private original outputs/rubrics/reviews, one primary record per request')
    parser.add_argument('--sampling-rule', help='Actual review selection rule; not a quality or real-user assertion')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new window report')
    report = summarize_window(read_jsonl(args.ledger), observation_scope=args.scope,
                              scored_answers=read_jsonl(args.scored_answers) if args.scored_answers else None,
                              sampling_rule=args.sampling_rule)
    report['ledger_sha256'] = sha256(args.ledger)
    report['scored_answers_sha256'] = sha256(args.scored_answers) if args.scored_answers else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: report[key] for key in ('observation_scope', 'requests', 'answer_requests',
                      'http_failed_requests', 'provider_attempts', 'money_cost', 'human_quality')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
