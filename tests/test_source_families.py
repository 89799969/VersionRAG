"""Conservative split policy checks with fixtures, not new acceptance data."""
import copy

import pytest

from app.source_families import build_source_families, audit_questions, assign_grouped_splits


def docs():
    return [{'id': '1', 'contents': '软件 工程\\n每年 8000元', 'metadata': {'source_id': 's1'}},
            {'id': '2', 'contents': '软件工程每年8000元', 'metadata': {'source_id': 's2'}},
            {'id': '3', 'contents': '转载第二段', 'metadata': {'source_id': 's2'}},
            {'id': '4', 'contents': '转载第二段', 'metadata': {'source_id': 's3'}},
            {'id': '5', 'contents': '独立的另一份资料', 'metadata': {'source_id': 's4'}}]


def case(cid, sources, question, exposed=False, split='dev'):
    return {'id': cid, 'source_ids': sources, 'question': question,
            'previously_exposed': exposed, 'split': split}


def test_normalized_duplicate_links_join_entire_sources_transitively():
    families = build_source_families(docs())
    ids = families['source_to_family']
    assert ids['s1'] == ids['s2'] == ids['s3'] != ids['s4']
    assert len(families['duplicate_links']) == 2
    assert build_source_families(list(reversed(docs())))['source_to_family'] == ids


def test_audit_warns_without_rewriting_the_old_split():
    families = build_source_families(docs())
    cases = [case('old', ['s1'], '学费是多少', True), case('other', ['s3'], '教学语言是什么', True, 'holdout')]
    before = copy.deepcopy(cases)
    audit = audit_questions(cases, families)
    assert len(audit['links']) == 1 and audit['links'][0]['cross_split']
    assert cases == before


def test_rephrased_or_shared_source_components_cannot_be_new_acceptance():
    families = build_source_families(docs())
    cases = [case('old', ['s1'], '软件工程专业的学费是多少', True),
             case('shared-source', ['s3'], '请列出教学语言'),
             case('rewritten', ['s4'], '软件工程专业的学费是多少呢')]
    assigned = assign_grouped_splits(cases, families)
    assert {c['split'] for c in assigned} == {'regression'}
    assert len({c['group_id'] for c in assigned}) == 1


def test_unknown_source_and_unstated_exposure_cannot_enter_a_split():
    families = build_source_families(docs())
    with pytest.raises(ValueError, match='unknown'):
        assign_grouped_splits([case('x', ['missing'], '问题')], families)
    unknown = case('x', ['s1'], '问题')
    unknown.pop('previously_exposed')
    with pytest.raises(ValueError, match='explicit'):
        assign_grouped_splits([unknown], families)


def test_split_is_deterministic_under_input_order_and_no_family_crosses():
    families = build_source_families(docs())
    cases = [case('a', ['s1'], '程序语言'), case('b', ['s3'], '学位要求'), case('c', ['s4'], '独立事件日期')]
    first = assign_grouped_splits(cases, families, seed='frozen')
    second = assign_grouped_splits(cases[::-1], families, seed='frozen')
    assert {c['id']: c['split'] for c in first} == {c['id']: c['split'] for c in second}
    assert first[0]['split'] == first[1]['split']
