from app.task_splitting import balance_task_groups


def test_source_components_stay_together_and_both_sides_cover_available_types():
    tasks = [{'id': str(i), 'group_id': g, 'question_type': kind, 'split': 'dev'} for i, (g, kind) in enumerate([
        ('a', 'single'), ('a', 'multi'), ('b', 'single'), ('b', 'multi'), ('c', 'compare'), ('d', 'compare'),
        ('e', 'refuse'), ('f', 'refuse')])]
    result, audit = balance_task_groups(tasks)
    assert audit['objective'][0] == 0 and audit['retrieval_results_used'] is False
    assert {t['split'] for t in result if t['group_id'] == 'a'} in ({'dev'}, {'acceptance'})
    assert set(audit['type_counts_by_split']['dev']) == set(audit['type_counts_by_split']['acceptance'])
    reversed_result, _ = balance_task_groups(tasks[::-1])
    assert {t['id']: t['split'] for t in result} == {t['id']: t['split'] for t in reversed_result}


def test_public_regressions_are_not_moved_to_balance_fresh_counts():
    tasks = [{'id': 'old', 'group_id': 'exposed', 'question_type': 'compare', 'split': 'regression'},
             {'id': 'fresh', 'group_id': 'fresh', 'question_type': 'single', 'split': 'acceptance'}]
    result, audit = balance_task_groups(tasks)
    assert result[0]['split'] == 'regression'
    assert 'single' in audit['types_not_splittable_across_both']


def test_expansion_keeps_queried_dev_cases_and_their_new_group_mates_in_development():
    tasks = [{'id': 'queried', 'group_id': 'a', 'question_type': 'single', 'split': 'dev'},
             {'id': 'new-same-source', 'group_id': 'a', 'question_type': 'multi', 'split': 'acceptance'},
             {'id': 'other', 'group_id': 'b', 'question_type': 'single', 'split': 'dev'}]
    result, audit = balance_task_groups(tasks, locked_dev_ids=['queried'])
    assert result[0]['split'] == result[1]['split'] == 'dev'
    assert audit['locked_previously_queried_dev_ids'] == ['queried']
