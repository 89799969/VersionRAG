"""Pre-execution gates: draft diagnostics and once-frozen acceptance differ."""
from app.review_workflow import record_hash


def validate_task_run(labels, manifest, split, *, allow_pending_dev=False, freeze=None):
    if len({r['id'] for r in labels}) != len(labels):
        raise ValueError('Duplicate case identities')
    selected = [r for r in labels if r['split'] == split]
    if not selected:
        raise ValueError('Selected split has no cases')
    if split == 'acceptance':
        if not manifest.get('acceptance_ready_for_later_freeze') or not freeze or freeze.get('acceptance_frozen') is not True:
            raise ValueError('Acceptance requires human gold and a frozen selected configuration')
        if any(r['label_status'] != 'human_confirmed' or r.get('human_confirmed') is not True for r in selected):
            raise ValueError('All planned acceptance labels require actual human confirmation')
        if freeze.get('case_ids') != [r['id'] for r in selected] or freeze.get('task_records_sha256') != record_hash(selected):
            raise ValueError('Acceptance cases changed after freezing')
    elif split == 'dev':
        if any(r['label_status'] != 'human_confirmed' for r in selected) and not allow_pending_dev:
            raise ValueError('Pending dev labels are only usable with explicit draft diagnostics')
    else:
        raise ValueError('Use dev or acceptance; known exposed tasks belong to regression')
    return selected
