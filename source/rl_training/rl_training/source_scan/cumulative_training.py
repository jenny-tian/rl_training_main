"""Authorization v2 for a fresh, bounded numeric trial with prior charges."""
from __future__ import annotations

from .training_lifecycle import SourceRunSeeds
from .source_selection import validate_protocol, NODES
from .runtime_readiness import validate_numerics_runtime_receipt
from .training_budget_ledger import BudgetLedger

SCHEMA = 'lite3_source_ppo_cumulative_authorization_v2'
NUMERICS_SCHEMA = 'lite3_source_ppo_bounded_numerics_authorization_v3'


def validate_evidence(manifest, input_sha256, budget):
    from .training_runner import _receipt
    runtime = _receipt(manifest.get('runtime_qualification'), 'bounded numeric runtime')
    validate_numerics_runtime_receipt(runtime, input_sha256)
    if manifest.get('schema')==NUMERICS_SCHEMA:
        from .delivery_stage import validate_grant,validate_receipt
        validate_grant(manifest['delivery_stage'])
        validate_receipt(manifest['predecessor_qualification'],'g4',inputs=input_sha256)
        if (manifest.get('formal_selection_authorized') is not False
                or manifest.get('metrics_scope')!='diagnostic_only' or budget['run']!=1
                or manifest.get('delivery_job')!='numerics'
                or 'source_selection_lock' in manifest or 'metrics_qualification' in manifest):
            raise ValueError('numeric-only authorization cannot claim formal selection or metrics qualification')
        _validate_dependencies(manifest)
        return
    protocol = _receipt(manifest.get('source_selection_lock'), 'Source selection protocol')
    validate_protocol(protocol, require_locked=True)
    if protocol['budget_nodes'][str(budget['run'])] != [budget['nodes'][name]['remaining_updates'] for name in ('numerics','midpoint','pilot')]:
        raise ValueError('Source selection nodes differ from cumulative budget')
    metrics = _receipt(manifest.get('metrics_qualification'), 'Source metrics qualification')
    if (metrics.get('schema') != 'lite3_source_metrics_qualification_v1'
            or metrics.get('status') != 'qualified' or metrics.get('input_sha256') != input_sha256
            or metrics.get('live_normalized_metric_collector_qualified') is not True
            or metrics.get('registered_cells') != protocol['registered_cells']
            or metrics.get('front_foot_detector_force_N') != 20.
            or metrics.get('normalized_episode_schema') != 'lite3_source_episode_metrics_v1'):
        raise ValueError('Source metric collection has not been qualified')
    _validate_dependencies(manifest)


def _validate_dependencies(manifest):
    dependencies = manifest.get('dependency_sha256')
    if not isinstance(dependencies, dict) or len(dependencies) < 44:
        raise ValueError('cumulative numeric dependency inventory is incomplete')
    import hashlib
    from pathlib import Path
    for name, digest in dependencies.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('cumulative runtime dependency changed: '+name)


def validate_authorization(manifest, *, run, budget_node, output_dir, checkpoint_sha256,
                           revision, input_sha256, ledger_path):
    if manifest.get('schema') not in (SCHEMA,NUMERICS_SCHEMA): raise ValueError('invalid cumulative authorization schema')
    if manifest.get('user_approved') is not True or manifest.get('status') != 'approved':
        raise PermissionError('bounded numeric launch requires separate approval')
    if budget_node != 'numerics':
        raise NotImplementedError('midpoint/pilot continuation requires a separate reviewed implementation')
    ledger = BudgetLedger(ledger_path)
    if ledger.reference() != manifest.get('budget_ledger'):
        raise ValueError('cumulative ledger snapshot changed')
    budget = ledger.numerics_plan(run)
    node = budget['nodes']['numerics']; seeds = SourceRunSeeds(run)
    expected = {'action': 'source_ppo_bounded_numerics', 'run': run, 'budget_node': 'numerics',
        'ppo_seed': seeds.ppo, 'geometry_seed': seeds.geometry, 'sampler_seed': seeds.sampler,
        'reset_state_id': seeds.reset_state_id, 'num_envs': 120, 'rollout_length': 24,
        'level': 'L1', 'registered_cells': 15, 'six_families_retained': True,
        'device': 'cuda:0', 'minimum_free_vram_mib': 12288,
        'retry': False, 'resume': False, 'automatic_scale_increase': False,
        'development_or_sealed_feedback_allowed': False,
        'base_revision': revision, 'remote_revision_verified': revision,
        'checkpoint_sha256': checkpoint_sha256, 'input_sha256': input_sha256,
        'output_dir': str(output_dir.resolve()), 'cumulative_budget': budget,
        'max_updates': node['remaining_updates'], 'max_transitions': node['planned_transitions'],
        'hard_transition_ceiling': node['ceiling'],
        'initialization': 'registered_actor_fresh_critic_optimizer',
        'diagnostic_weights_reused': False, 'checkpoint_nodes': [node['remaining_updates']],
        'startup_timeout_s': 120, 'wall_timeout_s': 900, 'cleanup_grace_s': 15}
    for key, value in expected.items():
        if type(manifest.get(key)) is not type(value) or manifest[key] != value:
            raise ValueError('cumulative authorization mismatch: '+key)
    validate_evidence(manifest, input_sha256, budget)
    keys=(
        'schema','status','user_approved','budget_ledger','runtime_qualification',
        'source_selection_lock','metrics_qualification','dependency_sha256')
    if manifest['schema']==NUMERICS_SCHEMA:
        keys=('schema','status','user_approved','budget_ledger','runtime_qualification','dependency_sha256',
              'delivery_stage','delivery_job','predecessor_qualification','formal_selection_authorized','metrics_scope')
    return expected | {key:manifest[key] for key in keys}


def validate_supervised_child(manifest, ticket, *, output_dir, revision, input_sha256, ledger_path):
    lock = ticket.get('authorization')
    if ticket.get('schema') != 'lite3_source_numerics_supervision_v1' or not isinstance(lock, dict):
        raise ValueError('invalid Source numerics supervision ticket')
    if lock.get('schema') not in (SCHEMA,NUMERICS_SCHEMA) or lock.get('status') != 'approved' or lock.get('user_approved') is not True:
        raise PermissionError('unapproved Source numerics supervision')
    if any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k, v in lock.items()):
        raise ValueError('supervised authorization differs from approved manifest')
    if (lock['base_revision'] != revision or lock['remote_revision_verified'] != revision
            or lock['input_sha256'] != input_sha256 or lock['output_dir'] != str(output_dir.resolve())):
        raise ValueError('supervised Source inputs or output changed')
    ledger = BudgetLedger(ledger_path)
    if lock['budget_ledger']['path'] != str(ledger.path): raise ValueError('supervised budget path differs')
    reservation = ticket.get('reservation')
    if not isinstance(reservation, dict): raise ValueError('supervision reservation missing')
    rows = [r for r in ledger.read()['entries'] if r['attempt_id'] == reservation.get('attempt_id')]
    if rows != [reservation] or reservation.get('state') != 'reserved':
        raise ValueError('supervisor budget reservation missing or changed')
    budget = lock['cumulative_budget']; node = budget['nodes']['numerics']
    expected = {'run': lock['run'], 'purpose': 'source_numerics_cumulative', 'output_dir': lock['output_dir'],
        'base_revision': revision, 'reserved_transitions': node['planned_transitions'],
        'max_updates': node['remaining_updates'], 'num_envs': 120, 'rollout_length': 24,
        'baseline_charged_transitions': budget['baseline_charged_transitions'],
        'node_ceiling': node['ceiling'], 'initialization': budget['initialization'], 'automatic_refund': False}
    if any(reservation.get(key) != value for key, value in expected.items()):
        raise ValueError('supervised reservation contract changed')
    prior_charge = sum(r['reserved_transitions'] for r in ledger.read()['entries']
                       if r['run'] == lock['run'] and r['attempt_id'] != reservation['attempt_id'])
    if prior_charge != budget['baseline_charged_transitions']:
        raise ValueError('supervised prior budget changed')
    validate_evidence(lock, input_sha256, budget)
    return lock
