"""Proposed Source-only selection rules and offline checks of normalized episodes."""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .frozen_actor import CHECKPOINT_SHA256
from .parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS

SCHEMA = 'lite3_six_family_source_selection_v1'
NODES = {'1': [90, 363, 728], '2': [91, 364, 728], '3': [91, 364, 728]}
LIMITS = {'velocity_error_mean_mps': .20, 'velocity_error_rms_mps': .25,
          'roll_peak_rad': .40, 'pitch_peak_rad': .40, 'lateral_peak_m': .30,
          'slip_contact_fraction': .10, 'projection_fraction': .10}
HARD_COUNTS = ('emergency_count', 'invalid_state_count', 'nonfoot_contact_termination_count')
RETENTION = {key: (.10 if key in ('roll_peak_rad', 'pitch_peak_rad') else .05) for key in LIMITS}


def registered_validation_cells():
    order = ['stairs_ascending', 'stairs_descending', 'flat']
    order.extend(name for name in SOURCE_RESET_BLOCK_COUNTS if name not in order)
    return [{'subterrain': name, 'level': 1,
        'selection_geometry_seeds': list(range(50000+5*i, 50005+5*i)),
        'selection_state_ids': list(range(60000+5*i, 60005+5*i)),
        'confirmation_geometry_seeds': list(range(55000+5*i, 55005+5*i)),
        'confirmation_state_ids': list(range(65000+5*i, 65005+5*i))}
        for i, name in enumerate(order)]


def family_of(cell):
    if cell == 'flat': return 'flat'
    if cell == 'multi_scale_rough': return 'multi_scale_rough'
    if cell.startswith('slope_rough_'): return 'slope_plus_rough'
    if cell.startswith('slope_'): return 'slope'
    if cell.startswith('stairs_'): return 'multi_step_stairs'
    if cell.startswith('friction_'): return 'friction_variation'
    raise ValueError('unknown Source cell')


def proposal():
    return {'schema': SCHEMA, 'status': 'proposed', 'user_approved': False,
        'scope': 'source_only', 'registered_cells': list(SOURCE_RESET_BLOCK_COUNTS),
        'budget_nodes': deepcopy(NODES), 'seed_splits': registered_validation_cells(),
        'baseline_checkpoint_sha256': CHECKPOINT_SHA256, 'episodes_per_cell': 5,
        'required_complete_and_quality_count': 4, 'quality_gates': dict(LIMITS),
        'hard_zero_counts': list(HARD_COUNTS), 'flat_paired_degradation_limits': dict(RETENTION),
        'completion_progress_m': 9., 'completion_hold_steps': 50, 'episode_step_limit': 2000,
        'stairs': {'completed_step_index': 3, 'stuck_count': 0, 'max_collision_episodes': 1,
                   'front_foot_rising_edge_force_N': 20., 'runtime_detector_qualification_required': True},
        'metric_qualification_required': True,
        'normalized_episode_schema': 'lite3_source_episode_metrics_v1',
        'selection_rule': ['maximize_worst_cell_quality_count', 'maximize_equal_cell_quality_count_sum',
                           'minimize_equal_cell_mean_rms_velocity_error', 'prefer_earlier_update'],
        'early_stop': 'midpoint_only_if_both_numerics_and_midpoint_pass_all_15_cells',
        'confirmation': 'one_selected_checkpoint_per_run_no_reselection',
        'state_seed_semantics': 'bookkeeping_only_deterministic_reset',
        'geometry_identity': 'sha256_of_canonical_local_mesh_material_and_parameters',
        'fixed_geometry_interpretation': 'repeated_trials_not_independent_geometry_samples',
        'development_or_sealed_feedback_allowed': False,
        'evaluation_episodes_max': {'selection_all_runs': 675, 'confirmation_all_runs': 225,
            'shared_frozen_flat_baseline_selection': 5, 'shared_frozen_flat_baseline_confirmation': 5,
            'total': 910}}


def validate_protocol(protocol, *, require_locked=False):
    expected = proposal()
    for key, value in expected.items():
        if key in ('status', 'user_approved'): continue
        if type(protocol.get(key)) is not type(value) or protocol[key] != value:
            raise ValueError('Source selection protocol mismatch: '+key)
    if set(protocol) - set(expected) - {'approval_message'}:
        raise ValueError('unknown Source selection protocol fields')
    if require_locked and (protocol.get('status') != 'locked' or protocol.get('user_approved') is not True):
        raise PermissionError('Source selection thresholds require explicit protocol approval')
    if not require_locked and (protocol.get('status'), protocol.get('user_approved')) not in (
            ('proposed', False), ('locked', True)):
        raise ValueError('invalid protocol approval state')
    return protocol


def _number(value, name, *, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or upper is not None and value > upper:
        raise ValueError('invalid Source metric: '+name)
    return value


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid Source evidence digest')


def _episode(record, cell, split, checkpoint):
    if record.get('schema') != 'lite3_source_episode_metrics_v1':
        raise ValueError('episode metrics have not been normalized to the registered schema')
    _sha(record.get('geometry_sha256'))
    if (record.get('cell') != cell or record.get('split') != split
            or record.get('checkpoint_sha256') != checkpoint or record.get('state_randomized') is not False):
        raise ValueError('episode belongs to a different checkpoint, cell or split')
    metrics = record.get('metrics')
    if not isinstance(metrics, dict): raise ValueError('missing Source episode metrics')
    if type(metrics.get('completed')) is not bool: raise ValueError('completed must be boolean')
    for key in LIMITS: _number(metrics.get(key), key, upper=1 if key.endswith('fraction') else None)
    for key in (*HARD_COUNTS, 'stable_hold_steps', 'steps'):
        if type(metrics.get(key)) is not int or metrics[key] < 0: raise ValueError('invalid metric count: '+key)
    if not 1 <= metrics['steps'] <= 2000: raise ValueError('episode step count outside registered bound')
    progress = metrics.get('forward_progress_m')
    if type(progress) not in (int, float) or not math.isfinite(progress):
        raise ValueError('invalid signed forward progress')
    if metrics['stable_hold_steps'] > metrics['steps'] or metrics['velocity_error_rms_mps']+1e-8 < metrics['velocity_error_mean_mps']:
        raise ValueError('episode duration or velocity statistics contradict each other')
    completed = metrics['completed']
    if completed and (metrics['forward_progress_m'] < 9 or metrics['stable_hold_steps'] < 50):
        raise ValueError('completion contradicts progress/hold evidence')
    quality = completed and all(metrics[key] <= value for key, value in LIMITS.items())
    hard = all(metrics[key] == 0 for key in HARD_COUNTS)
    family = family_of(cell)
    if family == 'multi_step_stairs':
        for key in ('completed_step_index', 'stuck_count', 'front_foot_collision_count'):
            if type(metrics.get(key)) is not int or metrics[key] < 0: raise ValueError('missing stair metric: '+key)
        if completed and metrics['completed_step_index'] != 3: raise ValueError('stairs completion is not three steps')
        hard &= metrics['stuck_count'] == 0
    if family in ('slope', 'slope_plus_rough'):
        direction = cell.removeprefix('slope_rough_').removeprefix('slope_')
        if metrics.get('slope_direction') != direction or metrics.get('slope_angle_deg') != 5.:
            raise ValueError('slope metric direction/angle mismatch')
        if type(metrics.get('transition_completed')) is not bool: raise ValueError('missing slope transition metric')
        quality &= metrics['transition_completed']
    if family in ('multi_scale_rough', 'slope_plus_rough'):
        _number(metrics.get('audited_rough_peak_valley_m'), 'audited_rough_peak_valley_m')
        scales = metrics.get('audited_spatial_scales_m')
        if not isinstance(scales, list) or len(scales) < 2 or len(set(scales)) != len(scales):
            raise ValueError('rough spatial scale audit missing')
        for scale in scales:
            if not _number(scale, 'spatial_scale_m'): raise ValueError('spatial scale must be positive')
    if family == 'friction_variation':
        if metrics.get('region_layout') != cell.removeprefix('friction_') or metrics.get('material_binding_verified') is not True:
            raise ValueError('friction region/material evidence missing')
        slips = metrics.get('slip_by_region')
        if not isinstance(slips, dict) or not slips: raise ValueError('missing region slip measurements')
        for value in slips.values(): _number(value, 'region_slip_fraction', upper=1)
        if completed and len(slips) < 2: raise ValueError('completed friction episode did not observe multiple regions')
        quality &= max(slips.values()) <= LIMITS['slip_contact_fraction']
    return bool(quality), bool(hard)


def evaluate_batch(batch, protocol, *, baseline_flat, split='selection'):
    validate_protocol(protocol)
    if split not in ('selection', 'confirmation'): raise ValueError('Source-only split required')
    run, update = batch.get('run'), batch.get('update')
    if type(run) is not int or type(update) is not int or str(run) not in NODES or update not in NODES[str(run)]:
        raise ValueError('unregistered training run/update')
    checkpoint = batch.get('checkpoint_sha256'); _sha(checkpoint)
    if batch.get('split') != split: raise ValueError('batch uses wrong split')
    episodes = batch.get('episodes')
    if not isinstance(episodes, list) or len(episodes) != 75: raise ValueError('expected exactly five episodes in all 15 cells')
    if any(row.get('run') != run or row.get('update') != update
            or type(row.get('geometry_seed')) is not int or type(row.get('state_id')) is not int for row in episodes):
        raise ValueError('episode run/update or seed identity mismatch')
    results = {}; all_passed = True
    for cell in registered_validation_cells():
        name = cell['subterrain']; rows = [record for record in episodes if record.get('cell') == name]
        pairs = list(zip(cell[split+'_geometry_seeds'], cell[split+'_state_ids']))
        if len(rows) != 5 or {(r.get('geometry_seed'), r.get('state_id')) for r in rows} != set(pairs):
            raise ValueError('missing, duplicate or unregistered episode seed: '+name)
        flags = [_episode(row, name, split, checkpoint) for row in rows]
        count = sum(quality for quality, hard in flags)
        passed = count >= 4 and all(hard for quality, hard in flags)
        if name.startswith('stairs_'):
            passed &= sum(row['metrics']['front_foot_collision_count'] > 0 for row in rows) <= 1
        if name == 'flat':
            if len(baseline_flat) != 5: raise ValueError('five paired frozen flat baseline records required')
            baseline_by_pair = {(row['geometry_seed'], row['state_id']): row for row in baseline_flat}
            if set(baseline_by_pair) != set(pairs): raise ValueError('flat baseline pairing differs from candidate split')
            for row in rows:
                base = baseline_by_pair[(row['geometry_seed'], row['state_id'])]
                _episode(base, 'flat', split, CHECKPOINT_SHA256)
            for key, limit in RETENTION.items():
                delta = sum(row['metrics'][key] - baseline_by_pair[(row['geometry_seed'], row['state_id'])]['metrics'][key] for row in rows)/5
                passed &= delta <= limit
        distinct = len({row['geometry_sha256'] for row in rows})
        results[name] = {'passed': bool(passed), 'complete_and_quality_count': count,
            'mean_rms_velocity_error_mps': sum(row['metrics']['velocity_error_rms_mps'] for row in rows)/5,
            'distinct_local_geometry_hashes': distinct,
            'trial_interpretation': 'repeated_trials' if distinct < 5 else 'distinct_geometry_instances',
            'population_success_confidence_interval': None}
        all_passed &= passed
    return {'run': run, 'update': update, 'checkpoint_sha256': checkpoint, 'split': split,
            'passed': bool(all_passed), 'cells': results,
            'worst_cell_quality_count': min(r['complete_and_quality_count'] for r in results.values()),
            'quality_count_sum': sum(r['complete_and_quality_count'] for r in results.values()),
            'equal_cell_mean_rms_velocity_error_mps': sum(r['mean_rms_velocity_error_mps'] for r in results.values())/15}


def select_checkpoint(batches, protocol, *, baseline_flat, schedule_end='pilot'):
    validate_protocol(protocol, require_locked=True)
    if schedule_end not in ('midpoint', 'pilot'): raise ValueError('invalid selection schedule end')
    if not batches: raise ValueError('no selection batches')
    run = batches[0].get('run')
    if any(batch.get('run') != run or batch.get('split') != 'selection' for batch in batches):
        raise ValueError('cannot pool training runs or use confirmation feedback for selection')
    expected = NODES.get(str(run), [])[:2 if schedule_end == 'midpoint' else 3]
    if len(batches) != len(expected) or sorted(batch.get('update') for batch in batches) != expected:
        raise ValueError('selection schedule incomplete, duplicated or changed')
    evaluated = [evaluate_batch(batch, protocol, baseline_flat=baseline_flat) for batch in batches]
    if schedule_end == 'midpoint' and not all(row['passed'] for row in evaluated):
        raise ValueError('midpoint early stop requires both registered nodes to pass')
    eligible = [row for row in evaluated if row['passed']]
    if not eligible: return {'run': run, 'status': 'no_eligible_checkpoint', 'selected': None}
    best = min(eligible, key=lambda row: (-row['worst_cell_quality_count'], -row['quality_count_sum'],
        row['equal_cell_mean_rms_velocity_error_mps'], row['update']))
    return {'run': run, 'status': 'selected_for_confirmation',
            'selected': {key: best[key] for key in ('update', 'checkpoint_sha256')},
            'selection_rule': protocol['selection_rule'], 'confirmation_used': False}


def confirm_checkpoint(batch, decision, protocol, *, baseline_flat):
    validate_protocol(protocol, require_locked=True)
    selected = decision.get('selected')
    if (decision.get('status') != 'selected_for_confirmation' or not selected
            or batch.get('run') != decision.get('run') or any(batch.get(key) != value for key, value in selected.items())):
        raise ValueError('confirmation does not belong to the selected checkpoint')
    result = evaluate_batch(batch, protocol, baseline_flat=baseline_flat, split='confirmation')
    return {'run': result['run'], 'checkpoint_sha256': result['checkpoint_sha256'],
            'status': 'confirmed' if result['passed'] else 'confirmation_failed',
            'reselection_allowed': False, 'result': result}
