"""Source-only selection tests use fabricated normalized episode metrics on CPU."""
from copy import deepcopy
import hashlib

import pytest

from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
from rl_training.source_scan.source_selection import (
    LIMITS, NODES, proposal, registered_validation_cells, family_of,
    validate_protocol, evaluate_batch, select_checkpoint, confirm_checkpoint)


def protocol():
    return proposal() | {'status':'locked','user_approved':True}


def batch(update=90, split='selection', run=1):
    checkpoint = hashlib.sha256(f'CPU fixture {run} {update}'.encode()).hexdigest()
    episodes = []
    for cell in registered_validation_cells():
        name = cell['subterrain']; family = family_of(name)
        for geometry,state in zip(cell[split+'_geometry_seeds'],cell[split+'_state_ids']):
            metrics = {key:.05 for key in LIMITS}
            metrics.update(velocity_error_rms_mps=.10,completed=True,forward_progress_m=9.1,
                stable_hold_steps=50,steps=1600,emergency_count=0,invalid_state_count=0,
                nonfoot_contact_termination_count=0)
            if family=='multi_step_stairs':
                metrics.update(completed_step_index=3,stuck_count=0,front_foot_collision_count=0)
            if family in ('slope','slope_plus_rough'):
                metrics.update(slope_direction=name.removeprefix('slope_rough_').removeprefix('slope_'),
                    slope_angle_deg=5.,transition_completed=True)
            if family in ('multi_scale_rough','slope_plus_rough'):
                metrics.update(audited_rough_peak_valley_m=.02,audited_spatial_scales_m=[.1,.5])
            if family=='friction_variation':
                metrics.update(region_layout=name.removeprefix('friction_'),material_binding_verified=True,
                    slip_by_region={'first':.03,'second':.05})
            episodes.append(dict(schema='lite3_source_episode_metrics_v1',run=run,update=update,
                cell=name,split=split,checkpoint_sha256=checkpoint,geometry_seed=geometry,state_id=state,
                geometry_sha256=hashlib.sha256(name.encode()).hexdigest(),state_randomized=False,metrics=metrics))
    return dict(run=run,update=update,split=split,checkpoint_sha256=checkpoint,episodes=episodes)


def baseline(split='selection'):
    rows = [deepcopy(row) for row in batch(split=split)['episodes'] if row['cell']=='flat']
    for row in rows: row['checkpoint_sha256'] = CHECKPOINT_SHA256
    return rows


def cell_rows(value,cell):
    return [row for row in value['episodes'] if row['cell']==cell]


def evaluate(value):
    return evaluate_batch(value,protocol(),baseline_flat=baseline())


def test_protocol_preserves_registered_seeds_and_all_six_families():
    from scripts.tools.source_training_preflight import validation_cells
    cells = registered_validation_cells()
    assert cells == validation_cells() and len(cells)==15
    assert [cell['subterrain'] for cell in cells[:3]] == ['stairs_ascending','stairs_descending','flat']
    assert cells[0]['selection_geometry_seeds'] == list(range(50000,50005))
    assert cells[0]['confirmation_state_ids'] == list(range(65000,65005))
    assert len({family_of(cell['subterrain']) for cell in cells})==6
    assert proposal()['evaluation_episodes_max']['total']==910
    with pytest.raises(PermissionError): validate_protocol(proposal(),require_locked=True)
    changed = protocol(); changed['quality_gates']['projection_fraction'] = .2
    with pytest.raises(ValueError): validate_protocol(changed)


def test_all_cells_required_and_repeated_geometry_does_not_create_population_interval():
    result = evaluate(batch())
    assert result['passed'] and result['quality_count_sum']==75
    for cell in result['cells'].values():
        assert cell['distinct_local_geometry_hashes']==1
        assert cell['trial_interpretation']=='repeated_trials'
        assert cell['population_success_confidence_interval'] is None


@pytest.mark.parametrize('failures',[1,2])
def test_four_out_of_five_boundary_retains_failed_and_backward_episodes(failures):
    value = batch()
    for row in cell_rows(value,'multi_scale_rough')[:failures]:
        row['metrics'].update(completed=False,forward_progress_m=-.2,stable_hold_steps=0)
    result = evaluate(value)
    assert result['passed'] == (failures==1)
    assert result['cells']['multi_scale_rough']['complete_and_quality_count']==5-failures


@pytest.mark.parametrize('key',['emergency_count','invalid_state_count','nonfoot_contact_termination_count'])
def test_any_hard_safety_failure_disqualifies_node(key):
    value = batch(); value['episodes'][-1]['metrics'][key] = 1
    assert not evaluate(value)['passed']


@pytest.mark.parametrize('fault',['missing_cell','duplicate_seed','pooled_run','wrong_split','nan','invalid_completion',
    'wrong_checkpoint','unnormalized','state_randomized','steps','inconsistent_rms'])
def test_missing_or_invalid_episode_evidence_is_rejected(fault):
    value = batch(); row = value['episodes'][0]
    if fault=='missing_cell': value['episodes'].pop()
    if fault=='duplicate_seed': value['episodes'][1] = deepcopy(row)
    if fault=='pooled_run': row['run']=2
    if fault=='wrong_split': row['split']='confirmation'
    if fault=='nan': row['metrics']['velocity_error_mean_mps']=float('nan')
    if fault=='invalid_completion': row['metrics']['forward_progress_m']=1.
    if fault=='wrong_checkpoint': row['checkpoint_sha256']='f'*64
    if fault=='unnormalized': row['schema']='raw'
    if fault=='state_randomized': row['state_randomized']=True
    if fault=='steps': row['metrics']['steps']=2001
    if fault=='inconsistent_rms': row['metrics']['velocity_error_mean_mps']=.15
    with pytest.raises(ValueError): evaluate(value)


@pytest.mark.parametrize('fault',['stair_index','stairs_stuck','stairs_collision','slope_direction','slope_transition',
    'rough_scales','rough_audit','friction_material','friction_layout','friction_region_slip','friction_missing_region'])
def test_family_specific_evidence_and_quality_rules(fault):
    value = batch()
    if fault.startswith('stair'):
        rows = cell_rows(value,'stairs_ascending')
        if fault=='stair_index': rows[0]['metrics']['completed_step_index']=2
        if fault=='stairs_stuck': rows[0]['metrics']['stuck_count']=1
        if fault=='stairs_collision':
            for row in rows[:2]: row['metrics']['front_foot_collision_count']=1
    if fault.startswith('slope'):
        for row in cell_rows(value,'slope_uphill'):
            row['metrics']['slope_direction' if fault=='slope_direction' else 'transition_completed'] = 'downhill' if fault=='slope_direction' else False
    if fault.startswith('rough'):
        row = cell_rows(value,'multi_scale_rough')[0]
        if fault=='rough_scales': row['metrics']['audited_spatial_scales_m']=[.1]
        else: del row['metrics']['audited_rough_peak_valley_m']
    if fault.startswith('friction'):
        for row in cell_rows(value,'friction_left_right_bands'):
            if fault=='friction_material': row['metrics']['material_binding_verified']=False
            if fault=='friction_layout': row['metrics']['region_layout']='front_back_bands'
            if fault=='friction_region_slip': row['metrics']['slip_by_region']['first']=.11
            if fault=='friction_missing_region': row['metrics']['slip_by_region']={'first':.01}
    disqualifies = fault in ('stairs_stuck','stairs_collision','slope_transition','friction_region_slip')
    if disqualifies: assert not evaluate(value)['passed']
    else:
        with pytest.raises(ValueError): evaluate(value)


def test_flat_paired_retention_and_baseline_need_not_complete():
    value = batch(); base = baseline()
    for row in base: row['metrics'].update(completed=False,forward_progress_m=-.1,stable_hold_steps=0)
    assert evaluate_batch(value,protocol(),baseline_flat=base)['passed']
    for row in cell_rows(value,'flat'): row['metrics']['velocity_error_rms_mps']=.16
    assert not evaluate_batch(value,protocol(),baseline_flat=base)['passed']
    base[0]['geometry_seed'] += 100
    with pytest.raises(ValueError): evaluate_batch(value,protocol(),baseline_flat=base)


def test_midpoint_requires_both_nodes_and_ranking_does_not_depend_on_input_order():
    values = [batch(update=node) for node in NODES['1']]
    decision = select_checkpoint(values,protocol(),baseline_flat=baseline())
    assert decision['selected']['update']==90
    assert decision==select_checkpoint(list(reversed(values)),protocol(),baseline_flat=baseline())
    assert select_checkpoint(values[:2],protocol(),baseline_flat=baseline(),schedule_end='midpoint')['selected']['update']==90
    for row in cell_rows(values[0],'multi_scale_rough')[:2]: row['metrics']['completed']=False
    with pytest.raises(ValueError,match='both'): select_checkpoint(values[:2],protocol(),baseline_flat=baseline(),schedule_end='midpoint')
    assert select_checkpoint(values,protocol(),baseline_flat=baseline())['selected']['update']==363


def test_ranking_prefers_worst_cell_then_sum_then_error_before_earlier_update():
    values = [batch(update=node) for node in NODES['1']]
    cell_rows(values[0],'stairs_ascending')[0]['metrics']['completed']=False
    for row in values[1]['episodes']: row['metrics']['velocity_error_rms_mps']=.12
    decision = select_checkpoint(values,protocol(),baseline_flat=baseline())
    assert decision['selected']['update']==728


@pytest.mark.parametrize('fault',['incomplete_schedule','pooled_run','confirmation_feedback','all_fail'])
def test_selection_schedule_and_feedback_boundaries(fault):
    values = [batch(update=node) for node in NODES['1']]
    if fault=='incomplete_schedule': values.pop()
    if fault=='pooled_run': values[0]['run']=2
    if fault=='confirmation_feedback': values[0]['split']='confirmation'
    if fault=='all_fail':
        for value in values:
            for row in cell_rows(value,'multi_scale_rough')[:2]: row['metrics']['completed']=False
        assert select_checkpoint(values,protocol(),baseline_flat=baseline())['status']=='no_eligible_checkpoint'
    else:
        with pytest.raises(ValueError): select_checkpoint(values,protocol(),baseline_flat=baseline())


def test_confirmation_uses_only_selected_checkpoint_and_does_not_reselect():
    values = [batch(update=node) for node in NODES['1']]
    decision = select_checkpoint(values,protocol(),baseline_flat=baseline())
    confirm = batch(split='confirmation')
    assert confirm_checkpoint(confirm,decision,protocol(),baseline_flat=baseline('confirmation'))['status']=='confirmed'
    for row in cell_rows(confirm,'multi_scale_rough')[:2]: row['metrics']['completed']=False
    result = confirm_checkpoint(confirm,decision,protocol(),baseline_flat=baseline('confirmation'))
    assert result['status']=='confirmation_failed' and result['reselection_allowed'] is False
    with pytest.raises(ValueError):
        confirm_checkpoint(batch(update=363,split='confirmation'),decision,protocol(),baseline_flat=baseline('confirmation'))
