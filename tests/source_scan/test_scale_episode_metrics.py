from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from scale_cpu_fixture import make_environment
from rl_training.source_scan.episode_metrics import (
    geometry_digest,material_binding_audit,native_metric_rows,MetricEpisode,PreResetMetricCapture)
from rl_training.source_scan.source_selection import registered_validation_cells,family_of,proposal,evaluate_batch
from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256


def descriptor(cell):
    digest=geometry_digest([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],parameters={'cell':cell},materials=[[.8,.8,0]])
    return dict(cell=cell,geometry_sha256=digest,riser_progress_m=[2,2.5,3],
        tread_intervals_m=[[2,2.5],[2.5,3],[3,3.5]],slope_angle_deg=5.,exit_transition_progress_m=8.5,
        audited_rough_peak_valley_m=.01,audited_spatial_scales_m=[.1,.5],
        material_binding_audit={'status':'verified_direct_USD_binding','records':[{'cpu_fixture':True}]},
        regions=[{'id':'left','bounds_reset_xy_m':[-1.5,10.5,-2,0]},
                 {'id':'right','bounds_reset_xy_m':[-1.5,10.5,0,2]}])


def identity(cell,seed=50000,state=60000):
    return dict(cell=cell,geometry_sha256=descriptor(cell)['geometry_sha256'],run=1,update=90,
        split='selection',checkpoint_sha256='a'*64,geometry_seed=seed,state_id=state,state_randomized=False)


def row(step=1,**changes):
    value=dict(step=step,forward_progress_m=9.1,lateral_m=.01,velocity_error_mps=.1,
        forward_velocity_mps=.3,roll_rad=.02,pitch_rad=-.03,
        foot_speeds_mps=[.1]*4,foot_contact=[True]*4,foot_horizontal_forces_N=[0.]*4,
        foot_positions_reset_xy_m=[[2.,-.5],[2.,.5],[2.,-.5],[2.,.5]])
    return value|changes


@pytest.mark.parametrize('cell',[c['subterrain'] for c in registered_validation_cells()])
def test_all_fifteen_cells_emit_normalized_complete_episode(cell):
    episode=MetricEpisode(identity(cell),descriptor(cell))
    for step in range(1,51):episode.add(row(step),stable=True)
    record=episode.finish(completed=True)
    assert record['metrics']['completed'] and record['metrics']['steps']==50
    assert record['metrics']['velocity_error_rms_mps']==pytest.approx(.1)
    assert record['metrics']['slip_contact_fraction']==0 and record['metrics']['stable_hold_steps']==50
    assert episode.finish(completed=True)==record
    with pytest.raises(RuntimeError):episode.add(row(51),stable=True)
    with pytest.raises(ValueError):episode.finish(completed=False)


def test_generated_records_feed_the_existing_all_cell_selection_evaluator():
    episodes=[]
    for cell in registered_validation_cells():
        name=cell['subterrain']
        for seed,state in zip(cell['selection_geometry_seeds'],cell['selection_state_ids']):
            episode=MetricEpisode(identity(name,seed,state),descriptor(name))
            for step in range(1,51):episode.add(row(step),stable=True)
            episodes.append(episode.finish(completed=True))
    base=[deepcopy(r) for r in episodes if r['cell']=='flat']
    for record in base:record['checkpoint_sha256']=CHECKPOINT_SHA256
    result=evaluate_batch(dict(run=1,update=90,checkpoint_sha256='a'*64,split='selection',episodes=episodes),proposal(),baseline_flat=base)
    assert result['passed'] and result['quality_count_sum']==75


def test_geometry_identity_keeps_material_assignment_and_ignores_mesh_order():
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    faces=np.array([[0,1,2],[1,3,2]]);materials=np.array([[.7,.7,0],[1,1,0]])
    digest=geometry_digest(vertices,faces,parameters={},materials=materials)
    perm=[2,0,3,1];inverse=np.argsort(perm)
    assert geometry_digest(vertices[perm],inverse[faces[::-1]],parameters={},materials=materials[::-1])==digest
    assert geometry_digest(vertices,faces,parameters={},materials=materials[::-1])!=digest
    with pytest.raises(ValueError):geometry_digest(vertices,faces,parameters={'geometry_seed':100},materials=materials)
    with pytest.raises(ValueError):geometry_digest(vertices,faces,parameters={},materials=[[.7,.7,0]])


@pytest.mark.parametrize('fault',['unbound','nonfinite','valid'])
def test_material_audit_reads_bound_parameters_without_claiming_effective_friction(fault):
    material=NS(GetPath=lambda:'/World/materials/rubber',GetAttribute=lambda name:NS(Get=lambda:float('nan') if fault=='nonfinite' else .8))
    stage=NS(GetPrimAtPath=lambda p:object())
    if fault=='valid':
        result=material_binding_audit(stage,['/World/ground/region'],lambda p:material)
        assert result['records'][0]['dynamicFriction']==.8 and result['effective_contact_friction_measured'] is False
    else:
        with pytest.raises(ValueError):material_binding_audit(stage,['/World/ground/region'],lambda p:None if fault=='unbound' else material)


@pytest.mark.parametrize('fault',['step','nan','contact','missing_foot','bad_geometry','no_contact','false_completion'])
def test_incomplete_or_invalid_metrics_do_not_silently_become_zero(fault):
    geometry=descriptor('stairs_ascending')
    if fault=='bad_geometry':
        geometry['riser_progress_m'][0]=float('nan')
        with pytest.raises(ValueError):MetricEpisode(identity('stairs_ascending'),geometry)
        return
    episode=MetricEpisode(identity('stairs_ascending'),geometry);value=row()
    if fault=='step':value['step']=2
    if fault=='nan':value['roll_rad']=float('nan')
    if fault=='contact':value['foot_contact']=[1]*4
    if fault=='missing_foot':value['foot_speeds_mps'].pop()
    if fault in ('no_contact','false_completion'):
        if fault=='no_contact':value['foot_contact']=[False]*4
        episode.add(value,stable=True)
        with pytest.raises(ValueError):episode.finish(completed=True)
    else:
        with pytest.raises(ValueError):episode.add(value,stable=True)
        assert not episode.rows and episode.contacts==0


def test_stair_rising_edges_dwell_and_per_tread_completion():
    episode=MetricEpisode(identity('stairs_ascending'),descriptor('stairs_ascending'))
    for step in range(1,27):
        forces=[20.,0.,0.,0.] if step in (1,2,4) else [0.]*4
        episode.add(row(step,forward_progress_m=2.1,forward_velocity_mps=0.,foot_horizontal_forces_N=forces),stable=True)
    result=episode.finish(completed=False)['metrics']
    assert result['front_foot_collision_count']==2 and result['stuck_count']==1 and result['completed_step_index']==0


def test_friction_slip_is_counted_per_contact_and_region():
    episode=MetricEpisode(identity('friction_left_right_bands'),descriptor('friction_left_right_bands'))
    episode.add(row(foot_speeds_mps=[.36,0,.36,0],foot_contact=[True,True,False,False]),stable=True,projected=True)
    result=episode.finish(completed=False)['metrics']
    assert result['slip_contact_fraction']==.5 and result['slip_by_region']=={'left':1.,'right':0.}
    assert result['projection_fraction']==1.


def test_native_sampling_maps_feet_by_names_and_preserves_signed_progress():
    env,_,_,_=make_environment()
    env.episode_length_buf[:]=1;env.scene['robot'].data.root_link_pos_w[:,0]-=.1
    env.scene['robot'].data.body_link_lin_vel_w[:,13,0]=.5
    rows=native_metric_rows(env)
    assert len(rows)==16 and rows[0]['forward_progress_m']==pytest.approx(-.1)
    assert rows[0]['foot_speeds_mps'][0]==pytest.approx(.5)
    env.scene['robot'].data.projected_gravity_b[15,0]=float('nan')
    with pytest.raises(FloatingPointError):native_metric_rows(env)


def test_terminal_metrics_are_copied_before_selective_reset_and_hook_restored():
    env,_,_,wrapped=make_environment();capture=PreResetMetricCapture(env)
    original=env._reset_idx
    with capture.installed():
        capture.begin_control();env.episode_length_buf[15]=1999
        _,_,done,_=wrapped.step(torch.zeros(16,12))
        rows=capture.collect_control(done)
        assert rows[15]['step']==2000 and int(env.episode_length_buf[15])==0
        assert rows[0]['step']==1
        with pytest.raises(RuntimeError):capture.collect_control(done)
    assert env._reset_idx==original


def test_missing_terminal_capture_is_refused():
    env,_,_,_=make_environment();capture=PreResetMetricCapture(env);capture.begin_control()
    done=torch.zeros(16,dtype=torch.long);done[15]=1
    with pytest.raises(RuntimeError):capture.collect_control(done)
