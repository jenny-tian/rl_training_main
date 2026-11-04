"""Regress the real G2 callback omission without initializing CUDA or Isaac."""
import json

import pytest
import torch

from scale_reset_cpu_fixture import environment,build_fixture,cpu_plan
from reset_contact_cpu_fixture import FAILURE,contact_sensor
from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
from rl_training.source_scan.scale_diagnostic import ScaleObserver
from rl_training.source_scan.scale_reset import validate_plan
from rl_training.source_scan.moving_diagnostic import EventJournal,restore_tensor
from source_parallel_reset_lifecycle import snapshot,check_transition
from source_scale_reset_validation import validate


def test_original_gpu_snapshots_still_fail_the_strict_boundary_check():
    events=[json.loads(line) for line in (FAILURE/'events.jsonl').read_text().splitlines()]
    before=next(v['before'] for v in events if v['phase']=='scale_reset_attempt')
    after=next(v['after'] for v in events if v['phase']=='scale_reset_after')
    with pytest.raises(AssertionError,match='reset changed untouched buffer: sensor/contact_forces/_is_outdated'):
        check_transition({k:restore_tensor(v) for k,v in before.items()},{k:restore_tensor(v) for k,v in after.items()},[0])


@pytest.mark.parametrize('recorded_cache',[False,True])
def test_real_adapter_pre_reset_callback_uses_a_fresh_comparable_baseline(tmp_path,recorded_cache):
    env,_,wrapped=environment();adapter=wrapped.runtime.adapter
    assert isinstance(adapter,Lite3IsaacLabAdapter)
    assert env._cpu_reset_callback_owner._pre_reset_callbacks==[adapter.capture_terminal_contact,wrapped.runtime.before_reset]
    sensor=env.scene.sensors['contact_forces']
    if recorded_cache:
        events=[json.loads(line) for line in (FAILURE/'events.jsonl').read_text().splitlines()]
        after=next(v['after'] for v in events if v['phase']=='scale_reset_after')
        sensor=contact_sensor(120,net=restore_tensor(after['cache/contact_forces/net_forces_w']),
            history=restore_tensor(after['cache/contact_forces/net_forces_w_history']))
        env.scene.sensors['contact_forces']=adapter.contact_sensor=sensor
    assert sensor._is_outdated.all() and not sensor._data.net_forces_w.any()
    with EventJournal(tmp_path/'events.jsonl') as journal:
        observer=ScaleObserver(env,wrapped.runtime,journal,snapshot,check_transition)
        with observer.installed():observer.direct_resets([[0]])
        assert observer.resets==1
    events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    before=next(v['before'] for v in events if v['phase']=='scale_reset_attempt')
    assert not restore_tensor(before['sensor/contact_forces/_is_outdated']).any()
    assert restore_tensor(before['cache/contact_forces/net_forces_w'])[1:].any()
    assert not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled()


@pytest.mark.parametrize('field',['root','action','sensor_flag','contact_net','contact_history','history','filter','lifecycle'])
def test_priming_keeps_strict_detection_of_real_unselected_changes(tmp_path,field):
    env,_,wrapped=environment();original=env._reset_idx
    def reset(ids):
        original(ids)
        sensor=env.scene.sensors['contact_forces']
        if field=='root':env.scene['robot'].data.root_state_w[119,0]+=1
        if field=='action':env.action_manager.action[119,0]=.1
        if field=='sensor_flag':sensor._is_outdated[119]=True
        if field=='contact_net':sensor._data.net_forces_w[119,0,0]=1
        if field=='contact_history':sensor._data.net_forces_w_history[119,0,0,0]=1
        if field=='history':env.observation_manager._group_obs_term_history_buffer['policy']['actions']._buffer[0,119,0]=1
        if field=='filter':wrapped.runtime.safety._previous_residual[119,0]=.1
        if field=='lifecycle':wrapped.runtime.lifecycle.hold[119]=1
    env._reset_idx=reset
    with EventJournal(tmp_path/'events.jsonl') as journal:
        observer=ScaleObserver(env,wrapped.runtime,journal,snapshot,check_transition)
        if field=='filter':
            wrapped.runtime.safety.temporal_snapshot(env.action_manager.action)
        with pytest.raises((AssertionError,RuntimeError),match='reset changed untouched buffer'):
            with observer.installed():observer.direct_resets([[0]])
        assert observer.resets==0 and observer.reset_attempts==1
    assert env._reset_idx==reset


@pytest.mark.parametrize('failed_batch',[1,2,3])
def test_partial_failure_preserves_actual_counts_and_primary_error(tmp_path,failed_batch):
    def fault(env,batch):
        if batch==failed_batch:env.scene['robot'].data.root_state_w[118 if batch==2 else 119,0]+=1
    folder=tmp_path/'failed';plan,execution=build_fixture(folder,reset_fault=fault,allow_failure=True)
    report=json.loads((folder/'result.json').read_text());completed=failed_batch-1
    assert report['status']=='failed'
    assert report['planned_reset_batches']==plan['planned_reset_batches']
    assert report['reset_attempts']==failed_batch and report['reset_batches']==completed
    assert report['simulation_control_steps']==report['steps_written']==completed
    assert report['completed_diagnostic_transitions']==120*completed
    assert report['motion_samples']==report['contact_capture']['samples']==4*completed
    assert report['ticket_summary']['assignment_count']==120+sum(map(len,plan['planned_reset_batches'][:completed]))
    result=validate(folder,plan,execution)
    assert result['accepted'] is False and result['probe_error']==report['error']
    assert 'reset changed untouched buffer: root' in result['reasons'][0]
    assert 'report mismatch' not in result['reasons'][0]
    assert validate(folder,plan,execution|dict(returncode=17,monitor_errors=['secondary monitor error']))['probe_error']==report['error']
    (folder/'result.cleanup.json').unlink()
    assert validate(folder,plan,execution)['probe_error']==report['error']
    assert not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled()


def test_v2_cannot_reuse_the_v1_approval_or_output_identity(tmp_path):
    old=json.loads((FAILURE/'run_manifest.json').read_text());new=cpu_plan(tmp_path/'v2')
    assert new['schema'].endswith('_v2') and new['attempt_id'].endswith('-v2')
    assert old['attempt_id']!=new['attempt_id'] and old['output_dir']!=new['output_dir']
    assert new['status']=='proposed' and new['user_approved'] is False
    assert 'reset_batches' not in new
    with pytest.raises(ValueError,match='G2 contract mismatch: schema'):
        validate_plan(old,revision=old['base_revision'],inputs=old['input_sha256'],dependencies=old['dependency_sha256'],
            budget_reference=old['budget_ledger'],output_dir=old['output_dir'])
