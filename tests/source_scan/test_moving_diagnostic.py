import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/tools'))
from moving_cpu_fixture import environment, build_artifacts
from rl_training.source_scan.moving_diagnostic import (
    EventJournal, MovingObserver, check_scan, motion_sample, restore_tensor,
)
from source_parallel_reset_lifecycle import snapshot, check_transition
from source_eight_env_moving_validation import validate


@pytest.mark.parametrize('fault',['nonfinite','foreign_mesh','bad_ids','foreign_tile','stale','lagging_base'])
def test_scan_ownership_and_freshness_fail_closed(fault):
    root=torch.zeros(8,3);root[:,2]=.4
    hits=torch.zeros(8,9,3);ids=torch.zeros(8,9,1,dtype=torch.int16)
    meshes=torch.full((8,1),17,dtype=torch.int64);age=torch.zeros(8)
    if fault=='nonfinite':hits[7,0,0]=float('nan')
    if fault=='foreign_mesh':meshes[7]=18
    if fault=='bad_ids':ids[7,0]=-1
    if fault=='foreign_tile':hits[7,0,1]=2.01
    if fault=='stale':age[7]=.005
    if fault=='lagging_base':hits[7,0,0]=.09
    with pytest.raises((ValueError,RuntimeError,FloatingPointError)):
        check_scan(root,hits,ids,meshes,17,torch.zeros(8,3),age=age,max_age=1e-6,base=True)


def test_prediction_is_episode_feedback_but_actual_envelope_escape_is_integrity_failure():
    pos=torch.zeros(8,17,3);pos[:,:,2]=.4;vel=torch.zeros_like(pos)
    quat=torch.zeros(8,17,4);quat[:,:,0]=1
    pos[0,0,0]=10.20;vel[0,0,0]=10.
    grid=torch.zeros(1,15,3);index=torch.zeros(8,dtype=torch.long)
    sample=motion_sample(pos,vel,quat,grid,index,index)
    assert sample['outside_prediction'][0] is True
    assert sample['prediction_is_integrity_stop'] is False
    pos[0,0,0]=10.24
    with pytest.raises(RuntimeError,match='actual collision envelope'):
        motion_sample(pos,vel,quat,grid,index,index)
    pos[0,0,0]=0;vel[7,16,0]=float('nan')
    with pytest.raises(FloatingPointError):motion_sample(pos,vel,quat,grid,index,index)


def test_reset_rebases_motion_and_preserves_unaffected_rows(tmp_path):
    env,_,_,wrapped=environment()
    with EventJournal(tmp_path/'events.jsonl') as journal:
        observer=MovingObserver(env,wrapped.runtime,journal,snapshot,check_transition)
        other=env.scene['robot'].data.body_link_pos_w[[1,3,4,5,6]].clone()
        with observer.installed():
            env._reset_idx(torch.tensor([7,0,2]))
            assert torch.equal(env.scene['robot'].data.body_link_pos_w[[1,3,4,5,6]],other)
            env._sim_step_counter+=1
            env.scene.update(.005)
            assert not observer.pending_recovery
    rows=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    motion=next(row for row in rows if row['phase']=='motion_substep')
    assert motion['recovery_ids']==[0,2,7]
    assert restore_tensor(motion['displacements_m']).abs().max()<.001
    assert 'reset_checked' in [row['phase'] for row in rows]


def test_reset_mutation_is_recorded_and_hooks_restore_after_failure(tmp_path):
    env,_,_,wrapped=environment()
    original_reset=env._reset_idx;original_update=env.scene.update
    def corrupt(ids):
        original_reset(ids)
        env.scene['robot'].data.joint_vel[7,0]=123.
    env._reset_idx=corrupt
    with EventJournal(tmp_path/'events.jsonl') as journal:
        observer=MovingObserver(env,wrapped.runtime,journal,snapshot,check_transition)
        with pytest.raises(RuntimeError,match='unaffected'):
            with observer.installed():env._reset_idx(torch.tensor([0]))
    assert env._reset_idx is corrupt and env.scene.update is original_update
    rows=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert rows[-1]['phase']=='reset_boundary'
    assert rows[-1]['after']['joint_vel']['values'][7][0]==123.
    assert not any(row['phase']=='reset_checked' for row in rows)


def test_motion_fault_restores_instance_hooks_without_extra_physics(tmp_path):
    env,interface,_,wrapped=environment()
    original_reset,original_update=env._reset_idx,env.scene.update
    with EventJournal(tmp_path/'events.jsonl') as journal:
        observer=MovingObserver(env,wrapped.runtime,journal,snapshot,check_transition)
        with pytest.raises(FloatingPointError):
            with observer.installed():
                env.scene['robot'].data.body_link_pos_w[7,16,0]=float('nan')
                env._sim_step_counter+=1
                env.scene.update(.005)
    assert env._reset_idx is original_reset and env.scene.update is original_update
    assert interface.simulations==0
    failure=json.loads((tmp_path/'events.jsonl').read_text().splitlines()[-1])
    assert failure['phase']=='motion_error'
    assert failure['body_positions_w']['values'][7][16][0]=='NaN'


def test_incorrect_selected_spawn_pose_is_rejected_before_recovery_step(tmp_path):
    env,_,_,wrapped=environment()
    original=env._reset_idx
    def wrong_pose(ids):
        original(ids)
        env.scene['robot'].data.body_link_pos_w[ids,0,0]+=.1
    env._reset_idx=wrong_pose
    with EventJournal(tmp_path/'events.jsonl') as journal:
        observer=MovingObserver(env,wrapped.runtime,journal,snapshot,check_transition)
        with pytest.raises(RuntimeError,match='spawn origin or root pose'):
            with observer.installed():env._reset_idx(torch.tensor([0]))
    rows=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert rows[-1]['phase']=='reset_boundary'


def test_event_byte_budget_preserves_completed_records_and_refuses_overwrite(tmp_path):
    path=tmp_path/'events.jsonl'
    with pytest.raises(RuntimeError,match='ceiling'):
        with EventJournal(path,byte_limit=40) as journal:
            journal.record({'phase':'good'});journal.flush()
            journal.record({'large':'x'*40})
    assert [json.loads(line) for line in path.read_text().splitlines()]==[{'phase':'good'}]
    with pytest.raises(FileExistsError):EventJournal(path)


@pytest.fixture(scope='module')
def complete_cpu_artifacts(tmp_path_factory):
    folder=tmp_path_factory.mktemp('eight_env_synthetic_cpu')
    output,events=build_artifacts(folder)
    return output,events


def test_complete_pinned_step_contact_motion_reset_and_lossless_archive_acceptance(complete_cpu_artifacts):
    output,events=complete_cpu_artifacts
    result=validate(output,returncode=0,attempt_id='cpu-eight',revision='a'*40,checkpoint_sha256='b'*64)
    assert result['accepted'],result
    assert result['ordinary_failure_episodes']==1
    assert result['total_episode_endings']==9 and result['event_review']['reset_batches']==4
    assert result['event_review']['motion_samples']==result['event_review']['contact_samples']==384
    assert result['event_review']['pending_final_recovery_ids']==[]
    assert events[:9]==['action','write','simulate','fetch','tensor_net','tensor_matrix','tensor_data','scene_update','motion']
    assert Path(str(output.with_suffix('.events.jsonl'))+'.gz').exists()
    with (output.parent/'cpu_acceptance.json').open('x') as stream:json.dump(result,stream,indent=2)


@pytest.mark.parametrize('fault',['incomplete','missing_terminal','wrong_counters','wrong_columns','wrong_probe',
                                 'bad_timeout_mask','bad_exposure','nonzero_exit','wrong_identity','missing_events'])
def test_parent_refuses_bad_controls_without_promoting_policy_failures(complete_cpu_artifacts,tmp_path,fault):
    output,_=complete_cpu_artifacts
    report=json.loads(output.read_text());rows=[json.loads(line) for line in output.with_suffix('.steps.jsonl').read_text().splitlines()]
    if fault=='incomplete':rows.pop()
    if fault=='missing_terminal':rows[12]['terminal']=None
    if fault=='wrong_counters':rows[12]['episode_counters'][3]=13
    if fault=='wrong_columns':rows[2]['terrain_columns_after'][7]=14
    if fault=='wrong_probe':rows[23]['forced_ids']=[]
    if fault=='bad_timeout_mask':rows[12]['time_outs'][3]=True
    if fault=='bad_exposure':report['exposure']['flat']['transitions']+=1
    target=tmp_path/'result.json'
    target.with_suffix('.steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report['steps_sha256']=hashlib.sha256(target.with_suffix('.steps.jsonl').read_bytes()).hexdigest()
    target.write_text(json.dumps(report))
    target.with_suffix('.cleanup.json').write_bytes(output.with_suffix('.cleanup.json').read_bytes())
    if fault!='missing_events':
        Path(str(target.with_suffix('.events.jsonl'))+'.gz').symlink_to(Path(str(output.with_suffix('.events.jsonl'))+'.gz'))
    result=validate(target,returncode=1 if fault=='nonzero_exit' else 0,
                    attempt_id='wrong' if fault=='wrong_identity' else 'cpu-eight',revision='a'*40,checkpoint_sha256='b'*64)
    assert not result['accepted'] and result['reasons'],fault


def test_eight_env_supervisor_rejects_unapproved_or_increased_budget_before_gpu_calls(tmp_path,monkeypatch):
    import source_eight_env_moving_supervisor as module
    manifest=tmp_path/'plan.json';manifest.write_text(json.dumps({'user_approved':False}))
    def forbidden(*args,**kwargs):raise AssertionError('CPU preparation must not query GPU or launch runtime')
    monkeypatch.setattr(module.subprocess,'Popen',forbidden)
    monkeypatch.setattr(module.subprocess,'check_output',forbidden)
    with pytest.raises(ValueError,match='unapproved'):module.run(NS(manifest=manifest))
    assert module.startup_expired('not ready',120.1)
    assert not module.startup_expired('[eight-env-probe] initial-ready',121)
    assert not module.startup_expired('not ready',119.9)


def test_actual_eight_env_entrypoint_keeps_failure_before_immediate_fake_close(tmp_path):
    code=r'''
import builtins,os,runpy,sys,types
class Launcher:
    @staticmethod
    def add_app_launcher_args(parser):parser.add_argument('--device',default='cpu')
    def __init__(self,args):self.app=types.SimpleNamespace(close=lambda:os._exit(0))
sys.modules['isaaclab']=types.ModuleType('isaaclab')
sys.modules['isaaclab.app']=types.SimpleNamespace(AppLauncher=Launcher)
old=builtins.__import__
def injected(name,*args,**kwargs):
    if name.startswith('carb'):raise RuntimeError('synthetic CPU startup failure')
    return old(name,*args,**kwargs)
builtins.__import__=injected
sys.argv=[sys.argv[1],'--output',sys.argv[2],'--checkpoint','unused','--attempt-id','cpu-eight']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    output=tmp_path/'result.json'
    process=subprocess.run([sys.executable,'-c',code,str(ROOT/'scripts/tools/source_eight_env_moving_probe.py'),str(output)],
        capture_output=True,text=True,timeout=30,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    assert process.returncode==0
    report=json.loads(output.read_text())
    assert report['status']=='failed' and 'synthetic CPU startup failure' in report['error']
    assert report['schema']=='lite3_source_eight_env_motion_v1'


def test_supervisor_pins_installed_dependency_bytes_without_mutating_them():
    import source_eight_env_moving_supervisor as module
    old=json.loads((ROOT/'docs/lite3-source-shared-bank-g2-8env-recheck-20261029/manifest.json').read_text())
    dependencies=dict(old['dependency_file_sha256'])
    app=Path('/home/tjx/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py')
    dependencies[str(app)]=hashlib.sha256(app.read_bytes()).hexdigest()
    module.check_dependencies({'dependency_sha256':dependencies})
    changed=dict(dependencies);changed[str(app)]='0'*64
    with pytest.raises(ValueError,match='changed runtime dependency'):
        module.check_dependencies({'dependency_sha256':changed})
    with pytest.raises(ValueError,match='must lock'):
        module.check_dependencies({'dependency_sha256':{}})
