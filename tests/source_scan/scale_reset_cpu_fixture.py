"""Synthetic G2 state/buffers and observation history, never an Isaac/CUDA launch."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import sys
import time
import traceback

import torch

from scale_cpu_fixture import make_environment,install_tensor_backend,BODY_NAMES,GROUND,ROOT
sys.path.insert(0,str(ROOT/'scripts/tools'))
from rl_training.source_scan.scale_reset import proposal
from rl_training.source_scan.scale_diagnostic import ScaleObserver
from rl_training.source_scan.moving_diagnostic import EventJournal
from rl_training.source_scan.smoke_artifacts import SmokeArtifactWriter,finalize_and_close
from source_parallel_reset_lifecycle import snapshot,check_transition
from source_sparse_tensor_contact import SparseTensorContactCapture
from source_scale_reset_collection import collect


def cpu_plan(output):
    return proposal(revision='a'*40,remote='a'*40,inputs={str(i):'b'*64 for i in range(243)},
        dependencies={str(i):'c'*64 for i in range(51)},budget_reference={'path':'synthetic_CPU_fixture','sha256':'d'*64},output_dir=output)


def environment():
    env,interface,_,wrapped=make_environment(120)
    history=NS(_buffer=torch.full((1,120,12),.25),_num_pushes=torch.ones(120,dtype=torch.long))
    env.observation_manager._group_obs_term_history_buffer={'policy':{'actions':history}}
    original_compute=env.observation_manager.compute;original_reset=env._reset_idx
    def compute(update_history=False):
        result=original_compute(update_history=update_history)
        if update_history:
            history._buffer[0]=env.action_manager.action;history._num_pushes+=1
        return result
    def reset(ids):
        result=original_reset(ids);history._buffer[:,ids]=0;history._num_pushes[ids]=0;return result
    env.observation_manager.compute=compute;env._reset_idx=reset
    from reset_contact_cpu_fixture import install_real_reset_callback
    install_real_reset_callback(env,wrapped)
    return env,interface,wrapped


def build_fixture(folder,*,contact_fault=None,reset_fault=None,allow_failure=False):
    folder=Path(folder);folder.mkdir();plan=cpu_plan(folder)|dict(status='approved',user_approved=True)
    env,interface,wrapped=environment();runtime=install_tensor_backend(env,interface,fault=contact_fault)
    if reset_fault is not None:
        original_reset=env._reset_idx;reset_calls=0
        def reset(ids):
            nonlocal reset_calls
            result=original_reset(ids);reset_calls+=1;reset_fault(env,reset_calls);return result
        env._reset_idx=reset
    report=plan|dict(status='running',simulation_control_steps=0,reset_batches=0,reset_attempts=0,
        motion_samples=0,completed_diagnostic_transitions=0,pending_recovery_ids=[],
        cpu_fixture=True,application_start_s=.1,scene_build_s=.2)
    writer=SmokeArtifactWriter(folder/'result.json',report,schema=plan['schema']);started=time.monotonic()
    try:
        with EventJournal(folder/'events.jsonl',byte_limit=plan['events_byte_limit']) as journal:
            observer=ScaleObserver(env,wrapped.runtime,journal,snapshot,check_transition)
            sensors=sorted(f'/World/envs/env_{i}/Robot/{name}' for i in range(120) for name in BODY_NAMES)
            capture=SparseTensorContactCapture(physics_view=env.sim.physics_sim_view,ground_colliders=[GROUND],num_envs=120,
                physics_dt=.005,fault_path=folder/'contact_fault.npz',require_ground_positive=False,diagnostic_scope='G2_reset_recovery',
                stage_id=73,reporting_bodies=sensors,record=journal.record,start_step=env._sim_step_counter,decimation=4,control_steps=4)
            world='/World/collisions/global_group';groups={world:dict(includes=['/World/ground'],allowed_groups=[world,*[f'/World/collisions/group{i}' for i in range(120)]])}
            groups.update({f'/World/collisions/group{i}':dict(includes=[f'/World/envs/env_{i}'],allowed_groups=[f'/World/collisions/group{i}',world]) for i in range(120)})
            report.update(collision_groups=dict(inverted=True,groups=groups),observation_widths=dict(policy=45,critic=393))
            collect(env,wrapped,observer,capture,journal,writer,report,interface,runtime)
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
        if not allow_failure:raise
    finally:
        report['events_sha256']=hashlib.sha256((folder/'events.jsonl').read_bytes()).hexdigest()
        report['elapsed_seconds_before_shutdown']=time.monotonic()-started
        finalize_and_close(writer,report,environment_close=lambda:None,application_close=lambda:None)
    return plan,dict(returncode=0,process_group_empty=True,timed_out=False,monitor_errors=[])
