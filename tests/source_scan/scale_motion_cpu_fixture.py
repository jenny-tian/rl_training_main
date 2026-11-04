"""Short synthetic G3 collection through real adapter and pinned CPU step bodies."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch

from scale_reset_cpu_fixture import environment
from scale_cpu_fixture import ROOT,BODY_NAMES,GROUND,install_tensor_backend
sys.path.insert(0,str(ROOT/'scripts/tools'))
from rl_training.source_scan.scale_motion import proposal
from rl_training.source_scan.scale_motion_evidence import MotionObserver,CompressedJournal,digest_stream,file_digest,MotionArtifactWriter
from rl_training.source_scan.frozen_actor import load_frozen_source_actor
from rl_training.source_scan.moving_diagnostic import EXPECTED_TERMS
from rl_training.source_scan.smoke_artifacts import finalize_and_close
from source_parallel_reset_lifecycle import snapshot,check_transition
from source_scale_motion_collection import collect,MotionContactCapture

CHECKPOINT=Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt')


def cpu_plan(output):
    return proposal(revision='a'*40,remote='a'*40,inputs={str(i):'b'*64 for i in range(249)},
        dependencies={str(i):'c'*64 for i in range(51)},budget_reference=dict(path='synthetic_CPU_fixture',sha256='d'*64),output_dir=output)


def build_fixture(folder,*,contact_fault=None,app_running=lambda:True):
    folder=Path(folder);folder.mkdir();plan=cpu_plan(folder)|dict(status='approved',user_approved=True)
    env,interface,wrapped=environment();runtime=install_tensor_backend(env,interface,fault=contact_fault)
    original_compute=env.termination_manager.compute;original_get=env.termination_manager.get_term
    forced={2:[0],4:[1,17,119],5:[3]};mask=torch.zeros(120,dtype=torch.bool)
    def compute():
        original_compute();mask.zero_();mask[forced.get(env.common_step_counter,[])]=True
        env.termination_manager.terminated|=mask
        return env.termination_manager.terminated|env.termination_manager.time_outs
    env.termination_manager.compute=compute
    env.termination_manager.get_term=lambda name:original_get(name)|mask if name=='illegal_contact' else original_get(name)
    actor=load_frozen_source_actor(CHECKPOINT)
    world='/World/collisions/global_group'
    groups={world:dict(includes=['/World/ground'],allowed_groups=[world,*[f'/World/collisions/group{i}' for i in range(120)]])}
    groups.update({f'/World/collisions/group{i}':dict(includes=[f'/World/envs/env_{i}'],allowed_groups=[f'/World/collisions/group{i}',world]) for i in range(120)})
    report=plan|dict(status='running',synthetic_cpu_fixture=True,body_names=BODY_NAMES,
        termination_terms=sorted(EXPECTED_TERMS),collision_groups=dict(inverted=True,groups=groups))
    writer=MotionArtifactWriter(folder/'result.json',report,schema=plan['schema'])
    try:
        with CompressedJournal(folder/'events.jsonl.gz',byte_limit=plan['events_byte_limit'],
                uncompressed_limit=plan['events_uncompressed_byte_limit'],record_limit=plan['event_record_byte_limit']) as journal:
            observer=MotionObserver(env,wrapped.runtime,journal,snapshot,check_transition)
            capture=MotionContactCapture(physics_view=env.sim.physics_sim_view,ground_colliders=[GROUND],num_envs=120,
                physics_dt=.005,fault_path=folder/'contact_fault.npz',stage_id=73,
                reporting_bodies=[f'/World/envs/env_{i}/Robot/{name}' for i in range(120) for name in BODY_NAMES],
                record=journal.record,start_step=env._sim_step_counter,decimation=4,control_steps=5)
            collect(env,wrapped,actor,observer,capture,journal,writer,report,interface,runtime,cpu_controls=5,app_running=app_running)
    finally:
        report.update(events_sha256=file_digest(folder/'events.jsonl.gz'),events_uncompressed_sha256=digest_stream(folder/'events.jsonl.gz'))
        finalize_and_close(writer,report,environment_close=lambda:None,application_close=lambda:None)
    return plan,dict(returncode=0,process_group_empty=True,timed_out=False,monitor_errors=[])
