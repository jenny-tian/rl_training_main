"""Child of the selected 16/32-env G1 supervisor; no Isaac before authorization."""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import resource
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,choices=(16,32),default=16)
    args=parser.parse_args()
    from source_scale_startup_supervisor import preflight
    plan=preflight(args,child=True)
    num_envs=plan['num_envs']
    from rl_training.source_scan.smoke_artifacts import SmokeArtifactWriter,finalize_and_close
    from rl_training.source_scan.moving_diagnostic import EventJournal,tensor_record
    report=dict(plan,status='starting',simulation_control_steps=0,base_revision=plan['base_revision'])
    folder=Path(plan['output_dir']);writer=SmokeArtifactWriter(folder/'result.json',report,schema=plan['schema'])
    env=base=app=None;started=time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        app_started=time.monotonic();app=AppLauncher(headless=True,device='cuda:0').app
        report['application_start_s']=time.monotonic()-app_started
        import carb.settings
        import gymnasium as gym
        import omni.usd
        import torch
        import rl_training.tasks  # noqa: F401
        from omni.physx import get_physx_interface,get_physx_simulation_interface
        from pxr import Usd,UsdPhysics,UsdUtils,PhysxSchema
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rl_training.source_scan.training_env import SourceTrainingVecEnv
        from rl_training.source_scan.scale_diagnostic import ScaleObserver
        from rl_training.source_scan.episode_metrics import native_metric_rows
        from rl_training.source_scan.shared_terrain_bank import audit_shared_collision_groups
        from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_training_env_cfg import DeeproboticsLite3SourceTrainingEnvCfg
        from source_parallel_reset_lifecycle import snapshot,check_transition
        from source_shared_contact_report import ContactRuntime,capture_substeps
        from source_sparse_tensor_contact import SparseTensorContactCapture
        cfg=DeeproboticsLite3SourceTrainingEnvCfg(source_training_run=1,source_parallel_sampler_enabled=True,
            source_parallel_sampler_num_envs=num_envs,source_shared_bank_rows=1)
        cfg.sim.device='cuda:0';scene_started=time.monotonic()
        base=gym.make('SourceTerrain-Deeprobotics-Lite3-v0',cfg=cfg)
        env=SourceTrainingVecEnv(RslRlVecEnvWrapper(base,clip_actions=None));base=env.unwrapped
        report['scene_build_s']=time.monotonic()-scene_started
        robot=base.scene['robot'];bodies=[]
        for path in base.scene.env_prim_paths:
            found=[p for p in Usd.PrimRange(base.sim.stage.GetPrimAtPath(path+'/Robot')) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(found)!=17 or any(not p.HasAPI(PhysxSchema.PhysxContactReportAPI) or
                PhysxSchema.PhysxContactReportAPI(p).GetThresholdAttr().Get()!=0. for p in found):raise ValueError('all rigid bodies require zero-threshold reporting')
            bodies.extend(str(p.GetPath()) for p in found)
        ground=[str(p.GetPath()) for p in Usd.PrimRange(base.sim.stage.GetPrimAtPath('/World/ground')) if
            p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        interface=get_physx_simulation_interface()
        runtime=ContactRuntime(base,stage_id_for=lambda s:UsdUtils.StageCache.Get().GetId(s).ToLongInt(),
            usd_context_stage_id=lambda:omni.usd.get_context().get_stage_id(),
            get_disabled=lambda:carb.settings.get_settings().get('/physics/disableContactProcessing'),
            get_backend_state=lambda:dict(suppress_readback_setting=carb.settings.get_settings().get('/physics/suppressReadback'),
                readback_suppressed=get_physx_interface().is_readback_suppressed(),tensor_device=str(base.sim.physics_sim_view.device)))
        with EventJournal(folder/'events.jsonl',byte_limit=plan['events_byte_limit']) as journal:
            observer=ScaleObserver(base,env.runtime,journal,snapshot,check_transition)
            capture=SparseTensorContactCapture(physics_view=base.sim.physics_sim_view,ground_colliders=ground,num_envs=num_envs,
                physics_dt=.005,fault_path=folder/'contact_fault.npz',require_ground_positive=False,
                stage_id=UsdUtils.StageCache.Get().GetId(base.sim.get_initial_stage()).ToLongInt(),
                reporting_bodies=bodies,record=journal.record,start_step=base._sim_step_counter,decimation=4,control_steps=1)
            if capture.capacity!=plan['contact_capacity']:raise ValueError('G1 contact capacity differs from approved scale')
            report.update(collision_groups=audit_shared_collision_groups(base.scene),body_names=list(robot.body_names),
                initial_assignments=[asdict(t) for t in observer.tickets],start_sim_step=base._sim_step_counter,
                terrain_origins=tensor_record(base.scene.terrain.terrain_origins),
                initial_root_positions_w=tensor_record(robot.data.root_link_pos_w),
                observation_widths={k:v.shape[1] for k,v in env.get_observations().items()})
            journal.record(dict(phase='scale_initial',scanners=observer.scans()))
            report['status']='running';writer.persist(report)
            print('[scale-g1] initial-ready',flush=True)
            first_started=time.monotonic()
            with capture_substeps(base,interface,capture,runtime),observer.installed():
                if not app.is_running():raise RuntimeError('simulator stopped before G1 step')
                raw=torch.zeros((num_envs,12),device='cuda:0')
                with torch.inference_mode():obs,reward,done,extras=env.step(raw)
                capture.check_control(base,1)
                if bool(done.any()) or env.runtime.terminal is not None:raise RuntimeError('unexpected G1 episode ending')
                journal.record(dict(phase='scale_control_scans',sim_step=base._sim_step_counter,scanners=observer.scans()))
                journal.flush()
                writer.append_step(dict(step=1,done=done.cpu().tolist(),time_outs=extras['time_outs'].cpu().tolist(),
                    raw_action=raw.cpu().tolist(),executed_action=env.last_trace.executed_action.cpu().tolist(),
                    projected=env.last_trace.decision.projected.cpu().tolist(),paused=env.last_trace.decision.paused.cpu().tolist(),
                    metric_rows=native_metric_rows(base)))
            report.update(status='passed',simulation_control_steps=1,first_step_s=time.monotonic()-first_started,
                contact_capture=capture.summary(),motion_samples=observer.samples,reset_batches=observer.resets,
                pending_recovery_ids=sorted(observer.pending_recovery))
        report.update(events_sha256=hashlib.sha256((folder/'events.jsonl').read_bytes()).hexdigest(),
            process_RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    except Exception:
        report.update(status='failed',error=traceback.format_exc());traceback.print_exc()
    finally:
        report['elapsed_seconds_before_shutdown']=time.monotonic()-started
        finalize_and_close(writer,report,environment_close=env.close if env is not None else base.close if base is not None else None,
            application_close=app.close if app is not None else None)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
