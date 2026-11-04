"""Child of the separately approved G3 supervisor; no Isaac before preflight."""
import argparse
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','checkpoint'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--num-envs',type=int,choices=(120,),required=True);args=parser.parse_args()
    from source_scale_motion_supervisor import preflight
    plan=preflight(args,child=True)
    from rl_training.source_scan.smoke_artifacts import finalize_and_close
    from rl_training.source_scan.scale_motion_evidence import CompressedJournal,MotionObserver,digest_stream,file_digest,MotionArtifactWriter
    report=dict(plan,status='starting',simulation_control_steps=0,completed_diagnostic_transitions=0,
        motion_samples=0,reset_attempts=0,reset_batches=0,pending_main_recovery_ids=[],pending_recovery_ids=[])
    folder=Path(plan['output_dir']);writer=MotionArtifactWriter(folder/'result.json',report,schema=plan['schema'])
    env=base=app=None;started=time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        tick=time.monotonic();app=AppLauncher(headless=True,device='cuda:0').app;report['application_start_s']=time.monotonic()-tick
        import carb.settings
        import gymnasium as gym
        import omni.usd
        import torch
        import rl_training.tasks  # noqa: F401
        from omni.physx import get_physx_interface,get_physx_simulation_interface
        from pxr import Usd,UsdPhysics,UsdUtils,PhysxSchema
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rl_training.source_scan.frozen_actor import load_frozen_source_actor
        from rl_training.source_scan.training_env import SourceTrainingVecEnv
        from rl_training.source_scan.shared_terrain_bank import audit_shared_collision_groups
        from rl_training.source_scan.moving_diagnostic import EXPECTED_TERMS
        from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_training_env_cfg import DeeproboticsLite3SourceTrainingEnvCfg
        from source_parallel_reset_lifecycle import snapshot,check_transition
        from source_shared_contact_report import ContactRuntime
        from source_scale_motion_collection import collect,MotionContactCapture
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        actor=load_frozen_source_actor(args.checkpoint,device='cuda:0')
        cfg=DeeproboticsLite3SourceTrainingEnvCfg(source_training_run=1,source_parallel_sampler_enabled=True,
            source_parallel_sampler_num_envs=120,source_shared_bank_rows=1)
        cfg.sim.device='cuda:0'
        ranges=cfg.commands.base_velocity.ranges
        ranges.lin_vel_x=(.3,.3);ranges.lin_vel_y=(0.,0.);ranges.ang_vel_z=(0.,0.)
        tick=time.monotonic();base=gym.make('SourceTerrain-Deeprobotics-Lite3-v0',cfg=cfg)
        env=SourceTrainingVecEnv(RslRlVecEnvWrapper(base,clip_actions=None));base=env.unwrapped
        report['scene_build_s']=time.monotonic()-tick
        robot=base.scene['robot'];guard=base.termination_manager.get_term_cfg('source_terrain_tile_bounds').params
        ids=list(range(17))[guard['asset_cfg'].body_ids] if isinstance(guard['asset_cfg'].body_ids,slice) else list(guard['asset_cfg'].body_ids)
        if (sorted(ids)!=list(range(17)) or len(robot.body_names)!=17 or guard['collision_radius_m']!=.27
                or guard['numerical_margin_m']!=.001 or tuple(guard['size'])!=(12.,4.) or guard['flat_length_m']!=3.
                or set(base.termination_manager.active_terms)!=EXPECTED_TERMS):raise ValueError('G3 guard/body/termination contract differs')
        if plan.get('delivery_stage') and guard.get('scanner_footprint_margin_m')!=.1:
            raise ValueError('delivery G3 requires the scanner footprint boundary guard')
        bodies=[]
        for path in base.scene.env_prim_paths:
            found=[p for p in Usd.PrimRange(base.sim.stage.GetPrimAtPath(path+'/Robot')) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(found)!=17 or any(not p.HasAPI(PhysxSchema.PhysxContactReportAPI) or
                PhysxSchema.PhysxContactReportAPI(p).GetThresholdAttr().Get()!=0. for p in found):raise ValueError('all-body reporting required')
            bodies.extend(str(p.GetPath()) for p in found)
        ground=[str(p.GetPath()) for p in Usd.PrimRange(base.sim.stage.GetPrimAtPath('/World/ground')) if
            p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        interface=get_physx_simulation_interface()
        runtime=ContactRuntime(base,stage_id_for=lambda s:UsdUtils.StageCache.Get().GetId(s).ToLongInt(),
            usd_context_stage_id=lambda:omni.usd.get_context().get_stage_id(),
            get_disabled=lambda:carb.settings.get_settings().get('/physics/disableContactProcessing'),
            get_backend_state=lambda:dict(suppress_readback_setting=carb.settings.get_settings().get('/physics/suppressReadback'),
                readback_suppressed=get_physx_interface().is_readback_suppressed(),tensor_device=str(base.sim.physics_sim_view.device)))
        with CompressedJournal(folder/'events.jsonl.gz',byte_limit=plan['events_byte_limit'],
                uncompressed_limit=plan['events_uncompressed_byte_limit'],record_limit=plan['event_record_byte_limit'],
                binary_tensors=plan.get('events_encoding')=='concatenated_gzip_JSONL_base64_tensors_v1') as journal:
            observer=MotionObserver(base,env.runtime,journal,snapshot,check_transition,scan_guard_enabled=bool(plan.get('delivery_stage')))
            capture=MotionContactCapture(physics_view=base.sim.physics_sim_view,ground_colliders=ground,num_envs=120,
                physics_dt=.005,fault_path=folder/'contact_fault.npz',stage_id=UsdUtils.StageCache.Get().GetId(base.sim.get_initial_stage()).ToLongInt(),
                reporting_bodies=bodies,record=journal.record,start_step=base._sim_step_counter,decimation=4,control_steps=plan['control_steps'])
            report.update(body_names=list(robot.body_names),collision_groups=audit_shared_collision_groups(base.scene),
                termination_terms=sorted(base.termination_manager.active_terms),checkpoint_sha256=plan['checkpoint_sha256'])
            def ready():
                writer.persist(report);print('[scale-g3] initial-ready',flush=True)
            collect(base,env,actor,observer,capture,journal,writer,report,interface,runtime,app_running=app.is_running,on_ready=ready)
    except Exception:
        report.update(status='failed',error=traceback.format_exc());traceback.print_exc()
    finally:
        if (folder/'events.jsonl.gz').is_file():
            report['events_sha256']=file_digest(folder/'events.jsonl.gz')
            try:report['events_uncompressed_sha256']=digest_stream(folder/'events.jsonl.gz')
            except Exception as error:report['event_stream_error']=repr(error)
        report.update(elapsed_seconds_before_shutdown=time.monotonic()-started,
            process_RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        finalize_and_close(writer,report,environment_close=env.close if env is not None else base.close if base is not None else None,
            application_close=app.close if app is not None else None)
    return 0 if report['status']=='collected' else 1


if __name__=='__main__':raise SystemExit(main())
