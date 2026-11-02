"""One bounded eight-environment motion/reset integrity probe; no PPO construction.

No runtime is started on import. Requires separate user authorization for launch.
Ordinary policy failures are individual episode outcomes; integrity failures stop.
"""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/rl_training'))

from rl_training.source_scan.smoke_artifacts import SmokeArtifactWriter, finalize_and_close
from rl_training.source_scan.moving_diagnostic import (
    SCHEMA, CONTROL_STEPS, TIMEOUT_PROBES, EventJournal, MovingObserver, digest_file, tensor_record,
)


def serial_terminal(terminal):
    if terminal is None:
        return None
    result = {key: terminal[key].cpu().tolist() for key in (
        'ids', 'steps', 'hold', 'completed', 'terminated', 'truncated')}
    result['observations'] = {name: value.cpu().tolist() for name, value in terminal['observations'].items()}
    result['terms'] = {name: value.cpu().tolist() for name, value in terminal['terms'].items()}
    return result


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--attempt-id', required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    events_path = args.output.with_suffix('.events.jsonl')
    if args.output.exists() or events_path.exists():
        parser.error('refusing to overwrite an existing attempt')
    report = {'status': 'starting', 'attempt_id': args.attempt_id,
              'base_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'num_envs': 8, 'shared_bank_rows': 1, 'control_steps_limit': CONTROL_STEPS,
              'simulation_control_steps': 0, 'ppo_updates': 0, 'training_ready': False,
              'g3_g4_qualified': False, 'diagnostic_only': True, 'noise_rng_seed': 132001,
              'geometry_seed': 30000, 'sampler_seed': 131001, 'ppo_seed': 130001,
              'reset_state_id': 40000, 'reset_randomized': False, 'stochastic_std': 1.0}
    writer = SmokeArtifactWriter(args.output, report, schema=SCHEMA)
    env = app = None
    started = time.monotonic()
    try:
        print('[eight-env-probe] launching simulator', flush=True)
        app = AppLauncher(args).app
        import carb.settings
        import gymnasium as gym
        import omni.usd
        from omni.physx import get_physx_interface, get_physx_simulation_interface
        from pxr import Usd, UsdPhysics, UsdUtils, PhysxSchema
        import torch
        import rl_training.tasks  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_training_env_cfg import DeeproboticsLite3SourceTrainingEnvCfg
        from rl_training.source_scan.frozen_actor import load_frozen_source_actor
        from rl_training.source_scan.training_env import SourceTrainingVecEnv
        from rl_training.source_scan.shared_terrain_bank import audit_shared_collision_groups
        from source_parallel_reset_lifecycle import snapshot, check_transition
        from source_shared_contact_report import ContactRuntime, capture_substeps
        from source_shared_tensor_contact import TensorContactCapture, contact_capacity_for

        cfg = DeeproboticsLite3SourceTrainingEnvCfg(source_training_run=1,
            source_parallel_sampler_enabled=True, source_parallel_sampler_num_envs=8, source_shared_bank_rows=1)
        cfg.sim.device = args.device
        actor = load_frozen_source_actor(args.checkpoint, device=args.device)
        env = gym.make('SourceTerrain-Deeprobotics-Lite3-v0', cfg=cfg)
        env = SourceTrainingVecEnv(RslRlVecEnvWrapper(env, clip_actions=None))
        base = env.unwrapped
        report.update(critic_width=env.get_observations()['critic'].shape[1],
                      collision_groups=audit_shared_collision_groups(base.scene),
                      rsl_rl=importlib.metadata.version('rsl-rl-lib'))
        robot = base.scene['robot']
        guard = base.termination_manager.get_term_cfg('source_terrain_tile_bounds').params
        body_ids = list(range(17))[guard['asset_cfg'].body_ids] if isinstance(guard['asset_cfg'].body_ids, slice) else list(guard['asset_cfg'].body_ids)
        if (len(robot.body_names) != 17 or sorted(body_ids) != list(range(17))
                or guard['collision_radius_m'] != .27 or guard['numerical_margin_m'] != .001
                or tuple(guard['size']) != (12., 4.) or guard['flat_length_m'] != 3.):
            raise RuntimeError('body mapping or tile guard contract changed')
        report['body_names'] = list(robot.body_names)
        bodies = []
        for path in base.scene.env_prim_paths:
            found = [prim for prim in Usd.PrimRange(base.sim.stage.GetPrimAtPath(path+'/Robot'))
                     if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(found) != 17 or any(not prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                    or PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr().Get() != 0.0 for prim in found):
                raise RuntimeError('contact reporting does not cover every body')
            bodies.extend(str(prim.GetPath()) for prim in found)
        ground = [str(prim.GetPath()) for prim in Usd.PrimRange(base.sim.stage.GetPrimAtPath('/World/ground'))
                  if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
        interface = get_physx_simulation_interface()
        runtime = ContactRuntime(base, stage_id_for=lambda stage: UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
            usd_context_stage_id=lambda: omni.usd.get_context().get_stage_id(),
            get_disabled=lambda: carb.settings.get_settings().get('/physics/disableContactProcessing'),
            get_backend_state=lambda: {'suppress_readback_setting': carb.settings.get_settings().get('/physics/suppressReadback'),
                'readback_suppressed': get_physx_interface().is_readback_suppressed(),
                'tensor_device': str(base.sim.physics_sim_view.device)})
        with EventJournal(events_path) as journal:
            observer = MovingObserver(base, env.runtime, journal, snapshot, check_transition)
            capture = TensorContactCapture(physics_view=base.sim.physics_sim_view, ground_colliders=ground,
                physics_dt=.005, num_envs=8, capacity=contact_capacity_for(8),
                stage_id=UsdUtils.StageCache.Get().GetId(base.sim.get_initial_stage()).ToLongInt(),
                reporting_bodies=bodies, record=journal.record, start_step=base._sim_step_counter,
                decimation=4, control_steps=CONTROL_STEPS)
            report.update(initial_assignments=observer.initial_assignments, start_sim_step=base._sim_step_counter,
                          terrain_origins=tensor_record(base.scene.terrain.terrain_origins),
                          initial_positions_w=tensor_record(observer.previous[0]),
                          initial_velocities_w=tensor_record(observer.previous[1]))
            journal.record({'phase': 'initial', 'initial_assignments': observer.initial_assignments,
                            'body_names': report['body_names'], 'scanners': observer.scans(('height_scanner', 'height_scanner_base'))})
            journal.flush()
            obs = env.get_observations()
            generator = torch.Generator(device=args.device).manual_seed(132001)
            report['status'] = 'running'
            writer.persist(report)
            print('[eight-env-probe] initial-ready', flush=True)
            with capture_substeps(base, interface, capture, runtime), observer.installed():
                for step in range(1, CONTROL_STEPS+1):
                    if not app.is_running():
                        raise RuntimeError('simulator exited before probe completed')
                    forced = TIMEOUT_PROBES.get(step, [])
                    if forced:
                        base.episode_length_buf[forced] = 1999
                    before = base.scene.terrain.terrain_types.clone()
                    with torch.inference_mode():
                        raw = actor(obs) + torch.randn((8, 12), generator=generator, device=args.device)
                        obs, rewards, done, extras = env.step(raw)
                    capture.check_control(base, step)
                    scans = observer.scans(('height_scanner', 'height_scanner_base'))
                    journal.record({'phase': 'control_scans', 'step': step, 'sim_step': base._sim_step_counter,
                                    'root_positions_w': base.scene['robot'].data.root_link_pos_w.cpu().tolist(),
                                    'origins_w': base.scene.terrain.env_origins.cpu().tolist(), 'scanners': scans})
                    journal.flush()
                    terminal = serial_terminal(extras['source_terminal'])
                    writer.append_step({'step': step, 'forced_ids': forced, 'done': done.cpu().tolist(),
                        'time_outs': extras['time_outs'].cpu().tolist(), 'terminal': terminal,
                        'episode_counters': base.episode_length_buf.cpu().tolist(),
                        'terrain_columns_before': before.cpu().tolist(),
                        'terrain_columns_after': base.scene.terrain.terrain_types.cpu().tolist(),
                        'raw_action': raw.cpu().tolist(), 'base_action': env.last_trace.base_action.cpu().tolist(),
                        'requested_action': env.last_trace.requested_action.cpu().tolist(),
                        'executed_action': env.last_trace.executed_action.cpu().tolist(),
                        'projected': env.last_trace.decision.projected.cpu().tolist(),
                        'paused': env.last_trace.decision.paused.cpu().tolist()})
                    # Pure or compound forced timeouts are recorded; ordinary
                    # safety/fall/bounds endings in other rows are learning outcomes.
                    if forced and (terminal is None or any(
                            i not in terminal['ids'] or not terminal['truncated'][i] for i in forced)):
                        raise RuntimeError('forced timeout failed to reset the registered rows')
                    report.update(simulation_control_steps=step, total_transitions=step*8,
                                  exposure=env.exposure.report())
                    writer.persist(report)
            report.update(contact_capture=capture.summary(), motion_samples=observer.samples,
                          reset_batches=observer.resets, pending_recovery_ids=sorted(observer.pending_recovery))
        report.update(status='passed', checkpoint_sha256=digest_file(args.checkpoint),
                      events_file=events_path.name, events_sha256=digest_file(events_path))
        print('[eight-env-probe] assertions complete; parent acceptance pending', flush=True)
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report['elapsed_seconds_before_shutdown'] = time.monotonic()-started
        finalize_and_close(writer, report, environment_close=env.close if env is not None else None,
                           application_close=app.close if app is not None else None)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
