"""Child of the approved live-PPO supervisor; importing starts no simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'source/rl_training'), str(ROOT / 'scripts/tools')]
from rl_training.source_scan.smoke_artifacts import SmokeArtifactWriter, finalize_and_close
from rl_training.source_scan.live_ppo_diagnostic import SCHEMA, LivePPOJournal


def check_child_authorization(plan, checkpoint):
    from source_live_ppo_supervisor import CONTRACT, dependencies, _input_hashes
    from rl_training.source_scan.training_budget_ledger import BudgetLedger, canonical_budget_path
    if plan.get('user_approved') is not True or plan.get('status') != 'approved':
        raise PermissionError('separate live PPO launch approval required')
    for key, value in CONTRACT.items():
        if type(plan.get(key)) is not type(value) or plan[key] != value:
            raise ValueError('child contract mismatch: ' + key)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if (plan['base_revision'] != revision or plan['remote_revision_verified'] != revision
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
            or plan['input_sha256'] != _input_hashes() or plan['dependency_sha256'] != dependencies()
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != plan['checkpoint_sha256']):
        raise ValueError('child inputs changed after supervisor approval')
    ledger = BudgetLedger(canonical_budget_path(ROOT))
    rows = [row for row in ledger.read()['entries'] if row['run'] == 1]
    expected = {'attempt_id': plan['attempt_id'], 'run': 1, 'reserved_transitions': 384,
        'state': 'reserved', 'purpose': plan['purpose'], 'base_revision': revision,
        'output_dir': plan['output_dir'], 'automatic_refund': False}
    if len(rows) != 1 or rows[0] != expected or plan['budget_ledger']['path'] != str(ledger.path):
        raise ValueError('child requires the exact durable supervisor budget reservation')
    if json.loads((Path(plan['output_dir'])/'budget_reservation.json').read_text()) != expected:
        raise ValueError('output reservation ticket mismatch')


def motion_components(env, journal, report):
    import carb.settings
    import omni.usd
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from pxr import Usd, UsdPhysics, UsdUtils, PhysxSchema
    from rl_training.source_scan.moving_diagnostic import MovingObserver, tensor_record
    from rl_training.source_scan.shared_terrain_bank import audit_shared_collision_groups
    from source_parallel_reset_lifecycle import snapshot, check_transition
    from source_shared_contact_report import ContactRuntime
    from source_shared_tensor_contact import TensorContactCapture
    base = env.unwrapped
    robot = base.scene['robot']
    guard = base.termination_manager.get_term_cfg('source_terrain_tile_bounds').params
    ids = list(range(17))[guard['asset_cfg'].body_ids] if isinstance(guard['asset_cfg'].body_ids, slice) else list(guard['asset_cfg'].body_ids)
    if (len(robot.body_names) != 17 or sorted(ids) != list(range(17))
            or guard['collision_radius_m'] != .27 or guard['numerical_margin_m'] != .001
            or tuple(guard['size']) != (12., 4.) or guard['flat_length_m'] != 3.):
        raise RuntimeError('live PPO body/tile guard contract changed')
    bodies = []
    for path in base.scene.env_prim_paths:
        found = [prim for prim in Usd.PrimRange(base.sim.stage.GetPrimAtPath(path+'/Robot'))
                 if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(found) != 17 or any(not prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                or PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr().Get() != 0.0 for prim in found):
            raise RuntimeError('contact reports do not cover every body')
        bodies.extend(str(prim.GetPath()) for prim in found)
    ground = [str(prim.GetPath()) for prim in Usd.PrimRange(base.sim.stage.GetPrimAtPath('/World/ground'))
              if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
    runtime = ContactRuntime(base, stage_id_for=lambda stage: UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
        usd_context_stage_id=lambda: omni.usd.get_context().get_stage_id(),
        get_disabled=lambda: carb.settings.get_settings().get('/physics/disableContactProcessing'),
        get_backend_state=lambda: {'suppress_readback_setting': carb.settings.get_settings().get('/physics/suppressReadback'),
            'readback_suppressed': get_physx_interface().is_readback_suppressed(),
            'tensor_device': str(base.sim.physics_sim_view.device)})
    observer = MovingObserver(base, env.runtime, journal, snapshot, check_transition)
    capture = TensorContactCapture(physics_view=base.sim.physics_sim_view, ground_colliders=ground,
        physics_dt=.005, num_envs=8, capacity=65536,
        stage_id=UsdUtils.StageCache.Get().GetId(base.sim.get_initial_stage()).ToLongInt(),
        reporting_bodies=bodies, record=journal.record, start_step=base._sim_step_counter,
        decimation=4, control_steps=48)
    report.update(body_names=list(robot.body_names), collision_groups=audit_shared_collision_groups(base.scene),
        initial_assignments=observer.initial_assignments, start_sim_step=base._sim_step_counter,
        terrain_origins=tensor_record(base.scene.terrain.terrain_origins),
        initial_positions_w=tensor_record(observer.previous[0]), initial_velocities_w=tensor_record(observer.previous[1]))
    return observer, capture, runtime, get_physx_simulation_interface()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    check_child_authorization(plan, args.checkpoint)
    output = Path(plan['output_dir']) / 'result.json'
    report = {**plan, 'status': 'starting', 'diagnostic_only': True, 'synthetic_cpu_fixture': False,
              'simulation_control_steps': 0, 'total_transitions': 0, 'ppo_updates': 0,
              'critic_width': 393, 'data_kind': 'source_simulation'}
    writer = SmokeArtifactWriter(output, report, schema=SCHEMA)
    app = env = base_env = training = None
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        print('[live-ppo] launching simulator', flush=True)
        app = AppLauncher(headless=True, device='cuda:0').app
        import gymnasium as gym
        import torch
        import numpy as np
        import rl_training.tasks  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from rl_training.source_scan.training_env import SourceTrainingVecEnv
        from rl_training.source_scan.source_training import TransitionBudget, load_source_actor_only
        from rl_training.source_scan.training_runner import SourceTrainingRunner
        from rl_training.source_scan.moving_diagnostic import EventJournal, digest_file
        from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_training_env_cfg import DeeproboticsLite3SourceTrainingEnvCfg
        from source_training_preflight import runner_config
        from source_shared_contact_report import capture_substeps
        cfg = DeeproboticsLite3SourceTrainingEnvCfg(source_training_run=1,
            source_parallel_sampler_enabled=True, source_parallel_sampler_num_envs=8, source_shared_bank_rows=1)
        cfg.sim.device = 'cuda:0'
        base_env = gym.make('SourceTerrain-Deeprobotics-Lite3-v0', cfg=cfg)
        env = SourceTrainingVecEnv(RslRlVecEnvWrapper(base_env, clip_actions=None))
        torch.manual_seed(130001); np.random.seed(130001); random.seed(130001)
        config = runner_config()
        config['algorithm']['class_name'] = 'rl_training.source_scan.training_ppo:SourcePPO'
        runner = OnPolicyRunner(env, config, log_dir=None, device='cuda:0')
        report['actor_initialization'] = load_source_actor_only(runner.alg.actor, args.checkpoint)
        training = SourceTrainingRunner(runner, run=1, budget=TransitionBudget(8, 24, 384),
            output_dir=output.parent/'training', checkpoint_sha256=plan['checkpoint_sha256'], checkpoint_nodes=(1, 2))
        events_path = output.with_suffix('.events.jsonl')
        with EventJournal(events_path, byte_limit=plan['events_byte_limit']) as journal:
            motion, capture, runtime, interface = motion_components(env, journal, report)
            def check_step(step):
                capture.check_control(env.unwrapped, step)
                journal.record({'phase': 'control_scans', 'step': step, 'sim_step': env.unwrapped._sim_step_counter,
                    'root_positions_w': env.unwrapped.scene['robot'].data.root_link_pos_w.cpu().tolist(),
                    'origins_w': env.unwrapped.scene.terrain.env_origins.cpu().tolist(),
                    'scanners': motion.scans(('height_scanner', 'height_scanner_base'))})
                journal.flush()
            observer = LivePPOJournal(writer, report, app_running=app.is_running, step_check=check_step)
            report['status'] = 'running'; writer.persist(report)
            print('[live-ppo] initial-ready', flush=True)
            with capture_substeps(env.unwrapped, interface, capture, runtime), motion.installed():
                training.run(observer=observer)
            report.update(contact_capture=capture.summary(), motion_samples=motion.samples,
                          reset_batches=motion.resets, pending_recovery_ids=sorted(motion.pending_recovery))
        report.update(status='passed', events_file=events_path.name, events_sha256=digest_file(events_path))
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if training is not None: report['training_snapshot'] = dict(training.report)
        report['elapsed_seconds_before_shutdown'] = time.monotonic() - started
        closing = env if env is not None else base_env
        finalize_and_close(writer, report, environment_close=closing.close if closing is not None else None,
                           application_close=app.close if app is not None else None)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
