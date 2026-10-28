"""Bounded two-robot shared-bank diagnostic; imports do not launch Isaac Sim."""

from dataclasses import asdict
import json

import torch

from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.shared_terrain_bank import SharedBankTerrainAllocator, audit_shared_collision_groups
from source_parallel_reset_lifecycle import snapshot, check_transition, summarize_tickets, _digest
from source_shared_contact_report import (
    ContactRuntime, capture_substeps, classify_contact,
)
from source_shared_tensor_contact import TensorContactCapture, tensor_summary_passed

SHARED_DIAGNOSTIC_SEED = 130196
SHARED_CONTROL_STEPS = 20
SHARED_RESET_BATCHES = ((0,), (1,), (1, 0), (0,), (1,))


def contact_sensor_snapshot(env):
    """Body-origin height and net force telemetry; neither proves pairwise isolation."""
    robot = env.scene['robot']
    sensor = env.scene.sensors['contact_forces']
    body_ids, names = robot.find_bodies(env.cfg.foot_link_name)
    sensor_names = sensor.body_names
    if len(names) != 4 or any(name not in sensor_names for name in names):
        raise AssertionError('expected four feet in both robot and contact sensor')
    sensor_ids = [sensor_names.index(name) for name in names]
    heights = robot.data.body_pos_w[:, body_ids, 2] - env.scene.terrain.env_origins[:, 2, None]
    forces = sensor.data.net_forces_w
    if (forces.shape != (2, len(sensor_names), 3) or heights.shape != (2, 4)
            or not torch.isfinite(heights).all() or not torch.isfinite(forces).all()):
        raise AssertionError('invalid foot-height or contact-force telemetry')
    return {'foot_names': names, 'foot_body_origin_height_above_tile_m': heights.cpu().tolist(),
            'foot_net_contact_forces_w_N': forces[:, sensor_ids].cpu().tolist(),
            'contact_sensor_body_names': sensor_names, 'net_contact_forces_w_N': forces.cpu().tolist(),
            'collider_clearance_measured': False, 'pairwise_isolation_evidence': False}


def check_scanners(env, *, flat_entry=False):
    """Require one real Warp ground mesh, valid hit indices and finite hits."""
    result = {}
    for name in ("height_scanner", "height_scanner_base"):
        sensor = env.scene.sensors[name]
        data = sensor.data
        ids = sensor._mesh_ids_wp.numpy()
        ground_id = sensor.meshes["/World/ground"].id
        if ids.shape != (2, 1) or not (ids == ground_id).all():
            raise AssertionError(f"{name}: runtime ray targets are not the shared ground mesh")
        if (data.ray_hits_w.ndim != 3 or data.ray_hits_w.shape[0] != 2
                or data.ray_hits_w.shape[1] == 0 or data.ray_hits_w.shape[2] != 3
                or not isinstance(data.ray_mesh_ids, torch.Tensor)
                or data.ray_mesh_ids.shape != (*data.ray_hits_w.shape[:2], 1)):
            raise AssertionError(f"{name}: invalid ray-hit or mesh-ID cache dimensions")
        if not torch.isfinite(data.ray_hits_w).all() or not (data.ray_mesh_ids == 0).all():
            raise AssertionError(f"{name}: missing ground hit or nonfinite ray")
        if flat_entry and not torch.allclose(
            data.ray_hits_w[..., 2], env.scene.terrain.env_origins[:, 2, None].expand_as(data.ray_hits_w[..., 2]),
            atol=1e-4, rtol=0,
        ):
            raise AssertionError(f"{name}: flat entry rays did not hit the ground height")
        result[name] = {"shape": list(data.ray_hits_w.shape), "ground_mesh_id": int(ground_id),
                        "hit_indices": [0], "hits_sha256": _digest(data.ray_hits_w)}
    return result


def check_shared_resets(env, seed, record):
    """Exercise the real _reset_idx with independent schedule replay and snapshots."""
    oracle = SharedBankTerrainAllocator(2, SOURCE_RESET_BLOCK_COUNTS, seed, bank_rows=1)
    tickets = list(oracle.reset([0, 1]))
    if tuple(tickets) != env._source_parallel_terrain_allocator.assignments:
        raise AssertionError("unexpected assignments before shared reset checks")
    history_buffers = []
    for batch, ids in enumerate(SHARED_RESET_BATCHES):
        selected = torch.tensor(sorted(ids), device=env.device, dtype=torch.long)
        before = snapshot(env)
        prior = env._source_parallel_terrain_allocator.assignments
        record({"phase": "before_reset", "batch": batch, "ids": list(ids),
                "assignments": [asdict(t) for t in prior],
                "buffer_sha256": {k: _digest(v) for k, v in before.items()}})
        env._reset_idx(torch.tensor(ids, device=env.device, dtype=torch.long))
        current = env._source_parallel_terrain_allocator.assignments
        actual = tuple(current[i] for i in sorted(ids))
        expected = oracle.reset(ids)
        record({"phase": "assignment_comparison", "batch": batch,
                "actual": [asdict(t) for t in actual], "expected": [asdict(t) for t in expected]})
        after = snapshot(env)
        check_transition(before, after, ids)
        if actual != expected or any(current[i] != prior[i] for i in set(range(2)) - set(ids)):
            raise AssertionError("shared reset assignment or untouched ticket changed")
        terrain = env.scene.terrain
        origins = terrain.terrain_origins[terrain.terrain_levels[selected], terrain.terrain_types[selected]]
        if not torch.equal(origins, terrain.env_origins[selected]):
            raise AssertionError("shared reset origin mismatch")
        robot = env.scene['robot'].data
        if not torch.allclose(robot.root_pos_w[selected], robot.default_root_state[selected, :3] + origins,
                              atol=1e-5, rtol=0):
            raise AssertionError("shared reset root mismatch")
        history_buffers = [k for k in after if k.startswith('history/')]
        record({"phase": "reset_checked", "batch": batch, "ids": list(ids),
                "buffer_sha256": {k: _digest(v) for k, v in after.items()},
                "history_buffers": history_buffers, "untouched_buffers_equal": True})
        tickets.extend(actual)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.0)
        record({"phase": "reset_scanners", "batch": batch, "scanners": check_scanners(env, flat_entry=True)})
        observations = env.observation_manager.compute(update_history=True)
        if not all(torch.isfinite(v).all() for v in observations.values()):
            raise AssertionError("shared reset observations are nonfinite")
    return {**summarize_tickets(tickets), "reset_batch_count": len(SHARED_RESET_BATCHES),
            "direct_reset_count": sum(map(len, SHARED_RESET_BATCHES)), "history_buffers": history_buffers,
            "automatic_reset_tested": False}


def run_shared_diagnostic(env, seed, journal_path, trace):
    """20 zero-action steps at identical spawns, ray occlusion probe, five resets."""
    import carb.settings
    import omni.usd
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdUtils

    if env.num_envs != 2 or seed != SHARED_DIAGNOSTIC_SEED:
        raise ValueError("shared diagnostic requires two environments and seed 130196")
    terrain = env.scene.terrain
    robot = env.scene['robot']
    if tuple(terrain.terrain_origins.shape) != (1, 15, 3):
        raise AssertionError("expected one shared bank row with fifteen columns")
    assignments = env._source_parallel_terrain_allocator.assignments
    expected = SharedBankTerrainAllocator(2, SOURCE_RESET_BLOCK_COUNTS, seed, bank_rows=1).reset([0, 1])
    if assignments != expected or any(t.subterrain != 'flat' for t in assignments):
        raise AssertionError("initial shared assignments must overlap on flat")
    if not torch.equal(terrain.env_origins[0], terrain.env_origins[1]) or not torch.allclose(
        robot.data.root_state_w[0], robot.data.root_state_w[1], atol=1e-5, rtol=0,
    ):
        raise AssertionError("robots do not have identical overlapping initial states")
    if not torch.allclose(robot.data.root_pos_w, robot.data.default_root_state[:, :3] + terrain.env_origins,
                          atol=1e-5, rtol=0):
        raise AssertionError("shared initial roots do not match origins")
    guard = env.termination_manager.get_term_cfg('source_terrain_tile_bounds').params['asset_cfg'].body_ids
    all_ids = list(range(robot.num_bodies))
    guard_ids = all_ids[guard] if isinstance(guard, slice) else list(guard)
    if sorted(guard_ids) != all_ids:
        raise AssertionError("tile guard does not cover all robot bodies")
    reporting_bodies = []
    for path in env.scene.env_prim_paths:
        bodies = [p for p in Usd.PrimRange(env.sim.stage.GetPrimAtPath(path + '/Robot'))
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != robot.num_bodies or any(not p.HasAPI(PhysxSchema.PhysxContactReportAPI) for p in bodies):
            raise AssertionError("contact reporting is not enabled for every runtime rigid body")
        if any(PhysxSchema.PhysxContactReportAPI(p).GetThresholdAttr().Get() != 0.0 for p in bodies):
            raise AssertionError("contact reporting must use a zero force threshold")
        reporting_bodies.extend(str(p.GetPath()) for p in bodies)
    ground_colliders = [str(p.GetPath()) for p in Usd.PrimRange(env.sim.stage.GetPrimAtPath('/World/ground'))
                        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    if not ground_colliders:
        raise AssertionError('no enabled static ground colliders for tensor contact filters')

    with journal_path.open('x', encoding='utf-8') as journal:
        def record(value):
            journal.write(json.dumps(value, allow_nan=False) + '\n')
            journal.flush()

        record({"phase": "initial", "collision_groups": audit_shared_collision_groups(env.scene),
                "assignments": [asdict(t) for t in assignments], "reporting_bodies": reporting_bodies,
                "ground_colliders": ground_colliders,
                "roots": robot.data.root_state_w.cpu().tolist(), "scanners": check_scanners(env, flat_entry=True),
                "contact_sensor": contact_sensor_snapshot(env)})

        # At a fixed env-0 pose, moving env-1 vertically through its ray paths
        # must not change env-0 ground hits. No physics integration in this probe.
        baseline = {name: env.scene.sensors[name].data.ray_hits_w[0].clone()
                    for name in ('height_scanner', 'height_scanner_base')}
        pose = robot.data.root_state_w[1:2, :7].clone()
        lifted = pose.clone()
        lifted[:, 2] += 0.8
        ids = torch.tensor([1], device=env.device)
        robot.write_root_pose_to_sim(lifted, env_ids=ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.0)
        for name, expected_hits in baseline.items():
            sensor = env.scene.sensors[name]
            sensor.update(0.0, force_recompute=True)
            if not torch.allclose(sensor.data.ray_hits_w[0], expected_hits, atol=1e-5, rtol=0):
                raise AssertionError("another robot changed stationary ground scans")
        record({"phase": "vertical_occlusion_probe", "lift_m": 0.8, "scanners": check_scanners(env, flat_entry=True)})
        robot.write_root_pose_to_sim(pose, env_ids=ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.0)

        interface = get_physx_simulation_interface()
        runtime = ContactRuntime(
            env, stage_id_for=lambda stage: UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
            usd_context_stage_id=lambda: omni.usd.get_context().get_stage_id(),
            get_disabled=lambda: carb.settings.get_settings().get('/physics/disableContactProcessing'),
            get_backend_state=lambda: {
                'suppress_readback_setting': carb.settings.get_settings().get('/physics/suppressReadback'),
                'readback_suppressed': get_physx_interface().is_readback_suppressed(),
                'tensor_device': str(env.sim.physics_sim_view.device),
            },
        )
        capture = TensorContactCapture(
            stage_id=UsdUtils.StageCache.Get().GetId(env.sim.get_initial_stage()).ToLongInt(),
            reporting_bodies=reporting_bodies, ground_colliders=ground_colliders,
            physics_view=env.sim.physics_sim_view, physics_dt=env.physics_dt, record=record,
            start_step=env._sim_step_counter, decimation=4, control_steps=SHARED_CONTROL_STEPS,
        )
        with capture_substeps(env, interface, capture, runtime):
            for step in range(SHARED_CONTROL_STEPS):
                obs, reward, terminated, truncated, _ = env.step(torch.zeros_like(env.action_manager.action))
                capture.check_control(env, step + 1)
                record({"phase": "control_step", "step": step + 1,
                        "roots": robot.data.root_state_w.cpu().tolist(),
                        "ground_contact_points": dict(capture.ground['poll']),
                        "contact_sensor": contact_sensor_snapshot(env),
                        "terminated": terminated.cpu().tolist(), "truncated": truncated.cpu().tolist()})
                if terminated.any() or truncated.any():
                    raise AssertionError("unexpected termination during shared overlap probe")
                if not all(torch.isfinite(v).all() for v in [*obs.values(), reward, robot.data.root_state_w,
                                                                           robot.data.body_pos_w, robot.data.body_lin_vel_w]):
                    raise AssertionError("nonfinite state during shared overlap probe")
                check_scanners(env)
                trace(f'shared-control-step-checked: {step + 1}')
        ground_contacts = dict(capture.ground['poll'])
        resets = check_shared_resets(env, seed, record)
        record({"phase": "complete", "reset_checks": resets})
    return {"status": "shared_bank_smoke_passed", "num_envs": 2, "bank_rows": 1,
            "terrain_origin_shape": [1, 15, 3], "initial_assignments": [asdict(t) for t in assignments],
            "overlapping_spawn_checked": True, "cross_environment_contacts": 0,
            "ground_contact_points": ground_contacts, "contact_capture": capture.summary(),
            "zero_action_steps": SHARED_CONTROL_STEPS,
            "vertical_occlusion_probe_passed": True, "reset_checks": resets,
            "collision_groups": audit_shared_collision_groups(env.scene),
            "journal": str(journal_path), "policy_loaded": False, "ppo_updates": 0,
            "motion_isolation_qualified": False, "qualification_claim": False}


def shared_report_passed(report):
    """Fail closed if a success file omits any required shared-bank coverage."""
    if not isinstance(report, dict):
        return False
    contacts = report.get('ground_contact_points', {})
    resets = report.get('reset_checks', {})
    if not isinstance(contacts, dict) or not isinstance(resets, dict):
        return False
    return (
        report.get('status') == 'shared_bank_smoke_passed'
        and report.get('num_envs') == 2 and report.get('bank_rows') == 1
        and report.get('zero_action_steps') == SHARED_CONTROL_STEPS
        and report.get('overlapping_spawn_checked') is True
        and report.get('cross_environment_contacts') == 0
        and all(type(contacts.get(str(i), contacts.get(i))) is int
                and contacts.get(str(i), contacts.get(i, 0)) > 0 for i in (0, 1))
        and tensor_summary_passed(report.get('contact_capture'), contacts)
        and report.get('vertical_occlusion_probe_passed') is True
        and resets.get('assignment_count') == 8 and resets.get('reset_batch_count') == 5
        and resets.get('direct_reset_count') == 6
        and report.get('policy_loaded') is False and report.get('ppo_updates') == 0
    )
