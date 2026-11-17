"""Current-memory frozen evaluation; geometry/outcomes match the locked protocol.

Derived from reference_online_collect.py at 8ec25a9. Only the approved safety
memory repair, richer evidence, and the new registration directory differ.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import fields

from reference_lite3_loader import ROOT, load_reference, namespace, resolve_lite3_asset, source_manifest
from reference_stairs_eval import MODELS, load_actor, save, sha
from reference_online_targets import MODEL_PATHS, HASHES, protocol, rows, read

OUTPUT = ROOT / 'logs/reference_six/online_targets_current_20261117'


def collect(args):
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh
    from isaaclab.utils.math import quat_from_euler_xyz
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = protocol()
    winner = read(args.winner) if args.winner else None
    specs = rows(args.kind, args.condition, winner)
    if args.kind == 'smoke':
        for spec in specs:
            if spec['slot'] == 1:
                spec.update(u=[1., 0.], code='smoke_fixed')
    reference, _ = load_reference()
    from reference_six_terrain_cfg import replace_terrain
    from lite3_terrain_reuse.heldout_boxes import HeldoutBoxesTrackCfg
    from lite3_terrain_reuse.rocky_road_track import RockyRoadTrackCfg
    cfg = replace_terrain(reference)
    training = read(MODEL_PATHS[args.model].parent / 'configuration.json')['environment']
    for key in ('actions', 'observations', 'terminations'):
        assert json.loads(json.dumps(cfg.to_dict()[key], default=str)) == training[key], key
    # Reference tasks remain isolated. Import only the project's pure residual helpers.
    package = ROOT / 'source/rl_training/rl_training'
    sys.modules['rl_training'].__path__.append(str(package))
    namespace('rl_training.online_adaptation', package / 'online_adaptation')
    namespace('rl_training.ea_lpl', package / 'ea_lpl')
    from rl_training.online_adaptation.safety import SafetyLimits
    from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter
    from rl_training.ea_lpl.residual_bridge import ResidualSafetyBridge
    from rl_training.ea_lpl.basis import BASIS_NAMES, stacked_bases, residual_request
    from reference_online_interface import (CurrentIsaacAdapter, EpisodeLedger, clip_action,
                                            safety_diagnostics, wrap_manager_compute)

    cfg.seed = p['seeds']['environment']
    cfg.scene.num_envs = len(specs)
    cfg.sim.device = 'cuda:0'
    cfg.episode_length_s = 20.
    resolve_lite3_asset(cfg)
    gen = cfg.scene.terrain.terrain_generator
    conditions = list(dict.fromkeys(s['condition'] for s in specs))
    gen.num_rows, gen.num_cols = 1, len(conditions)
    gen.size = (8., 8.)
    gen.curriculum, gen.use_cache = True, False
    gen.seed, gen.difficulty_range = p['seeds']['environment'], (.5, .5)
    stairs = copy.deepcopy(gen.sub_terrains['stairs_up'])
    subs = {}
    for c in conditions:
        if c.startswith('stairs'):
            sub = copy.deepcopy(stairs)
            sub.step_height_range = (int(c[-2:]) / 100.,) * 2
        elif c == 'boxes':
            sub = HeldoutBoxesTrackCfg(**p['geometry']['boxes'])
        else:
            sub = RockyRoadTrackCfg(**p['geometry']['rocky'])
        sub.proportion = 1. / len(conditions)
        subs[c] = sub
    gen.sub_terrains = subs
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = cfg.curriculum.gait_level = None
    cfg.commands.base_velocity.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    for name in ('randomize_rigid_body_mass', 'randomize_rigid_body_inertia',
                 'randomize_com_positions', 'randomize_actuator_gains'):
        setattr(cfg.events, name, None)
    cfg.events.randomize_rigid_body_material.params.update(static_friction_range=(.9, .9),
        dynamic_friction_range=(.9, .9), restitution_range=(0., 0.), make_consistent=True)
    cfg.events.randomize_reset_base.params.update(
        pose_range={k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')},
        velocity_range={k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')})
    config = json.loads(json.dumps(cfg.to_dict(), default=str))
    for name in ('Hip', 'Knee'):
        assert config['scene']['robot']['actuators'][name] == training['scene']['robot']['actuators'][name]
    save(args.output / 'config.json', dict(configuration=config, specs=specs, model=args.model,
        checkpoint_sha256=HASHES[args.model], reward_scoring='reference_frozen_common_not_selection_metric',
        protocol_sha256=sha(OUTPUT / 'lock.json'), sources=source_manifest(),
        safety_memory=CurrentModelSafetyFilter.SEMANTICS))
    base = ManagerBasedRLEnv(cfg=cfg)
    obs, _ = base.reset()
    device, n, dt = base.device, base.num_envs, base.step_dt
    robot, terrain, sensor = base.scene['robot'], base.scene.terrain, base.scene['contact_forces']
    assert cfg.scene.filter_collisions
    expected_columns = torch.tensor([conditions.index(s['condition']) for s in specs], device=device)
    assert torch.equal(terrain.terrain_types, expected_columns)
    adapter = CurrentIsaacAdapter(base)
    assert obs['policy'].shape == (n, 45) and obs['critic'].shape == (n, 393)
    MODELS[args.model] = MODEL_PATHS[args.model]
    assert sha(MODELS[args.model]) == HASHES[args.model]
    actor = load_actor(args.model).to(device)
    def actor_digest():
        return hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in actor.state_dict().values())).hexdigest()
    original_digest = actor_digest()
    safety = CurrentModelSafetyFilter(SafetyLimits(residual_limits=torch.tensor(p['residual_limits'], device=device)))
    bridge = ResidualSafetyBridge(adapter, safety, action_clip=clip_action)
    timeout = torch.zeros(n, 1, dtype=torch.bool, device=device)
    basis_ids = torch.tensor([BASIS_NAMES.index(s['basis']) for s in specs], device=device)
    codes = torch.tensor([s['u'] for s in specs], device=device)
    request = residual_request(basis_ids, codes, stacked_bases(device=device))
    assert (request.abs() <= torch.tensor(p['residual_limits'], device=device) + 1e-7).all()

    # Ray-audit the actual collider mesh and fixed completion plane before any rollout.
    mesh = RayCaster.meshes[cfg.scene.terrain.prim_path]
    geometries = []
    for col, c in enumerate(conditions):
        origin = terrain.terrain_origins[0, col]
        xs = torch.linspace(0., 3.8 if c.startswith('stairs') else (6.8 if c == 'boxes' else 6.3),
                            3401, device=device)
        starts = origin + torch.stack((xs, torch.zeros_like(xs), torch.full_like(xs, 20.)), -1)
        directions = torch.zeros_like(starts); directions[:, 2] = -1
        hits = raycast_mesh(starts, directions, mesh)[0]
        assert torch.isfinite(hits).all()
        z = (hits[:, 2] - origin[2]).cpu().numpy()
        xx = xs.cpu().numpy()
        fronts, tops = [], []
        if c.startswith('stairs'):
            h = int(c[-2:]) / 100.
            edges = np.flatnonzero(np.diff(z) > h * .5) + 1
            fronts = ((xx[edges] + xx[edges-1]) / 2).tolist()
            tops = z[edges].tolist()
            assert len(fronts) >= 5 and np.allclose(np.diff([0.] + tops), h, atol=.002)
            goal = fronts[-1] + .30
        else:
            goal = p['geometry'][c + '_goal_x']
            assert np.ptp(z) > .01
        geometries.append(dict(condition=c, fronts=fronts, tops=tops, goal_x=goal,
            centerline_min_z=float(z.min()), centerline_max_z=float(z.max())))
    save(args.output / 'geometry.json', geometries)

    ledger = EpisodeLedger([s['slot'] for s in specs], p['namespaces'][args.kind], n)
    offsets = torch.tensor(ledger.reset(range(n)), device=device, dtype=torch.float32)
    roots = robot.data.default_root_state.clone()
    roots[:, :3] += base.scene.env_origins
    roots[:, :2] += offsets[:, :2]
    roots[:, 3:7] = quat_from_euler_xyz(offsets[:, 2]*0, offsets[:, 2]*0, offsets[:, 2])
    roots[:, 7:] = 0
    robot.write_root_state_to_sim(roots)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    # Preserve native 0/1 delay support, paired by reset key across codes and models.
    delay_table = []
    for actuator_index, actuator in enumerate(robot.actuators.values()):
        lag = torch.tensor([int(np.random.default_rng([*ledger.keys[i], actuator_index]).integers(0, 2))
                            for i in range(n)], device=device)
        delay_table.append(lag.cpu().tolist())
        for name in ('positions_delay_buffer', 'velocities_delay_buffer', 'efforts_delay_buffer'):
            buffer = getattr(actuator, name)
            buffer.set_time_lag(lag); buffer.reset()
    base.sim.forward(); base.scene.update(dt)
    command = base.command_manager.get_term('base_velocity')
    tick = -1
    def set_command(next_tick):
        command.is_heading_env[:] = command.is_standing_env[:] = False
        command.vel_command_b[:] = 0
        command.vel_command_b[:, 0] = .4 * min(max((next_tick*dt - .5) / .5, 0.), 1.) if next_tick >= 0 else 0.
    def compute_command(step_dt):
        set_command(tick + 1 if tick >= 0 else -1)
    wrap_manager_compute(base.command_manager, compute_command)
    set_command(-1)
    preparation_steps = 0
    with torch.inference_mode():
        for _ in range(50):
            obs, _, term, trunc, _ = base.step(torch.zeros(n, 12, device=device))
            preparation_steps += 1
            if (term | trunc).any():
                raise RuntimeError('standing preparation terminated')
    base.episode_length_buf[:] = 0
    base.action_manager.reset()
    safety.reset_temporal_state()
    tick = 0; set_command(tick)
    obs = base.observation_manager.compute()
    initial = dict(root=robot.data.root_state_w.cpu().numpy(), joint_pos=robot.data.joint_pos.cpu().numpy(),
        joint_vel=robot.data.joint_vel.cpu().numpy(), observation=obs['policy'].cpu().numpy(),
        origins=base.scene.env_origins.cpu().numpy(), delays=np.asarray(delay_table),
        keys=np.asarray([ledger.keys[i] for i in range(n)]))
    np.savez_compressed(args.output / 'initial.npz', **initial)
    previous = torch.zeros(n, 12, device=device)
    adapter.check_observation(obs['policy'], previous)
    foot_ids = [robot.body_names.index(f'{leg}_FOOT') for leg in ('FL', 'FR', 'HL', 'HR')]
    contact_ids = [sensor.body_names.index(f'{leg}_FOOT') for leg in ('FL', 'FR', 'HL', 'HR')]
    torso = sensor.body_names.index('TORSO')
    active = np.ones(n, bool)
    records = [None] * n
    dwell, maximum = np.zeros(n), np.zeros(n)
    x0 = (robot.data.root_pos_w - base.scene.env_origins)[:, 0].cpu().numpy().copy()
    counts = {key: np.zeros(n, int) for key in ('active_steps', 'projected', 'paused', 'base_projection', 'zero_changed')}
    residual_norm = np.zeros(n)
    learned_norm = np.zeros(n)
    buffers = {key: [] for key in ('root', 'gravity', 'feet', 'action', 'base_action', 'residual_request',
        'applied_torque', 'joint_pos', 'joint_vel', 'base_projection', 'paused', 'projected', 'active',
        'terminated', 'truncated', 'base_correction', 'learned_effect', 'observation')}
    steps = 550 if args.kind == 'smoke' else p['episode_steps']
    trace = None
    base_projection = None
    pre_decision = {}
    auto_resets = 0
    def capture(step_dt):
        data = robot.data
        relative = data.root_pos_w - base.scene.env_origins
        feet = data.body_pos_w[:, foot_ids] - base.scene.env_origins[:, None, :]
        forces = sensor.data.net_forces_w
        quantities = dict(root=relative, gravity=data.projected_gravity_b, feet=feet,
            action=base.action_manager.action, base_action=trace.base_action, residual_request=request,
            applied_torque=data.applied_torque[:, adapter.ids], joint_pos=data.joint_pos[:, adapter.ids],
            joint_vel=data.joint_vel[:, adapter.ids], base_projection=base_projection,
            paused=trace.decision.paused[:, 0], projected=trace.decision.projected[:, 0],
            terminated=base.reset_terminated, truncated=base.reset_time_outs,
            **safety.execution_components(), observation=obs['policy'])
        arrays = {k: v.detach().cpu().numpy().copy() for k, v in quantities.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        assert torch.equal(base.action_manager.action, trace.executed_action)
        for k, v in arrays.items(): buffers[k].append(v)
        for k, v in pre_decision.items(): buffers.setdefault(k, []).append(v)
        buffers['active'].append(active.copy())
        ledger.count_steps(active)
        counts['active_steps'] += active
        for k in ('projected', 'paused', 'base_projection'):
            counts[k] += arrays[k] & active
        counts['zero_changed'] += ((arrays['action'] != arrays['base_action']).any(-1)
            & (arrays['residual_request'] == 0).all(-1) & active)
        residual_norm[:] += np.linalg.norm(arrays['action'] - arrays['base_action'], axis=-1) * active
        learned_norm[:] += np.linalg.norm(arrays['learned_effect'], axis=-1) * active
        ground = base.scene['height_scanner_base'].data.ray_hits_w[:, :, 2].amax(-1)
        clearance = (data.root_pos_w[:, 2] - ground).cpu().numpy()
        supports = (forces[:, contact_ids, 2] > 1.).sum(-1).cpu().numpy()
        torso_contact = (forces[:, torso].norm(dim=-1) > 1.).cpu().numpy()
        trunc = base.reset_time_outs.cpu().numpy()
        for i in np.flatnonzero(active):
            spec = specs[i]; geom = geometries[conditions.index(spec['condition'])]
            root = arrays['root'][i]
            tilt = float(np.degrees(np.arccos(np.clip(-arrays['gravity'][i, 2], -1., 1.))))
            maximum[i] = max(maximum[i], float(root[0] - x0[i]))
            clear = root[0] > geom['goal_x'] and .18 < clearance[i] < .55 and tilt < 44.
            if geom['fronts']:
                clear &= (arrays['feet'][i, :, 0].min() > geom['fronts'][-1] + .05
                          and arrays['feet'][i, :, 2].min() > geom['tops'][-1] - .01)
            clear &= supports[i] >= 2 and not torso_contact[i]
            dwell[i] = dwell[i] + dt if clear else 0.
            outcome = None
            if arrays['terminated'][i]: outcome = 'posture_failure'
            elif abs(root[1]) > 1.: outcome = 'lateral_exit'
            elif dwell[i] >= .26 - 1e-8: outcome = 'success'
            elif trunc[i] or tick == steps - 1:
                recent = max(0, len(buffers['root']) - 251)
                outcome = 'stuck' if root[0] - buffers['root'][recent][i, 0] < .15 else 'timeout'
            if outcome:
                records[i] = dict(**spec, outcome=outcome, duration_s=(tick+1)*dt,
                    final_root=root.tolist(), progress_fraction=float(min(1., maximum[i] / (geom['goal_x'] - x0[i]))),
                    max_forward_m=float(maximum[i]), reset_key=list(ledger.keys[i]))
        active[:] = [r is None for r in records]
    wrap_manager_compute(base.reward_manager, capture)
    with torch.inference_mode():
        for tick in range(steps):
            # A bridge exception must not label the preceding decision as the
            # failing one in the traceback capture.
            trace = None
            adapter.check_observation(obs['policy'], previous)
            trace = bridge.step(actor(obs['policy']), request, timeout)
            base_projection, _ = safety_diagnostics(trace)
            # SafetyState contains simulator views: copy BEFORE base.step.
            pre_decision = {f'state.{f.name}': getattr(trace.safety_state, f.name).detach().cpu().numpy().copy()
                            for f in fields(trace.safety_state)}
            pre_decision.update({f'memory.{k}': v.cpu().numpy().copy()
                                 for k, v in trace.filter_state_before.items()})
            obs, _, term, trunc, _ = base.step(trace.executed_action)
            assert np.array_equal(term.cpu().numpy(), buffers['terminated'][-1])
            assert np.array_equal(trunc.cpu().numpy(), buffers['truncated'][-1])
            done = term | trunc
            # Native automatic reset produces a zero last action; reset only those filter rows.
            previous = trace.executed_action.clone()
            previous[done] = 0
            if done.any():
                ids = torch.where(done)[0]
                safety.reset_temporal_state(ids)
                auto_resets += len(ids)
            if not active.any(): break
            if (tick + 1) % 250 == 0:
                print(f'CURRENT INTERFACE {args.model} {args.kind}: {tick+1}/{steps}, active={active.sum()}', flush=True)
    assert all(r is not None for r in records)
    for i, record in enumerate(records):
        record['safety_counts'] = {k: int(v[i]) for k, v in counts.items()}
        record['mean_executed_delta_norm'] = float(residual_norm[i] / max(1, counts['active_steps'][i]))
        record['mean_learned_effect_norm'] = float(learned_norm[i] / max(1, counts['active_steps'][i]))
    assert actor_digest() == original_digest and sha(MODELS[args.model]) == HASHES[args.model]
    np.savez_compressed(args.output / 'trace.npz', **{k: np.asarray(v) for k, v in buffers.items()})
    report = dict(status='passed', kind=args.kind, model=args.model, episodes=n, control_steps=tick+1,
        preparation_steps=preparation_steps, scored_transitions=ledger.steps, automatic_reset_events=auto_resets,
        first_episodes_only=True, actor_unchanged=True, optimizer_updates=0,
        checkpoint_sha256=HASHES[args.model], actor_digest=original_digest, records=records,
        interface=dict(current_observation_checked_every_step=True, zero_feasible_action_exact=True,
            safety_memory=CurrentModelSafetyFilter.SEMANTICS, pre_reset_flags_exact=True,
            safety_snapshots_before_step=True, base_and_learned_components_separate=True,
            reset_filter_per_environment=True, action_clip=[-100, 100], dt=dt,
            base_projection_steps=int(counts['base_projection'].sum()),
            zero_changed_steps=int(counts['zero_changed'].sum())),
        caveats=['Nominal domain with paired native 0/1 actuator delays.',
            'Native automatic resets after scored first episodes are unscored and reported separately.',
            'Boxes generator has a continuous base at z=0; negative box draws are buried, verified via ray heights.',
            'Safety joint projection can modify even a zero residual and is separately recorded.'])
    save(args.output / 'report.json', report)
    print(json.dumps({c: {o: sum(r['condition'] == c and r['outcome'] == o for r in records)
        for o in ('success', 'posture_failure', 'lateral_exit', 'stuck', 'timeout')} for c in conditions}), flush=True)


