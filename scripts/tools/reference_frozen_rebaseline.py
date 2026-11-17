"""B10200 冻结再评估：可选航向闭环，零更新。方案见 docs/lite3-frozen-rebaseline-20261117/PLAN.md。

采集循环改写自 reference_stair_allocation_retention.py（训练地图取样）和
reference_online_current_collect.py（固定高度台阶、箱体）。基础策略直接输出动作，
不经过在线阶段的残差安全过滤。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
import traceback
import zlib

from reference_stairs_eval import ROOT, save, sha, load_actor, MODELS
from reference_online_targets import MODEL_PATHS, HASHES

OUTPUT = ROOT / 'logs/reference_six/frozen_rebaseline_20261117'
STARTS = 64
MAP_COLUMNS = dict(flat=0, rough=4, slope_up=8, slope_down=10, friction_lateral=16,
                   friction_longitudinal=17, slope_rough_up=18, slope_rough_down=19)
BASE_FIXED = ('stairs_up_05', 'stairs_down_05')
TARGETS = ('stairs_up_06', 'stairs_up_07', 'stairs_up_08', 'boxes_08', 'stairs_down_08', 'slope_rough_up_x13')
# 训练上限：坡度参数 0.4，粗糙幅度 0.06 m。候选取约 1.3 倍。
BEYOND = {'x13': (.52, (.02, .08)), 'x11': (.44, (.01, .07)), 'x12': (.48, (.01, .07))}
DENSE = ('stairs_up_065', 'boxes_05', 'boxes_06', 'slope_rough_up_x11', 'slope_rough_up_x12', 'stairs_down_10')


def height(c):
    d = c.rsplit('_', 1)[1]
    return int(d) / (100. if len(d) == 2 else 1000.)
HEAD_TARGETS = ('stairs_up_06', 'stairs_up_065', 'boxes_06', 'slope_rough_up_x11')
RUNS = {
    'map_nominal_on': dict(layout='map', profile='nominal', heading=True, conditions=tuple(MAP_COLUMNS)),
    'map_randomized_on': dict(layout='map', profile='training_randomized', heading=True, conditions=tuple(MAP_COLUMNS)),
    'fixed_nominal_on': dict(layout='fixed', profile='nominal', heading=True, conditions=BASE_FIXED + TARGETS),
    'fixed_randomized_on': dict(layout='fixed', profile='training_randomized', heading=True, conditions=BASE_FIXED),
    'fixed_nominal_off': dict(layout='fixed', profile='nominal', heading=False, conditions=TARGETS),
    'dense_nominal_on': dict(layout='fixed', profile='nominal', heading=True, conditions=DENSE, namespace=42),
    'dense_map_nominal_on': dict(layout='map', profile='nominal', heading=True, conditions=('slope_rough_down',), namespace=42, row=8),
    'smoke': dict(layout='fixed', profile='nominal', heading=True, conditions=BASE_FIXED + TARGETS, starts=15),
    'headroom_smoke': dict(layout='fixed', profile='nominal', heading=True, conditions=HEAD_TARGETS, namespace=45, residual='grid', starts=2),
    'headroom_grid': dict(layout='fixed', profile='nominal', heading=True, conditions=HEAD_TARGETS, namespace=43, residual='grid', starts=16),
    'headroom_confirm': dict(layout='fixed', profile='nominal', heading=True, conditions=HEAD_TARGETS, namespace=44, residual='confirm', starts=64),
}
GRID_VALUES = (-1.5, 0., 1.5)
BASES = ('sag', 'roll', 'hipx')
RESIDUAL_LIMITS = [.24, .20, .20] * 4


def codes_for(run, condition):
    kind = RUNS[run].get('residual')
    if kind is None:
        return [None]
    if kind == 'grid':
        return [dict(basis=b, u=[u1, u2]) for b in BASES for u1 in GRID_VALUES for u2 in GRID_VALUES]
    import json
    winner = json.loads((OUTPUT / 'winners.json').read_text())[condition]
    return [dict(basis='sag', u=[0., 0.]), dict(basis=winner['basis'], u=winner['u'])]


def namespace_of(condition):
    return 41 if condition in TARGETS else 40


def start_offset(condition, slot, ns=None):
    """起点只由（命名空间、条件名、编号）决定，所以闭环开/关两组逐一配对。"""
    import numpy as np
    rng = np.random.default_rng([ns or namespace_of(condition), zlib.crc32(condition.encode()), slot])
    return rng.uniform([.20, -.10, -.03], [.30, .10, .03]).tolist()


def specs_for(run):
    r = RUNS[run]
    starts = r.get('starts', STARTS)
    out = []
    for condition in r['conditions']:
      for code in codes_for(run, condition):
        for slot in range(starts):
            row = r.get('row', 1 if slot < starts // 2 else 8) if r['layout'] == 'map' else 0
            ns = r.get('namespace', namespace_of(condition))
            out.append(dict(env_id=len(out), condition=condition, slot=slot, row=row,
                            namespace=ns, offset=start_offset(condition, slot, ns), **(code or {})))
    return out


def command_speed(t):
    return .4 * min(max((t - .5) / .5, 0.), 1.)


def feedback(heading, forward):
    import torch
    error = torch.remainder(-heading + torch.pi, 2 * torch.pi) - torch.pi
    return torch.where(forward > 0, (.5 * error).clamp(-.8, .8), torch.zeros_like(error))


def outcome(*, terminated, lateral, reached, support, dwell, expired, last5):
    if terminated: return 'posture_failure'
    if abs(lateral) > 1.: return 'lateral_exit'
    if reached and support >= 2 and dwell >= .25: return 'success'
    if expired: return 'stuck' if last5 < .15 else 'timeout'
    return None


def collect(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0',
        kit_args=f'--/renderer/multiGpu/enabled=False --/renderer/activeGpu={args.gpu}').app
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh
    from isaaclab.utils.math import quat_from_euler_xyz
    from reference_lite3_loader import load_reference, resolve_lite3_asset, source_manifest
    reference, _ = load_reference()
    from reference_six_terrain_cfg import replace_terrain
    from reference_low_stairs_cfg import apply_terrain
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as rewards_module

    run = RUNS[args.run]
    smoke = 'smoke' in args.run
    specs = specs_for(args.run)
    n = len(specs)
    conditions = list(run['conditions'])
    if run['layout'] == 'map':
        cfg, _ = apply_terrain(replace_terrain(reference))
        gen = cfg.scene.terrain.terrain_generator
        gen.seed = 42
        columns = MAP_COLUMNS
    else:
        from lite3_terrain_reuse.heldout_boxes import HeldoutBoxesTrackCfg
        cfg = replace_terrain(reference)
        gen = cfg.scene.terrain.terrain_generator
        source = gen.sub_terrains
        subs = {}
        for c in conditions:
            if c.startswith('stairs'):
                sub = copy.deepcopy(source['stairs_up' if '_up_' in c else 'stairs_down'])
                sub.step_height_range = (height(c),) * 2
            elif c.startswith('boxes'):
                sub = HeldoutBoxesTrackCfg(flat_length=3., lane_width=2., grid_width=.4,
                                           grid_height_range=[height(c)] * 2, seed=101)
            else:
                sub = copy.deepcopy(source['slope_rough_up'])
                assert tuple(sub.slope_range) == (0., .4) and tuple(sub.noise_range) == (.01, .06)
                slope, noise = BEYOND[c.rsplit('_', 1)[1]]
                sub.slope_range = (slope,) * 2
                sub.noise_range = noise
            sub.proportion = 1. / len(conditions)
            subs[c] = sub
        gen.sub_terrains = subs
        gen.num_rows, gen.num_cols = 1, len(conditions)
        gen.size = (8., 8.)
        gen.curriculum = True
        gen.seed, gen.difficulty_range = 20261116, (.5, .5)
        cfg.scene.terrain.max_init_terrain_level = 0
        columns = {c: i for i, c in enumerate(conditions)}
    gen.use_cache = False
    cfg.seed = 20261117
    cfg.scene.num_envs = n
    cfg.sim.device = 'cuda:0'
    cfg.episode_length_s = 20.
    cfg.commands.base_velocity.debug_vis = False
    cfg.curriculum.terrain_levels = cfg.curriculum.gait_level = None
    zero = {k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.events.randomize_reset_base.params['pose_range'] = dict(zero)
    cfg.events.randomize_reset_base.params['velocity_range'] = dict(zero)
    if run['profile'] == 'nominal':
        cfg.observations.policy.enable_corruption = False
        for name in ('randomize_rigid_body_mass', 'randomize_rigid_body_inertia',
                     'randomize_com_positions', 'randomize_actuator_gains'):
            setattr(cfg.events, name, None)
        cfg.events.randomize_rigid_body_material.params.update(static_friction_range=(.9, .9),
            dynamic_friction_range=(.9, .9), restitution_range=(0., 0.), make_consistent=True)
    resolve_lite3_asset(cfg)
    base = ManagerBasedRLEnv(cfg=cfg)
    base.reset()
    device, dt = base.device, base.step_dt
    robot, terrain, sensor = base.scene['robot'], base.scene.terrain, base.scene['contact_forces']
    levels = torch.tensor([s['row'] for s in specs], device=device)
    types = torch.tensor([columns[s['condition']] for s in specs], device=device)
    terrain.terrain_levels.copy_(levels); terrain.terrain_types.copy_(types)
    terrain.env_origins.copy_(terrain.terrain_origins[levels, types])
    origins = terrain.env_origins.clone()

    # 固定布局：射线核对实际碰撞网格，终点取最后一个立面之后 0.30 m。
    goals = {c: 3.25 for c in conditions}
    geometry = []
    if run['layout'] == 'fixed':
        mesh = RayCaster.meshes[cfg.scene.terrain.prim_path]
        for c in conditions:
            origin = terrain.terrain_origins[0, columns[c]]
            xs = torch.linspace(0., 6.8 if c.startswith('boxes') else 3.8, 3401, device=device)
            starts = origin + torch.stack((xs, torch.zeros_like(xs), torch.full_like(xs, 20.)), -1)
            directions = torch.zeros_like(starts); directions[:, 2] = -1
            z = (raycast_mesh(starts, directions, mesh)[0][:, 2] - origin[2]).cpu().numpy()
            assert np.isfinite(z).all()
            xx = xs.cpu().numpy()
            fronts = []
            if c.startswith('stairs'):
                h = height(c)
                edges = np.flatnonzero(np.abs(np.diff(z)) > h * .5) + 1
                fronts = ((xx[edges] + xx[edges - 1]) / 2).tolist()
                steps = np.diff(z)[edges - 1]
                assert len(fronts) >= 5 and np.allclose(np.abs(steps), h, atol=.003), (c, steps)
                assert (steps > 0).all() if '_up_' in c else (steps < 0).all()
                goals[c] = fronts[-1] + .30
            elif c.startswith('boxes'):
                assert np.ptp(z) > .01
                goals[c] = 5.5
            else:
                goals[c] = 3.25
            geometry.append(dict(condition=c, fronts=fronts, goal_x=goals[c],
                z_min=float(z.min()), z_max=float(z.max()), z_at_goal=float(np.interp(goals[c], xx, z))))
    save(args.output / 'geometry.json', geometry)
    goal = np.asarray([goals[s['condition']] for s in specs])

    offsets = torch.tensor([s['offset'] for s in specs], device=device, dtype=torch.float32)
    roots = robot.data.default_root_state.clone()
    roots[:, :3] += origins
    roots[:, :2] += offsets[:, :2]
    roots[:, 7:] = 0
    roots[:, 3:7] = quat_from_euler_xyz(offsets[:, 2] * 0, offsets[:, 2] * 0, offsets[:, 2])
    robot.write_root_state_to_sim(roots)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    base.sim.forward(); base.scene.update(dt)

    command = base.command_manager.get_term('base_velocity')
    tick = -1
    feedback_error = 0.

    def set_command(t):
        nonlocal feedback_error
        command.is_heading_env[:] = command.is_standing_env[:] = False
        command.vel_command_b[:] = 0
        if t >= 0:
            command.vel_command_b[:, 0] = command_speed(t * dt)
            if run['heading']:
                heading = robot.data.heading_w
                wz = feedback(heading, command.vel_command_b[:, 0])
                command.vel_command_b[:, 2] = wz
                # 独立复算一遍公式（numpy，不用 torch 的 remainder），记录最大偏差。
                h = heading.double().cpu().numpy()
                expect = np.clip(.5 * np.arctan2(np.sin(-h), np.cos(-h)), -.8, .8) * (command_speed(t * dt) > 0)
                feedback_error = max(feedback_error, float(np.abs(wz.double().cpu().numpy() - expect).max()))

    native_compute = base.command_manager.compute

    def compute(dt):
        native_compute(dt)
        set_command(tick + 1 if tick >= 0 else -1)

    base.command_manager.compute = compute
    set_command(-1)
    rewards_module.update_gait_level_from_terrain_mean(torch.tensor(5.))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    MODELS['B'] = MODEL_PATHS['B']
    assert sha(MODELS['B']) == HASHES['B']
    actor = load_actor('B').to(device)
    digest = lambda: hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in actor.state_dict().values())).hexdigest()
    before = digest()
    bridge = None
    if run.get('residual'):
        from reference_lite3_loader import namespace
        package = ROOT / 'source/rl_training/rl_training'
        sys.modules['rl_training'].__path__.append(str(package))
        namespace('rl_training.online_adaptation', package / 'online_adaptation')
        namespace('rl_training.ea_lpl', package / 'ea_lpl')
        from rl_training.online_adaptation.safety import SafetyLimits
        from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter
        from rl_training.ea_lpl.residual_bridge import ResidualSafetyBridge
        from rl_training.ea_lpl.basis import BASIS_NAMES, stacked_bases, residual_request
        from reference_online_interface import CurrentIsaacAdapter, clip_action
        safety = CurrentModelSafetyFilter(SafetyLimits(residual_limits=torch.tensor(RESIDUAL_LIMITS, device=device)))
        bridge = ResidualSafetyBridge(CurrentIsaacAdapter(base), safety, action_clip=clip_action)
        request = residual_request(torch.tensor([BASIS_NAMES.index(s['basis']) for s in specs], device=device),
                                   torch.tensor([s['u'] for s in specs], device=device), stacked_bases(device=device))
        no_timeout = torch.zeros(n, 1, dtype=torch.bool, device=device)
    delta_sum = torch.zeros(n, device=device); delta_steps = torch.zeros(n, device=device)
    with torch.inference_mode():
        for _ in range(50):
            obs, _, terminated, truncated, _ = base.step(torch.zeros(n, 12, device=device))
            if (terminated | truncated).any():
                raise RuntimeError('standing preparation terminated')
    base.episode_length_buf[:] = 0
    base.action_manager.reset()
    if bridge: safety.reset_temporal_state()
    tick = 0; set_command(0)
    obs = base.observation_manager.compute()
    start = (robot.data.root_pos_w - origins).cpu().numpy().copy()
    np.savez_compressed(args.output / 'initial.npz', root=robot.data.root_state_w.cpu().numpy(),
        joint_pos=robot.data.joint_pos.cpu().numpy(), joint_vel=robot.data.joint_vel.cpu().numpy(),
        origins=origins.cpu().numpy(), offsets=offsets.cpu().numpy())
    feet_ids, _ = sensor.find_bodies(['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT'], preserve_order=True)
    active = np.ones(n, bool)
    records = [None] * n
    dwell, maximum = np.zeros(n), start[:, 0].copy()
    history = []          # 每步相对地块原点的 x，用于判断最后 5 秒是否停滞
    speed_error = np.zeros(n); speed_steps = np.zeros(n)
    yaw_abs = np.zeros(n)
    total = 550 if smoke else 1000
    native_reward = base.reward_manager.compute
    step = base.step_dt

    def capture(dt=None):
        result = native_reward(dt)
        data = robot.data
        rel = (data.root_pos_w - origins).cpu().numpy()
        vx = data.root_lin_vel_b[:, 0].cpu().numpy()
        heading = data.heading_w.cpu().numpy()
        terminated = base.reset_terminated.cpu().numpy()
        truncated = base.reset_time_outs.cpu().numpy()
        support = (sensor.data.net_forces_w[:, feet_ids, 2] > 1.).sum(-1).cpu().numpy()
        assert np.isfinite(rel).all() and np.isfinite(vx).all()
        history.append(rel[:, 0].copy())
        dt = step
        t = (tick + 1) * dt
        for i in np.flatnonzero(active):
            maximum[i] = max(maximum[i], rel[i, 0])
            yaw_abs[i] = max(yaw_abs[i], abs(np.arctan2(np.sin(heading[i]), np.cos(heading[i]))))
            if t >= 4.:
                speed_error[i] += abs(vx[i] - .4); speed_steps[i] += 1
            reached = rel[i, 0] >= goal[i]
            clear = reached and support[i] >= 2 and abs(rel[i, 1]) <= 1.
            dwell[i] = dwell[i] + dt if clear else 0.
            expired = bool(truncated[i]) or tick == total - 1
            last5 = rel[i, 0] - history[max(len(history) - 251, 0)][i]
            name = outcome(terminated=bool(terminated[i]), lateral=rel[i, 1], reached=reached,
                           support=support[i], dwell=dwell[i], expired=expired, last5=last5)
            if name:
                span = max(goal[i] - start[i, 0], 1e-6)
                records[i] = dict(**specs[i], outcome=name, duration_s=t, final_x=float(rel[i, 0]),
                    final_y=float(rel[i, 1]), progress=float(np.clip((maximum[i] - start[i, 0]) / span, 0., 1.)),
                    max_abs_heading_rad=float(yaw_abs[i]),
                    speed_mae=float(speed_error[i] / speed_steps[i]) if speed_steps[i] else None)
                active[i] = False
        return result

    base.reward_manager.compute = capture
    with torch.inference_mode():
        for tick in range(total):
            assert torch.equal(obs['policy'][:, 6:9], command.command)
            if bridge is None:
                obs, *_ = base.step(actor(obs['policy']).clamp(-100, 100))
            else:
                trace = bridge.step(actor(obs['policy']), request, no_timeout)
                live = torch.as_tensor(active, device=device)
                delta_sum += (trace.executed_action - trace.base_action).norm(dim=-1) * live; delta_steps += live
                obs, _, term, trunc, _ = base.step(trace.executed_action)
                if (term | trunc).any():
                    safety.reset_temporal_state(torch.where(term | trunc)[0])
            if not active.any():
                break
    assert all(r is not None for r in records)
    for i, r in enumerate(records):
        r['mean_executed_delta'] = float(delta_sum[i] / delta_steps[i].clamp(min=1))
    assert digest() == before and sha(MODELS['B']) == HASHES['B']
    assert feedback_error <= 1e-5, feedback_error
    save(args.output / 'report.json', dict(status='passed', run=args.run, **{k: run[k] for k in ('layout', 'profile', 'heading')},
        model='B10200', model_sha256=HASHES['B'], episodes=n, records=records, actor_unchanged=True,
        first_episodes_only=True, optimizer_updates=0, step_dt=dt, feedback_max_error=feedback_error,
        safety_filter=bridge is not None, source_sha256=sha(__file__), sources=source_manifest(), goals=goals))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', choices=tuple(RUNS), required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    args.output = OUTPUT / args.run
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=args.gpu, output=args.output, timeout=3600, minimum_free_mib=10240)
    try:
        collect(args); sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc(); save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
