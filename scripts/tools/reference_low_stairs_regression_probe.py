"""冻结回放与单因素诊断；复用历史协议，不覆盖正式验收，也不训练。"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from reference_stairs_eval import ROOT, MODELS, save, sha
from reference_eval_checkpoint import register_checkpoint

PARENT = ROOT / 'logs/reference_six/low_stairs_training/scratch_seed42_10000/model_updates_10000.pt'
CANDIDATE = ROOT / 'logs/reference_six/low_stairs_clearance_trial_20261115/clearance_seed42_500/model_updates_10500.pt'
HASHES = {'parent': '419921855e24b40af058ac6642083eca8538f639e596606c51c2150ae7549a8f',
          'candidate': '6f5be217cd012d678e0528bc7b9c8d528001083f5ae48697585bf7e1944fea8e'}
OUTPUT = ROOT / 'logs/reference_six/low_stairs_regression_preparation_20261115'


def heading_command(heading, gain=.5, limit=.8):
    """朝台阶的世界坐标 +x 方向，沿用训练中的增益和限幅。"""
    error = (-np.asarray(heading) + np.pi) % (2 * np.pi) - np.pi
    return np.clip(gain * error, -limit, limit)


def ramp_fraction(tick, dt=.02, duration=.5):
    if tick < 0 or dt <= 0 or duration <= 0:
        raise ValueError('无效的动作过渡参数')
    return min((tick + 1) * dt / duration, 1.)


def collect(args, app):
    import torch
    import isaaclab.envs as envs
    from reference_stairs_eval import load_actor
    module = importlib.import_module('reference_turn_diagnose' if args.kind == 'turn'
                                    else 'reference_isaac_stairs_diagnose')
    native_environment = envs.ManagerBasedRLEnv
    native_loader = module.load_actor
    arrays, preparation, observations, parent_actions, candidate_actions, sent_actions = {}, {}, [], [], [], []
    state = {'steps': 0, 'actor_steps': 0}
    register_checkpoint('regression_parent', PARENT, HASHES['parent'])
    register_checkpoint('regression_candidate', CANDIDATE, HASHES['candidate'])
    shadow = None

    def snap(values):
        return {k: v.detach().cpu().numpy().copy() for k, v in values.items()}

    def instrument(base):
        state['base'] = base
        robot = base.scene['robot']
        sensor = base.scene['contact_forces']
        feet = [robot.body_names.index(f'{leg}_FOOT') for leg in ('FL', 'FR', 'HL', 'HR')]
        contact_ids = [sensor.body_names.index(f'{leg}_FOOT') for leg in ('FL', 'FR', 'HL', 'HR')]
        native_step, native_reward = base.step, base.reward_manager.compute

        def physics_snapshot():
            values = dict(masses=robot.root_physx_view.get_masses(),
                          inertias=robot.root_physx_view.get_inertias(),
                          coms=robot.root_physx_view.get_coms(),
                          stiffness=robot.data.joint_stiffness, damping=robot.data.joint_damping,
                          default_joint_pos=robot.data.default_joint_pos)
            for name, actuator in robot.actuators.items():
                values[f'delay_{name}'] = actuator.positions_delay_buffer.time_lags
                values[f'pd_stiffness_{name}'] = actuator.stiffness
                values[f'pd_damping_{name}'] = actuator.damping
            return snap(values)

        def capture(dt):
            value = native_reward(dt)
            if state['steps'] < 50:
                values = dict(root=robot.data.root_state_w, gravity=robot.data.projected_gravity_b,
                    joint_pos=robot.data.joint_pos, force=sensor.data.net_forces_w[:, contact_ids])
                for key, array in snap(values).items():
                    preparation.setdefault(key, []).append(array)
            if state['steps'] >= 50:
                data = robot.data
                command = base.command_manager.get_command('base_velocity')
                stand = (command[:, :2].norm(dim=1) < .1) & (command[:, 2].abs() < .1)
                height = data.root_pos_w[:, 2] - base.scene['height_scanner_base'].data.ray_hits_w[:, :, 2].mean(-1)
                values = dict(joint_pos=data.joint_pos, joint_vel=data.joint_vel,
                    joint_target=data.joint_pos_target, torque=data.applied_torque,
                    feet=data.body_pos_w[:, feet] - base.scene.env_origins[:, None],
                    force=sensor.data.net_forces_w[:, contact_ids],
                    root=data.root_state_w, gravity=data.projected_gravity_b,
                    train_stand_rate=-.5 * (data.joint_pos - data.default_joint_pos).abs().sum(-1) * stand,
                    train_height_rate=-50 * (height - .32).square(),
                    command=command, terminated=base.reset_terminated)
                for key, array in snap(values).items():
                    arrays.setdefault(key, []).append(array)
            return value

        def step(action):
            if state['steps'] == 50:
                np.savez_compressed(args.output / 'physics_initial.npz', **physics_snapshot())
                save(args.output / 'names.json', dict(joint_names=robot.joint_names,
                    body_names=robot.body_names, actuator_names=list(robot.actuators),
                    action_joint_ids=base.action_manager.get_term('joint_pos')._joint_ids))
                if args.intervention == 'no_delay':
                    for actuator in robot.actuators.values():
                        for name in ('positions_delay_buffer', 'velocities_delay_buffer', 'efforts_delay_buffer'):
                            getattr(actuator, name).set_time_lag(0)
                if args.intervention == 'heading':
                    original_compute = base.command_manager.compute
                    command = base.command_manager.get_term('base_velocity')
                    def compute(dt):
                        original_compute(dt)
                        # 历史协议先写入下一帧速度；随后加航向反馈，观测与奖励读同一命令。
                        moving = command.vel_command_b[:, 0] > 0
                        heading = robot.data.heading_w
                        error = torch.remainder(-heading + torch.pi, 2 * torch.pi) - torch.pi
                        command.vel_command_b[:, 2] = torch.where(moving, (.5 * error).clamp(-.8, .8), 0.)
                    base.command_manager.compute = compute
                np.savez_compressed(args.output / 'physics_effective.npz', **physics_snapshot())
            result = native_step(action)
            state['steps'] += 1
            return result
        base.step = step
        base.reward_manager.compute = capture
        return base

    class ActorProbe(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, obs):
            nonlocal shadow
            if shadow is None:
                # 加载网络会初始化随机权重；保存并恢复 RNG，避免扰动观测噪声与重置采样。
                with torch.random.fork_rng(devices=[obs.device.index]):
                    shadow = load_actor('regression_parent' if args.policy == 'candidate'
                                        else 'regression_candidate').to(obs.device)
            output, other = self.actor(obs), shadow(obs)
            parent, candidate = (other, output) if args.policy == 'candidate' else (output, other)
            if args.intervention == 'ramp':
                output = output * ramp_fraction(state['actor_steps'])
            observations.append(obs.detach().cpu().numpy().copy())
            parent_actions.append(parent.detach().cpu().numpy().copy())
            candidate_actions.append(candidate.detach().cpu().numpy().copy())
            sent_actions.append(output.detach().cpu().numpy().copy())
            state['actor_steps'] += 1
            return output

    class ObservedEnvironment(native_environment):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            instrument(self)

    envs.ManagerBasedRLEnv = ObservedEnvironment
    module.load_actor = lambda name: ActorProbe(native_loader(name))
    try:
        module.collect(args, app)
    finally:
        envs.ManagerBasedRLEnv = native_environment
        module.load_actor = native_loader
    np.savez_compressed(args.output / 'diagnostic.npz',
        **{k: np.asarray(v) for k, v in arrays.items()}, observation=np.asarray(observations),
        parent_action=np.asarray(parent_actions), candidate_action=np.asarray(candidate_actions),
        sent_action=np.asarray(sent_actions))
    np.savez_compressed(args.output / 'preparation.npz', **{k: np.asarray(v) for k, v in preparation.items()})
    assert len(observations) == len(arrays['root']) == args.steps
    assert sha(PARENT) == HASHES['parent'] and sha(CANDIDATE) == HASHES['candidate']
    report = json.loads((args.output / 'report.json').read_text())
    save(args.output / 'diagnostic_report.json', dict(status='passed', training=False, optimizer_updates=0,
        kind=args.kind, policy=args.policy, intervention=args.intervention, profile=args.profile,
        steps=args.steps, episodes=report['episodes'], initial_preparation_steps=50,
        model_hashes=HASHES, diagnostic_only=True,
        intervention_scope={'native': '纯记录', 'ramp': '前 0.5 秒动作由默认 PD 线性过渡到策略动作',
          'no_delay': '准备后仅将执行器延迟设为零，保留历史缓冲',
          'heading': '运动后加入训练增益 0.5、限幅 0.8 的世界 +x 航向反馈'}[args.intervention],
        source_sha256=sha(__file__), original_protocol_sha256=sha(module.__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('turn', 'stairs'), required=True)
    parser.add_argument('--policy', choices=('parent', 'candidate'), required=True)
    parser.add_argument('--intervention', choices=('native', 'ramp', 'no_delay', 'heading'), default='native')
    parser.add_argument('--profile', choices=('nominal', 'training_randomized'), required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or (args.intervention == 'heading' and args.kind != 'stairs'):
        parser.error('步数应为 1–1000，航向对照只用于台阶')
    args.output = args.output.resolve()
    args.model = 'regression_' + args.policy
    args.repeats, args.gpu, args.video_env = 20, 1, None
    args.direction, args.height_set, args.action_mode = 'up', 'low_stairs', 'mean'
    register_checkpoint(args.model, PARENT if args.policy == 'parent' else CANDIDATE, HASHES[args.policy])
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=1, output=args.output, timeout=1800, minimum_free_mib=10240)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device='cuda:0',
            kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        collect(args, app)
        sys.stdout.flush()
        os._exit(0)
    except Exception:
        traceback.print_exc()
        save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush()
        os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
