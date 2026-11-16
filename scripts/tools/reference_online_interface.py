"""Current reference-trained actor interface; no simulator import or learning."""
from __future__ import annotations

import numpy as np
import torch

JOINTS = [f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'HL', 'HR')
          for joint in ('HipX', 'HipY', 'Knee')]
DEFAULT = [0., -.65, 1.3] * 4
SCALE = [.125, .25, .25] * 4
RESET_SEED = 20261116
RESET_RANGES = {'x': (.20, .30), 'y': (-.10, .10), 'yaw': (-.03, .03)}


def clip_action(action):
    if action.ndim != 2 or action.shape[1] != 12 or not torch.isfinite(action).all():
        raise ValueError('invalid current actor action')
    return action.clamp(-100., 100.)


def observation(robot, ids, command, previous):
    """Native terms clip before scaling; noise is disabled for this experiment."""
    data = robot.data
    return torch.cat((data.root_ang_vel_b.clamp(-100, 100) * .25,
        data.projected_gravity_b.clamp(-100, 100), command.clamp(-100, 100),
        (data.joint_pos[:, ids] - data.default_joint_pos[:, ids]).clamp(-100, 100),
        (data.joint_vel[:, ids] - data.default_joint_vel[:, ids]).clamp(-100, 100) * .05,
        previous.clamp(-100, 100)), -1)


class EpisodeLedger:
    """Reset identities persist for untouched rows; reserve attempts before running.

    Conditions/model arms share slot keys intentionally for paired comparisons.
    A reset does not create free budget: the caller must reserve each attempt.
    """
    def __init__(self, slots, namespace, limit):
        self.slots = list(slots)
        self.namespace = int(namespace)
        self.next_episode = [0] * len(slots)
        self.keys = {}
        self.limit, self.started, self.steps = int(limit), 0, 0

    def reset(self, ids):
        from rl_training.ea_lpl.reset_table import initial_state_offsets
        ids = list(ids)
        if len(set(ids)) != len(ids) or any(i < 0 or i >= len(self.slots) for i in ids):
            raise ValueError('invalid reset IDs')
        if self.started + len(ids) > self.limit:
            raise RuntimeError('episode budget exhausted')
        slots, episodes = [self.slots[i] for i in ids], [self.next_episode[i] for i in ids]
        offsets = initial_state_offsets(RESET_SEED, self.namespace, slots, episodes, RESET_RANGES)
        for i, slot, episode in zip(ids, slots, episodes):
            self.keys[i] = (RESET_SEED, self.namespace, slot, episode)
            self.next_episode[i] += 1
        self.started += len(ids)
        return offsets

    def count_steps(self, active):
        self.steps += int(np.count_nonzero(active))


class CurrentIsaacAdapter:
    def __init__(self, env):
        self.env, self.robot = env, env.scene['robot']
        self.sensor = env.scene['contact_forces']
        term = env.action_manager.get_term('joint_pos')
        self.ids = list(term._joint_ids)
        if [self.robot.joint_names[i] for i in self.ids] != JOINTS:
            raise ValueError('current actor joint order mismatch')
        self.scale = torch.as_tensor(SCALE, device=env.device)[None]
        if not torch.allclose(term._scale, self.scale.expand(env.num_envs, -1), atol=1e-7, rtol=0):
            raise ValueError('current actor scale mismatch')
        default = torch.tensor(DEFAULT, device=env.device)[None]
        if not torch.allclose(self.robot.data.default_joint_pos[:, self.ids], default, atol=1e-7, rtol=0):
            raise ValueError('current actor default joint position mismatch')
        if abs(env.step_dt - .02) > 1e-10 or abs(env.cfg.sim.dt - .005) > 1e-10:
            raise ValueError('current actor control period mismatch')
        self.nonfoot_ids = [i for i, name in enumerate(self.sensor.body_names) if not name.endswith('_FOOT')]

    def safety_state(self, control_timed_out):
        from rl_training.online_adaptation.types import SafetyState
        data = self.robot.data
        hits = self.env.scene['height_scanner_base'].data.ray_hits_w
        # Match the registered terrain-relative 10 cm patch clearance semantics.
        if hits.shape != (self.env.num_envs, 9, 3) or not torch.isfinite(hits).all():
            raise ValueError('invalid base safety height scan')
        if (torch.linalg.vector_norm(hits[:, :, :2] - data.root_pos_w[:, None, :2], dim=-1) > .08).any():
            raise ValueError('base safety height scan not under the robot')
        forces = self.sensor.data.net_forces_w
        if not torch.isfinite(forces).all():
            raise ValueError('nonfinite contact forces')
        limits = data.soft_joint_pos_limits[:, self.ids]
        return SafetyState(data.projected_gravity_b[:, :2],
            data.root_pos_w[:, 2:3] - hits[:, :, 2].amax(-1, keepdim=True),
            data.joint_pos[:, self.ids], data.joint_vel[:, self.ids], limits[..., 0], limits[..., 1],
            data.default_joint_pos[:, self.ids], self.scale,
            (forces[:, self.nonfoot_ids].norm(dim=-1) > 1.).any(-1, keepdim=True), control_timed_out)

    def check_observation(self, native, previous):
        expected = observation(self.robot, self.ids, self.env.command_manager.get_command('base_velocity'), previous)
        if native.shape != expected.shape or not torch.allclose(native, expected, atol=1e-6, rtol=0):
            raise ValueError('native observation differs from current actor contract')
        if not torch.equal(native[:, 33:45], previous.clamp(-100, 100)):
            raise ValueError('actor last_action is not the previous executed action')


def safety_diagnostics(trace):
    """Distinguish base-target projection from a guarded/rate-limited residual."""
    s = trace.safety_state
    base_target = s.default_joint_pos + s.action_scale * trace.base_action
    base_projection = ((base_target < s.joint_lower + .02) | (base_target > s.joint_upper - .02)).any(-1)
    changed = (trace.executed_action != trace.base_action).any(-1)
    zero = (trace.residual_request == 0).all(-1)
    if (changed & zero & ~base_projection).any():
        raise ValueError('zero residual changes a feasible base action')
    return base_projection, changed
