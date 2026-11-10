"""Source-only corrections at the RSL-RL learning boundary, without Isaac imports."""
from __future__ import annotations

import torch


class FailurePriorityVecEnv:
    """Keep raw Gym endings, but never bootstrap a transition that also failed."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def step(self, actions):
        observations, rewards, dones, extras = self.env.step(actions)
        if 'time_outs' in extras:
            truncated = extras['time_outs']
            terminated = self.unwrapped.reset_terminated
            if (not isinstance(truncated, torch.Tensor) or not isinstance(terminated, torch.Tensor)
                    or truncated.shape != terminated.shape or truncated.ndim != 1):
                raise ValueError('termination and timeout masks must have the same one-dimensional shape')
            # Allocate a new mask: the original termination manager buffers are
            # also used by curriculum updates and must retain both raw flags.
            simultaneous = truncated.bool() & terminated.bool()
            extras['time_outs'] = truncated.bool() & ~terminated.bool()
            extras['log'] = dict(extras.get('log', {}))
            extras['log']['Diagnostics/source_failure_timeouts'] = simultaneous.float().sum()
        return observations, rewards, dones, extras


def wrap_source_curriculum_learning(env):
    """Opt in via the task configuration; preserve other RSL-RL tasks."""
    enabled = getattr(env.unwrapped.cfg, 'source_failure_priority_bootstrap', False)
    if type(enabled) is not bool:
        raise ValueError('source_failure_priority_bootstrap must be boolean')
    return FailurePriorityVecEnv(env) if enabled else env
