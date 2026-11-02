"""Source training adapter around the installed RSL-RL environment wrapper.

Imported only after Isaac Sim startup by simulator entrypoints. Duck-typed CPU
fixtures exercise the same reset/capture/action code without importing Isaac.
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from .bridge import FrozenPolicySafetyBridge
from .parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from .training_lifecycle import SourceEpisodeLifecycle, SourceExposureLedger, SourceRunSeeds


def require_finite_observations(observations):
    for key, value in observations.items():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"invalid Source observation group: {key}")


class SourceTrainingRuntime:
    def __init__(self, env, adapter):
        self.env, self.adapter = env, adapter
        self.device = torch.device(env.device)
        self.zeros = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=self.device)
        self.safety = SafetyFilter(SafetyLimits(
            residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4, device=self.device)))
        # Gaussian sampling has already happened in PPO.act(). The identity
        # callable only reuses the deployment clipping/validation/filter trace.
        self.bridge = FrozenPolicySafetyBridge(lambda action: action, self, self.safety)
        self.lifecycle = SourceEpisodeLifecycle(env.num_envs, env.device, self.safety.limits)
        self.last_step = None
        self.terminal = None
        env.add_pre_reset_callback(self.before_reset)

    def safety_state(self, control_timed_out):
        # NaN > threshold is False: validate raw forces before constructing a
        # boolean contact flag, including rows not ending on this control step.
        forces = self.adapter.contact_sensor.data.net_forces_w
        if (forces.ndim != 3 or forces.shape[0] != self.env.num_envs
                or forces.shape[-1] != 3 or forces.device != self.device):
            raise ValueError("Source contact force batch is malformed")
        if not bool(torch.isfinite(forces).all()):
            raise FloatingPointError("Source contact forces contain non-finite values")
        return self.adapter.safety_state(control_timed_out)

    def evaluate(self):
        if self.last_step != self.env.common_step_counter:
            state = self.safety_state(self.zeros)
            progress = (self.env.scene["robot"].data.root_link_pos_w[:, 0]
                        - self.env.scene.terrain.env_origins[:, 0])
            self.lifecycle.update(state, progress, self.env.episode_length_buf)
            self.last_step = self.env.common_step_counter
        return self.lifecycle

    def before_reset(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if getattr(self.env, "_source_training_in_step", False):
            if self.terminal is not None:
                raise RuntimeError("multiple reset callbacks in one Source control step")
            observations = self.env.observation_manager.compute(update_history=False)
            require_finite_observations(observations)
            terminated = self.env.reset_terminated.detach().clone()
            truncated = self.env.reset_time_outs.detach().clone()
            terms = {name: self.env.termination_manager.get_term(name)[ids].clone()
                     for name in self.env.termination_manager.active_terms}
            other_failure = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
            for name, active in terms.items():
                if name not in ("source_training_completion", "source_training_timeout"):
                    other_failure |= active
            self.terminal = {
                "ids": ids.clone(),
                "observations": TensorDict(observations, batch_size=[self.env.num_envs]).clone(),
                "terminated": terminated, "truncated": truncated,
                "steps": self.env.episode_length_buf[ids].clone(),
                "hold": self.lifecycle.hold[ids].clone(),
                "completed": self.lifecycle.completed[ids].clone() & ~other_failure,
                "terms": terms,
            }
        self.safety.reset_temporal_state(ids)
        self.lifecycle.reset(ids)
        self.last_step = None


def source_training_runtime(env):
    runtime = getattr(env, "_source_training_runtime", None)
    if runtime is None:
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        runtime = SourceTrainingRuntime(env, Lite3IsaacLabAdapter(env, safety_height_source="base_scan"))
        env._source_training_runtime = runtime
    return runtime


def source_training_completion(env):
    return source_training_runtime(env).evaluate().completed.clone()


def source_training_safety(env):
    return source_training_runtime(env).evaluate().unsafe.clone()


def source_training_timeout(env):
    return source_training_runtime(env).evaluate().timeout.clone()


class SourceTrainingVecEnv:
    """Retain PPO requests; send only copied, filtered actions to the simulator.

    A pre-action pause means the post-step safety termination failed to end a
    bad state (or reset supplied one). Stop before executing the paused action.
    Ordinary post-step task/safety endings auto-reset only the affected rows.
    """
    def __init__(self, wrapped_env, *, runtime=None):
        self.env = wrapped_env
        cfg = wrapped_env.unwrapped.cfg
        seeds = SourceRunSeeds(getattr(cfg, "source_training_run", None))
        if (cfg.source_terrain_level != "L1" or cfg.seed != seeds.ppo
                or cfg.source_parallel_sampler_seed != seeds.sampler
                or cfg.scene.terrain.terrain_generator.seed != seeds.geometry):
            raise ValueError("Source training configuration has unregistered level or seeds")
        required_terms = {"source_training_safety", "source_training_completion", "source_training_timeout"}
        if not required_terms.issubset(wrapped_env.unwrapped.termination_manager.active_terms):
            raise ValueError("Source training requires registered manager episode endings")
        self.runtime = runtime or source_training_runtime(wrapped_env.unwrapped)
        labels = tuple(wrapped_env.unwrapped.cfg.scene.terrain.terrain_generator.sub_terrains)
        if labels != tuple(SOURCE_RESET_BLOCK_COUNTS):
            raise ValueError("Source training must retain all 15 registered L1 cells in order")
        if getattr(wrapped_env, "clip_actions", None) is not None:
            raise ValueError("Source adapter owns clipping; underlying wrapper clipping must be disabled")
        self.exposure = SourceExposureLedger(labels, wrapped_env.device)
        self.exposure.record(self.unwrapped.scene.terrain.terrain_types, reset=True)
        self.last_trace = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        observations = self.env.get_observations()
        require_finite_observations(observations)
        if observations["policy"].shape != (self.num_envs, 45):
            raise ValueError("Source actor requires 45 proprioceptive inputs")
        return observations

    def reset(self):
        observations, extras = self.env.reset()
        require_finite_observations(observations)
        self.exposure.record(self.unwrapped.scene.terrain.terrain_types, reset=True)
        return observations, extras

    def step(self, actions):
        if actions.shape != (self.num_envs, 12):
            raise ValueError("Source PPO requires [num_envs, 12] sampled actions")
        trace = self.runtime.bridge.step(actions, self.runtime.zeros)
        if bool(trace.decision.paused.any()):
            raise RuntimeError("Source pre-action safety pause: invalid reset or missing post-step termination")
        old_columns = self.unwrapped.scene.terrain.terrain_types.clone()
        self.runtime.terminal = None
        self.unwrapped._source_training_in_step = True
        try:
            observations, rewards, dones, extras = self.env.step(trace.executed_action.clone())
        finally:
            self.unwrapped._source_training_in_step = False
        require_finite_observations(observations)
        if not bool(torch.isfinite(rewards).all()):
            raise FloatingPointError("Source rewards contain non-finite values")
        ended_ids = dones.nonzero(as_tuple=False).flatten()
        terminal = self.runtime.terminal
        if len(ended_ids) and (terminal is None or not torch.equal(terminal["ids"], ended_ids)):
            raise RuntimeError("Source auto-reset did not capture exactly the terminal rows")
        if not len(ended_ids) and terminal is not None:
            raise RuntimeError("Source reset callback occurred without an episode ending")
        self.exposure.record(old_columns)
        self.exposure.record(self.unwrapped.scene.terrain.terrain_types[ended_ids], reset=True)
        extras = dict(extras)
        extras["source_terminal"] = terminal
        extras["time_outs"] = (terminal["truncated"] & ~terminal["terminated"] if terminal is not None
                              else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.last_trace = trace
        return observations, rewards, dones, extras
