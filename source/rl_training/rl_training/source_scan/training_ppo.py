"""RSL-RL 5.0.1 adapter using actual pre-reset values for timeouts."""
from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Mapping

import torch
from torch.overrides import TorchFunctionMode
from rsl_rl.algorithms import PPO


def _assert_finite(value, label):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite Source {label}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")


class _FiniteLossBackward(TorchFunctionMode):
    """Check the installed PPO's scalar loss before its backward call."""

    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor.backward:
            _assert_finite(args[0], "loss before backward")
            self.algorithm.source_backward_checks += 1
        return func(*args, **(kwargs or {}))


class SourcePPO(PPO):
    def __init__(self, *args, **kwargs):
        if importlib.metadata.version("rsl-rl-lib") != "5.0.1":
            raise RuntimeError("Source PPO adapter is audited only for RSL-RL 5.0.1")
        super().__init__(*args, **kwargs)
        self.source_backward_checks = 0
        self.source_optimizer_steps_started = 0
        self.source_optimizer_steps_completed = 0
        self._source_optimizer_hooks = [
            self.optimizer.register_step_pre_hook(self._before_optimizer_step),
            self.optimizer.register_step_post_hook(self._after_optimizer_step),
        ]

    def _model_state(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def _optimizer_state(self):
        return self.optimizer.state

    def _before_optimizer_step(self, optimizer, args, kwargs):
        _assert_finite(self._model_state(), "model parameter")
        _assert_finite(self._optimizer_state(), "optimizer state")
        for name, parameter in (*self.actor.named_parameters(), *self.critic.named_parameters()):
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"non-finite Source gradient: {name}")
        self.source_optimizer_steps_started += 1

    def _after_optimizer_step(self, optimizer, args, kwargs):
        _assert_finite(self._model_state(), "model parameter after update")
        _assert_finite(self._optimizer_state(), "optimizer state after update")
        self.source_optimizer_steps_completed += 1

    def compute_returns(self, obs):
        _assert_finite(obs, "return observation")
        super().compute_returns(obs)
        for name in ("values", "rewards", "dones", "returns", "advantages"):
            _assert_finite(getattr(self.storage, name), f"rollout {name}")

    def update(self):
        for name in ("observations", "actions", "actions_log_prob", "values", "rewards", "dones",
                     "returns", "advantages", "distribution_params"):
            _assert_finite(getattr(self.storage, name, None), f"rollout {name}")
        _assert_finite(self._model_state(), "model parameter before update")
        _assert_finite(self._optimizer_state(), "optimizer state before update")
        with _FiniteLossBackward(self):
            losses = super().update()
        for name, value in losses.items():
            if isinstance(value, torch.Tensor):
                finite = bool(torch.isfinite(value).all())
            else:
                finite = isinstance(value, (int, float)) and math.isfinite(float(value))
            if not finite:
                raise FloatingPointError(f"non-finite Source loss: {name}")
        _assert_finite(self._model_state(), "model parameter after update")
        _assert_finite(self._optimizer_state(), "optimizer state after update")
        return losses

    def process_env_step(self, obs, rewards, dones, extras):
        if "source_terminal" not in extras or "time_outs" not in extras:
            raise RuntimeError("SourcePPO requires the Source training environment adapter")
        timeouts = extras["time_outs"].to(self.device)
        if timeouts.shape != dones.shape or timeouts.dtype != torch.bool:
            raise ValueError("Source timeout mask has invalid shape/dtype")
        if bool((timeouts & ~dones.bool()).any()):
            raise ValueError("Source timeout without a done flag")
        adjusted = rewards.clone()
        if bool(timeouts.any()):
            terminal = extras["source_terminal"]
            if terminal is None or bool((timeouts & terminal["terminated"].to(self.device)).any()):
                raise RuntimeError("missing or contradictory Source timeout snapshot")
            with torch.no_grad():
                terminal_values = self.critic(terminal["observations"].to(self.device)).squeeze(-1)
            if not bool(torch.isfinite(terminal_values).all()):
                raise FloatingPointError("Source terminal critic values are non-finite")
            adjusted += self.gamma * torch.where(timeouts, terminal_values, 0.0)
        # Disable RSL-RL's current-state timeout shortcut. The captured terminal
        # value was added above; done still cuts GAE across the reset boundary.
        forwarded = {key: value for key, value in extras.items() if key != "time_outs"}
        super().process_env_step(obs, adjusted, dones, forwarded)
