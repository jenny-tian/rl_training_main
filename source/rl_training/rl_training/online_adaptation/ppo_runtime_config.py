"""Apply explicit PPO runtime settings after restoring an episode checkpoint."""

from __future__ import annotations

import math
from dataclasses import replace
from numbers import Real
from typing import Any

import torch

from .online_session import OnlinePPOSession


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _std_parameter(policy_state: dict[str, Any]) -> tuple[str, torch.Tensor, bool]:
    fields = [
        key for key in policy_state
        if key.rsplit(".", 1)[-1] in {"log_std_param", "std_param"}
    ]
    if len(fields) != 1:
        raise ValueError("PPO policy must expose exactly one Gaussian log_std_param or std_param")
    key = fields[0]
    value = policy_state[key]
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.numel() == 0:
        raise ValueError("Gaussian standard deviation must be a nonempty floating-point tensor")
    return key, value, key.rsplit(".", 1)[-1] == "log_std_param"


def _scalar_or_list(values: list[float]) -> float | list[float]:
    if not values:
        raise ValueError("PPO runtime setting has no values")
    return values[0] if all(value == values[0] for value in values) else values


def read_ppo_runtime_config(session: OnlinePPOSession) -> dict[str, float | int | list[float]]:
    """Read effective settings without changing a frozen evaluation checkpoint.

    Uniform action standard deviations and optimizer learning rates are
    returned as scalars. Nonuniform values remain lists in action/parameter
    group order so the record does not conceal differences with an average.
    """
    if not isinstance(session, OnlinePPOSession):
        raise TypeError("session must be an OnlinePPOSession")
    updater = session.updater
    _, value, logarithmic = _std_parameter(updater.policy.state_dict())
    std = value.detach().exp() if logarithmic else value.detach()
    return {
        "action_std": _scalar_or_list(std.reshape(-1).cpu().tolist()),
        "actor_lr": _scalar_or_list([float(group["lr"]) for group in updater.actor_optimizer.param_groups]),
        "critic_lr": _scalar_or_list([float(group["lr"]) for group in updater.critic_optimizer.param_groups]),
        "policy_version": updater.policy_version,
    }


def apply_ppo_runtime_config(
    session: OnlinePPOSession,
    *,
    action_std: float,
    actor_lr: float,
    critic_lr: float,
) -> dict[str, float | int | list[float]]:
    """Override resumed settings at an empty rollout boundary, then publish.

    Both the active checkpoint and its existing rollback target receive the
    new settings. Their policy/critic weights, Adam moments, counters, and
    candidate verification status are preserved. A changed action distribution
    advances the active policy version; changing only learning rates does not.
    No pending rollout may be relabeled with a different sampling distribution.
    """
    if not isinstance(session, OnlinePPOSession):
        raise TypeError("session must be an OnlinePPOSession")
    if session.pending_transitions:
        raise ValueError("cannot apply PPO runtime settings with pending rollout transitions")
    action_std = _positive_float(action_std, "action_std")
    actor_lr = _positive_float(actor_lr, "actor_lr")
    critic_lr = _positive_float(critic_lr, "critic_lr")
    updated_config = replace(session.updater.config, actor_lr=actor_lr, critic_lr=critic_lr)

    # state_dict() owns independent tensors. Validate and edit both branches
    # before load_state_dict() atomically restores and republishes the session.
    state = session.state_dict()
    active_std_changed = False
    for branch_name in ("updater", "stable_state"):
        branch = state[branch_name]
        key, previous, logarithmic = _std_parameter(branch["policy"])
        try:
            replacement = torch.full_like(previous, math.log(action_std) if logarithmic else action_std)
        except (RuntimeError, OverflowError) as error:
            raise ValueError("action_std is not representable in the policy dtype") from error
        effective_std = replacement.exp() if logarithmic else replacement
        if not bool((torch.isfinite(effective_std) & (effective_std > 0.0)).all().item()):
            raise ValueError("action_std is not representable in the policy dtype")
        if branch_name == "updater":
            previous_std = previous.exp() if logarithmic else previous
            active_std_changed = not torch.equal(previous_std, effective_std)
        branch["policy"][key] = replacement
        for optimizer_name, learning_rate in (("actor_optimizer", actor_lr), ("critic_optimizer", critic_lr)):
            for group in branch[optimizer_name]["param_groups"]:
                group["lr"] = learning_rate
    if active_std_changed:
        state["updater"]["policy_version"] += 1

    session.load_state_dict(state)
    session.updater.config = updated_config
    session.updater.actor_optimizer.defaults["lr"] = actor_lr
    session.updater.critic_optimizer.defaults["lr"] = critic_lr
    return read_ppo_runtime_config(session)


__all__ = ["apply_ppo_runtime_config", "read_ppo_runtime_config"]
