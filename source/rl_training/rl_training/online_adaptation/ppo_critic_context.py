"""Critic-only velocity/time observations and lossless checkpoint migration."""

from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Any

import torch
from torch import Tensor


def build_critic_observation(
    policy_obs: Tensor,
    body_lin_vel_xy: Tensor,
    *,
    step: int,
    max_steps: int,
) -> Tensor:
    """Append body-frame planar velocity and elapsed fraction to policy input.

    The actor observation is preserved. No position, terrain descriptor, or
    heading is introduced, and the returned tensor keeps its dtype/device.
    """
    if not isinstance(policy_obs, Tensor) or policy_obs.ndim != 2 or policy_obs.shape[1] == 0:
        raise ValueError("policy_obs must have shape [N, policy_obs_dim]")
    if not policy_obs.is_floating_point() or not bool(torch.isfinite(policy_obs).all().item()):
        raise ValueError("policy_obs must contain finite real floating-point values")
    if not isinstance(body_lin_vel_xy, Tensor) or body_lin_vel_xy.shape != (policy_obs.shape[0], 2):
        raise ValueError("body_lin_vel_xy must have shape [N, 2] matching policy_obs")
    if body_lin_vel_xy.device != policy_obs.device:
        raise ValueError("body_lin_vel_xy and policy_obs must be on the same device")
    if not body_lin_vel_xy.is_floating_point() or not bool(torch.isfinite(body_lin_vel_xy).all().item()):
        raise ValueError("body_lin_vel_xy must contain finite real floating-point values")
    if isinstance(max_steps, bool) or not isinstance(max_steps, Real) or not math.isfinite(max_steps) or max_steps <= 0:
        raise ValueError("max_steps must be finite and positive")
    if isinstance(step, bool) or not isinstance(step, Real) or not math.isfinite(step) or not 0 <= step <= max_steps:
        raise ValueError("step must be finite and in [0, max_steps]")
    velocity = body_lin_vel_xy.to(dtype=policy_obs.dtype)
    if not bool(torch.isfinite(velocity).all().item()):
        raise ValueError("body_lin_vel_xy is not finite in the policy observation dtype")
    elapsed = policy_obs.new_full((policy_obs.shape[0], 1), float(step / max_steps))
    return torch.cat((policy_obs, velocity, elapsed), dim=-1)


def _positive_dimension(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _pad_columns(value: Tensor, count: int) -> Tensor:
    with torch.no_grad():
        return torch.cat((value, value.new_zeros((value.shape[0], count))), dim=1)


def _migrate_updater(state: Any, *, policy_obs_dim: int, context_dim: int, label: str) -> None:
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError(f"{label} has an incompatible online PPO updater schema")
    critic = state.get("critic")
    if not isinstance(critic, dict):
        raise ValueError(f"{label}.critic must be a state dictionary")
    weight = critic.get("0.weight")
    widths = (policy_obs_dim, policy_obs_dim + context_dim)
    if (
        not isinstance(weight, Tensor)
        or weight.ndim != 2
        or weight.shape[0] != 128
        or weight.shape[1] not in widths
        or not weight.is_floating_point()
        or not bool(torch.isfinite(weight).all().item())
    ):
        raise ValueError(f"{label}.critic 0.weight must be finite with shape [128, {widths[0]}] or [128, {widths[1]}]")
    needs_padding = weight.shape[1] == policy_obs_dim

    optimizer = state.get("critic_optimizer")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError(f"{label}.critic_optimizer must contain an Adam state dictionary")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{label}.critic_optimizer has no parameter groups")
    parameter_ids = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise ValueError(f"{label}.critic_optimizer has an invalid parameter group")
        parameter_ids.extend(group["params"])
    # OnlinePPOUpdater constructs Adam from critic.parameters() in order.
    # Its sequential Linear/ELU critic registers 0.weight first. Check that
    # this mapping still describes the checkpoint instead of assuming id=0.
    parameter_names = [name for name in critic if name.endswith((".weight", ".bias"))]
    if (
        not parameter_names
        or parameter_names[0] != "0.weight"
        or len(parameter_ids) != len(parameter_names)
        or any(isinstance(key, bool) or not isinstance(key, int) for key in parameter_ids)
        or len(set(parameter_ids)) != len(parameter_ids)
    ):
        raise ValueError(f"{label}.critic_optimizer parameter mapping is incompatible with the sequential critic")
    named_groups = ["param_names" in group for group in groups]
    if any(named_groups):
        if not all(named_groups) or any(not isinstance(group["param_names"], list) for group in groups):
            raise ValueError(f"{label}.critic_optimizer has incomplete parameter names")
        optimizer_names = [name for group in groups for name in group["param_names"]]
        if optimizer_names != parameter_names:
            raise ValueError(f"{label}.critic_optimizer parameter names do not match critic order")

    moment_state = optimizer["state"].get(parameter_ids[0])
    if moment_state is not None:
        if not isinstance(moment_state, dict):
            raise ValueError(f"{label}.critic_optimizer first-weight state is invalid")
        if moment_state and not {"step", "exp_avg", "exp_avg_sq"}.issubset(moment_state):
            raise ValueError(f"{label}.critic_optimizer first-weight Adam state is incomplete")
        for name, value in moment_state.items():
            if name == "step":
                if isinstance(value, Tensor) and value.numel() != 1:
                    raise ValueError(f"{label}.critic_optimizer Adam step must be scalar")
                continue
            if name in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} and not isinstance(value, Tensor):
                raise ValueError(f"{label}.critic_optimizer {name} must be a tensor")
            if not isinstance(value, Tensor) or value.ndim == 0:
                if name in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
                    raise ValueError(f"{label}.critic_optimizer {name} has an incompatible shape")
                continue
            if value.shape != weight.shape:
                raise ValueError(f"{label}.critic_optimizer {name} must match 0.weight shape {tuple(weight.shape)}")
            if needs_padding:
                moment_state[name] = _pad_columns(value, context_dim)
    if needs_padding:
        critic["0.weight"] = _pad_columns(weight, context_dim)


def migrate_critic_context(
    session_state: dict[str, Any],
    *,
    policy_obs_dim: int,
    context_dim: int = 3,
) -> dict[str, Any]:
    """Copy a session and zero-extend active/stable critics and Adam moments.

    Only an explicit empty publication boundary (pending_rollout=None) is
    supported. Existing context-width checkpoints are validated and copied.
    Versions, actor state, remaining critic weights, counters, candidate
    status, optimizer steps, and optimizer hyperparameters are preserved.
    """
    _positive_dimension(policy_obs_dim, "policy_obs_dim")
    _positive_dimension(context_dim, "context_dim")
    if not isinstance(session_state, dict) or session_state.get("schema_version") != 1:
        raise ValueError("incompatible online PPO session schema")
    if "pending_rollout" not in session_state or session_state["pending_rollout"] is not None:
        raise ValueError("critic context migration requires pending_rollout=None")
    migrated = copy.deepcopy(session_state)
    for name in ("updater", "stable_state"):
        _migrate_updater(
            migrated.get(name), policy_obs_dim=policy_obs_dim,
            context_dim=context_dim, label=name,
        )
    return migrated


__all__ = ["build_critic_observation", "migrate_critic_context"]
