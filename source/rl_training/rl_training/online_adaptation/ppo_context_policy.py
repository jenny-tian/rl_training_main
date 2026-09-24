"""A zero-output context head around the existing 45D Lite3 PPO actor."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from .rsl_rl_compat import RslMLPPolicyAdapter


CONTEXT_POLICY_OBSERVATION_SCHEMA = "base45_context3_gated_residual_v1"
HEIGHT_SCAN_POLICY_OBSERVATION_SCHEMA = "base45_heightscan10_gated_residual_v2"
_HEAD_NAMES = ("context_A.weight", "context_B.weight")
_SCAN_HEAD_NAMES = ("scan_A.weight", "scan_B.weight")


class ContextGatedPolicy(nn.Module):
    """Keep the base actor unchanged and gate three low-rank action branches.

    Input columns 0:45 are the original policy observation; columns 45:48
    are caller-provided context. The wrapper does not obtain world pose or
    any other environment state. Gaussian scale belongs only to base_policy.
    """

    base_observation_dim = 45
    context_dim = 3
    action_dim = 12

    def __init__(self, base_policy: RslMLPPolicyAdapter, *, rank: int = 2) -> None:
        super().__init__()
        if not isinstance(base_policy, RslMLPPolicyAdapter):
            raise TypeError("base_policy must be an RslMLPPolicyAdapter")
        if base_policy.action_transform is not clip_lite3_policy_action:
            raise ValueError("base_policy must already use the canonical Lite3 action transform")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= self.base_observation_dim:
            raise ValueError("rank must be a positive integer <= 45")
        std_keys = [key for key in base_policy.state_dict() if key.rsplit(".", 1)[-1] in {"std_param", "log_std_param"}]
        if len(std_keys) != 1:
            raise ValueError("base_policy must own exactly one Gaussian standard-deviation parameter")
        reference = next((p for p in base_policy.parameters() if p.ndim == 2 and p.shape[1] == 45), None)
        if reference is None or not reference.is_floating_point():
            raise ValueError("base_policy must expose a floating-point 45D input layer")
        # Registration order is part of the optimizer migration contract.
        self.base_policy = base_policy
        self.context_A = nn.Linear(45, rank, bias=False, device=reference.device, dtype=reference.dtype)
        self.context_B = nn.Linear(rank, 3 * 12, bias=False, device=reference.device, dtype=reference.dtype)
        self.rank = rank
        nn.init.normal_(self.context_A.weight, mean=0.0, std=1.0 / math.sqrt(45))
        nn.init.zeros_(self.context_B.weight)

    def _split_observations(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(observations, Tensor) or observations.ndim != 2 or observations.shape[1] != 48:
            raise ValueError("context policy observations must have shape [N, 48] = [base45, context3]")
        if not observations.is_floating_point() or not bool(torch.isfinite(observations).all().item()):
            raise ValueError("context policy observations must contain finite real floating-point values")
        if observations.device != self.context_A.weight.device or observations.dtype != self.context_A.weight.dtype:
            raise ValueError("context policy observations must match the policy dtype and device")
        return observations[:, :45], observations[:, 45:]

    def _gated_mean(self, base_observations: Tensor, context: Tensor, base_mean: Tensor) -> Tensor:
        if base_mean.shape != (base_observations.shape[0], 12):
            raise ValueError("base_policy must return [N, 12] mean actions")
        branches = self.context_B(self.context_A(base_observations)).reshape(-1, 3, 12)
        correction = (branches * context.unsqueeze(-1)).sum(dim=1)
        return clip_lite3_policy_action(base_mean + correction)

    def forward(self, observations: Tensor) -> Tensor:
        base_observations, context = self._split_observations(observations)
        return self._gated_mean(base_observations, context, self.base_policy.mean_action(base_observations))

    def mean_action(self, observations: Tensor) -> Tensor:
        return self.forward(observations)

    def get_distribution(self, observations: Tensor) -> Normal:
        base_observations, context = self._split_observations(observations)
        base_distribution = self.base_policy.get_distribution(base_observations)
        if not isinstance(base_distribution, Normal):
            raise TypeError("ContextGatedPolicy requires the base actor's diagonal Gaussian Normal")
        mean = self._gated_mean(base_observations, context, base_distribution.mean)
        return Normal(mean, base_distribution.scale, validate_args=base_distribution._validate_args)

    def evaluate_actions(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        distribution = self.get_distribution(observations)
        return distribution.log_prob(actions).sum(dim=-1), distribution.entropy().sum(dim=-1)


class HeightScanGatedPolicy(nn.Module):
    """Zero-output terrain adapter that preserves the frozen 45D actor.

    The ten scan summary features are gated through a low-rank branch, so the
    initial policy is exactly the proprio-only actor and only the new branch
    needs online adaptation.
    """

    base_observation_dim = 45
    scan_dim = 10
    action_dim = 12

    def __init__(self, base_policy: RslMLPPolicyAdapter, *, rank: int = 2) -> None:
        super().__init__()
        if not isinstance(base_policy, RslMLPPolicyAdapter):
            raise TypeError("base_policy must be an RslMLPPolicyAdapter")
        if base_policy.action_transform is not clip_lite3_policy_action:
            raise ValueError("base_policy must already use the canonical Lite3 action transform")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= self.scan_dim:
            raise ValueError("rank must be a positive integer <= 10")
        reference = next((p for p in base_policy.parameters() if p.ndim == 2 and p.shape[1] == 45), None)
        if reference is None or not reference.is_floating_point():
            raise ValueError("base_policy must expose a floating-point 45D input layer")
        self.base_policy = base_policy
        self.scan_A = nn.Linear(self.scan_dim, rank, bias=False, device=reference.device, dtype=reference.dtype)
        self.scan_B = nn.Linear(rank, self.action_dim, bias=False, device=reference.device, dtype=reference.dtype)
        self.rank = rank
        nn.init.normal_(self.scan_A.weight, mean=0.0, std=1.0 / math.sqrt(self.scan_dim))
        nn.init.zeros_(self.scan_B.weight)

    def _split(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(observations, Tensor) or observations.ndim != 2 or observations.shape[1] != 55:
            raise ValueError("height-scan policy observations must have shape [N, 55] = [base45, scan10]")
        if (
            not observations.is_floating_point()
            or observations.device != self.scan_A.weight.device
            or observations.dtype != self.scan_A.weight.dtype
            or not bool(torch.isfinite(observations).all().item())
        ):
            raise ValueError("height-scan observations must be finite and match the policy dtype/device")
        return observations[:, :45], observations[:, 45:]

    def _mean(self, base: Tensor, scan: Tensor, base_mean: Tensor) -> Tensor:
        correction = self.scan_B(self.scan_A(scan))
        return clip_lite3_policy_action(base_mean + correction)

    def mean_action(self, observations: Tensor) -> Tensor:
        base, scan = self._split(observations)
        return self._mean(base, scan, self.base_policy.mean_action(base))

    def forward(self, observations: Tensor) -> Tensor:
        return self.mean_action(observations)

    def get_distribution(self, observations: Tensor) -> Normal:
        base, scan = self._split(observations)
        dist = self.base_policy.get_distribution(base)
        return Normal(self._mean(base, scan, dist.mean), dist.scale, validate_args=dist._validate_args)

    def evaluate_actions(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        distribution = self.get_distribution(observations)
        return distribution.log_prob(actions).sum(dim=-1), distribution.entropy().sum(dim=-1)


def _validate_policy_state(actual: Any, template: dict[str, Tensor], *, label: str) -> None:
    if not isinstance(actual, dict) or set(actual) != set(template):
        raise ValueError(f"{label} policy state keys do not match the target policy")
    for name, reference in template.items():
        value = actual[name]
        if (
            not isinstance(value, Tensor) or value.shape != reference.shape or value.dtype != reference.dtype
            or value.is_complex() or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(f"{label} policy tensor {name} has an incompatible shape/dtype or non-finite value")


def _target_trainables(target: ContextGatedPolicy) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    base = [(name, value) for name, value in target.base_policy.named_parameters() if value.requires_grad]
    if not base or any(name.rsplit(".", 1)[-1] not in {"lora_A", "lora_B"} for name, _ in base):
        raise ValueError("legacy base trainables must consist only of the existing LoRA parameters")
    full = [(name, value) for name, value in target.named_parameters() if value.requires_grad]
    expected = ["base_policy." + name for name, _ in base] + list(_HEAD_NAMES)
    if [name for name, _ in full] != expected:
        raise ValueError("target trainable order must be base LoRA parameters followed by context_A/context_B")
    return base, full


def _validate_actor_optimizer(
    optimizer: Any,
    policy_state: dict[str, Tensor],
    named_parameters: list[tuple[str, Tensor]],
    *,
    label: str,
) -> list[int]:
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError(f"{label} actor optimizer must contain an Adam state dictionary")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
        raise ValueError(f"{label} actor optimizer must have one unambiguous parameter group")
    group = groups[0]
    ids = group.get("params")
    names = [name for name, _ in named_parameters]
    if (
        not isinstance(ids, list) or len(ids) != len(names)
        or any(isinstance(key, bool) or not isinstance(key, int) or key < 0 for key in ids)
        or len(set(ids)) != len(ids)
        or not set(optimizer["state"]).issubset(ids)
    ):
        raise ValueError(f"{label} actor optimizer parameter IDs/count do not match the target trainables")
    if [name for name in policy_state if name in set(names)] != names:
        raise ValueError(f"{label} policy parameter order does not match the target trainables")
    if "param_names" in group and group["param_names"] != names:
        raise ValueError(f"{label} actor optimizer parameter names do not match the target order")
    for key, (name, parameter) in zip(ids, named_parameters):
        if key not in optimizer["state"]:
            continue  # Adam initializes moments only after a parameter first receives a gradient.
        moments = optimizer["state"][key]
        if not isinstance(moments, dict):
            raise ValueError(f"{label} Adam state for {name} must be a dictionary")
        if not moments:
            continue
        required = {"step", "exp_avg", "exp_avg_sq"}
        if group.get("amsgrad", False):
            required.add("max_exp_avg_sq")
        if not required.issubset(moments) or set(moments) - {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
            raise ValueError(f"{label} Adam state for {name} is incomplete or not a supported Adam state")
        step = moments["step"]
        if isinstance(step, Tensor):
            if step.numel() != 1 or step.is_complex() or not bool(torch.isfinite(step).all().item()) or step.item() < 0:
                raise ValueError(f"{label} Adam step for {name} must be a finite non-negative scalar")
        elif isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step < 0:
            raise ValueError(f"{label} Adam step for {name} must be a finite non-negative scalar")
        for moment_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if moment_name not in moments:
                continue
            value = moments[moment_name]
            if (
                not isinstance(value, Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype
                or not bool(torch.isfinite(value).all().item())
                or (moment_name != "exp_avg" and bool((value < 0).any().item()))
            ):
                raise ValueError(f"{label} Adam {moment_name} shape/dtype/value does not match {name}")
    return ids


def migrate_context_policy(
    session_state: dict[str, Any],
    *,
    target_policy: ContextGatedPolicy,
) -> dict[str, Any]:
    """Copy a session and wrap legacy actor states without changing outputs.

    Both branches receive the same new A and zero B when they are legacy.
    Already wrapped branches retain their saved, possibly trained heads.
    Existing Adam IDs/moments/steps and learning rates are preserved; the two
    new parameters are appended with lazy Adam state. Critic state is opaque.
    """
    if not isinstance(target_policy, ContextGatedPolicy):
        raise TypeError("target_policy must be a ContextGatedPolicy")
    if not isinstance(session_state, dict) or session_state.get("schema_version") != 1:
        raise ValueError("incompatible online PPO session schema")
    if "pending_rollout" not in session_state or session_state["pending_rollout"] is not None:
        raise ValueError("context policy migration requires pending_rollout=None")
    base_trainables, full_trainables = _target_trainables(target_policy)
    base_template = target_policy.base_policy.state_dict()
    wrapped_template = target_policy.state_dict()
    head_template = {
        "context_A.weight": target_policy.context_A.weight.detach().clone(),
        "context_B.weight": torch.zeros_like(target_policy.context_B.weight.detach()),
    }
    if not all(bool(torch.isfinite(value).all().item()) for value in head_template.values()):
        raise ValueError("new context head must contain finite values")
    result = copy.deepcopy(session_state)
    for branch_name in ("updater", "stable_state"):
        branch = result.get(branch_name)
        if not isinstance(branch, dict) or branch.get("schema_version") != 1:
            raise ValueError(f"{branch_name} has an incompatible updater schema")
        policy_state = branch.get("policy")
        if not isinstance(policy_state, dict):
            raise ValueError(f"{branch_name} policy state must be a dictionary")
        if set(policy_state) == set(wrapped_template):
            _validate_policy_state(policy_state, wrapped_template, label=branch_name)
            _validate_actor_optimizer(
                branch.get("actor_optimizer"), policy_state, full_trainables, label=branch_name,
            )
            continue
        _validate_policy_state(policy_state, base_template, label=branch_name)
        optimizer = branch.get("actor_optimizer")
        ids = _validate_actor_optimizer(optimizer, policy_state, base_trainables, label=branch_name)
        branch["policy"] = {"base_policy." + name: value for name, value in policy_state.items()}
        branch["policy"].update({name: value.clone() for name, value in head_template.items()})
        new_ids = [max(ids) + 1, max(ids) + 2]
        optimizer["param_groups"][0]["params"] = ids + new_ids
        if "param_names" in optimizer["param_groups"][0]:
            optimizer["param_groups"][0]["param_names"] = [name for name, _ in full_trainables]
        _validate_actor_optimizer(optimizer, branch["policy"], full_trainables, label=branch_name)
    return result


def migrate_height_scan_policy(
    session_state: dict[str, Any],
    *,
    target_policy: HeightScanGatedPolicy,
) -> dict[str, Any]:
    """Wrap a legacy/base LoRA session with a zero-output scan adapter."""
    if not isinstance(target_policy, HeightScanGatedPolicy):
        raise TypeError("target_policy must be a HeightScanGatedPolicy")
    if not isinstance(session_state, dict) or session_state.get("schema_version") != 1:
        raise ValueError("incompatible online PPO session schema")
    if session_state.get("pending_rollout") is not None:
        raise ValueError("height-scan migration requires pending_rollout=None")
    full_template = target_policy.state_dict()
    base_template = target_policy.base_policy.state_dict()
    base_trainables = [(name, value) for name, value in target_policy.base_policy.named_parameters() if value.requires_grad]
    full_trainables = [(name, value) for name, value in target_policy.named_parameters() if value.requires_grad]
    expected = ["base_policy." + name for name, _ in base_trainables] + list(_SCAN_HEAD_NAMES)
    if [name for name, _ in full_trainables] != expected:
        raise ValueError("height-scan trainable order is incompatible")
    result = copy.deepcopy(session_state)
    for branch_name in ("updater", "stable_state"):
        branch = result.get(branch_name)
        policy_state = branch.get("policy") if isinstance(branch, dict) else None
        optimizer = branch.get("actor_optimizer") if isinstance(branch, dict) else None
        if set(policy_state or {}) == set(full_template):
            _validate_policy_state(policy_state, full_template, label=branch_name)
            _validate_actor_optimizer(optimizer, policy_state, full_trainables, label=branch_name)
            continue
        _validate_policy_state(policy_state, base_template, label=branch_name)
        ids = _validate_actor_optimizer(optimizer, policy_state, base_trainables, label=branch_name)
        branch["policy"] = {"base_policy." + name: value for name, value in policy_state.items()}
        branch["policy"].update({
            "scan_A.weight": target_policy.scan_A.weight.detach().clone(),
            "scan_B.weight": torch.zeros_like(target_policy.scan_B.weight.detach()),
        })
        new_ids = [max(ids) + 1, max(ids) + 2]
        optimizer["param_groups"][0]["params"] = ids + new_ids
        if "param_names" in optimizer["param_groups"][0]:
            optimizer["param_groups"][0]["param_names"] = expected
        _validate_actor_optimizer(optimizer, branch["policy"], full_trainables, label=branch_name)
    return result


__all__ = ["CONTEXT_POLICY_OBSERVATION_SCHEMA", "HEIGHT_SCAN_POLICY_OBSERVATION_SCHEMA", "ContextGatedPolicy", "HeightScanGatedPolicy", "migrate_context_policy", "migrate_height_scan_policy"]
