"""Small, auditable contracts for starting Source PPO runs.

These helpers deliberately do not construct Isaac Lab environments or call the
generic resume path.  They cover the two pieces that must be unambiguous before
the first rollout: actor-only initialization and transition accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
from typing import Any

import torch

from .frozen_actor import CHECKPOINT_SHA256, FrozenSourceActor


@dataclass(frozen=True)
class TransitionBudget:
    """An exact rollout budget; the runner may only complete whole updates."""

    num_envs: int
    rollout_length: int
    max_transitions: int

    def __post_init__(self) -> None:
        for name, value in (
            ("num_envs", self.num_envs),
            ("rollout_length", self.rollout_length),
            ("max_transitions", self.max_transitions),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def transitions_per_update(self) -> int:
        return self.num_envs * self.rollout_length

    @property
    def max_updates(self) -> int:
        return self.max_transitions // self.transitions_per_update

    @property
    def committed_transitions(self) -> int:
        return self.max_updates * self.transitions_per_update

    @property
    def unused_transitions(self) -> int:
        return self.max_transitions - self.committed_transitions

    def assert_update_allowed(self, completed_updates: int) -> None:
        """Reject an update that would exceed the registered hard ceiling."""
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError("completed_updates must be a nonnegative integer")
        if completed_updates >= self.max_updates:
            raise RuntimeError(
                "transition budget exhausted; do not start a partial or extra rollout"
            )

    def transitions_after(self, completed_updates: int) -> int:
        if type(completed_updates) is not int or not 0 <= completed_updates <= self.max_updates:
            raise ValueError("completed_updates is outside this budget")
        return completed_updates * self.transitions_per_update


def load_source_actor_only(
    actor: torch.nn.Module,
    checkpoint: Path,
    *,
    action_std: float | None = 1.0,
    expected_sha256: str = CHECKPOINT_SHA256,
) -> dict[str, Any]:
    """Load only a registered actor state; None preserves its learned exploration std.

    ``actor`` is the freshly constructed Source PPO actor.  The function never
    receives a critic, optimizer, or iteration counter, so those states cannot
    be restored accidentally.  The installed RSL-RL actor exposes the same
    ``state_dict`` keys as the pinned checkpoint.
    """
    if not isinstance(checkpoint, Path):
        checkpoint = Path(checkpoint)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected_sha256 must be a 64-character digest")
    if action_std is not None and (type(action_std) not in (int, float) or not 0.0 < float(action_std) < float("inf")):
        raise ValueError("action_std must be finite and positive")
    payload_bytes = checkpoint.read_bytes()
    observed_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("Source checkpoint hash differs from the registered digest")
    payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"
    }:
        raise ValueError("Source checkpoint schema is not the audited RSL-RL checkpoint")
    state = payload["actor_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint actor_state_dict is not a mapping")
    # Validate every source and destination tensor before mutating the live actor.
    FrozenSourceActor(state)
    layers = list(getattr(actor, "mlp", torch.nn.Sequential()))
    if (
        len(layers) != 7
        or any(not isinstance(layers[i], torch.nn.Linear) for i in (0, 2, 4, 6))
        or any(not isinstance(layers[i], torch.nn.ELU) or layers[i].alpha != 1.0 for i in (1, 3, 5))
        or getattr(actor, "obs_normalization", None) is not False
        or getattr(actor, "obs_groups", None) != ["policy"]
    ):
        raise ValueError("fresh actor must use the unnormalized policy-only ELU MLP")
    destination = actor.state_dict()
    if set(destination) != set(state) or any(
        destination[key].shape != value.shape or destination[key].dtype != value.dtype
        for key, value in state.items()
    ):
        raise ValueError("fresh actor schema differs from the registered actor")
    distribution = getattr(actor, "distribution", None)
    log_std = getattr(distribution, "log_std_param", None)
    if log_std is None or not isinstance(log_std, torch.Tensor):
        raise ValueError("fresh actor does not expose log_std_param")
    if not log_std.requires_grad or not any(parameter is log_std for parameter in actor.parameters()):
        raise ValueError("actor log_std_param must be a trainable parameter")
    initialized = dict(state)
    if action_std is not None:
        initialized["distribution.log_std_param"] = torch.full_like(
            state["distribution.log_std_param"], math.log(action_std)
        )
    actual_std = initialized["distribution.log_std_param"].exp()
    if not bool(torch.isfinite(actual_std).all()) or not bool((actual_std > 0).all()):
        raise FloatingPointError("actor exploration std must be finite and positive in float32")
    actor.load_state_dict(initialized, strict=True)
    return {
        "checkpoint_sha256": observed_sha256,
        "checkpoint_iteration_ignored": payload["iter"],
        "loaded": "actor_state_dict_only",
        "critic_restored": False,
        "optimizer_restored": False,
        "iteration_restored": False,
        "action_std": actual_std.cpu().tolist() if action_std is None else float(action_std),
        "action_std_mode": "checkpoint" if action_std is None else "constant",
    }


__all__ = ["TransitionBudget", "load_source_actor_only"]
