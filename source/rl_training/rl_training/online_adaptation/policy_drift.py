from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .networks import ResidualActor
from .types import TransitionBatch


def calibrate_drift_threshold(
    samples: torch.Tensor, quantile: float = 0.99, multiplier: float = 1.25
) -> float:
    if not isinstance(samples, torch.Tensor) or samples.numel() == 0:
        raise ValueError("samples must be a nonempty tensor")
    if samples.dtype == torch.bool or samples.is_complex():
        raise ValueError("samples must be finite and nonnegative")
    values = samples.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(values).all().item()) or bool((values < 0).any().item()):
        raise ValueError("samples must be finite and nonnegative")
    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not 0.0 <= float(quantile) <= 1.0
    ):
        raise ValueError("quantile must be in [0, 1]")
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, (int, float))
        or not torch.isfinite(torch.tensor(float(multiplier))).item()
        or multiplier <= 0.0
    ):
        raise ValueError("multiplier must be finite and positive")
    return float(torch.quantile(values, float(quantile)) * float(multiplier))


@dataclass(frozen=True)
class PolicyDrift:
    residual_mse: float
    joint_abs_change_mean: float
    joint_abs_change_max: float
    parameter_l2: float
    adaptation_episode: int
    actor_version: int
    update_step: int


class ActorDriftMonitor:
    def __init__(self) -> None:
        self._reference_actor: ResidualActor | None = None
        self._diagnostic_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._adaptation_episode = 0

    @property
    def adaptation_episode(self) -> int:
        return self._adaptation_episode

    def start_episode(self, actor: ResidualActor, *, adaptation_episode: int) -> None:
        if adaptation_episode <= 0:
            raise ValueError("adaptation_episode must be positive")
        self._reference_actor = copy.deepcopy(actor)
        self._reference_actor.eval()
        self._reference_actor.requires_grad_(False)
        self._diagnostic_inputs = None
        self._adaptation_episode = adaptation_episode

    def clear_episode(self, *, adaptation_episode: int) -> None:
        if adaptation_episode != self._adaptation_episode:
            return
        self._reference_actor = None
        self._diagnostic_inputs = None
        self._adaptation_episode = 0

    def measure(
        self,
        actor: ResidualActor,
        batch: TransitionBatch,
        *,
        actor_version: int,
        update_step: int,
        adaptation_episode: int,
    ) -> PolicyDrift:
        if self._reference_actor is None or adaptation_episode != self._adaptation_episode:
            raise RuntimeError("no reference actor for adaptation episode")
        if self._diagnostic_inputs is None:
            self._diagnostic_inputs = (
                batch.history.detach().clone(),
                batch.current_frame.detach().clone(),
                batch.base_action.detach().clone(),
            )

        history, current_frame, base_action = self._diagnostic_inputs
        with torch.inference_mode():
            if any(not torch.isfinite(parameter).all().item() for parameter in actor.parameters()):
                raise FloatingPointError("non-finite policy drift")
            candidate_residual, _, _ = actor.sample(
                history, current_frame, base_action, deterministic=True
            )
            reference_residual, _, _ = self._reference_actor.sample(
                history, current_frame, base_action, deterministic=True
            )
            delta = candidate_residual - reference_residual
            joint_mean_abs = delta.abs().mean(dim=0)

            reference_parameters = dict(self._reference_actor.named_parameters())
            parameter_squared = next(actor.parameters()).new_zeros(())
            for name, parameter in actor.named_parameters():
                parameter_squared += (parameter - reference_parameters[name]).square().sum()

            values = (
                delta.square().mean(),
                joint_mean_abs.mean(),
                joint_mean_abs.max(),
                parameter_squared.sqrt(),
            )

        if not all(torch.isfinite(value).item() for value in values):
            raise FloatingPointError("non-finite policy drift")
        return PolicyDrift(
            residual_mse=float(values[0]),
            joint_abs_change_mean=float(values[1]),
            joint_abs_change_max=float(values[2]),
            parameter_l2=float(values[3]),
            adaptation_episode=adaptation_episode,
            actor_version=actor_version,
            update_step=update_step,
        )
