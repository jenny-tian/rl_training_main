"""Prevent resumed critics and replay from crossing reward/objective changes."""

from dataclasses import asdict
from typing import Any

from .reward import RewardWeights
from .sac import SACConfig


def training_contract(
    sac: SACConfig, reward: RewardWeights, *, target_semantics: str,
    risk_gate: dict[str, Any] | None = None,
    actor_architecture: str = "direct_residual_v1",
    latent_code_dim: int | None = None,
    latent_residual_fraction: float | None = None,
) -> dict[str, Any]:
    contract = {
        "version": 1,
        "reward_semantics": "contact_swing_lateral_current_pose_completion_v2",
        "entropy_coordinates": (
            "normalized_tanh_latent_code_v1"
            if actor_architecture.startswith("latent_residual_")
            else "normalized_tanh_action_v1"
        ),
        "sac": asdict(sac),
        "reward": asdict(reward),
        "target_semantics": target_semantics,
    }
    if actor_architecture != "direct_residual_v1" or latent_code_dim is not None:
        contract["actor_architecture"] = actor_architecture
        contract["latent_code_dim"] = latent_code_dim
    if actor_architecture == "latent_residual_v1":
        contract["latent_decoder"] = "fixed_orthogonal_initialization_v1"
        contract["latent_residual_fraction"] = latent_residual_fraction
    elif actor_architecture == "latent_residual_teacher_decoder_v1":
        contract["latent_decoder"] = "teacher_svd_decoder_v1"
        contract["latent_residual_fraction"] = latent_residual_fraction
    # Preserve old disabled-gate runs, but never mix enabled replay/critics with
    # a different state-to-action transformation on learning resume.
    if risk_gate is not None:
        contract["risk_gate"] = risk_gate
    return contract


def validate_training_contract(saved: object, expected: dict[str, Any]) -> None:
    if saved != expected:
        raise ValueError(
            "training contract differs or is missing: reward/SAC/risk-gate changes require "
            "a new run and compatible clean initializer; historical checkpoints "
            "remain usable for frozen evaluation"
        )
