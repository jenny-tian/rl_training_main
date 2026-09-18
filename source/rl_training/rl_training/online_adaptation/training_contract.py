"""Prevent resumed critics and replay from crossing reward/objective changes."""

from dataclasses import asdict
from typing import Any

from .reward import RewardWeights
from .sac import SACConfig


def training_contract(sac: SACConfig, reward: RewardWeights, *, target_semantics: str) -> dict[str, Any]:
    return {
        "version": 1,
        "reward_semantics": "contact_swing_lateral_current_pose_completion_v2",
        "entropy_coordinates": "normalized_tanh_action_v1",
        "sac": asdict(sac),
        "reward": asdict(reward),
        "target_semantics": target_semantics,
    }


def validate_training_contract(saved: object, expected: dict[str, Any]) -> None:
    if saved != expected:
        raise ValueError(
            "training contract differs or is missing: reward/SAC changes require "
            "a new run and compatible clean initializer; historical checkpoints "
            "remain usable for frozen evaluation"
        )
