"""Explicit low-speed Source rewards and actor initialization, without simulator imports.

The reward baseline is model_1998's saved env.yaml (SHA256
860a7cb7f8c0949cc431b1f8738960b1849a5970aa2b906aac6d953de9902a31).
Only the height reference changes: a local ground scan replaces world height.
"""
from __future__ import annotations

import math
from pathlib import Path

from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES

SOURCE_ADAPTATION_PROFILE = "lite3_source_lowspeed_v1"
_REWARDS = "rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards:"


def source_adaptation_reward_specs() -> dict:
    """Return independent, serializable definitions; do not inherit Rough weights."""
    def term(function, weight, **params):
        return dict(func=function if ":" in function else _REWARDS + function, weight=weight, params=params)

    def joints():
        return dict(name="robot", joint_names=list(LITE3_POLICY_JOINT_NAMES))

    def feet(name):
        return dict(name=name, body_names=[".*_FOOT"])

    return {
        "track_lin_vel_xy_exp": term("track_lin_vel_xy_exp", 4.0, command_name="base_velocity", std=0.25),
        "track_ang_vel_z_exp": term("track_ang_vel_z_exp", 1.5, command_name="base_velocity", std=0.5),
        "lin_vel_z_l2": term("lin_vel_z_l2", -2.0),
        "ang_vel_xy_l2": term("ang_vel_xy_l2", -0.25),
        "flat_orientation_l2": term("flat_orientation_l2", -10.0),
        "base_height_l2": term("base_height_l2", -25.0, target_height=0.375,
                               asset_cfg=dict(name="robot", body_names=["TORSO"]),
                               sensor_cfg=dict(name="height_scanner_base")),
        "joint_torques_l2": term("joint_torques_l2", -2.5e-4, asset_cfg=joints()),
        "joint_acc_l2": term("isaaclab.envs.mdp.rewards:joint_acc_l2", -1.0e-8, asset_cfg=joints()),
        "feet_air_time": term("feet_air_time_including_ang_z", 2.0, command_name="base_velocity",
                              threshold=0.35, command_threshold=0.05, sensor_cfg=feet("contact_forces")),
        "feet_air_time_variance": term("feet_air_time_variance_penalty", -5.0, sensor_cfg=feet("contact_forces")),
        "undesired_contacts": term("undesired_contacts", -1.0, threshold=0.1,
                                   sensor_cfg=dict(name="contact_forces", body_names=[".*_(THIGH|SHANK)"])),
        "feet_slide": term("feet_slide", -0.05, sensor_cfg=feet("contact_forces"), asset_cfg=feet("robot")),
        "foot_impact_velocity": term("foot_impact_velocity", -0.5, speed_threshold=0.1,
                                     sensor_cfg=feet("contact_forces"), asset_cfg=feet("robot")),
        "contact_forces": term("contact_forces", -0.01, threshold=100.0, sensor_cfg=feet("contact_forces")),
        "action_rate_l2": term("action_rate_l2", -0.05),
        "joint_pos_limits": term("isaaclab.envs.mdp.rewards:joint_pos_limits", -5.0, asset_cfg=joints()),
        "stand_still": term("isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards:stand_still_joint_deviation_l1",
                            -0.2, command_name="base_velocity", command_threshold=0.1, asset_cfg=joints()),
    }


def build_source_adaptation_rewards(reward_term_type, scene_entity_type) -> dict:
    """Construct manager terms using IsaacLab types supplied by the config module."""
    result = {}
    for name, spec in source_adaptation_reward_specs().items():
        params = {key: scene_entity_type(**value) if key.endswith("_cfg") else value
                  for key, value in spec["params"].items()}
        result[name] = reward_term_type(func=spec["func"], weight=spec["weight"], params=params)
    return result


def resolve_reward_gait_level(cfg, legacy_level: float) -> float:
    """A configured environment owns its gate; old environments retain their curriculum."""
    level = getattr(cfg, "reward_gait_level", legacy_level)
    if type(level) not in (int, float) or not math.isfinite(level) or not 0.0 <= level <= 1.0:
        raise ValueError("reward_gait_level must be a finite scalar in [0, 1]")
    return float(level)


def validate_source_adaptation_config(cfg) -> dict:
    """Audit configured terms, including manager-resolved entity configurations."""
    if getattr(cfg, "source_reward_profile", None) != SOURCE_ADAPTATION_PROFILE:
        raise ValueError("unexpected Source adaptation reward profile")
    if getattr(cfg, "reward_gait_level", None) is None or resolve_reward_gait_level(cfg, 0.0) != 1.0:
        raise ValueError("Source low-speed v1 requires explicit reward_gait_level=1.0")
    expected = source_adaptation_reward_specs()
    if not isinstance(cfg.rewards, dict) or set(cfg.rewards) != set(expected):
        raise ValueError("Source adaptation active reward set differs from the profile")
    for name, spec in expected.items():
        term = cfg.rewards[name]
        function = term.func
        if callable(function):
            function = function.__module__ + ":" + function.__qualname__
        if function != spec["func"] or term.weight != spec["weight"] or set(term.params) != set(spec["params"]):
            raise ValueError("Source adaptation reward changed: " + name)
        for key, value in spec["params"].items():
            actual = term.params[key]
            if key.endswith("_cfg"):
                if any(getattr(actual, field, None) != field_value for field, field_value in value.items()):
                    raise ValueError("Source adaptation entity changed: " + name + "." + key)
            elif actual != value:
                raise ValueError("Source adaptation parameter changed: " + name + "." + key)
    return dict(profile=SOURCE_ADAPTATION_PROFILE, reward_gait_level=1.0, rewards=expected,
                height_reference="height_scanner_base", action_std_initialization="checkpoint")


def initialize_source_adaptation_actor(actor, checkpoint: Path, cfg, **kwargs) -> dict:
    """Load all actor tensors, including its learned std; do not restore a critic or optimizer."""
    from .source_training import load_source_actor_only

    validate_source_adaptation_config(cfg)
    if "action_std" in kwargs:
        raise ValueError("Source low-speed v1 preserves checkpoint action std")
    report = load_source_actor_only(actor, checkpoint, action_std=None, **kwargs)
    return dict(report, reward_profile=SOURCE_ADAPTATION_PROFILE)
