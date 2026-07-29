import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_training.online_adaptation import terrain_guards
from rl_training.online_adaptation.terrain_guards import (
    assert_heldout_terrains_disabled,
    deterministic_joint_reset_params,
    deterministic_root_reset_params,
)


ROOT = Path(__file__).resolve().parents[2]


def terrain(boxes=0.0, stairs=0.0, stairs_inv=0.0):
    sub = {
        "boxes": SimpleNamespace(proportion=boxes),
        "pyramid_stairs": SimpleNamespace(proportion=stairs),
        "pyramid_stairs_inv": SimpleNamespace(proportion=stairs_inv),
    }
    return SimpleNamespace(sub_terrains=sub)


def test_guard_accepts_zero_heldout_proportions():
    assert_heldout_terrains_disabled(terrain())


def test_guard_rejects_any_heldout_probability():
    with pytest.raises(ValueError, match="held-out terrain leakage"):
        assert_heldout_terrains_disabled(terrain(boxes=0.01))


def test_plane_only_guard_accepts_plane_without_generator():
    terrain_cfg = SimpleNamespace(terrain_type="plane", terrain_generator=None)

    terrain_guards.assert_plane_only_terrain(terrain_cfg)


@pytest.mark.parametrize(
    ("terrain_type", "terrain_generator"),
    [
        ("generator", None),
        ("plane", object()),
        ("generator", object()),
    ],
)
def test_plane_only_guard_rejects_non_strict_configuration(terrain_type, terrain_generator):
    terrain_cfg = SimpleNamespace(terrain_type=terrain_type, terrain_generator=terrain_generator)

    with pytest.raises(ValueError, match="strict plane-only"):
        terrain_guards.assert_plane_only_terrain(terrain_cfg)


@pytest.mark.parametrize(
    "task",
    [
        "OnlineAdapt-Deeprobotics-Lite3-v0",
        "HeldoutBoxes-Deeprobotics-Lite3-v0",
        "Rough-Deeprobotics-Lite3-v0",
        "FlatOnline-Deeprobotics-Lite3-v0",
    ],
)
def test_strict_flat_baseline_task_rejects_every_other_lite3_task(task):
    assert terrain_guards.is_strict_flat_baseline_task(task) is False


def test_strict_flat_baseline_task_accepts_exact_registered_task():
    assert terrain_guards.is_strict_flat_baseline_task("Flat-Deeprobotics-Lite3-v0") is True


def test_flat_baseline_train_guards_terrain_immediately_before_gym_make():
    source = (ROOT / "scripts/reinforcement_learning/rsl_rl/train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    make_index = next(
        index
        for index, statement in enumerate(main.body)
        if isinstance(statement, ast.Assign) and "gym.make" in ast.unparse(statement.value)
    )

    guard = main.body[make_index - 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == "is_strict_flat_baseline_task(args_cli.task)"
    assert len(guard.body) == 1
    assert ast.unparse(guard.body[0]) == "assert_plane_only_terrain(env_cfg.scene.terrain)"


def test_deterministic_root_reset_keeps_spawn_at_terrain_origin():
    params = deterministic_root_reset_params()

    assert params["pose_range"] == {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    assert all(bounds == (0.0, 0.0) for bounds in params["velocity_range"].values())


def test_deterministic_joint_reset_preserves_default_standing_pose():
    assert deterministic_joint_reset_params() == {
        "position_range": (1.0, 1.0),
        "velocity_range": (0.0, 0.0),
    }


@pytest.mark.isaaclab
@pytest.mark.parametrize("config_module, config_name", [
    (
        "rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.online_adapt_env_cfg",
        "DeeproboticsLite3OnlineAdaptEnvCfg",
    ),
    (
        "rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.heldout_boxes_env_cfg",
        "DeeproboticsLite3HeldoutBoxesEnvCfg",
    ),
])
def test_subclass_configs_remove_zero_weight_rewards_before_manager_resolution(config_module, config_name):
    pytest.importorskip("pxr")
    module = __import__(config_module, fromlist=[config_name])
    config_type = getattr(module, config_name)
    config = config_type()

    assert config.rewards.body_lin_acc_l2 is None
    assert config.rewards.base_height_l2 is not None
