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


def test_guard_accepts_generator_without_heldout_entries():
    flat_only = SimpleNamespace(
        sub_terrains={"flat_online": SimpleNamespace(proportion=1.0)}
    )
    assert_heldout_terrains_disabled(flat_only)


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


def _flat_online_deployment_env_cfg():
    flat_online = SimpleNamespace(
        proportion=1.0,
        seed=0,
        step_height=0.0,
        rough_amplitude=0.0,
    )
    generator = SimpleNamespace(
        seed=0,
        curriculum=False,
        size=(20.0, 4.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        use_cache=False,
        sub_terrains={"flat_online": flat_online},
    )
    return SimpleNamespace(
        scene=SimpleNamespace(
            num_envs=1,
            terrain=SimpleNamespace(
                terrain_type="generator",
                terrain_generator=generator,
                max_init_terrain_level=0,
            ),
        ),
        commands=SimpleNamespace(
            base_velocity=SimpleNamespace(
                resampling_time_range=(1.0e9, 1.0e9),
                rel_standing_envs=0.0,
                rel_heading_envs=0.0,
                heading_command=False,
                command_acceleration_limits=(0.5, 0.5, 0.5),
                ranges=SimpleNamespace(
                    lin_vel_x=(0.5, 0.5),
                    lin_vel_y=(0.0, 0.0),
                    ang_vel_z=(0.0, 0.0),
                    heading=(0.0, 0.0),
                ),
            )
        ),
        observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=False)),
        curriculum=SimpleNamespace(
            terrain_levels=None,
            gait_level=None,
            command_levels=None,
        ),
        events=SimpleNamespace(
            randomize_apply_external_force_torque=None,
            randomize_push_robot=None,
            randomize_rigid_body_material=SimpleNamespace(
                func=SimpleNamespace(__name__="randomize_rigid_body_material"),
                params={
                    "asset_cfg": SimpleNamespace(name="robot", body_names=".*"),
                    **terrain_guards.deterministic_rigid_body_material_params(),
                },
            ),
            randomize_rigid_body_mass=None,
            randomize_rigid_body_mass_base=None,
            randomize_rigid_body_inertia=None,
            randomize_com_positions=None,
            randomize_actuator_gains=None,
            randomize_reset_base=SimpleNamespace(params=deterministic_root_reset_params()),
            randomize_reset_joints=SimpleNamespace(
                params=deterministic_joint_reset_params(),
                func=SimpleNamespace(__name__="reset_joints_by_scale"),
            ),
        ),
        terminations=SimpleNamespace(
            time_out=SimpleNamespace(),
            terrain_out_of_bounds=SimpleNamespace(params={"distance_buffer": 0.5}),
            illegal_contact=SimpleNamespace(),
        ),
    )


def test_flat_online_deployment_guard_accepts_deterministic_generator_configuration():
    guard = getattr(terrain_guards, "assert_flat_online_deployment_config", None)

    assert guard is not None, "FlatOnline deployment guard is missing"
    guard(_flat_online_deployment_env_cfg())


def test_flat_online_deployment_guard_rejects_non_fixed_forward_command():
    cfg = _flat_online_deployment_env_cfg()
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)

    with pytest.raises(ValueError, match="lin_vel_x"):
        terrain_guards.assert_flat_online_deployment_config(cfg)


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("static_friction_range", (0.64, 0.65)),
        ("dynamic_friction_range", (0.50, 0.55)),
        ("restitution_range", (0.0, 0.1)),
        ("num_buckets", 2),
    ],
)
def test_flat_online_deployment_guard_rejects_changed_material(key, invalid):
    cfg = _flat_online_deployment_env_cfg()
    cfg.events.randomize_rigid_body_material.params[key] = invalid

    with pytest.raises(ValueError, match=key):
        terrain_guards.assert_flat_online_deployment_config(cfg)


def test_flat_online_deployment_guard_rejects_wrong_material_function():
    cfg = _flat_online_deployment_env_cfg()
    cfg.events.randomize_rigid_body_material.func.__name__ = "wrong_material"

    with pytest.raises(ValueError, match="randomize_rigid_body_material.func"):
        terrain_guards.assert_flat_online_deployment_config(cfg)


@pytest.mark.parametrize(
    "invalid",
    [None, (0.6, 0.5, 0.5), (0.5, 0.5), [0.5, 0.5, 0.5]],
)
def test_flat_online_deployment_guard_rejects_changed_command_ramp(invalid):
    cfg = _flat_online_deployment_env_cfg()
    cfg.commands.base_velocity.command_acceleration_limits = invalid

    with pytest.raises(ValueError, match="command_acceleration_limits"):
        terrain_guards.assert_flat_online_deployment_config(cfg)


def test_flat_online_deployment_guard_rejects_oversized_out_of_bounds_buffer():
    cfg = _flat_online_deployment_env_cfg()
    cfg.terminations.terrain_out_of_bounds.params["distance_buffer"] = 3.0

    with pytest.raises(ValueError, match="distance_buffer"):
        terrain_guards.assert_flat_online_deployment_config(cfg)


def test_deployment_qualification_uses_flat_online_guard_before_shared_factory():
    source = (
        ROOT / "scripts/reinforcement_learning/online_adaptation/qualify_flat_deployment.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    try_body = next(statement for statement in main.body if isinstance(statement, ast.Try)).body
    parse_index = next(
        index
        for index, statement in enumerate(try_body)
        if any(
            isinstance(node, ast.Call) and ast.unparse(node.func) == "parse_env_cfg"
            for node in ast.walk(statement)
        )
    )
    make_index = next(
        index
        for index, statement in enumerate(try_body)
        if any(
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "make_lite3_deployment_env"
            for node in ast.walk(statement)
        )
    )
    guard_indices = [
        index
        for index, statement in enumerate(try_body)
        if ast.unparse(statement) == "assert_flat_online_deployment_config(env_cfg)"
    ]

    assert guard_indices == [parse_index + 2]
    assert guard_indices[0] < make_index
    assert "assert_plane_only_terrain" not in source


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


def test_deterministic_rigid_body_material_uses_one_fixed_bucket():
    helper = getattr(
        terrain_guards, "deterministic_rigid_body_material_params", None
    )

    assert helper is not None, "deterministic material helper is missing"
    assert helper() == {
        "static_friction_range": (0.65, 0.65),
        "dynamic_friction_range": (0.55, 0.55),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
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
