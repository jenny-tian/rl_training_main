import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest

from rl_training.online_adaptation import terrain_protocol
from rl_training.online_adaptation.terrain_guards import (
    deterministic_joint_reset_params,
    deterministic_root_reset_params,
)
from rl_training.online_adaptation.terrain_protocol import TerrainGeometry


ROOT = Path(__file__).resolve().parents[2]
LITE3_CONFIG_DIR = (
    ROOT
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped"
    / "deeprobotics_lite3"
)
EXPECTED_ILLEGAL_CONTACT_LINKS = [
    "TORSO",
    "FL_HIP",
    "FR_HIP",
    "HL_HIP",
    "HR_HIP",
    "FL_THIGH",
    "FR_THIGH",
    "HL_THIGH",
    "HR_THIGH",
    "FL_SHANK",
    "FR_SHANK",
    "HL_SHANK",
    "HR_SHANK",
]


class _ReadOnlyRoughAmplitudeSubterrain:
    def __init__(self):
        self.seed = 1
        self.step_height = 0.01
        self._rough_amplitude = 0.02

    @property
    def rough_amplitude(self):
        return self._rough_amplitude


class _ReadOnlySeedGenerator:
    def __init__(self, sub_terrains):
        self._seed = 2
        self.sub_terrains = sub_terrains

    @property
    def seed(self):
        return self._seed


class _RollbackRejectingSubterrain(_ReadOnlyRoughAmplitudeSubterrain):
    def __init__(self):
        self._seed = 1
        self.step_height = 0.01
        self._rough_amplitude = 0.02

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        if value == 1 and self._seed != 1:
            raise RuntimeError("seed rollback blocked")
        self._seed = value


class _MutatingThenRaisingRoughAmplitudeSubterrain:
    def __init__(self, *, raise_after_restore=False):
        self.seed = 1
        self.step_height = 0.01
        self._rough_amplitude = 0.02
        self._raise_after_restore = raise_after_restore

    @property
    def rough_amplitude(self):
        return self._rough_amplitude

    @rough_amplitude.setter
    def rough_amplitude(self, value):
        self._rough_amplitude = value
        if value != 0.02 or self._raise_after_restore:
            raise RuntimeError("rough amplitude setter failed")


class _RollbackLeavesWrongRoughAmplitudeSubterrain:
    def __init__(self):
        self.seed = 1
        self.step_height = 0.01
        self._rough_amplitude = 0.02

    @property
    def rough_amplitude(self):
        return self._rough_amplitude

    @rough_amplitude.setter
    def rough_amplitude(self, value):
        if value == 0.02 and self._rough_amplitude != 0.02:
            raise RuntimeError("rough amplitude restore blocked")
        self._rough_amplitude = value
        if value != 0.02:
            raise RuntimeError("rough amplitude setter failed")


def _flat_online_subterrain(**overrides):
    values = {"seed": 1, "step_height": 0.01, "rough_amplitude": 0.02}
    values.update(overrides)
    return SimpleNamespace(**values)


def _env_cfg(*, generator=...):
    if generator is ...:
        generator = SimpleNamespace(
            seed=2,
            sub_terrains={"flat_online": _flat_online_subterrain()},
        )
    return SimpleNamespace(
        scene=SimpleNamespace(
            terrain=SimpleNamespace(terrain_generator=generator),
        )
    )


def test_apply_geometry_updates_generator_and_sole_flat_online_subterrain():
    env_cfg = _env_cfg()
    geometry = TerrainGeometry(step_height=0.06, rough_amplitude=0.04, seed=1009)

    result = terrain_protocol.apply_geometry_to_env_cfg(env_cfg, geometry)

    generator = env_cfg.scene.terrain.terrain_generator
    subterrain = generator.sub_terrains["flat_online"]
    assert result is None
    assert generator.seed == 1009
    assert subterrain.seed == 1009
    assert subterrain.step_height == 0.06
    assert subterrain.rough_amplitude == 0.04


def test_apply_geometry_rejects_wrong_geometry_type_without_mutation():
    env_cfg = _env_cfg()
    generator = env_cfg.scene.terrain.terrain_generator
    before = (generator.seed, vars(generator.sub_terrains["flat_online"]).copy())

    with pytest.raises(TypeError, match="TerrainGeometry"):
        terrain_protocol.apply_geometry_to_env_cfg(env_cfg, SimpleNamespace())

    assert (generator.seed, vars(generator.sub_terrains["flat_online"])) == before


def test_apply_geometry_rejects_missing_generator():
    env_cfg = _env_cfg(generator=None)

    with pytest.raises(ValueError, match="terrain_generator"):
        terrain_protocol.apply_geometry_to_env_cfg(
            env_cfg,
            TerrainGeometry(step_height=0.06, rough_amplitude=0.04, seed=1009),
        )


@pytest.mark.parametrize(
    "sub_terrains",
    [
        {},
        {
            "flat_online": _flat_online_subterrain(),
            "unexpected": SimpleNamespace(seed=9),
        },
    ],
)
def test_apply_geometry_requires_exactly_one_flat_online_subterrain_without_mutation(
    sub_terrains,
):
    generator = SimpleNamespace(seed=2, sub_terrains=sub_terrains)
    env_cfg = _env_cfg(generator=generator)
    before_generator_seed = generator.seed
    before_subterrains = {
        name: vars(subterrain).copy() for name, subterrain in sub_terrains.items()
    }

    with pytest.raises(ValueError, match="sole 'flat_online' subterrain"):
        terrain_protocol.apply_geometry_to_env_cfg(
            env_cfg,
            TerrainGeometry(step_height=0.06, rough_amplitude=0.04, seed=1009),
        )

    assert generator.seed == before_generator_seed
    assert {
        name: vars(subterrain) for name, subterrain in sub_terrains.items()
    } == before_subterrains


def test_apply_geometry_validates_mutable_fields_before_any_mutation():
    subterrain = SimpleNamespace(seed=1, step_height=0.01)
    generator = SimpleNamespace(seed=2, sub_terrains={"flat_online": subterrain})
    env_cfg = _env_cfg(generator=generator)

    with pytest.raises(ValueError, match="rough_amplitude"):
        terrain_protocol.apply_geometry_to_env_cfg(
            env_cfg,
            TerrainGeometry(step_height=0.06, rough_amplitude=0.04, seed=1009),
        )

    assert generator.seed == 2
    assert vars(subterrain) == {"seed": 1, "step_height": 0.01}


def test_apply_geometry_rolls_back_when_subterrain_setter_is_read_only():
    subterrain = _ReadOnlyRoughAmplitudeSubterrain()
    sub_terrains = {"flat_online": subterrain}
    generator = SimpleNamespace(seed=2, sub_terrains=sub_terrains)
    env_cfg = _env_cfg(generator=generator)

    with pytest.raises(ValueError, match=r"flat_online\.rough_amplitude") as exc_info:
        terrain_protocol.apply_geometry_to_env_cfg(
            env_cfg,
            TerrainGeometry(step_height=0.06, rough_amplitude=0.04, seed=1009),
        )

    assert isinstance(exc_info.value.__cause__, AttributeError)
    assert env_cfg.scene.terrain.terrain_generator is generator
    assert generator.sub_terrains is sub_terrains
    assert generator.sub_terrains["flat_online"] is subterrain
    assert generator.seed == 2
    assert subterrain.seed == 1
    assert subterrain.step_height == 0.01
    assert subterrain.rough_amplitude == 0.02


def test_apply_geometry_rolls_back_subterrain_when_generator_seed_is_read_only():
    subterrain = _flat_online_subterrain()
    sub_terrains = {"flat_online": subterrain}
    generator = _ReadOnlySeedGenerator(sub_terrains)
    env_cfg = _env_cfg(generator=generator)

    with pytest.raises(ValueError, match="terrain_generator.seed") as exc_info:
        terrain_protocol.apply_geometry_to_env_cfg(
            env_cfg,
            TerrainGeometry(step_height=0.06, rough_amplitude=0.04, seed=1009),
        )

    assert isinstance(exc_info.value.__cause__, AttributeError)
    assert env_cfg.scene.terrain.terrain_generator is generator
    assert generator.sub_terrains is sub_terrains
    assert generator.sub_terrains["flat_online"] is subterrain
    assert generator.seed == 2
    assert vars(subterrain) == {
        "seed": 1,
        "step_height": 0.01,
        "rough_amplitude": 0.02,
    }


def test_apply_geometry_reports_rollback_setter_failures():
    subterrain = _RollbackRejectingSubterrain()
    generator = SimpleNamespace(
        seed=2,
        sub_terrains={"flat_online": subterrain},
    )

    with pytest.raises(
        ValueError,
        match=r"rollback failed.*flat_online\.seed",
    ) as exc_info:
        terrain_protocol.apply_geometry_to_env_cfg(
            _env_cfg(generator=generator),
            TerrainGeometry(step_height=0.06, rough_amplitude=0.04, seed=1009),
        )

    assert isinstance(exc_info.value.__cause__, AttributeError)
    assert generator.seed == 2
    assert subterrain.seed == 1009


def test_apply_geometry_restores_failed_field_when_setter_mutates_then_raises():
    subterrain = _MutatingThenRaisingRoughAmplitudeSubterrain()
    sub_terrains = {"flat_online": subterrain}
    generator = SimpleNamespace(seed=2, sub_terrains=sub_terrains)
    env_cfg = _env_cfg(generator=generator)

    with pytest.raises(ValueError, match="previous values restored") as exc_info:
        terrain_protocol.apply_geometry_to_env_cfg(
            env_cfg,
            TerrainGeometry(step_height=0.06, rough_amplitude=0.04, seed=1009),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert env_cfg.scene.terrain.terrain_generator is generator
    assert generator.sub_terrains is sub_terrains
    assert generator.sub_terrains["flat_online"] is subterrain
    assert generator.seed == 2
    assert subterrain.seed == 1
    assert subterrain.step_height == 0.01
    assert subterrain.rough_amplitude == 0.02


def test_apply_geometry_accepts_rollback_setter_that_raises_after_restoring():
    subterrain = _MutatingThenRaisingRoughAmplitudeSubterrain(raise_after_restore=True)
    generator = SimpleNamespace(
        seed=2,
        sub_terrains={"flat_online": subterrain},
    )

    with pytest.raises(ValueError, match="previous values restored") as exc_info:
        terrain_protocol.apply_geometry_to_env_cfg(
            _env_cfg(generator=generator),
            TerrainGeometry(step_height=0.06, rough_amplitude=0.04, seed=1009),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "rollback failed" not in str(exc_info.value)
    assert generator.seed == 2
    assert subterrain.seed == 1
    assert subterrain.step_height == 0.01
    assert subterrain.rough_amplitude == 0.02


def test_apply_geometry_reports_rollback_that_leaves_failed_field_changed():
    subterrain = _RollbackLeavesWrongRoughAmplitudeSubterrain()
    generator = SimpleNamespace(
        seed=2,
        sub_terrains={"flat_online": subterrain},
    )

    with pytest.raises(
        ValueError,
        match=r"rollback failed.*flat_online\.rough_amplitude",
    ) as exc_info:
        terrain_protocol.apply_geometry_to_env_cfg(
            _env_cfg(generator=generator),
            TerrainGeometry(step_height=0.06, rough_amplitude=0.04, seed=1009),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "previous values restored" not in str(exc_info.value)
    assert subterrain.rough_amplitude == 0.04


def test_flat_online_registration_uses_manager_env_and_flat_ppo_runner():
    package_name = (
        "rl_training.tasks.manager_based.locomotion.velocity.config.quadruped."
        "deeprobotics_lite3"
    )

    importlib.import_module(package_name)
    spec = gym.spec("FlatOnline-Deeprobotics-Lite3-v0")

    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.disable_env_checker is True
    assert spec.kwargs == {
        "env_cfg_entry_point": (
            f"{package_name}.flat_online_env_cfg:DeeproboticsLite3FlatOnlineEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{package_name}.agents.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPORunnerCfg"
        ),
    }


def test_flat_online_source_disables_inherited_dynamics_randomization():
    source = (LITE3_CONFIG_DIR / "flat_online_env_cfg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    disabled_events = {
        ast.unparse(target)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
        for target in node.targets
        if ast.unparse(target).startswith("self.events.")
    }

    assert {
        "self.events.randomize_rigid_body_material",
        "self.events.randomize_rigid_body_mass",
        "self.events.randomize_rigid_body_inertia",
        "self.events.randomize_com_positions",
        "self.events.randomize_actuator_gains",
    } <= disabled_events


def test_flat_online_source_selects_all_non_foot_links_for_illegal_contacts():
    source = (LITE3_CONFIG_DIR / "flat_online_env_cfg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            ast.unparse(target)
            == "self.terminations.illegal_contact.params['sensor_cfg'].body_names"
            for target in node.targets
        )
    )

    assert isinstance(assignment.value, ast.ListComp)
    assert ast.unparse(assignment.value.elt) == "link_name"
    assert len(assignment.value.generators) == 1
    generator = assignment.value.generators[0]
    assert ast.unparse(generator.target) == "link_name"
    assert ast.unparse(generator.iter) == "self.link_names"
    assert [ast.unparse(condition) for condition in generator.ifs] == [
        "not link_name.endswith('_FOOT')"
    ]


@pytest.fixture(scope="module")
def flat_online_env_cfg():
    pytest.importorskip("pxr")
    module = importlib.import_module(
        "rl_training.tasks.manager_based.locomotion.velocity.config.quadruped."
        "deeprobotics_lite3.flat_online_env_cfg"
    )
    return module.DeeproboticsLite3FlatOnlineEnvCfg()


@pytest.mark.isaaclab
def test_flat_online_config_has_single_runtime_mutable_track(flat_online_env_cfg):
    cfg = flat_online_env_cfg
    generator = cfg.scene.terrain.terrain_generator

    assert cfg.scene.num_envs == 1
    assert generator.size == (20.0, 4.0)
    assert generator.border_width == 0.0
    assert generator.num_rows == 1
    assert generator.num_cols == 1
    assert generator.curriculum is False
    assert generator.use_cache is False
    assert generator.seed == 0
    assert set(generator.sub_terrains) == {"flat_online"}
    assert generator.sub_terrains["flat_online"].seed == 0


@pytest.mark.isaaclab
def test_flat_online_config_is_deterministic_without_curricula_or_disturbances(
    flat_online_env_cfg,
):
    cfg = flat_online_env_cfg

    assert cfg.curriculum.terrain_levels is None
    assert cfg.curriculum.gait_level is None
    assert cfg.curriculum.command_levels is None
    assert cfg.events.randomize_apply_external_force_torque is None
    assert cfg.events.randomize_push_robot is None
    assert cfg.events.randomize_rigid_body_material is None
    assert cfg.events.randomize_rigid_body_mass is None
    assert cfg.events.randomize_rigid_body_mass_base is None
    assert cfg.events.randomize_rigid_body_inertia is None
    assert cfg.events.randomize_com_positions is None
    assert cfg.events.randomize_actuator_gains is None
    assert cfg.events.randomize_reset_base.params == deterministic_root_reset_params()
    joint_reset = cfg.events.randomize_reset_joints.params
    assert joint_reset["position_range"] == deterministic_joint_reset_params()["position_range"]
    assert joint_reset["velocity_range"] == deterministic_joint_reset_params()["velocity_range"]
    assert cfg.observations.policy.enable_corruption is False


@pytest.mark.isaaclab
def test_flat_online_config_uses_fixed_forward_command_and_safety_terminations(
    flat_online_env_cfg,
):
    cfg = flat_online_env_cfg
    command = cfg.commands.base_velocity

    assert command.resampling_time_range == (1.0e9, 1.0e9)
    assert command.rel_standing_envs == 0.0
    assert command.rel_heading_envs == 0.0
    assert command.heading_command is False
    assert command.ranges.lin_vel_x == (0.5, 0.5)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.ranges.heading == (0.0, 0.0)
    assert cfg.terminations.time_out is not None
    assert cfg.terminations.terrain_out_of_bounds is not None
    assert cfg.terminations.terrain_out_of_bounds.params["distance_buffer"] == 0.5
    assert cfg.terminations.illegal_contact is not None
    assert (
        cfg.terminations.illegal_contact.params["sensor_cfg"].body_names
        == EXPECTED_ILLEGAL_CONTACT_LINKS
    )
    assert cfg.rewards.body_lin_acc_l2 is None
    assert cfg.rewards.base_height_l2 is not None
