"""CPU checks against the saved training config; no Isaac scene or PPO update."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from rl_training.source_scan.adaptation_profile import (
    SOURCE_ADAPTATION_PROFILE,
    build_source_adaptation_rewards,
    resolve_reward_gait_level,
    source_adaptation_reward_specs,
    validate_source_adaptation_config,
)

ROOT = Path(__file__).resolve().parents[2]
VELOCITY = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"


def configuration():
    return NS(source_reward_profile=SOURCE_ADAPTATION_PROFILE, reward_gait_level=1.0,
              rewards=build_source_adaptation_rewards(NS, NS))


def test_rewards_match_saved_checkpoint_configuration_except_local_height_reference():
    saved = json.loads((ROOT / "docs/lite3-source-actor90-regression-diagnosis-20261105/deliverables/reward_configuration.json").read_text())
    old = saved["old_rewards"]
    expected = deepcopy(old)
    for term in expected.values():
        for key, value in term["params"].items():
            if key.endswith("_cfg") and value is not None:
                term["params"][key] = {k: v for k, v in value.items()
                    if k in ("name", "joint_names", "body_names") and v is not None}
    expected["base_height_l2"]["params"]["sensor_cfg"] = {"name": "height_scanner_base"}
    assert source_adaptation_reward_specs() == expected
    cfg = configuration()
    assert validate_source_adaptation_config(cfg)["rewards"] == expected
    # Editing one environment cannot modify another or the saved profile.
    cfg.rewards["joint_torques_l2"].params["asset_cfg"].joint_names.clear()
    assert len(configuration().rewards["joint_torques_l2"].params["asset_cfg"].joint_names) == 12


@pytest.mark.parametrize("fault", ["gate", "extra_term", "weight", "threshold", "entity", "profile"])
def test_audit_rejects_silent_reward_drift(fault):
    cfg = configuration()
    if fault == "gate": cfg.reward_gait_level = 0.0
    if fault == "extra_term": cfg.rewards["knee_joint_pos_penalty"] = NS(weight=-2.0)
    if fault == "weight": cfg.rewards["track_lin_vel_xy_exp"].weight = 0.0
    if fault == "threshold": cfg.rewards["track_lin_vel_xy_exp"].params["std"] = 0.707
    if fault == "entity": cfg.rewards["base_height_l2"].params["sensor_cfg"].name = "height_scanner"
    if fault == "profile": cfg.source_reward_profile = "legacy"
    with pytest.raises(ValueError): validate_source_adaptation_config(cfg)


def reward_functions():
    """Execute the actual reward definitions on CPU data, without importing Kit."""
    path = VELOCITY / "mdp/rewards.py"
    names = {"get_gait_level_tensor", "track_lin_vel_xy_exp", "base_height_l2", "action_rate_l2"}
    tree = ast.parse(path.read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = dict(torch=torch, gait_level=0.0, resolve_reward_gait_level=resolve_reward_gait_level,
                     SceneEntityCfg=lambda name: NS(name=name))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_environment_gate_is_independent_of_global_curriculum_and_other_environments():
    functions = reward_functions()
    local = NS(num_envs=2, device="cpu", cfg=NS(reward_gait_level=1.0),
               action_manager=NS(action=torch.ones(2, 12), prev_action=torch.zeros(2, 12)))
    legacy = NS(num_envs=2, device="cpu", cfg=NS())
    disabled = NS(num_envs=2, device="cpu", cfg=NS(reward_gait_level=0.0))
    for global_value in (0.0, 0.2, 1.0):
        functions["gait_level"] = global_value
        assert torch.equal(functions["action_rate_l2"](local), torch.full((2,), 12.0))
        assert torch.equal(functions["get_gait_level_tensor"](legacy), torch.full((2,), global_value))
        assert not functions["get_gait_level_tensor"](disabled).any()
    for invalid in (float("nan"), float("inf"), -1.0, 1.1, True, None):
        with pytest.raises(ValueError): resolve_reward_gait_level(NS(reward_gait_level=invalid), 0.0)


def test_tracking_discriminates_standing_and_height_is_translation_invariant():
    functions = reward_functions()
    data = NS(root_lin_vel_b=torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]),
              root_link_pos_w=torch.tensor([[0.0, 0.0, 0.4], [0.0, 0.0, 10.4]]))
    data.root_pos_w = data.root_link_pos_w
    scan = NS(data=NS(ray_hits_w=torch.tensor([[[0., 0., 0.]]*4, [[0., 0., 10.]]*4])))
    env = NS(num_envs=2, device="cpu", cfg=NS(reward_gait_level=1.0),
             scene={"robot": NS(data=data), "height_scanner_base": scan},
             command_manager=NS(get_command=lambda name: torch.tensor([[0.3, 0., 0.]]*2)))
    specs = source_adaptation_reward_specs()
    tracking = functions["track_lin_vel_xy_exp"](env, **specs["track_lin_vel_xy_exp"]["params"])
    assert tracking[0] == 1.0 and tracking[1].item() == pytest.approx(0.2369278)
    term = configuration().rewards["base_height_l2"]
    height = functions["base_height_l2"](env, **term.params)
    torch.testing.assert_close(height, torch.tensor([0.025**2]*2), rtol=4e-5, atol=1e-8)
    assert height.min() > 0  # The global default is zero; the local gate must still activate the term.


def test_adaptation_subclass_keeps_parent_scene_commands_and_restores_pruned_rewards():
    path = VELOCITY / "config/quadruped/deeprobotics_lite3/source_adaptation_env_cfg.py"

    class Parent:
        def __post_init__(self):
            assert not isinstance(self.rewards, dict)
            self.scene = NS(marker="parent L1 scene")
            self.commands = NS(marker="parent fixed commands")
            self.terminations = NS(marker="parent safety")
            self.rewards = {"feet_air_time": None, "knee_joint_pos_penalty": NS(weight=-2.0)}

    namespace = dict(configclass=lambda cls: cls, DeeproboticsLite3SourceTrainingEnvCfg=Parent,
        RewardTermCfg=NS, SceneEntityCfg=NS, RewardsCfg=NS, SOURCE_ADAPTATION_PROFILE=SOURCE_ADAPTATION_PROFILE,
        build_source_adaptation_rewards=build_source_adaptation_rewards,
        validate_source_adaptation_config=validate_source_adaptation_config)
    tree = ast.parse(path.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]),
                 str(path), "exec"), namespace)
    cfg = namespace["DeeproboticsLite3SourceAdaptationEnvCfg"]()
    cfg.__post_init__()
    assert cfg.scene.marker == "parent L1 scene" and cfg.commands.marker == "parent fixed commands"
    assert cfg.terminations.marker == "parent safety"
    assert cfg.source_parallel_sampler_num_envs == 120 and cfg.source_shared_bank_rows == 1
    assert cfg.rewards["feet_air_time"].weight == 2.0
    assert "knee_joint_pos_penalty" not in cfg.rewards
    assert validate_source_adaptation_config(cfg)["reward_gait_level"] == 1.0
    cfg.__post_init__()  # The already-resolved dict must not break parent reinitialization.
    assert validate_source_adaptation_config(cfg)["reward_gait_level"] == 1.0
