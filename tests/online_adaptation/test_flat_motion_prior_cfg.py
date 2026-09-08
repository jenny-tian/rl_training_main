from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3"


def test_flat_task_has_no_motion_prior_interface_or_phase_observation():
    source = (CFG / "flat_env_cfg.py").read_text(encoding="utf-8")
    assert "motion_prior_path" not in source
    assert "motion_prior_mode" not in source
    assert "configure_motion_prior_terms" not in source
    assert "self.observations.policy.motion_phase = None" in source
    assert "self.observations.policy.height_scan = None" in source
    assert "self.observations.policy.base_lin_vel = None" in source


def test_motion_imitation_task_alone_owns_reference_interface():
    source = (CFG / "motion_imitation_env_cfg.py").read_text(encoding="utf-8")
    reward_source = (ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py").read_text(encoding="utf-8")
    assert 'motion_prior_mode: str = "off"' in source
    assert "motion_prior_path: str | None = None" in source
    assert "configure_motion_prior_terms" in source
    for term in (
        "motion_prior_contact",
        "motion_prior_clearance",
    ):
        assert term in reward_source
        assert term in source


def test_flat_task_uses_unitree_go2_style_reward_set():
    source = (CFG / "flat_env_cfg.py").read_text(encoding="utf-8")
    expected = {
        '"track_lin_vel_xy_exp": 4.0',
        '"track_ang_vel_z_exp": 1.5',
        '"base_height_l2": -25.0',
        '"lin_vel_z_l2": -2.0',
        '"ang_vel_xy_l2": -0.25',
        '"joint_torques_l2": -2.5e-4',
        '"joint_acc_l2": -1.0e-8',
        '"feet_air_time": 2.0',
        '"feet_air_time_variance": -5.0',
        '"undesired_contacts": -1.0',
        '"action_rate_l2": -0.05',
        '"joint_pos_limits": -5.0',
    }
    assert all(item in source for item in expected)
    assert '".*_HipX_joint": 0.125' in source
    assert '"^(?!.*_HipX_joint).*": 0.25' in source
    assert "self.scene.height_scanner = None" in source
    assert "command.ranges.lin_vel_x = (-0.8, 0.8)" in source
    assert "command.ranges.lin_vel_y = (-0.8, 0.8)" in source
    assert "command.ranges.ang_vel_z = (-0.8, 0.8)" in source
    assert "command.planar_velocity_deadband = 0.05" in source
    assert "command.only_lin_x_magnitude_range = (0.08, 0.25)" in source
    assert 'self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25' in source
    assert 'self.rewards.feet_air_time.params["command_threshold"] = 0.05' in source
    assert "self.curriculum.gait_level = None" in source
    assert "reward_mdp.gait_level = 1.0" in source
