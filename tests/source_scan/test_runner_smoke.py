from __future__ import annotations

from types import SimpleNamespace

import torch

from rl_training.online_adaptation.types import RewardSignals, SafetyState
from scripts.reinforcement_learning.online_adaptation import run_source_qualification_scan as runner


class _TerminationManager:
    active_terms = ["bad_orientation_2", "illegal_contact"]

    def __init__(self, env):
        self.env = env

    def get_term(self, name):
        return torch.tensor([self.env.termination_term == name])

    @property
    def time_outs(self):
        return torch.tensor([self.env.termination_term == "time_out"])


class _Scene:
    def __init__(self):
        self.terrain = SimpleNamespace(env_origins=torch.zeros((1, 3)))
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.zeros((1, 3)),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
                root_ang_vel_b=torch.zeros((1, 3)),
                body_lin_vel_w=torch.zeros((1, 4, 3)),
            )
        )

    def __getitem__(self, name):
        assert name == "robot"
        return self.robot


class _Env:
    device = "cpu"

    def __init__(self, *, tilt=0.0, terminal_term="illegal_contact"):
        self.scene = _Scene()
        self.callbacks = []
        self.tilt = tilt
        self.termination_term = None
        self.terminal_term = terminal_term
        self.termination_manager = _TerminationManager(self)

    def add_pre_reset_callback(self, callback):
        self.callbacks.append(callback)

    def reset_before_auto_reset(self):
        self.termination_term = self.terminal_term
        for callback in self.callbacks:
            callback([0])
        self.scene.robot.data.root_pos_w.zero_()


class _WrappedEnv:
    def __init__(self, env):
        self.unwrapped = env
        self.seed_value = None
        self.step_calls = 0

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return torch.zeros((1, 45)), {}

    def step(self, action):
        self.step_calls += 1
        self.unwrapped.scene.robot.data.root_pos_w[0, 0] = 1.0
        self.unwrapped.reset_before_auto_reset()
        return torch.zeros((1, 45)), torch.zeros(1), torch.ones((1, 1), dtype=torch.bool), {}


class _Adapter:
    contact_threshold = 1.0

    def __init__(self, env, *, safety_height_source):
        assert safety_height_source == "base_scan"
        self.env = env
        self.robot = env.scene.robot
        self.robot_foot_ids = [0, 1, 2, 3]
        self.reset_count = 0

    def reset_contact_diagnostics(self):
        self.reset_count += 1

    def observe_nonfoot_contact(self):
        return {"max_nonfoot_contact_force_n": 0.0, "max_nonfoot_contact_body": None}

    def contact_diagnostics(self):
        return {
            "episode_max_nonfoot_contact_force_n": 0.0,
            "episode_max_nonfoot_contact_body": None,
        }

    def safety_state(self, control_timed_out):
        batch = control_timed_out.shape[0]
        return SafetyState(
            projected_gravity_xy=torch.tensor([[self.env.tilt, 0.0]]).expand(batch, -1),
            base_height=torch.full((batch, 1), 0.4),
            joint_pos=torch.zeros((batch, 12)),
            joint_vel=torch.zeros((batch, 12)),
            joint_lower=torch.full((batch, 12), -10.0),
            joint_upper=torch.full((batch, 12), 10.0),
            default_joint_pos=torch.zeros((batch, 12)),
            action_scale=torch.ones((batch, 12)),
            nonfoot_contact=torch.zeros((batch, 1), dtype=torch.bool),
            control_timed_out=control_timed_out,
        )

    def reward_signals(self):
        return RewardSignals(
            base_lin_vel_xy=torch.zeros((1, 2)),
            command_lin_vel_xy=torch.tensor([[0.3, 0.0]]),
            base_ang_vel_z=torch.zeros((1, 1)),
            command_ang_vel_z=torch.zeros((1, 1)),
            projected_gravity_xy=torch.tensor([[self.env.tilt, 0.0]]),
            base_lin_vel_z=torch.zeros((1, 1)),
            foot_slip=torch.zeros((1, 1)),
            foot_impact=torch.zeros((1, 1)),
            joint_power=torch.zeros((1, 1)),
            nonfoot_contact=torch.zeros((1, 1)),
            foot_height=torch.zeros((1, 4)),
            foot_contact=torch.ones((1, 4)),
        )


def _cell():
    return {
        "cell_id": "flat_L1_forward_flat",
        "terrain_family": "flat",
        "level": 1,
        "direction_or_layout": "forward_flat",
        "resolved_parameters": {
            "geometry": {"size": [10.0, 4.0], "flat_length_m": 3.0, "transition_length_m": 0.5},
            "track": {"start_x_m": 1.5, "completion_x_m": 9.5, "lateral_margin_m": 0.2},
        },
    }


def _install_adapter(monkeypatch):
    import rl_training.online_adaptation.isaaclab_adapter as adapter_module

    monkeypatch.setattr(adapter_module, "Lite3IsaacLabAdapter", _Adapter)


def test_filter_pause_stops_before_environment_action(monkeypatch):
    _install_adapter(monkeypatch)
    env = _Env(tilt=0.8)
    wrapped = _WrappedEnv(env)

    record = runner._run_episode(
        wrapped,
        lambda _observation: torch.zeros((1, 12)),
        _cell(),
        geometry_seed=1,
        initial_state_seed=2,
        episode_index=1,
        max_steps=10,
    )

    assert record["termination_reason"] == "safety_termination"
    assert record["safety_filter_rejection_or_emergency_count"] == 1
    assert wrapped.step_calls == 0
    assert record["action_trace"][0]["safety_paused"] is True
    assert "a_executed_command" not in record["action_trace"][0]


def test_environment_terminal_metrics_are_captured_before_auto_reset(monkeypatch):
    _install_adapter(monkeypatch)
    env = _Env(terminal_term="illegal_contact")
    wrapped = _WrappedEnv(env)

    record = runner._run_episode(
        wrapped,
        lambda _observation: torch.zeros((1, 12)),
        _cell(),
        geometry_seed=1,
        initial_state_seed=2,
        episode_index=1,
        max_steps=10,
    )

    assert wrapped.seed_value == 2
    assert wrapped.step_calls == 1
    assert env.scene.robot.data.root_pos_w[0, 0].item() == 0.0  # auto-reset state
    assert record["forward_progress_m"] == 1.0  # captured terminal position
    assert record["termination_reason"] == "body_contact_termination"
    assert record["termination_terms"] == ["illegal_contact"]


def test_sample_observer_receives_the_terminal_sample_before_episode_return(monkeypatch):
    _install_adapter(monkeypatch)
    env = _Env(terminal_term="illegal_contact")
    wrapped = _WrappedEnv(env)
    observed = []

    record = runner._run_episode(
        wrapped,
        lambda _observation: torch.zeros((1, 12)),
        _cell(),
        geometry_seed=1,
        initial_state_seed=2,
        episode_index=1,
        max_steps=10,
        sample_observer=lambda step, sample: observed.append((step, sample)),
    )

    assert record["steps"] == 1
    assert len(observed) == 1
    assert observed[0][0] == 1
    assert observed[0][1]["forward_progress_m"] == 1.0


def test_single_environment_scan_has_only_registered_forward_command():
    command = SimpleNamespace(
        ranges=SimpleNamespace(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        )
    )

    runner._configure_forward_command(command, 0.3)

    assert command.ranges.lin_vel_x == (0.3, 0.3)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.ranges.heading == (0.0, 0.0)
    assert command.rel_zero_vel_envs == 0.0
    assert command.rel_only_lin_y_envs == 0.0
    assert command.rel_only_lin_x_envs == 1.0
    assert command.rel_only_ang_z_envs == 0.0


def test_slope_family_metrics_use_the_registered_completion_track():
    cell = {
        "terrain_family": "slope",
        "direction_or_layout": "downhill",
        "resolved_parameters": {
            "geometry": {
                "size": [12.0, 4.0],
                "flat_length_m": 3.0,
                "transition_length_m": 0.5,
                "exit_flat_length_m": 2.0,
                "slope_angle_deg": 5.0,
            },
            "track": {"start_x_m": 1.5, "completion_x_m": 10.5},
        },
    }

    metrics = runner._family_metrics(
        cell,
        [0.0, 8.0, 8.5, 9.0, 10.0],
        pitch=[0.0, 0.1],
        roll=[0.0, 0.02],
    )

    assert metrics["exit_transition_progress_m"] == [0.0, 0.0, 0.5, 0.5, 0.5]
    assert metrics["terminal_flat_progress_m"] == [0.0, 0.0, 0.0, 0.5, 1.5]
    assert metrics.get("unavailable_metrics") is None


def test_rsl_wrapper_is_seeded_then_reset_without_seed_keyword():
    calls = []

    class Wrapped:
        def seed(self, seed):
            calls.append(("seed", seed))

        def reset(self):
            calls.append(("reset",))
            return "observation", {}

    assert runner._seeded_wrapper_reset(Wrapped(), 73) == ("observation", {})
    assert calls == [("seed", 73), ("reset",)]
