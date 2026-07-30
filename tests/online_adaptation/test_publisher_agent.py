from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch

from rl_training.online_adaptation.agent import AgentConfig, OnlineAdaptationAgent
from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher
from rl_training.online_adaptation.sac import SACConfig
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState
from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import (
    actor_version_for_record,
    enforce_control_timeout_fallback,
)


def make_agent(*, residual_gain=1.0, residual_rate_limit=0.02, replay_capacity=10_000, n_step=1, gamma=0.99):
    actor = ResidualActor(2, 1, 4, torch.tensor([0.2]))
    critic = TwinCritic(2, 1, 4)
    return OnlineAdaptationAgent(
        num_envs=1,
        actor=actor,
        critic=critic,
        safety_limits=SafetyLimits(
            residual_limits=torch.tensor([0.2]),
            residual_rate_limit=residual_rate_limit,
        ),
        config=AgentConfig(
            history_length=3,
            frame_dim=2,
            action_dim=1,
            batch_size=1,
            residual_gain=residual_gain,
            replay_capacity=replay_capacity,
            n_step=n_step,
            gamma=gamma,
        ),
        sac_config=SACConfig(gamma=gamma),
        device="cpu",
    )


def test_agent_requires_matching_nstep_and_sac_gamma():
    actor = ResidualActor(2, 1, 4, torch.tensor([0.2]))
    critic = TwinCritic(2, 1, 4)

    with pytest.raises(ValueError, match="gamma"):
        OnlineAdaptationAgent(
            num_envs=1,
            actor=actor,
            critic=critic,
            safety_limits=SafetyLimits(
                residual_limits=torch.tensor([0.2]),
                residual_rate_limit=0.02,
            ),
            config=AgentConfig(
                history_length=3,
                frame_dim=2,
                action_dim=1,
                batch_size=1,
                gamma=0.9,
            ),
        )


class FixedResidualActor:
    def __init__(self, residual):
        self.residual = residual

    def sample(self, history, frame, base_action, *, deterministic):
        residual = torch.full_like(base_action, self.residual)
        return residual, None, residual


def publish_fixed_residual(agent, residual, version=0):
    actor = FixedResidualActor(residual)
    agent.publisher.current = lambda: (actor, version)


def ending_fields(**overrides):
    fields = dict(
        terminated=torch.zeros(1, 1),
        truncated=torch.zeros(1, 1),
        bootstrap_mask=torch.ones(1, 1),
        discount=torch.full((1, 1), 0.99),
        safety_mask=torch.ones(1, 1),
        curriculum_stage_id=torch.zeros(1, 1, dtype=torch.int64),
        episode_id=torch.zeros(1, 1, dtype=torch.int64),
    )
    fields.update(overrides)
    return fields


def finish_agent_step(
    agent,
    next_frame,
    next_base_action,
    reward,
    *,
    adaptation_enabled=False,
    **ending_overrides,
):
    endings = ending_fields(**ending_overrides)
    agent.finish_step(
        next_frame=next_frame,
        next_base_action=next_base_action,
        reward=reward,
        terminated=endings["terminated"],
        truncated=endings["truncated"],
        bootstrap_mask=endings["bootstrap_mask"],
        discount=endings["discount"],
        safety_mask=endings["safety_mask"],
        curriculum_stage_id=endings["curriculum_stage_id"],
        episode_id=endings["episode_id"],
        adaptation_enabled=adaptation_enabled,
    )


def replay_state(replay):
    return {
        "size": replay._size,
        "cursor": replay._cursor,
        "storages": {
            name: value.clone()
            for name, value in vars(replay).items()
            if isinstance(value, torch.Tensor)
        },
    }


def assert_replay_state_equal(replay, expected):
    assert replay._size == expected["size"]
    assert replay._cursor == expected["cursor"]
    for name, expected_storage in expected["storages"].items():
        torch.testing.assert_close(
            getattr(replay, name),
            expected_storage,
            rtol=0,
            atol=0,
            equal_nan=True,
        )


def accumulate_residual_memory(agent, steps=3):
    frame = torch.zeros(1, 2)
    base_action = torch.zeros(1, 1)
    for _ in range(steps):
        agent.begin_step(frame, base_action, safety_state(), deterministic=True)
        finish_agent_step(
            agent,
            frame,
            base_action,
            torch.zeros(1, 1),
        )


def safety_state():
    return SafetyState(
        projected_gravity_xy=torch.zeros(1, 2),
        base_height=torch.full((1, 1), 0.35),
        joint_pos=torch.zeros(1, 1),
        joint_vel=torch.zeros(1, 1),
        joint_lower=torch.full((1, 1), -1.0),
        joint_upper=torch.full((1, 1), 1.0),
        default_joint_pos=torch.zeros(1, 1),
        action_scale=torch.ones(1, 1),
        nonfoot_contact=torch.zeros(1, 1, dtype=torch.bool),
        control_timed_out=torch.zeros(1, 1, dtype=torch.bool),
    )


def finish_transition(agent, *, adaptation_enabled):
    frame = torch.zeros(1, 2)
    action = torch.zeros(1, 1)
    agent.begin_step(frame, action, safety_state(), deterministic=True)
    finish_agent_step(
        agent,
        frame,
        action,
        torch.ones(1, 1),
        adaptation_enabled=adaptation_enabled,
    )


def test_publisher_swaps_complete_versions():
    actor = torch.nn.Linear(2, 1, bias=False)
    publisher = DoubleBufferedActorPublisher(actor)
    updated = torch.nn.Linear(2, 1, bias=False)
    updated.weight.data.fill_(3.0)
    publisher.publish(updated, version=4)
    published, version = publisher.current()
    assert version == 4
    assert torch.equal(published.weight, updated.weight)


def test_agent_does_not_request_update_before_minimum_replay():
    assert not OnlineAdaptationAgent.should_update(replay_size=255, min_replay_size=256, paused=False)
    assert OnlineAdaptationAgent.should_update(replay_size=256, min_replay_size=256, paused=False)
    assert not OnlineAdaptationAgent.should_update(replay_size=256, min_replay_size=256, paused=True)


def test_vectorized_learning_pauses_only_when_all_environments_are_paused():
    assert not OnlineAdaptationAgent.all_environments_paused(torch.tensor([[False], [True]]))
    assert OnlineAdaptationAgent.all_environments_paused(torch.tensor([[True], [True]]))


def test_agent_routes_only_adapting_transitions_to_adaptation_replay():
    agent = make_agent()

    finish_transition(agent, adaptation_enabled=False)
    finish_transition(agent, adaptation_enabled=True)

    assert len(agent.replay) == 2
    assert len(agent.adaptation_replay) == 1


def test_agent_accumulates_nstep_before_replay_insertion():
    agent = make_agent(n_step=3, gamma=0.9)
    frame = torch.zeros(1, 2)
    action = torch.zeros(1, 1)

    for index in range(2):
        agent.begin_step(frame + index, action, safety_state(), deterministic=True)
        finish_agent_step(
            agent,
            frame + index + 1,
            action,
            torch.ones(1, 1),
            adaptation_enabled=True,
        )

    assert len(agent.replay) == 0
    assert len(agent.adaptation_replay) == 0

    agent.begin_step(frame + 2, action, safety_state(), deterministic=True)
    finish_agent_step(
        agent,
        frame + 3,
        action,
        torch.ones(1, 1),
        adaptation_enabled=True,
    )

    assert len(agent.replay) == 1
    assert len(agent.adaptation_replay) == 1
    sampled = agent.replay.sample(1, "cpu")
    assert sampled.reward.item() == pytest.approx(1.0 + 0.9 + 0.9**2)
    assert sampled.discount.item() == pytest.approx(0.9**3)
    torch.testing.assert_close(sampled.next_frame, frame + 3)


def test_agent_routes_nstep_adaptation_replay_by_original_step():
    frame = torch.zeros(1, 2)
    action = torch.zeros(1, 1)

    enabled_first = make_agent(n_step=3, gamma=0.9)
    for adaptation_enabled in (True, False, False):
        enabled_first.begin_step(frame, action, safety_state(), deterministic=True)
        finish_agent_step(
            enabled_first,
            frame,
            action,
            torch.ones(1, 1),
            adaptation_enabled=adaptation_enabled,
        )

    disabled_first = make_agent(n_step=3, gamma=0.9)
    for adaptation_enabled in (False, True, True):
        disabled_first.begin_step(frame, action, safety_state(), deterministic=True)
        finish_agent_step(
            disabled_first,
            frame,
            action,
            torch.ones(1, 1),
            adaptation_enabled=adaptation_enabled,
        )

    assert len(enabled_first.replay) == 1
    assert len(enabled_first.adaptation_replay) == 1
    assert len(disabled_first.replay) == 1
    assert len(disabled_first.adaptation_replay) == 0


def test_agent_clears_adaptation_replay_without_touching_global_replay():
    agent = make_agent()
    finish_transition(agent, adaptation_enabled=True)

    agent.clear_adaptation_replay()

    assert len(agent.replay) == 1
    assert len(agent.adaptation_replay) == 0


def test_actor_update_requires_gate_replay_and_unpaused_control():
    should_update = OnlineAdaptationAgent.should_update_actor

    assert not should_update(127, 128, paused=False, actor_enabled=True)
    assert not should_update(128, 128, paused=True, actor_enabled=True)
    assert not should_update(128, 128, paused=False, actor_enabled=False)
    assert should_update(128, 128, paused=False, actor_enabled=True)


def test_begin_step_decision_keeps_version_used_even_after_publisher_updates():
    agent = make_agent()
    frame = torch.zeros(1, 2)
    base_action = torch.zeros(1, 1)

    decision = agent.begin_step(frame, base_action, safety_state(), deterministic=True)
    agent.publisher.publish(agent.updater.actor, version=4)

    assert decision.actor_version == 0
    assert actor_version_for_record(decision) == 0


def test_zero_residual_gain_suppresses_stochastic_actor_sample_exactly():
    agent = make_agent(residual_gain=0.0)
    base_action = torch.zeros(1, 1)

    decision = agent.begin_step(
        torch.randn(1, 2),
        base_action,
        safety_state(),
        deterministic=False,
    )

    assert torch.equal(decision.residual, torch.zeros_like(base_action))


def test_setting_zero_residual_gain_clears_accumulated_learned_residual_exactly():
    agent = make_agent(residual_gain=1.0, residual_rate_limit=0.02)
    publish_fixed_residual(agent, residual=0.2)
    accumulate_residual_memory(agent)
    assert agent.safety._previous_residual is not None
    assert torch.count_nonzero(agent.safety._previous_residual).item() > 0

    agent.set_residual_gain(0.0)
    decision = agent.begin_step(
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        safety_state(),
        deterministic=True,
    )

    assert torch.equal(decision.residual, torch.zeros(1, 1))
    assert torch.equal(decision.action, torch.zeros(1, 1))


def test_zero_residual_gain_retains_only_joint_projection_after_residual_memory():
    agent = make_agent(residual_gain=1.0, residual_rate_limit=0.02)
    publish_fixed_residual(agent, residual=0.2)
    accumulate_residual_memory(agent)
    agent.set_residual_gain(0.0)
    base_action = torch.tensor([[2.0]])
    state = safety_state()
    expected = SafetyFilter(agent.safety.limits).project(
        base_action,
        torch.zeros_like(base_action),
        state,
        force_zero_residual=True,
    )

    decision = agent.begin_step(
        torch.zeros(1, 2),
        base_action,
        state,
        deterministic=True,
    )

    assert torch.equal(decision.action, expected.action)
    assert torch.equal(decision.residual, expected.residual)
    assert agent._pending is not None
    assert torch.equal(agent._pending.residual_action, decision.residual)


def test_zero_residual_gain_preserves_safety_valid_base_action():
    agent = make_agent(residual_gain=0.0)
    publish_fixed_residual(agent, residual=0.2)
    base_action = torch.tensor([[0.25]])

    decision = agent.begin_step(
        torch.zeros(1, 2),
        base_action,
        safety_state(),
        deterministic=False,
    )

    assert torch.equal(decision.action, base_action)


def test_zero_residual_gain_still_projects_unsafe_base_action():
    agent = make_agent(residual_gain=0.0)
    publish_fixed_residual(agent, residual=0.2)
    base_action = torch.tensor([[2.0]])
    state = safety_state()
    expected = SafetyFilter(agent.safety.limits).project(
        base_action,
        torch.zeros_like(base_action),
        state,
    )

    decision = agent.begin_step(
        torch.zeros(1, 2),
        base_action,
        state,
        deterministic=False,
    )

    assert torch.equal(decision.action, expected.action)
    assert torch.equal(decision.residual, expected.residual)
    assert not torch.equal(decision.action, base_action)


def test_half_residual_gain_scales_published_actor_sample_before_safety():
    agent = make_agent(residual_gain=0.5, residual_rate_limit=1.0)
    publish_fixed_residual(agent, residual=0.12, version=7)

    decision = agent.begin_step(
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        safety_state(),
        deterministic=True,
    )

    assert torch.equal(decision.residual, torch.tensor([[0.06]]))
    assert decision.actor_version == 7


def test_set_residual_gain_takes_effect_on_next_begin_step():
    agent = make_agent(residual_gain=0.0, residual_rate_limit=1.0)
    publish_fixed_residual(agent, residual=0.12)
    frame = torch.zeros(1, 2)
    base_action = torch.zeros(1, 1)

    first = agent.begin_step(frame, base_action, safety_state(), deterministic=True)
    finish_agent_step(
        agent,
        frame,
        base_action,
        torch.zeros(1, 1),
    )
    agent.set_residual_gain(0.5)
    second = agent.begin_step(frame, base_action, safety_state(), deterministic=True)

    assert torch.equal(first.residual, torch.zeros_like(base_action))
    assert agent.residual_gain == 0.5
    assert torch.equal(second.residual, torch.tensor([[0.06]]))


def test_finish_step_resets_history_on_termination_without_truncation():
    agent = make_agent()
    frame = torch.tensor([[1.0, 2.0]])
    agent.begin_step(frame, torch.zeros(1, 1), safety_state(), deterministic=True)

    finish_agent_step(
        agent,
        torch.tensor([[3.0, 4.0]]),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        terminated=torch.ones(1, 1),
        truncated=torch.zeros(1, 1),
        bootstrap_mask=torch.zeros(1, 1),
    )

    assert torch.equal(agent.history.snapshot(), torch.zeros(1, 3, 2))


def test_finish_step_resets_history_on_truncation_even_when_bootstrapping():
    agent = make_agent()
    frame = torch.tensor([[1.0, 2.0]])
    agent.begin_step(frame, torch.zeros(1, 1), safety_state(), deterministic=True)

    finish_agent_step(
        agent,
        torch.tensor([[3.0, 4.0]]),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        truncated=torch.ones(1, 1),
        bootstrap_mask=torch.ones(1, 1),
    )

    assert torch.equal(agent.history.snapshot(), torch.zeros(1, 3, 2))


def test_finish_step_does_not_reset_history_from_bootstrap_mask_alone():
    agent = make_agent()
    frame = torch.tensor([[1.0, 2.0]])
    agent.begin_step(frame, torch.zeros(1, 1), safety_state(), deterministic=True)

    finish_agent_step(
        agent,
        torch.tensor([[3.0, 4.0]]),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        bootstrap_mask=torch.zeros(1, 1),
    )

    assert torch.count_nonzero(agent.history.snapshot()).item() > 0


def test_finish_step_routes_all_explicit_fields_to_both_replays():
    agent = make_agent(gamma=0.75)
    frame = torch.tensor([[1.0, 2.0]])
    agent.begin_step(frame, torch.zeros(1, 1), safety_state(), deterministic=True)
    explicit_fields = ending_fields(
        terminated=torch.ones(1, 1),
        truncated=torch.zeros(1, 1),
        bootstrap_mask=torch.zeros(1, 1),
        discount=torch.full((1, 1), 0.75),
        safety_mask=torch.ones(1, 1),
        curriculum_stage_id=torch.tensor([[7]], dtype=torch.int64),
        episode_id=torch.tensor([[11]], dtype=torch.int64),
    )

    finish_agent_step(
        agent,
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[0.5]]),
        torch.tensor([[2.0]]),
        **explicit_fields,
        adaptation_enabled=True,
    )

    for replay in (agent.replay, agent.adaptation_replay):
        sampled = replay.sample(1, "cpu")
        for name, expected in explicit_fields.items():
            torch.testing.assert_close(getattr(sampled, name), expected)


def test_finish_step_excludes_safety_invalid_rows_from_both_replays():
    agent = make_agent()
    frame = torch.zeros(1, 2)
    agent.begin_step(frame, torch.zeros(1, 1), safety_state(), deterministic=True)

    finish_agent_step(
        agent,
        frame,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        safety_mask=torch.zeros(1, 1),
        adaptation_enabled=True,
    )

    assert len(agent.replay) == 0
    assert len(agent.adaptation_replay) == 0


def test_finish_step_rolls_back_both_replays_when_second_commit_fails_and_retries_once(monkeypatch):
    agent = make_agent(replay_capacity=3)
    finish_transition(agent, adaptation_enabled=True)
    finish_transition(agent, adaptation_enabled=True)
    frame = torch.tensor([[1.0, 2.0]])
    agent.begin_step(frame, torch.zeros(1, 1), safety_state(), deterministic=True)
    history_before = agent.history.snapshot()
    replay_before = replay_state(agent.replay)
    adaptation_before = replay_state(agent.adaptation_replay)
    original_second_commit = agent.adaptation_replay._commit_prepared

    def commit_then_fail(prepared):
        original_second_commit(prepared)
        raise RuntimeError("injected second replay commit failure")

    monkeypatch.setattr(agent.adaptation_replay, "_commit_prepared", commit_then_fail)
    endings = ending_fields(
        terminated=torch.ones(1, 1),
        truncated=torch.zeros(1, 1),
        bootstrap_mask=torch.zeros(1, 1),
    )

    with pytest.raises(RuntimeError, match="injected second replay commit failure"):
        finish_agent_step(
            agent,
            torch.tensor([[3.0, 4.0]]),
            torch.zeros(1, 1),
            torch.full((1, 1), 2.0),
            **endings,
            adaptation_enabled=True,
        )

    assert_replay_state_equal(agent.replay, replay_before)
    assert_replay_state_equal(agent.adaptation_replay, adaptation_before)
    assert agent._pending is not None
    assert torch.equal(agent.history.snapshot(), history_before)

    monkeypatch.setattr(agent.adaptation_replay, "_commit_prepared", original_second_commit)
    finish_agent_step(
        agent,
        torch.tensor([[3.0, 4.0]]),
        torch.zeros(1, 1),
        torch.full((1, 1), 2.0),
        **endings,
        adaptation_enabled=True,
    )

    assert len(agent.replay) == 3
    assert len(agent.adaptation_replay) == 3
    assert agent.replay._cursor == 0
    assert agent.adaptation_replay._cursor == 0
    assert agent.replay._reward[2].item() == 2.0
    assert agent.adaptation_replay._reward[2].item() == 2.0
    assert agent._pending is None
    assert torch.equal(agent.history.snapshot(), torch.zeros(1, 3, 2))


def test_distinct_nstep_masks_roll_back_both_replays_on_second_commit_failure(monkeypatch):
    agent = make_agent(n_step=5, gamma=0.99, replay_capacity=4)
    frame = torch.zeros(1, 2)
    action = torch.zeros(1, 1)
    agent.begin_step(frame, action, safety_state(), deterministic=True)
    finish_agent_step(
        agent,
        frame,
        action,
        torch.ones(1, 1),
        adaptation_enabled=False,
    )
    agent.begin_step(frame, action, safety_state(), deterministic=True)
    history_before = agent.history.snapshot()
    replay_before = replay_state(agent.replay)
    adaptation_before = replay_state(agent.adaptation_replay)
    original_second_commit = agent.adaptation_replay._commit_prepared

    def commit_then_fail(prepared):
        original_second_commit(prepared)
        raise RuntimeError("injected distinct-mask commit failure")

    monkeypatch.setattr(agent.adaptation_replay, "_commit_prepared", commit_then_fail)
    with pytest.raises(RuntimeError, match="injected distinct-mask commit failure"):
        finish_agent_step(
            agent,
            frame,
            action,
            torch.ones(1, 1),
            truncated=torch.ones(1, 1),
            adaptation_enabled=True,
        )

    assert_replay_state_equal(agent.replay, replay_before)
    assert_replay_state_equal(agent.adaptation_replay, adaptation_before)
    assert agent._pending is not None
    assert torch.equal(agent.history.snapshot(), history_before)

    monkeypatch.setattr(agent.adaptation_replay, "_commit_prepared", original_second_commit)
    finish_agent_step(
        agent,
        frame,
        action,
        torch.ones(1, 1),
        truncated=torch.ones(1, 1),
        adaptation_enabled=True,
    )

    assert len(agent.replay) == 2
    assert len(agent.adaptation_replay) == 1
    assert agent._pending is None
    assert torch.equal(agent.history.snapshot(), torch.zeros(1, 3, 2))


@pytest.mark.parametrize(
    "invalid_gain",
    [True, float("nan"), float("inf"), float("-inf"), -0.01, 1.01, "0.5", None],
)
def test_agent_rejects_invalid_config_residual_gain(invalid_gain):
    with pytest.raises(ValueError, match="residual_gain"):
        make_agent(residual_gain=invalid_gain)


@pytest.mark.parametrize(
    "invalid_gain",
    [False, float("nan"), float("inf"), float("-inf"), -0.01, 1.01, "0.5", None],
)
def test_agent_rejects_invalid_runtime_residual_gain(invalid_gain):
    agent = make_agent()

    with pytest.raises(ValueError, match="residual_gain"):
        agent.set_residual_gain(invalid_gain)


@pytest.mark.parametrize(
    "valid_gain",
    [
        torch.tensor(0.5),
        torch.tensor(0, dtype=torch.int64),
        torch.tensor(1, dtype=torch.int32),
        np.float32(0.5),
        np.longdouble("0.5"),
    ],
)
def test_agent_accepts_real_numeric_scalar_residual_gain(valid_gain):
    agent = make_agent(residual_gain=valid_gain)

    assert agent.residual_gain == float(valid_gain)


@pytest.mark.parametrize(
    "invalid_gain",
    [
        torch.tensor(True),
        torch.tensor(0.5 + 0.0j),
        torch.tensor([0.5]),
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
        torch.tensor(float("-inf")),
        torch.tensor(-0.01),
        torch.tensor(1.01),
    ],
)
def test_agent_rejects_invalid_tensor_residual_gain(invalid_gain):
    with pytest.raises(ValueError, match="residual_gain"):
        make_agent(residual_gain=invalid_gain)


@pytest.mark.parametrize(
    "invalid_gain",
    [
        np.nextafter(np.longdouble(1), np.longdouble(2)),
        np.nextafter(np.longdouble(0), np.longdouble(-1)),
        Fraction(2**54 + 1, 2**54),
        Fraction(-1, 2**1075),
    ],
)
def test_agent_rejects_out_of_range_values_before_float_conversion(invalid_gain):
    assert float(invalid_gain) in (0.0, 1.0)

    with pytest.raises(ValueError, match="residual_gain"):
        make_agent(residual_gain=invalid_gain)


def test_fallback_pending_marks_residual_actor_version_as_unused():
    agent = make_agent()
    frame = torch.zeros(1, 2)
    base_action = torch.zeros(1, 1)
    normal_state = safety_state()
    learned_residual = torch.full_like(base_action, 0.2)
    for _ in range(10):
        agent.safety.project(base_action, learned_residual, normal_state)
    agent.begin_step(frame, base_action, normal_state, deterministic=True)

    fallback = agent.fallback_pending(normal_state)

    assert torch.equal(fallback.residual, torch.zeros_like(base_action))
    assert torch.equal(fallback.action, base_action)
    assert fallback.actor_version is None
    assert actor_version_for_record(fallback) is None


def test_timeout_after_rollback_fallback_pauses_and_excludes_transition_from_replay():
    agent = make_agent()
    frame = torch.zeros(1, 2)
    base_action = torch.zeros(1, 1)
    normal_state = safety_state()
    agent.begin_step(frame, base_action, normal_state, deterministic=True)
    agent.fallback_pending(normal_state)
    timed_out_state = replace(
        normal_state,
        control_timed_out=torch.ones_like(normal_state.control_timed_out),
    )

    decision = enforce_control_timeout_fallback(
        online_agent=agent,
        timed_out_state=timed_out_state,
    )
    finish_agent_step(
        agent,
        frame,
        base_action,
        torch.ones(1, 1),
        adaptation_enabled=True,
    )

    assert decision.paused.all()
    assert "control_timeout" in decision.reasons
    assert decision.actor_version is None
    assert len(agent.replay) == 0
    assert len(agent.adaptation_replay) == 0
