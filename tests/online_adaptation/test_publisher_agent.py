from dataclasses import replace

import torch

from rl_training.online_adaptation.agent import AgentConfig, OnlineAdaptationAgent
from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher
from rl_training.online_adaptation.safety import SafetyLimits
from rl_training.online_adaptation.types import SafetyState
from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import (
    actor_version_for_record,
    enforce_control_timeout_fallback,
)


def make_agent():
    actor = ResidualActor(2, 1, 4, torch.tensor([0.2]))
    critic = TwinCritic(2, 1, 4)
    return OnlineAdaptationAgent(
        num_envs=1,
        actor=actor,
        critic=critic,
        safety_limits=SafetyLimits(residual_limits=torch.tensor([0.2])),
        config=AgentConfig(history_length=3, frame_dim=2, action_dim=1, batch_size=1),
        device="cpu",
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
    agent.finish_step(
        frame,
        action,
        torch.ones(1, 1),
        torch.zeros(1, 1),
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


def test_fallback_pending_marks_residual_actor_version_as_unused():
    agent = make_agent()
    frame = torch.zeros(1, 2)
    base_action = torch.zeros(1, 1)
    agent.begin_step(frame, base_action, safety_state(), deterministic=True)

    fallback = agent.fallback_pending(safety_state())

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
    agent.finish_step(
        frame,
        base_action,
        torch.ones(1, 1),
        torch.zeros(1, 1),
        adaptation_enabled=True,
    )

    assert decision.paused.all()
    assert "control_timeout" in decision.reasons
    assert decision.actor_version is None
    assert len(agent.replay) == 0
    assert len(agent.adaptation_replay) == 0
