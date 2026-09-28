from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rl_training.online_adaptation.task_objective import FiniteHorizonObjective, resolve_task_objective, validate_objective_resume
from rl_training.online_adaptation.types import RewardSignals


def signals():
    return RewardSignals(
        base_lin_vel_xy=torch.tensor([[0.3, 0.]]), command_lin_vel_xy=torch.tensor([[0.3, 0.]]),
        base_ang_vel_z=torch.zeros(1, 1), command_ang_vel_z=torch.zeros(1, 1),
        projected_gravity_xy=torch.zeros(1, 2), base_lin_vel_z=torch.zeros(1, 1),
        foot_slip=torch.zeros(1, 1), foot_impact=torch.zeros(1, 1), joint_power=torch.zeros(1, 1),
        nonfoot_contact=torch.zeros(1, 1), foot_height=torch.zeros(1, 4), foot_contact=torch.ones(1, 4))


@pytest.mark.parametrize("kwargs,reason,terminal,truncated", [
    ({}, "collector_cutoff", False, True),
    ({"physical_terminated": True}, "physical_failure", True, False),
    ({"physical_truncated": True}, "physical_truncation", False, True),
    ({"success": True}, "completion", True, False),
    ({"safety_failure": True, "success": True}, "safety_failure", True, False),
    ({"qualification_failure": "lateral_violation", "success": True}, "lateral_violation", True, False),
])
def test_task_ends_and_sampling_cutoffs_have_distinct_bootstrap(kwargs, reason, terminal, truncated):
    boundary = FiniteHorizonObjective().boundary(step=128, collector_steps=128, **kwargs)
    assert boundary.reason == reason
    ended, cut, bootstrap = boundary.tensors(torch.zeros(1, 1))
    assert ended.item() is terminal and cut.item() is truncated
    assert bootstrap.item() == float(not terminal)


def test_success_on_deadline_wins_but_physical_failure_always_wins():
    objective = FiniteHorizonObjective()
    assert objective.boundary(step=2000, collector_steps=2000).reason == "task_timeout"
    assert objective.boundary(step=2000, collector_steps=2000, success=True).success
    assert not objective.boundary(step=2000, collector_steps=2000, success=True, physical_terminated=True).success
    assert not objective.boundary(step=1, collector_steps=2000).stop
    with pytest.raises(ValueError, match="deadline"):
        objective.boundary(step=2001, collector_steps=2001)


def test_common_reward_includes_feet_and_separates_explicit_terminal_reward():
    objective = FiniteHorizonObjective()
    inputs = dict(forward_progress=torch.ones(1, 1), stalled=False, base_y=torch.zeros(1, 1),
                  foot_y=torch.full((1, 4), .33), initial_y=0.)
    residual = torch.zeros(1, 12)
    for kwargs, expected in [({}, 2.96), ({"success": True}, 305.), ({"physical_terminated": True}, -8.)]:
        boundary = objective.boundary(step=1, collector_steps=128, **kwargs)
        reward, terms = objective.reward(signals(), residual, residual, boundary=boundary, **inputs)
        assert reward.item() == pytest.approx(expected)
        torch.testing.assert_close(reward, terms["ordinary_reward"] + terms["terminal_reward_override"])
    cutoff = objective.boundary(step=128, collector_steps=128)
    reward, _ = objective.reward(signals(), residual, residual, boundary=cutoff, **inputs)
    assert reward.item() == pytest.approx(2.96)


def test_deadline_changes_hash_and_old_learner_is_not_resumed():
    current = FiniteHorizonObjective()
    changed = replace(current, deadline_steps=1000)
    assert current.sha256() != changed.sha256()
    validate_objective_resume(current.contract(), current)
    for saved, objective in [(None, current), (current.contract(), None), (current.contract(), changed)]:
        with pytest.raises(ValueError, match="clean learner"):
            validate_objective_resume(saved, objective)


def test_common_objective_refuses_incompatible_collector_configuration():
    args = SimpleNamespace(task_objective="finite-horizon-v1", task_deadline_steps=2000,
                           task_objective_sha256=None, require_track_completion=True,
                           flat_retention_probe=False, max_steps=128)
    assert resolve_task_objective(args).deadline_steps == 2000
    args.max_steps = 2001
    with pytest.raises(ValueError, match="max_steps"):
        resolve_task_objective(args)


def test_flat_measurement_accepts_locked_objective_only_without_learning():
    args = SimpleNamespace(task_objective="finite-horizon-v1", task_deadline_steps=2000,
                           task_objective_sha256=None, require_track_completion=False,
                           flat_retention_probe=True, evaluation_only=True, max_steps=1000)
    assert resolve_task_objective(args).sha256() == FiniteHorizonObjective().sha256()
    args.evaluation_only = False
    with pytest.raises(ValueError, match="frozen evaluation"):
        resolve_task_objective(args)
