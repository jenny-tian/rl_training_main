"""Shared finite-horizon reward and ending contract for paired online trials."""

from dataclasses import asdict, dataclass
import hashlib
import json

import torch

from .reward import RewardWeights, compute_online_reward


@dataclass(frozen=True)
class TaskBoundary:
    reason: str | None
    success: bool = False
    terminated: bool = False
    truncated: bool = False

    @property
    def stop(self):
        return self.terminated or self.truncated

    def tensors(self, reference):
        return (
            torch.full_like(reference, self.terminated, dtype=torch.bool),
            torch.full_like(reference, self.truncated, dtype=torch.bool),
            torch.full_like(reference, not self.terminated, dtype=torch.float32),
        )


@dataclass(frozen=True)
class FiniteHorizonObjective:
    deadline_steps: int = 2000
    success_reward: float = 305.0
    failure_reward: float = -8.0

    def __post_init__(self):
        if type(self.deadline_steps) is not int or self.deadline_steps < 1:
            raise ValueError("task deadline must be a positive integer")
        if self.success_reward != 305.0 or self.failure_reward != -8.0:
            raise ValueError("finite-horizon-v1 fixes the terminal reward scale")

    def contract(self):
        return {
            "version": "lite3_finite_horizon_v1", **asdict(self),
            "reward_weights": asdict(RewardWeights()),
            "ordinary_reward": "velocity_progress_stall_body_and_foot_lateral_v1",
            "terminal_reward": "explicit_goal_value_not_unobserved_physics_tail",
            "bootstrap": "zero_for_task_terminal_one_for_collector_or_physical_truncation",
            "priority": ["physical_failure", "safety_failure", "qualification_failure", "success", "deadline", "physical_truncation", "collector_cutoff"],
        }

    def sha256(self):
        return hashlib.sha256(json.dumps(self.contract(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def boundary(self, *, step, collector_steps, physical_terminated=False,
                 physical_truncated=False, safety_failure=False, qualification_failure=None, success=False):
        if type(step) is not int or not 1 <= step <= self.deadline_steps:
            raise ValueError("transition step must be inside the declared task deadline")
        if type(collector_steps) is not int or not step <= collector_steps <= self.deadline_steps:
            raise ValueError("collector budget must cover the step without exceeding the task deadline")
        reason = ("physical_failure" if physical_terminated else
                  "safety_failure" if safety_failure else qualification_failure)
        if reason is not None:
            return TaskBoundary(reason, terminated=True)
        if success:
            return TaskBoundary("completion", success=True, terminated=True)
        if step == self.deadline_steps:
            return TaskBoundary("task_timeout", terminated=True)
        if physical_truncated:
            return TaskBoundary("physical_truncation", truncated=True)
        if step == collector_steps:
            return TaskBoundary("collector_cutoff", truncated=True)
        return TaskBoundary(None)

    def reward(self, signals, residual, previous_residual, *, forward_progress,
               stalled, base_y, foot_y, initial_y, boundary):
        lateral = torch.maximum((base_y - initial_y).abs(), (foot_y - initial_y).abs().amax(dim=-1, keepdim=True))
        ordinary = compute_online_reward(
            signals, residual, previous_residual, RewardWeights(),
            forward_progress=forward_progress,
            progress_stalled_mask=torch.full_like(forward_progress, float(stalled)),
            lateral_deviation=lateral,
        )
        total = (torch.full_like(ordinary, self.success_reward if boundary.success else self.failure_reward)
                 if boundary.terminated else ordinary)
        return total, {"ordinary_reward": ordinary, "terminal_reward_override": total - ordinary}


def qualification_failure(completion):
    if completion is None or not completion.strict_qualification_enabled:
        return None
    if completion.edge_slip_events:
        return "edge_slip"
    if completion.lateral_violation_steps:
        return "lateral_violation"
    return None


def add_task_objective_args(parser):
    parser.add_argument("--task-objective", choices=("legacy", "finite-horizon-v1"), default="legacy")
    parser.add_argument("--task-deadline-steps", type=int, default=2000)
    parser.add_argument("--task-objective-sha256", default=None)


def resolve_task_objective(args):
    if args.task_objective == "legacy":
        if args.task_objective_sha256 is not None:
            raise ValueError("legacy task has no common objective hash")
        return None
    objective = FiniteHorizonObjective(deadline_steps=args.task_deadline_steps)
    if args.flat_retention_probe:
        if not args.evaluation_only or args.require_track_completion:
            raise ValueError("flat measurement requires frozen evaluation without terrain completion")
    elif not args.require_track_completion:
        raise ValueError("finite-horizon-v1 requires strict terrain completion")
    if not 1 <= args.max_steps <= objective.deadline_steps:
        raise ValueError("max_steps must be inside the task deadline; use separate physical trials")
    if getattr(args, "continue_after_termination", False):
        raise ValueError("finite-horizon-v1 stops at each physical/task boundary")
    if getattr(args, "trial_deadline_steps", None) not in (None, objective.deadline_steps):
        raise ValueError("legacy and common task deadlines disagree")
    if getattr(args, "terminal_failure_penalty", -8.0) != -8.0 or getattr(args, "safety_projection_penalty", 0.0) != 0.0:
        raise ValueError("finite-horizon-v1 fixes failure reward and disables extra projection reward")
    if getattr(args, "target_semantics", "timeout-aware") != "timeout-aware":
        raise ValueError("finite-horizon-v1 requires timeout-aware targets")
    if getattr(args, "stall_penalty_weight", 2.0) != 2.0 or getattr(args, "completion_bonus", 5.0) != 5.0:
        raise ValueError("finite-horizon-v1 fixes the common reward weights")
    if args.task_objective_sha256 is not None and args.task_objective_sha256 != objective.sha256():
        raise ValueError("task objective SHA-256 mismatch")
    return objective


def validate_objective_resume(saved, objective):
    expected = None if objective is None else objective.contract()
    if saved != expected:
        raise ValueError("task objective differs: start a clean learner/replay for a changed reward or deadline")
