"""Experimental low-rank PPO worker for Lite3 online terrain adaptation.

The script is deliberately separate from ``play_online_adaptation.py``.  That
worker remains the SAC reference path; this entrypoint makes a paired PPO run
possible without changing its replay or safety semantics.  It uses the same
RSL-RL checkpoint and task, injects zero-output LoRA adapters into the actor,
and updates after on-policy blocks or explicit trial boundaries.
"""

from __future__ import annotations

import copy
import argparse
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


CONTACT_TERMINATION_PROTOCOL = "sustained_nonfoot_contact_5n_3steps_v1"
LEARNING_CONTRACT = "requested_ppo_absorbing_success_critic_context_v7"
REWARD_CONTRACT = "absorbing_success_explicit_task_deadline_v4"
CRITIC_OBSERVATION_SCHEMA = "policy_plus_body_velocity_xy_elapsed_v1"


def _prepare_evaluation_args(args: argparse.Namespace) -> None:
    if args.flat_retention_probe:
        if args.continue_after_termination:
            raise SystemExit("--flat-retention-probe cannot continue after a physical termination")
        args.evaluation_only = True
        args.require_track_completion = False
        args.trial_deadline_steps = None


def _evaluation_mode(args: argparse.Namespace) -> str:
    if not args.evaluation_only:
        return "online_learning"
    return "learned_frozen" if args.checkpoint_in else "base_frozen"


def _finalize_checkpoint(session, *, evaluation_only: bool,
                         retain_terminal_learning: bool, completion_stop: bool):
    # Frozen evaluation must save the same candidate it actually evaluated.
    return session.finalize_for_checkpoint(
        restore_unverified_candidate=not (evaluation_only or retain_terminal_learning or completion_stop)
    )


def _configure_forward_command(command_cfg, forward_speed: float) -> None:
    """Disable all sampled special modes for this fixed-forward experiment."""
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.ranges.lin_vel_x = (forward_speed, forward_speed)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    # UniformThresholdVelocityCommand samples these independently of the
    # standard standing ratio, even when all three ranges are fixed.
    for name in ("rel_zero_vel_envs", "rel_only_lin_y_envs",
                 "rel_only_lin_x_envs", "rel_only_ang_z_envs",
                 "planar_velocity_deadband"):
        if hasattr(command_cfg, name):
            setattr(command_cfg, name, 0.0)
    for name in ("command_acceleration_limits", "only_lin_x_magnitude_range"):
        if hasattr(command_cfg, name):
            setattr(command_cfg, name, None)


def _pin_forward_observation(wrapped_env, observation, forward_speed: float):
    """Refresh stale observations only if pinning changes their command."""
    import torch
    from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import pin_forward_command

    command = wrapped_env.unwrapped.command_manager.get_command("base_velocity")
    expected = command.new_tensor([forward_speed, 0.0, 0.0]).expand_as(command[:, :3])
    changed = not torch.equal(command[:, :3], expected)
    pin_forward_command(wrapped_env.unwrapped, forward_speed=forward_speed)
    return wrapped_env.get_observations() if changed else observation


def _assert_forward_command(signals, forward_speed: float) -> None:
    """Reject contaminated transitions before they enter the PPO buffer."""
    import torch

    expected = signals.command_lin_vel_xy.new_tensor([forward_speed, 0.0])
    if not (
        torch.isclose(signals.command_lin_vel_xy, expected, rtol=0.0, atol=1e-6).all()
        and torch.isclose(signals.command_ang_vel_z,
                          torch.zeros_like(signals.command_ang_vel_z), rtol=0.0, atol=1e-6).all()
    ):
        raise RuntimeError("fixed forward command changed during the physics step; refusing PPO data")


def _task_failure_reason(completion) -> str | None:
    # These events permanently disqualify the trial. Missing gait evidence
    # alone is provisional and must not terminate an otherwise valid approach.
    if completion is None or not completion.strict_qualification_enabled:
        return None
    if completion.edge_slip_events:
        return "edge_slip"
    if completion.lateral_violation_steps:
        return "lateral_violation"
    return None


def _deadline_failure(*, completed_steps: int, deadline_steps: int | None,
                      completion_passed: bool) -> str | None:
    if not completion_passed and deadline_steps is not None and completed_steps >= deadline_steps:
        return "task_timeout"
    return None


def _critic_time_horizon(*, max_steps: int, deadline_steps: int | None) -> int:
    return deadline_steps if deadline_steps is not None else max_steps


def _learning_termination(physical_terminated, *, rollback_reason: str | None,
                          task_failure_reason: str | None, completion_passed: bool):
    import torch

    task_ended = rollback_reason is not None or task_failure_reason is not None or completion_passed
    return physical_terminated | torch.full_like(physical_terminated, task_ended)


def _lateral_deviation(base_y, foot_y, initial_y: float):
    import torch

    return torch.maximum(
        (base_y - initial_y).abs(),
        (foot_y - initial_y).abs().amax(dim=-1, keepdim=True),
    )


def _capture_step_state(adapter):
    """Detach all diagnostics before Isaac Lab can overwrite them on reset."""
    from dataclasses import fields, replace

    signals = adapter.reward_signals()
    return {
        "signals": replace(signals, **{
            field.name: (None if getattr(signals, field.name) is None
                         else getattr(signals, field.name).detach().clone())
            for field in fields(signals)
        }),
        "base_x": adapter.base_x().detach().clone(),
        "base_y": adapter.base_y().detach().clone(),
        "foot_y": adapter.foot_y().detach().clone(),
        "contact": dict(adapter.observe_nonfoot_contact()),
    }


def _completion_passed(completion, *, fell: bool, safety_violation: bool, reset_count: int) -> bool:
    return bool(
        completion is not None
        and completion.strict_qualification_enabled
        and completion.reached
        and completion.qualification_satisfied
        and not fell and not safety_violation and reset_count == 0
    )


def _learning_boundary(*, retain_failures: bool, rollback_reason: str | None,
                       completion_passed: bool, environment_done: bool,
                       task_failure_reason: str | None = None) -> str | None:
    failure_reason = rollback_reason or task_failure_reason
    if failure_reason is not None:
        return failure_reason if retain_failures else None
    if completion_passed:
        return "completion"
    if environment_done and rollback_reason is None:
        return "truncated"
    return None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed >= 0:
        raise argparse.ArgumentTypeError("must be finite and negative")
    return parsed


def _rollback_reason(
    *,
    terminated: bool,
    nonfoot_contact: bool,
    max_nonfoot_contact_force_n: float,
    force_threshold_n: float,
) -> str | None:
    """Classify safety events before touching the on-policy session."""

    if not isinstance(terminated, bool) or not isinstance(nonfoot_contact, bool):
        raise TypeError("terminated and nonfoot_contact must be bool values")
    if (
        not math.isfinite(max_nonfoot_contact_force_n)
        or not math.isfinite(force_threshold_n)
        or force_threshold_n <= 0
    ):
        raise ValueError("contact force values must be finite and threshold must be positive")
    if terminated:
        return "terminated"
    if nonfoot_contact and max_nonfoot_contact_force_n >= force_threshold_n:
        return "severe_nonfoot_contact"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run low-rank online PPO adaptation from a frozen Lite3 PPO checkpoint.",
        allow_abbrev=False,
    )
    # AppLauncher inspects the parser while adding its own flags, so a
    # required application argument would be rejected before parse_args gets
    # the actual command line.  Validate it in main after parsing instead.
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--task", default="FlatOnline-Deeprobotics-Lite3-v0")
    parser.add_argument(
        "--terrain-stage",
        choices=("step_2cm", "step_4cm", "step_6cm", "rough_2cm", "rough_4cm"),
        default="rough_4cm",
    )
    parser.add_argument("--terrain-seed", type=int, default=101)
    parser.add_argument(
        "--protocol-sha256",
        default=None,
        help="expected formal terrain/completion protocol hash",
    )
    parser.add_argument(
        "--require-track-completion",
        action="store_true",
        help="record and enforce the formal reset-relative completion qualification",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--forward-speed", type=_positive_float, default=0.3)
    parser.add_argument("--max-steps", type=_positive_int, default=2_000)
    parser.add_argument(
        "--trial-deadline-steps", type=_positive_int, default=None,
        help="declare a task deadline as terminal failure; an earlier max-steps cut still bootstraps",
    )
    parser.add_argument("--rollout-steps", type=_positive_int, default=256)
    parser.add_argument("--output", default="logs/online_adaptation/lora_ppo")
    parser.add_argument(
        "--checkpoint-in",
        default=None,
        help="load a completed online PPO checkpoint before running",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="run a checkpoint, or the frozen base without one, deterministically without PPO updates",
    )
    parser.add_argument(
        "--flat-retention-probe",
        action="store_true",
        help="evaluate on an infinite plane with steady velocity/progress metrics and no track qualification",
    )
    parser.add_argument(
        "--continue-after-termination",
        action="store_true",
        help="keep consuming the step budget after a termination when a stop boundary is active",
    )
    parser.add_argument(
        "--stop-after-termination",
        action="store_true",
        help="end training at the first physical termination so a supervisor can own episode boundaries",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=_positive_int, default=None)
    parser.add_argument("--rank", type=_positive_int, default=2)
    parser.add_argument("--lora-alpha", type=_positive_float, default=None)
    parser.add_argument(
        "--action-std",
        type=_positive_float,
        default=0.01,
        help="Gaussian exploration standard deviation, applied after restoring training state",
    )
    parser.add_argument("--residual-rate-limit", type=_positive_float, default=0.03)
    parser.add_argument("--actor-lr", type=_positive_float, default=5e-5)
    parser.add_argument("--critic-lr", type=_positive_float, default=5e-5)
    parser.add_argument(
        "--nonfoot-contact-rollback-force",
        type=_positive_float,
        default=20.0,
        help="rollback threshold in N for non-foot contact during online PPO",
    )
    parser.add_argument("--ppo-epochs", type=_positive_int, default=2)
    parser.add_argument("--minibatch-size", type=_positive_int, default=64)
    parser.add_argument("--clip-eps", type=_positive_float, default=0.1)
    parser.add_argument("--target-kl", type=_positive_float, default=0.01)
    parser.add_argument("--gamma", type=_positive_float, default=0.99)
    parser.add_argument("--gae-lambda", type=_positive_float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--reference-anchor-coef",
        type=_nonnegative_float,
        default=1.0,
        help="penalty on mean-action drift from the frozen base actor",
    )
    parser.add_argument(
        "--flat-anchor-observations", default=None,
        help="verified reconstructed source-flat observations; teacher means are recomputed on these inputs",
    )
    parser.add_argument(
        "--max-reference-mse",
        type=_positive_float,
        default=1e-4,
        help="reject a PPO candidate whose mean-action MSE from the frozen base actor exceeds this value",
    )
    parser.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    parser.add_argument("--max-parameter-abs", type=_positive_float, default=None)
    parser.add_argument(
        "--terminal-failure-penalty",
        type=_negative_float,
        default=-8.0,
        help="reward assigned to a physical or strict qualification failure row",
    )
    parser.add_argument(
        "--retain-terminal-learning",
        action="store_true",
        help="end at the first failure and retain its PPO update for supervised simulation training",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.base_checkpoint:
        raise SystemExit("--base-checkpoint is required")
    _prepare_evaluation_args(args)
    if args.retain_terminal_learning and args.continue_after_termination:
        raise SystemExit("terminal learning cannot continue the failed physical trial")
    # AppLauncher consumes this flag while constructing the simulation.  Set
    # it before launch so ``--video`` can actually provision camera sensors;
    # changing the Namespace afterwards is too late for Isaac Lab startup.
    if args.video:
        args.enable_cameras = True
    simulation_app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch
        from torch import nn
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        import rl_training.tasks  # noqa: F401
        from rl_training.lite3_joint_conventions import clip_lite3_policy_action
        from rl_training.online_adaptation.lora import inject_lora, lora_parameter_count, mark_only_lora_trainable
        from rl_training.online_adaptation.online_session import OnlinePPOSession
        from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater, PPOConfig
        from rl_training.online_adaptation.ppo_safety import SAMPLING_CONTRACT, requested_action_rollout
        from rl_training.online_adaptation.ppo_critic_context import (
            build_critic_observation, migrate_critic_context,
        )
        from rl_training.online_adaptation.ppo_task_reward import success_absorbing_reward
        from rl_training.online_adaptation.ppo_source_anchor import load_source_anchor
        from rl_training.online_adaptation.flat_retention import (
            configure_flat_retention_plane, summarize_flat_retention,
        )
        from rl_training.online_adaptation.ppo_runtime_config import (
            apply_ppo_runtime_config,
            read_ppo_runtime_config,
        )
        from rl_training.online_adaptation.progress_novelty import ForwardProgressMonitor
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
        from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
        from rl_training.online_adaptation.rsl_rl_compat import (
            RslMLPPolicyAdapter,
            freeze_rsl_rl_models,
            load_frozen_rsl_rl_checkpoint,
            prepare_rsl_rl_agent_cfg,
            split_rsl_rl_endings,
        )
        from rl_training.online_adaptation.terrain_protocol import (
            apply_geometry_to_env_cfg,
            default_flat_online_protocol,
            development_geometry,
        )
        from rl_training.online_adaptation.swing_lift import SwingLiftTracker
        from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import (
            build_track_completion,
            normalized_forward_progress,
        )

        torch.manual_seed(args.seed)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = 1
        _configure_forward_command(env_cfg.commands.base_velocity, args.forward_speed)
        env_cfg.terminations.time_out = None
        if args.flat_retention_probe:
            terrain_geometry = configure_flat_retention_plane(env_cfg)
        else:
            geometry = development_geometry(args.terrain_stage, args.terrain_seed)
            apply_geometry_to_env_cfg(env_cfg, geometry)
            terrain_geometry = {
                "step_height": geometry.step_height,
                "rough_amplitude": geometry.rough_amplitude,
                "seed": geometry.seed,
            }
        protocol = default_flat_online_protocol()
        if args.protocol_sha256 is not None and args.protocol_sha256 != protocol.sha256():
            raise ValueError("protocol SHA-256 does not match the active formal protocol")
        agent_cfg = prepare_rsl_rl_agent_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"))
        def transform_video(environment):
            if not args.video:
                return environment
            return gym.wrappers.RecordVideo(
                environment,
                video_folder=str(output / "videos"),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length or args.max_steps,
                disable_logger=True,
            )

        env = gym.make(
            args.task,
            cfg=env_cfg,
            **({"render_mode": "rgb_array"} if args.video else {}),
        )
        env = transform_video(env)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        try:
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            checkpoint_load = load_frozen_rsl_rl_checkpoint(runner, args.base_checkpoint)
            source_model = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
            # Freeze the source model first, then inject adapters.  The RSL-RL
            # runner is not used for learning after this point.
            freeze_rsl_rl_models(runner)
            adapted_model = copy.deepcopy(source_model)
            adapters = inject_lora(adapted_model, rank=args.rank, alpha=args.lora_alpha)
            distribution = getattr(adapted_model, "distribution", None)
            if hasattr(distribution, "log_std_param"):
                distribution.log_std_param.data.fill_(math.log(args.action_std))
            elif hasattr(distribution, "std_param"):
                distribution.std_param.data.fill_(args.action_std)
            else:
                raise RuntimeError("RSL actor must expose a configurable Gaussian action standard deviation")
            mark_only_lora_trainable(adapted_model)
            # The adapter owns the trainable LoRA parameters while the
            # publisher holds a frozen inference copy.  Keep both on the same
            # device so publication is a pure state-dict swap.
            policy = RslMLPPolicyAdapter(
                adapted_model, action_transform=clip_lite3_policy_action
            ).to(args.device)
            reference_policy = RslMLPPolicyAdapter(
                source_model, action_transform=clip_lite3_policy_action
            ).to(args.device)
            reference_policy.eval()
            probe_obs, _ = wrapped_env.reset()
            probe_obs = _pin_forward_observation(wrapped_env, probe_obs, args.forward_speed)
            if not isinstance(probe_obs, dict) and not hasattr(probe_obs, "get"):
                raise RuntimeError("RSL-RL online PPO requires grouped policy observations")
            policy_obs = probe_obs["policy"]
            if not isinstance(policy_obs, torch.Tensor) or policy_obs.ndim != 2:
                raise RuntimeError("policy observation must have shape [1, obs_dim]")
            obs_dim = int(policy_obs.shape[-1])
            if args.flat_retention_probe and obs_dim != 45:
                raise RuntimeError("flat retention requires the deployed 45-dimensional Lite3 actor observation")
            source_anchor_observations = source_anchor_means = None
            source_anchor_metadata = None
            if args.flat_anchor_observations is not None:
                source_observations, source_anchor_metadata = load_source_anchor(
                    args.flat_anchor_observations, base_checkpoint_path=args.base_checkpoint,
                )
                if source_observations.shape[1] != obs_dim:
                    raise ValueError("source-flat anchor width differs from the frozen policy input")
                source_anchor_observations = source_observations.to(device=args.device)
                with torch.no_grad():
                    source_anchor_means = reference_policy.mean_action(source_anchor_observations).detach()
            critic = nn.Sequential(
                nn.Linear(obs_dim + 3, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1)
            ).to(args.device)
            updater = OnlinePPOUpdater(
                policy,
                critic,
                PPOConfig(
                    clip_eps=args.clip_eps,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    entropy_coef=args.entropy_coef,
                    value_coef=args.value_coef,
                    flat_anchor_coef=args.reference_anchor_coef,
                    max_anchor_mse=args.max_reference_mse,
                    max_grad_norm=args.max_grad_norm,
                    target_kl=args.target_kl,
                    max_parameter_abs=args.max_parameter_abs,
                ),
            )
            from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher

            publisher = DoubleBufferedActorPublisher(policy)
            session = OnlinePPOSession(updater, rollout_capacity=args.rollout_steps, publisher=publisher)
            loaded_checkpoint = None
            checkpoint_migration = None
            if args.checkpoint_in:
                loaded_checkpoint = torch.load(
                    args.checkpoint_in,
                    map_location=args.device,
                    weights_only=False,
                )
                if not isinstance(loaded_checkpoint, dict) or loaded_checkpoint.get("schema_version") != 1:
                    raise ValueError("unsupported online PPO checkpoint")
                if loaded_checkpoint.get("lora_rank") != args.rank:
                    raise ValueError("checkpoint LoRA rank differs from the requested rank")
                checkpoint_session = loaded_checkpoint.get("session")
                if not isinstance(checkpoint_session, dict):
                    raise ValueError("online PPO checkpoint is missing session state")
                checkpoint_migration = {
                    "source_learning_contract": loaded_checkpoint.get("learning_contract"),
                    "critic_input_from": int(checkpoint_session["updater"]["critic"]["0.weight"].shape[1]),
                    "critic_input_to": obs_dim + 3,
                }
                if (checkpoint_migration["critic_input_from"] == obs_dim + 3
                        and loaded_checkpoint.get("critic_observation_schema") != CRITIC_OBSERVATION_SCHEMA):
                    raise ValueError("checkpoint critic observation schema does not match the requested context")
                checkpoint_session = migrate_critic_context(checkpoint_session, policy_obs_dim=obs_dim)
                session.load_state_dict(checkpoint_session)
            effective_runtime_config = (
                read_ppo_runtime_config(session)
                if args.evaluation_only else apply_ppo_runtime_config(
                    session, action_std=args.action_std,
                    actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                )
            )
            initial_policy_version = effective_runtime_config.pop("policy_version")
            reward_weights = RewardWeights()
            completion_terminal_reward = None if args.flat_retention_probe else success_absorbing_reward(
                args.gamma,
                continuing_reward_upper_bound=(reward_weights.velocity_weight
                                               + reward_weights.yaw_weight
                                               + reward_weights.forward_progress_weight),
                completion_bonus=reward_weights.completion_bonus,
            )
            critic_time_horizon = _critic_time_horizon(
                max_steps=args.max_steps, deadline_steps=args.trial_deadline_steps,
            )
            (output / "config.json").write_text(
                json.dumps(
                    {**vars(args), "contact_termination": CONTACT_TERMINATION_PROTOCOL,
                     "mode": _evaluation_mode(args),
                     "evaluation_kind": "flat_retention" if args.flat_retention_probe else "terrain",
                     "terrain_geometry": terrain_geometry,
                     "learning_contract": LEARNING_CONTRACT,
                     "sampling_contract": SAMPLING_CONTRACT,
                     "reward_contract": REWARD_CONTRACT,
                     "critic_observation_schema": CRITIC_OBSERVATION_SCHEMA,
                     "checkpoint_migration": checkpoint_migration,
                     "completion_terminal_reward": completion_terminal_reward,
                     "critic_time_horizon_steps": critic_time_horizon,
                     "reference_anchor_distribution": (
                         "fixed_reconstructed_source_flat_v1" if source_anchor_metadata is not None
                         else "current_online_observations_v1"
                     ),
                     "source_anchor_metadata": source_anchor_metadata,
                     "effective_runtime_config": effective_runtime_config,
                     "initial_policy_version": initial_policy_version},
                    indent=2, default=str,
                ) + "\n", encoding="utf-8",
            )
            reward_adapter = Lite3IsaacLabAdapter(wrapped_env.unwrapped, terrain_feature_source="proprio")
            initial_x = float(reward_adapter.base_x().mean().item())
            initial_y = float(reward_adapter.base_y().mean().item())
            track_completion = None
            if args.require_track_completion:
                track_completion = build_track_completion(
                    env_cfg,
                    stage_name=args.terrain_stage,
                    initial_x=initial_x,
                    initial_y=initial_y,
                    protocol=protocol,
                )
            swing_lift_tracker = SwingLiftTracker()
            if not hasattr(wrapped_env.unwrapped, "add_pre_reset_callback"):
                raise RuntimeError("online PPO task must support pre-reset terminal capture")
            safety_filter = SafetyFilter(SafetyLimits(
                residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4, device=args.device),
                residual_rate_limit=args.residual_rate_limit,
            ))
            observation = probe_obs
            terminal_capture = {}
            current_residual = torch.zeros(1, 12, device=args.device)
            previous_residual = torch.zeros_like(current_residual)
            previous_x = reward_adapter.base_x().detach().clone()
            control_step_dt = float(env_cfg.sim.dt) * float(getattr(env_cfg, "decimation", 1))
            progress_monitor = ForwardProgressMonitor(min_delta_m=0.05, stall_steps=100)
            progress_monitor.observe(
                base_x=initial_x,
                commanded_forward=bool(args.forward_speed > 0.05),
            )
            anchor_observations = []
            anchor_means = []

            def reference_anchor_batch():
                if source_anchor_observations is not None:
                    return {
                        "flat_anchor_observations": source_anchor_observations,
                        "flat_anchor_means": source_anchor_means,
                    }
                return {
                    "flat_anchor_observations": torch.cat(anchor_observations, dim=0),
                    "flat_anchor_means": torch.cat(anchor_means, dim=0),
                }

            pending_valid_steps = 0
            invalid_rollout_blocks = 0

            def capture_terminal_state(env_ids):
                ids = torch.as_tensor(env_ids, dtype=torch.long, device=args.device)
                terminal_observation = wrapped_env.unwrapped.observation_manager.compute()["policy"]
                terminal_capture["ids"] = ids
                terminal_capture["observation"] = terminal_observation.detach().clone()
                terminal_capture["state"] = _capture_step_state(reward_adapter)

            wrapped_env.unwrapped.add_pre_reset_callback(capture_terminal_state)
            rows = []
            update_records = []
            rollback_records = []
            terminal_learning_records = []
            completion_stop = False
            task_failure_reason = None
            fell = False
            safety_violation = False
            reset_count = 0
            stop_reason = "step_budget"
            updates_stream = (output / "updates.jsonl").open("w", encoding="utf-8")
            for step in range(args.max_steps):
                previous_residual = current_residual.detach().clone()
                terminal_capture.clear()
                observation = _pin_forward_observation(wrapped_env, observation, args.forward_speed)
                obs_tensor = observation["policy"]
                critic_obs_tensor = build_critic_observation(
                    obs_tensor, reward_adapter.robot.data.root_lin_vel_b[:, :2],
                    step=step, max_steps=critic_time_horizon,
                )
                control_policy, version = publisher.current()
                with torch.no_grad():
                    reference_mean = reference_policy.mean_action(obs_tensor)
                    if args.evaluation_only:
                        requested_action = control_policy.mean_action(obs_tensor)
                        control_mean = requested_action
                        old_log_prob = torch.zeros(1, 1, device=args.device)
                        old_value = torch.zeros(1, 1, device=args.device)
                    else:
                        distribution = control_policy.get_distribution(obs_tensor)
                        control_mean = distribution.mean
                        requested_action = distribution.sample()
                        old_log_prob = distribution.log_prob(requested_action)
                        if old_log_prob.ndim > 1:
                            old_log_prob = old_log_prob.sum(dim=-1)
                        old_value = critic(critic_obs_tensor)
                    base_action = reference_mean
                    requested_residual = requested_action - base_action
                    safety_decision = safety_filter.project(
                        base_action,
                        requested_residual,
                        reward_adapter.safety_state(torch.zeros(1, 1, dtype=torch.bool, device=args.device)),
                    )
                    executed_action = safety_decision.action
                    current_residual = safety_decision.residual.detach().clone()
                result = wrapped_env.step(executed_action)
                next_observation, _, done, extras = result
                terminated, truncated, _ = split_rsl_rl_endings(done, extras)
                next_obs_tensor = next_observation["policy"]
                environment_done = bool((terminated | truncated).any().item())
                value_observation = next_obs_tensor
                if environment_done:
                    if "observation" not in terminal_capture or "state" not in terminal_capture:
                        raise RuntimeError("ended online PPO row has no pre-reset terminal capture")
                    value_observation = terminal_capture["observation"]
                    step_state = terminal_capture["state"]
                else:
                    step_state = _capture_step_state(reward_adapter)
                signals = step_state["signals"]
                _assert_forward_command(signals, args.forward_speed)
                swing_lift = swing_lift_tracker.observe(signals.foot_height, signals.foot_contact)
                velocity_error = torch.linalg.vector_norm(
                    signals.base_lin_vel_xy - signals.command_lin_vel_xy, dim=-1, keepdim=True
                )
                tilt_rms = torch.linalg.vector_norm(signals.projected_gravity_xy, dim=-1, keepdim=True)
                contact_diagnostics = step_state["contact"]
                max_nonfoot_contact_force_n = float(
                    contact_diagnostics["max_nonfoot_contact_force_n"]
                )
                rollback_reason = _rollback_reason(
                    terminated=bool(terminated.any().item()),
                    nonfoot_contact=bool(signals.nonfoot_contact.any().item()),
                    max_nonfoot_contact_force_n=max_nonfoot_contact_force_n,
                    force_threshold_n=args.nonfoot_contact_rollback_force,
                )
                fell |= bool(terminated.any().item())
                safety_violation |= (
                    rollback_reason == "severe_nonfoot_contact"
                    or bool(safety_decision.paused.any().item())
                )
                reset_count += int(environment_done)
                current_x = step_state["base_x"]
                progress_valid = (~done.bool()).reshape_as(current_x)
                if bool(safety_decision.paused.any().item()):
                    progress_valid = torch.zeros_like(progress_valid)
                forward_progress = normalized_forward_progress(
                    current_x - previous_x,
                    forward_speed=args.forward_speed,
                    step_dt=control_step_dt,
                    valid_mask=progress_valid,
                )
                progress_decision = progress_monitor.observe(
                    base_x=float(current_x.mean().item()),
                    commanded_forward=bool(args.forward_speed > 0.05),
                    paused=bool(safety_decision.paused.any().item()),
                )
                if track_completion is not None:
                    track_completion.observe_pose(
                        float(current_x.mean().item()),
                        float(step_state["base_y"].mean().item()),
                        foot_y=step_state["foot_y"].reshape(-1).cpu().tolist(),
                        swing_lift=swing_lift.detach().reshape(-1).cpu().tolist(),
                        velocity_error=float(velocity_error.mean().item()),
                        tilt_rms=float(tilt_rms.mean().item()),
                        foot_slip=float(signals.foot_slip.mean().item()),
                    )
                completion_passed = _completion_passed(
                    track_completion, fell=fell, safety_violation=safety_violation,
                    reset_count=reset_count,
                )
                task_failure_reason = _task_failure_reason(track_completion) or _deadline_failure(
                    completed_steps=step + 1, deadline_steps=args.trial_deadline_steps,
                    completion_passed=completion_passed,
                )
                lateral_deviation = _lateral_deviation(
                    step_state["base_y"], step_state["foot_y"], initial_y
                )
                # Failure learning must finish the current on-policy block
                # before any rollback changes its policy version.  Continuing
                # after the first severe contact used to erase this block,
                # leaving only masked rows when the fall finally arrived.
                learning_boundary = _learning_boundary(
                    retain_failures=args.retain_terminal_learning,
                    rollback_reason=rollback_reason,
                    completion_passed=completion_passed,
                    environment_done=environment_done,
                    task_failure_reason=task_failure_reason,
                )
                learning_terminated = _learning_termination(
                    terminated, rollback_reason=rollback_reason,
                    task_failure_reason=task_failure_reason, completion_passed=completion_passed,
                )
                learning_truncated = (truncated | torch.full_like(
                    truncated, step + 1 == args.max_steps
                )) & ~learning_terminated
                reward_tensor = compute_online_reward(
                    signals,
                    current_residual,
                    previous_residual,
                    reward_weights,
                    forward_progress=forward_progress,
                    progress_stalled_mask=torch.full_like(
                        forward_progress, float(progress_decision.stalled)
                    ),
                    completion_mask=torch.full_like(
                        forward_progress, float(completion_passed)
                    ),
                    lateral_deviation=lateral_deviation,
                ).to(dtype=torch.float32)
                previous_x = current_x
                if completion_passed:
                    # Include the known reward tail of the absorbing success
                    # state. The physical transition remains truly terminal;
                    # there is no critic or post-reset observation bootstrap.
                    reward_tensor = torch.full_like(reward_tensor, completion_terminal_reward)
                elif (bool(terminated.any().item()) or learning_boundary == "severe_nonfoot_contact"
                        or task_failure_reason is not None):
                    reward_tensor = torch.full_like(
                        reward_tensor, float(args.terminal_failure_penalty)
                    ).to(dtype=torch.float32)
                if not args.evaluation_only:
                    with torch.no_grad():
                        next_critic_observation = build_critic_observation(
                            value_observation, signals.base_lin_vel_xy,
                            step=step + 1, max_steps=critic_time_horizon,
                        )
                        next_value_raw = critic(next_critic_observation)
                        next_value = next_value_raw * (~learning_terminated).to(dtype=next_value_raw.dtype)
                    transition = requested_action_rollout(
                        observations=obs_tensor,
                        requested_actions=requested_action,
                        old_log_probs=old_log_prob,
                        old_values=old_value,
                        rewards=reward_tensor,
                        terminated=learning_terminated,
                        truncated=learning_truncated,
                        next_values=next_value,
                        policy_version=version,
                        safety_decision=safety_decision,
                        critic_observations=critic_obs_tensor,
                    )
                    session.append(transition)
                    anchor_observations.append(obs_tensor.detach())
                    anchor_means.append(reference_mean.detach())
                    pending_valid_steps += int(transition.valid_mask.sum().item())
                rows.append({
                    "step": step + 1,
                    "reward": float(reward_tensor.mean()),
                    "old_value": None if args.evaluation_only else float(old_value.mean()),
                    "terminal_raw_advantage": (
                        float((reward_tensor - old_value).mean())
                        if not args.evaluation_only and bool(learning_terminated.all()) else None
                    ),
                    "policy_reference_mse": float((control_mean - reference_mean).square().mean()),
                    "mean_residual_max_abs": float((control_mean - reference_mean).abs().max()),
                    "base_x": float(current_x.mean()),
                    "base_y": float(step_state["base_y"].mean()),
                    "foot_y": step_state["foot_y"].reshape(-1).cpu().tolist(),
                    "foot_slip": float(signals.foot_slip.mean()),
                    "velocity_error": float(velocity_error.mean()),
                    "command_lin_vel_xy": signals.command_lin_vel_xy.reshape(-1).cpu().tolist(),
                    "command_ang_vel_z": float(signals.command_ang_vel_z.mean()),
                    "base_lin_vel_xy": signals.base_lin_vel_xy.reshape(-1).cpu().tolist(),
                    "tilt_rms": float(tilt_rms.mean()),
                    "nonfoot_contact": bool(signals.nonfoot_contact.any().item()),
                    "max_nonfoot_contact_force_n": max_nonfoot_contact_force_n,
                    "done": bool(done.any()),
                    "policy_version": version,
                    "safety_projected": bool(safety_decision.projected.any()),
                    "safety_paused": bool(safety_decision.paused.any()),
                    "safety_reasons": list(safety_decision.reasons),
                    "requested_residual_max_abs": float(requested_residual.abs().max()),
                    "executed_residual_max_abs": float(current_residual.abs().max()),
                    "safety_projection_max_abs": float(
                        (current_residual - requested_residual).abs().max()
                    ),
                    "ppo_valid": bool((~safety_decision.paused).all()),
                    "forward_progress": float(forward_progress.mean()),
                    "lateral_deviation": float(lateral_deviation.mean()),
                    "progress_stalled": bool(progress_decision.stalled),
                    "rollback_reason": rollback_reason or "",
                    "task_failure_reason": task_failure_reason,
                    "completion_passed": completion_passed,
                    "learning_boundary": learning_boundary,
                    "learning_terminated": bool(learning_terminated.any()),
                    "learning_truncated": bool(learning_truncated.any()),
                })
                observation = next_observation
                if environment_done:
                    safety_filter.reset_temporal_state()
                    swing_lift_tracker.reset(done)
                    previous_x = reward_adapter.base_x().detach().clone()
                    progress_monitor = ForwardProgressMonitor(min_delta_m=0.05, stall_steps=100)
                    current_residual.zero_()
                    previous_residual.zero_()
                if (
                    not args.evaluation_only
                    and (learning_boundary is not None or (
                        args.retain_terminal_learning and step + 1 == args.max_steps
                    ))
                ):
                    # Keep the terminal transition in the learning signal. The
                    # supervisor still decides whether this candidate becomes
                    # the next long-lived checkpoint.
                    stats = None
                    boundary_transitions = session.pending_transitions
                    if pending_valid_steps > 0:
                        stats = session.update(
                            force=True,
                            **reference_anchor_batch(),
                        )
                        if stats is not None:
                            update_records.append(stats.__dict__)
                            updates_stream.write(json.dumps(stats.__dict__, sort_keys=True) + "\n")
                            updates_stream.flush()
                    else:
                        session.clear(discarded=True)
                    anchor_observations.clear()
                    anchor_means.clear()
                    pending_valid_steps = 0
                    boundary_record = {
                        "event": "boundary_learning_update",
                        "reason": learning_boundary or "step_budget",
                        "policy_version": updater.policy_version,
                        "transitions": boundary_transitions,
                        "update_attempted": stats is not None,
                        "accepted": stats.accepted if stats is not None else False,
                    }
                    terminal_learning_records.append(boundary_record)
                    updates_stream.write(json.dumps(boundary_record, sort_keys=True) + "\n")
                    updates_stream.flush()
                elif not args.evaluation_only and (rollback_reason or task_failure_reason) is not None:
                    rollback = session.rollback(rollback_reason or task_failure_reason)
                    anchor_observations.clear()
                    anchor_means.clear()
                    pending_valid_steps = 0
                    rollback_record = {"event": "rollback", **rollback.__dict__}
                    rollback_records.append(rollback_record)
                    updates_stream.write(json.dumps(rollback_record, sort_keys=True) + "\n")
                    updates_stream.flush()
                    print(
                        f"[LoRA-PPO] step={step + 1} rollback={rollback.reason} "
                        f"version={rollback.policy_version_before}->{rollback.policy_version_after}",
                        flush=True,
                    )
                elif not args.evaluation_only and session.ready:
                    stats = None
                    if pending_valid_steps == 0:
                        invalid_rollout_blocks += 1
                        discarded = session.pending_transitions
                        session.clear(discarded=True)
                        anchor_observations.clear()
                        anchor_means.clear()
                        pending_valid_steps = 0
                        record = {
                            "event": "discard_all_paused",
                            "discarded_transitions": discarded,
                            "policy_version": updater.policy_version,
                        }
                        updates_stream.write(json.dumps(record, sort_keys=True) + "\n")
                        updates_stream.flush()
                    else:
                        stats = session.update(
                            **reference_anchor_batch(),
                        )
                        anchor_observations.clear()
                        anchor_means.clear()
                        pending_valid_steps = 0
                    if stats is not None:
                        update_records.append(stats.__dict__)
                        updates_stream.write(json.dumps(stats.__dict__, sort_keys=True) + "\n")
                        updates_stream.flush()
                        print(
                            f"[LoRA-PPO] step={step + 1} version={updater.policy_version} "
                            f"accepted={stats.accepted} kl={stats.approx_kl:.6g}",
                            flush=True,
                        )
                if completion_passed:
                    completion_stop = True
                    stop_reason = "completion"
                    break
                if args.flat_retention_probe and (rollback_reason is not None or safety_violation):
                    stop_reason = rollback_reason or "safety_pause"
                    break
                if task_failure_reason is not None:
                    stop_reason = rollback_reason or task_failure_reason
                    break
                if args.retain_terminal_learning and learning_boundary is not None:
                    stop_reason = learning_boundary
                    break
                if (
                    environment_done
                    and (args.evaluation_only or args.stop_after_termination)
                    and not args.continue_after_termination
                ):
                    stop_reason = "terminated" if fell else "truncated"
                    break
            finalization = _finalize_checkpoint(
                session, evaluation_only=args.evaluation_only,
                retain_terminal_learning=args.retain_terminal_learning, completion_stop=completion_stop,
            )
            finalization_record = {"event": "finalize_stable", **finalization.__dict__}
            updates_stream.write(json.dumps(finalization_record, sort_keys=True) + "\n")
            updates_stream.flush()
            updates_stream.close()
            torch.save(
                {
                    "schema_version": 1,
                    "mode": _evaluation_mode(args),
                    "evaluation_kind": "flat_retention" if args.flat_retention_probe else "terrain",
                    "learning_contract": LEARNING_CONTRACT,
                    "sampling_contract": SAMPLING_CONTRACT,
                    "reward_contract": REWARD_CONTRACT,
                    "critic_observation_schema": CRITIC_OBSERVATION_SCHEMA,
                    "checkpoint_migration": checkpoint_migration,
                    "critic_time_horizon_steps": critic_time_horizon,
                    "trial_deadline_steps": args.trial_deadline_steps,
                    "discount_gamma": args.gamma,
                    "completion_terminal_reward": completion_terminal_reward,
                    "reference_anchor_distribution": (
                        "fixed_reconstructed_source_flat_v1" if source_anchor_metadata is not None
                        else "current_online_observations_v1"
                    ),
                    "source_anchor_metadata": source_anchor_metadata,
                    "effective_runtime_config": effective_runtime_config,
                    "base_checkpoint": str(args.base_checkpoint),
                    "checkpoint_load": checkpoint_load,
                    "contact_termination": CONTACT_TERMINATION_PROTOCOL,
                    "terrain_geometry": terrain_geometry,
                    "lora_rank": args.rank,
                    "lora_parameter_count": lora_parameter_count(adapted_model),
                    "session": session.state_dict(),
                    "rows": rows,
                    "updates": update_records,
                    "rollbacks": rollback_records,
                    "terminal_learning_updates": terminal_learning_records,
                    "finalization": finalization_record,
                },
                output / "latest.pt",
            )
            steps = len(rows)
            midpoint = max(1, steps // 2)
            def _mean(field, start, end):
                values = [float(row[field]) for row in rows[start:end]]
                return sum(values) / len(values) if values else 0.0

            max_base_x = max((float(row["base_x"]) for row in rows), default=initial_x)
            completion_evidence = (
                track_completion.evidence(
                    fell=fell,
                    safety_violation=safety_violation,
                )
                if track_completion is not None
                else {
                    "initial_x": initial_x,
                    "max_x": max_base_x,
                    "completion_target_x": None,
                    "completion_mode": "disabled",
                    "strict_qualification_enabled": False,
                    "completion_qualification_passed": False,
                    "completion_reached": False,
                }
            )
            summary = {
                "mode": _evaluation_mode(args),
                "evaluation_kind": "flat_retention" if args.flat_retention_probe else "terrain",
                "learning_contract": LEARNING_CONTRACT,
                "sampling_contract": SAMPLING_CONTRACT,
                "reward_contract": REWARD_CONTRACT,
                "critic_observation_schema": CRITIC_OBSERVATION_SCHEMA,
                "completion_terminal_reward": completion_terminal_reward,
                "critic_time_horizon_steps": critic_time_horizon,
                "reference_anchor_distribution": (
                    "fixed_reconstructed_source_flat_v1" if source_anchor_metadata is not None
                    else "current_online_observations_v1"
                ),
                "source_anchor_metadata": source_anchor_metadata,
                "effective_runtime_config": effective_runtime_config,
                "evaluation_only": bool(args.evaluation_only),
                "steps": steps,
                "max_steps": args.max_steps,
                "reset_count": reset_count,
                "fell": fell,
                "safety_violation": safety_violation,
                "stop_reason": stop_reason,
                "task_failure_reason": task_failure_reason,
                "command_verified_steps": steps,
                "command_forward_min": min(row["command_lin_vel_xy"][0] for row in rows),
                "command_forward_max": max(row["command_lin_vel_xy"][0] for row in rows),
                "updates": 0 if args.evaluation_only else len(update_records),
                "accepted_updates": sum(bool(record["accepted"]) for record in update_records),
                "rejected_updates": sum(not bool(record["accepted"]) for record in update_records),
                "invalid_rollout_blocks": invalid_rollout_blocks,
                "safety_projected_steps": sum(bool(row["safety_projected"]) for row in rows),
                "ppo_valid_steps": sum(bool(row["ppo_valid"]) for row in rows),
                "safety_projected_fraction": (
                    sum(bool(row["safety_projected"]) for row in rows) / steps if steps else 0.0
                ),
                "max_requested_residual_abs": max(
                    (float(row["requested_residual_max_abs"]) for row in rows), default=0.0
                ),
                "max_executed_residual_abs": max(
                    (float(row["executed_residual_max_abs"]) for row in rows), default=0.0
                ),
                "max_safety_projection_abs": max(
                    (float(row["safety_projection_max_abs"]) for row in rows), default=0.0
                ),
                "ppo_valid_fraction": (
                    sum(bool(row["ppo_valid"]) for row in rows) / steps if steps else 0.0
                ),
                "actor_updates_before_novelty": 0,
                "actor_updates_after_novelty": 0 if args.evaluation_only else len(update_records),
                "novelty_entries": 0,
                "rollbacks": 0 if args.evaluation_only else len(rollback_records),
                "terminal_learning_updates": (
                    sum(record["update_attempted"] and record["reason"] in (
                        "terminated", "severe_nonfoot_contact", "lateral_violation", "edge_slip", "task_timeout"
                    ) for record in terminal_learning_records)
                ),
                "boundary_learning_updates": sum(record["update_attempted"] for record in terminal_learning_records),
                "first_half_velocity_error": _mean("velocity_error", 0, midpoint),
                "second_half_velocity_error": _mean("velocity_error", midpoint, steps),
                "first_half_tilt_rms": _mean("tilt_rms", 0, midpoint),
                "second_half_tilt_rms": _mean("tilt_rms", midpoint, steps),
                "first_half_nonfoot_contacts": sum(bool(row["nonfoot_contact"]) for row in rows[:midpoint]),
                "second_half_nonfoot_contacts": sum(bool(row["nonfoot_contact"]) for row in rows[midpoint:]),
                **completion_evidence,
                "completion_qualified": _completion_passed(
                    track_completion, fell=fell, safety_violation=safety_violation, reset_count=reset_count,
                ),
                "passed": _completion_passed(
                    track_completion, fell=fell, safety_violation=safety_violation, reset_count=reset_count,
                ),
                "completion_qualification_version": protocol.completion_qualification_version,
                "completion_qualification_passed": bool(
                    completion_evidence["completion_qualification_passed"]
                ),
                "edge_slip_events": int(completion_evidence.get("edge_slip_events", 0)),
                "completion_evidence": completion_evidence,
                "completion_stop": completion_stop,
                "protocol_sha256": protocol.sha256(),
                "contact_termination": CONTACT_TERMINATION_PROTOCOL,
                "checkpoint_in": str(args.checkpoint_in) if args.checkpoint_in else None,
                "policy_version": int(updater.policy_version),
            }
            if args.flat_retention_probe:
                # A flat retention measurement is not a terrain qualification
                # or a retention pass. Paired acceptance is evaluated outside
                # this worker from its measured data and physical flags.
                for key in (
                    "passed", "completion_qualified", "completion_qualification_version",
                    "completion_qualification_passed", "completion_evidence", "edge_slip_events",
                    "completion_stop", "completion_mode", "completion_target_x", "completion_reached",
                    "strict_qualification_enabled", "protocol_sha256",
                ):
                    summary.pop(key, None)
                summary["flat_retention"] = summarize_flat_retention(
                    rows, initial_x=initial_x, step_dt=control_step_dt, max_steps=args.max_steps,
                    fell=fell, safety_violation=safety_violation, reset_count=reset_count,
                )
                summary["terrain_geometry"] = terrain_geometry
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        finally:
            wrapped_env.close()
    except BaseException:
        # Isaac Sim shutdown can otherwise hide the original worker failure in
        # a subprocess log, leaving only a missing summary artifact.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
