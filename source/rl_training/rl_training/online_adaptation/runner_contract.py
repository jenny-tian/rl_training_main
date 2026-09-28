"""Shared fixed-command and initial reset contract for paired Lite3 runners."""

import torch


def configure_forward_command(command_cfg, forward_speed: float) -> None:
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.ranges.lin_vel_x = (forward_speed, forward_speed)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    for name in ("rel_zero_vel_envs", "rel_only_lin_y_envs", "rel_only_lin_x_envs",
                 "rel_only_ang_z_envs", "planar_velocity_deadband"):
        if hasattr(command_cfg, name):
            setattr(command_cfg, name, 0.0)
    for name in ("command_acceleration_limits", "only_lin_x_magnitude_range"):
        if hasattr(command_cfg, name):
            setattr(command_cfg, name, None)


def pin_forward_command(env, forward_speed: float) -> None:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        raise AttributeError("deployment environment has no command_manager")
    command = command_manager.get_command("base_velocity")
    if not isinstance(command, torch.Tensor) or command.ndim != 2 or command.shape[1] < 3:
        raise ValueError("base_velocity command must be a batched tensor with at least 3 values")
    with torch.no_grad():
        command[:, 0].fill_(float(forward_speed))
        command[:, 1:3].zero_()
        term = command_manager.get_term("base_velocity")
        for name in ("vel_command_b", "target_vel_command_b"):
            value = getattr(term, name, None)
            if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] >= 3:
                value[:, 0].fill_(float(forward_speed))
                value[:, 1:3].zero_()
        for name in ("is_standing_env", "is_heading_env"):
            value = getattr(term, name, None)
            if isinstance(value, torch.Tensor):
                value.fill_(False)


def pin_forward_observation(wrapped_env, observation, forward_speed: float):
    command = wrapped_env.unwrapped.command_manager.get_command("base_velocity")
    expected = command.new_tensor([forward_speed, 0.0, 0.0]).expand_as(command[:, :3])
    changed = not torch.equal(command[:, :3], expected)
    pin_forward_command(wrapped_env.unwrapped, forward_speed)
    return wrapped_env.get_observations() if changed else observation


def seeded_initial_reset(wrapped_env, seed: int, *, preserve_learning_rng=False):
    """Isolate physical reset sampling from prior learner construction draws.

    RSL's wrapper reset does not accept seed=. Its public seed method forwards
    to Isaac's Python/NumPy/Torch seeding before actuator delays are sampled.
    Call only at an independent physical trial's initial reset, not each step.
    """
    if not preserve_learning_rng:
        wrapped_env.seed(seed)
        return wrapped_env.reset()
    from .checkpointing import capture_rng_state, restore_rng_state

    saved = capture_rng_state()
    try:
        wrapped_env.seed(seed)
        return wrapped_env.reset()
    finally:
        restore_rng_state(saved)


def actuator_delay_metadata(robot):
    return {
        name: {
            field: getattr(actuator, field).time_lags.detach().cpu().tolist()
            for field in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer")
            if hasattr(actuator, field)
        }
        for name, actuator in robot.actuators.items()
    }
