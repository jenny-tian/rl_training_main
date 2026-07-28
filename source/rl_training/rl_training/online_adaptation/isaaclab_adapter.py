from __future__ import annotations

import torch
from typing import Any

from .types import RewardSignals, SafetyState


def compose_proprio_frame(
    base_angular_velocity: torch.Tensor,
    projected_gravity: torch.Tensor,
    relative_joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    foot_contact: torch.Tensor,
    previous_final_action: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    frame = torch.cat(
        (
            base_angular_velocity,
            projected_gravity,
            relative_joint_position,
            joint_velocity,
            foot_contact,
            previous_final_action,
            command,
        ),
        dim=-1,
    )
    if frame.shape[-1] != 49:
        raise ValueError(f"expected 49-dimensional proprioceptive frame, got {frame.shape[-1]}")
    return frame


def base_x_position(root_pos_w: torch.Tensor) -> torch.Tensor:
    return root_pos_w[:, 0:1]


def heldout_mask_from_base_x(base_x: torch.Tensor, track_boundary_x: float) -> torch.Tensor:
    return base_x >= track_boundary_x


class Lite3IsaacLabAdapter:
    def __init__(
        self,
        env: Any,
        *,
        robot_name: str = "robot",
        contact_sensor_name: str = "contact_forces",
        foot_body_pattern: str = ".*_FOOT",
        nonfoot_body_pattern: str = "^(?!.*_FOOT).*",
        contact_threshold: float = 1.0,
        track_boundary_x: float | None = None,
    ):
        self.env = env
        self.robot = env.scene[robot_name]
        self.contact_sensor = env.scene.sensors[contact_sensor_name]
        self.foot_ids = self.contact_sensor.find_bodies(foot_body_pattern)[0]
        self.nonfoot_ids = self.contact_sensor.find_bodies(nonfoot_body_pattern)[0]
        self.contact_threshold = contact_threshold
        joint_pos = self.robot.data.joint_pos
        self.action_scale = torch.tensor(
            [0.125, 0.25, 0.25] * 4,
            device=joint_pos.device,
            dtype=joint_pos.dtype,
        ).unsqueeze(0)
        if track_boundary_x is None:
            track_boundary_x = float(getattr(env.cfg, "track_boundary_x", -7.0))
        self.track_boundary_x = track_boundary_x

    def frame(self, previous_action: torch.Tensor) -> torch.Tensor:
        robot = self.robot
        forces = self.contact_sensor.data.net_forces_w[:, self.foot_ids]
        contacts = (torch.linalg.norm(forces, dim=-1) > self.contact_threshold).float()
        command = self.env.command_manager.get_command("base_velocity")[:, :3]
        return compose_proprio_frame(
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel,
            contacts,
            previous_action,
            command,
        )

    def safety_state(self, control_timed_out: torch.Tensor) -> SafetyState:
        limits = self.robot.data.soft_joint_pos_limits
        nonfoot_forces = self.contact_sensor.data.net_forces_w[:, self.nonfoot_ids]
        return SafetyState(
            projected_gravity_xy=self.robot.data.projected_gravity_b[:, :2],
            base_height=self.robot.data.root_pos_w[:, 2:3],
            joint_pos=self.robot.data.joint_pos,
            joint_vel=self.robot.data.joint_vel,
            joint_lower=limits[..., 0],
            joint_upper=limits[..., 1],
            default_joint_pos=self.robot.data.default_joint_pos,
            action_scale=self.action_scale.expand_as(self.robot.data.joint_pos),
            nonfoot_contact=(torch.linalg.norm(nonfoot_forces, dim=-1) > self.contact_threshold).any(
                dim=1, keepdim=True
            ),
            control_timed_out=control_timed_out,
        )

    def reward_signals(self) -> RewardSignals:
        foot_forces = torch.linalg.norm(self.contact_sensor.data.net_forces_w[:, self.foot_ids], dim=-1)
        contacts = foot_forces > self.contact_threshold
        foot_speed = torch.linalg.norm(self.robot.data.body_lin_vel_w[:, self.foot_ids, :2], dim=-1)
        command = self.env.command_manager.get_command("base_velocity")
        nonfoot_forces = self.contact_sensor.data.net_forces_w[:, self.nonfoot_ids]
        return RewardSignals(
            base_lin_vel_xy=self.robot.data.root_lin_vel_b[:, :2],
            command_lin_vel_xy=command[:, :2],
            base_ang_vel_z=self.robot.data.root_ang_vel_b[:, 2:3],
            command_ang_vel_z=command[:, 2:3],
            projected_gravity_xy=self.robot.data.projected_gravity_b[:, :2],
            base_lin_vel_z=self.robot.data.root_lin_vel_b[:, 2:3],
            foot_slip=(foot_speed * contacts.float()).mean(dim=1, keepdim=True),
            foot_impact=foot_forces.max(dim=1, keepdim=True).values,
            joint_power=torch.abs(self.robot.data.joint_vel * self.robot.data.applied_torque).mean(
                dim=1, keepdim=True
            ),
            nonfoot_contact=(torch.linalg.norm(nonfoot_forces, dim=-1) > self.contact_threshold)
            .any(dim=1, keepdim=True)
            .float(),
        )

    def heldout_mask(self) -> torch.Tensor:
        return heldout_mask_from_base_x(self.base_x(), self.track_boundary_x)

    def base_x(self) -> torch.Tensor:
        return base_x_position(self.robot.data.root_pos_w)
