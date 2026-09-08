from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

import torch

from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES

from .terrain_observation import (
    HEIGHT_SCAN_SCHEMA_VERSION,
    HeightScanConfig,
    summarize_height_scan,
)
from .types import RewardSignals, SafetyState


LEGACY_PROPRIO_FRAME_DIM = 49
TERRAIN_FEATURE_DIM = 10
PROPRIO_FRAME_DIM = LEGACY_PROPRIO_FRAME_DIM + TERRAIN_FEATURE_DIM
TERRAIN_DISTANCE_SCALE_M = 1.0
TERRAIN_HEIGHT_SCALE_M = 0.10
TERRAIN_FORWARD_SCALE_M = 0.50
TerrainFeatureSource = Literal["proprio", "geometry", "height_scan"]
DEFAULT_HEIGHT_SCAN_CONFIG = HeightScanConfig(rows=23, cols=15)


def select_joint_order(value: torch.Tensor, joint_ids: list[int] | torch.Tensor) -> torch.Tensor:
    """Select robot-state columns in the PPO action/observation joint order."""
    if value.ndim < 2:
        raise ValueError("joint tensor must have batch and joint dimensions")
    indices = torch.as_tensor(joint_ids, dtype=torch.long, device=value.device)
    return value.index_select(1, indices)


def compose_proprio_frame(
    base_angular_velocity: torch.Tensor,
    projected_gravity: torch.Tensor,
    relative_joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    foot_contact: torch.Tensor,
    previous_final_action: torch.Tensor,
    command: torch.Tensor,
    terrain_features: torch.Tensor | None = None,
) -> torch.Tensor:
    terms = (
        base_angular_velocity,
        projected_gravity,
        relative_joint_position,
        joint_velocity,
        foot_contact,
        previous_final_action,
        command,
    )
    frame = torch.cat(terms if terrain_features is None else (*terms, terrain_features), dim=-1)
    expected_dim = LEGACY_PROPRIO_FRAME_DIM if terrain_features is None else PROPRIO_FRAME_DIM
    if frame.shape[-1] != expected_dim:
        raise ValueError(f"expected {expected_dim}-dimensional proprioceptive frame, got {frame.shape[-1]}")
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
        step_height: float | None = None,
        terrain_feature_source: TerrainFeatureSource = "geometry",
        height_scanner_name: str = "height_scanner",
        height_scan_config: HeightScanConfig | None = None,
    ):
        if terrain_feature_source not in {"proprio", "geometry", "height_scan"}:
            raise ValueError("terrain_feature_source must be proprio, geometry or height_scan")
        self.env = env
        self.robot = env.scene[robot_name]
        self.contact_sensor = env.scene.sensors[contact_sensor_name]
        self.foot_ids, sensor_foot_names = self.contact_sensor.find_bodies(foot_body_pattern)
        self.robot_foot_ids, robot_foot_names = self.robot.find_bodies(foot_body_pattern)
        sensor_foot_id_by_name = dict(zip(sensor_foot_names, self.foot_ids, strict=True))
        if set(sensor_foot_names) != set(robot_foot_names):
            raise ValueError(
                "contact sensor and robot expose different Lite3 feet: "
                f"sensor={tuple(sensor_foot_names)}, robot={tuple(robot_foot_names)}"
            )
        self.reward_foot_ids = [sensor_foot_id_by_name[name] for name in robot_foot_names]
        self.nonfoot_ids, self.nonfoot_names = self.contact_sensor.find_bodies(
            nonfoot_body_pattern
        )
        self.contact_threshold = contact_threshold
        self._episode_max_nonfoot_contact_force_n = 0.0
        self._episode_max_nonfoot_contact_body: str | None = None
        self._terminal_nonfoot_contact_force_n = 0.0
        self._terminal_nonfoot_contact_body: str | None = None
        add_pre_reset_callback = getattr(env, "add_pre_reset_callback", None)
        if callable(add_pre_reset_callback):
            add_pre_reset_callback(self.capture_terminal_contact)
        joint_pos = self.robot.data.joint_pos
        self.joint_ids, joint_names = self.robot.find_joints(
            list(LITE3_POLICY_JOINT_NAMES), preserve_order=True
        )
        if tuple(joint_names) != LITE3_POLICY_JOINT_NAMES:
            raise ValueError(
                "Lite3 robot joints do not match the frozen PPO joint order: "
                f"expected {LITE3_POLICY_JOINT_NAMES}, got {tuple(joint_names)}"
            )
        self.action_scale = torch.tensor(
            [0.125, 0.25, 0.25] * 4,
            device=joint_pos.device,
            dtype=joint_pos.dtype,
        ).unsqueeze(0)
        if track_boundary_x is None:
            track_boundary_x = float(getattr(env.cfg, "track_boundary_x", -7.0))
        self.track_boundary_x = track_boundary_x
        if step_height is not None:
            if isinstance(step_height, bool) or not torch.isfinite(torch.tensor(float(step_height))):
                raise ValueError("step_height must be a finite nonnegative scalar")
            if float(step_height) < 0.0:
                raise ValueError("step_height must be a finite nonnegative scalar")
        self.step_height = None if step_height is None else float(step_height)
        self.terrain_feature_source = terrain_feature_source
        self.height_scanner_name = height_scanner_name
        self.height_scan_config = height_scan_config or DEFAULT_HEIGHT_SCAN_CONFIG
        self._height_scanner = None
        if terrain_feature_source == "height_scan":
            sensors = getattr(env.scene, "sensors", None)
            try:
                self._height_scanner = sensors[height_scanner_name]
            except (AttributeError, IndexError, KeyError, TypeError):
                self._height_scanner = None
        batch_size = int(joint_pos.shape[0])
        self._terrain_valid = torch.ones(
            batch_size, 1, dtype=torch.bool, device=joint_pos.device
        )
        if terrain_feature_source == "height_scan":
            self._terrain_valid.zero_()

    @property
    def frame_dim(self) -> int:
        return (
            PROPRIO_FRAME_DIM
            if self.step_height is not None or self.terrain_feature_source == "height_scan"
            else LEGACY_PROPRIO_FRAME_DIM
        )

    def _nonfoot_contact_peak(self) -> tuple[float, str | None]:
        history = getattr(self.contact_sensor.data, "net_forces_w_history", None)
        forces = (
            history[:, :, self.nonfoot_ids]
            if isinstance(history, torch.Tensor)
            else self.contact_sensor.data.net_forces_w[:, self.nonfoot_ids].unsqueeze(1)
        )
        magnitudes = torch.linalg.norm(forces, dim=-1)
        if magnitudes.numel() == 0:
            return 0.0, None
        flat_index = int(magnitudes.argmax().item())
        body_axis_index = flat_index % len(self.nonfoot_ids)
        return float(magnitudes.flatten()[flat_index].item()), str(
            self.nonfoot_names[body_axis_index]
        )

    def observe_nonfoot_contact(self) -> dict[str, object]:
        force_n, body = self._nonfoot_contact_peak()
        if force_n > self._episode_max_nonfoot_contact_force_n:
            self._episode_max_nonfoot_contact_force_n = force_n
            self._episode_max_nonfoot_contact_body = body
        return {"max_nonfoot_contact_force_n": force_n, "max_nonfoot_contact_body": body}

    def capture_terminal_contact(self, _env_ids: object) -> None:
        diagnostics = self.observe_nonfoot_contact()
        self._terminal_nonfoot_contact_force_n = float(
            diagnostics["max_nonfoot_contact_force_n"]
        )
        self._terminal_nonfoot_contact_body = diagnostics["max_nonfoot_contact_body"]

    def reset_contact_diagnostics(self) -> None:
        self._episode_max_nonfoot_contact_force_n = 0.0
        self._episode_max_nonfoot_contact_body = None
        self._terminal_nonfoot_contact_force_n = 0.0
        self._terminal_nonfoot_contact_body = None

    def contact_diagnostics(self) -> dict[str, object]:
        return {
            "episode_max_nonfoot_contact_force_n": self._episode_max_nonfoot_contact_force_n,
            "episode_max_nonfoot_contact_body": self._episode_max_nonfoot_contact_body,
            "terminal_nonfoot_contact_force_n": self._terminal_nonfoot_contact_force_n,
            "terminal_nonfoot_contact_body": self._terminal_nonfoot_contact_body,
        }

    def _terrain_features(self) -> torch.Tensor | None:
        if self.terrain_feature_source == "height_scan":
            return self._height_scan_features()
        if self.terrain_feature_source == "proprio" or self.step_height is None:
            self._terrain_valid.fill_(True)
            return None
        root_x = self.robot.data.root_pos_w[:, 0:1]
        foot_positions = self.robot.data.body_pos_w[:, self.robot_foot_ids]
        distance_to_edge = (
            (self.track_boundary_x - root_x) / TERRAIN_DISTANCE_SCALE_M
        ).clamp(-1.0, 1.0)
        step_height = torch.full_like(
            root_x, self.step_height / TERRAIN_HEIGHT_SCALE_M
        ).clamp(-1.0, 1.0)
        foot_heights = (
            foot_positions[:, :, 2] / TERRAIN_HEIGHT_SCALE_M
        ).clamp(-1.0, 1.0)
        foot_edge_offsets = (
            (foot_positions[:, :, 0] - self.track_boundary_x) / TERRAIN_FORWARD_SCALE_M
        ).clamp(-1.0, 1.0)
        return torch.cat(
            (distance_to_edge, step_height, foot_heights, foot_edge_offsets), dim=-1
        )

    def _height_scan_features(self) -> torch.Tensor:
        zeros = torch.zeros(
            self.robot.data.root_pos_w.shape[0],
            TERRAIN_FEATURE_DIM,
            dtype=self.robot.data.root_pos_w.dtype,
            device=self.robot.data.root_pos_w.device,
        )
        scanner = self._height_scanner
        if scanner is None:
            self._terrain_valid = torch.zeros(
                zeros.shape[0], 1, dtype=torch.bool, device=zeros.device
            )
            return zeros
        try:
            hits = scanner.data.ray_hits_w
            if not isinstance(hits, torch.Tensor) or hits.ndim != 3 or hits.shape[-1] != 3:
                raise ValueError("height scanner ray_hits_w must be [batch, rays, 3]")
            if hits.shape[0] != zeros.shape[0]:
                raise ValueError("height scanner batch dimension differs from robot")
            reference = getattr(scanner.data, "pos_w", self.robot.data.root_pos_w)
            if not isinstance(reference, torch.Tensor) or reference.shape != (zeros.shape[0], 3):
                raise ValueError("height scanner reference position must be [batch, 3]")
            relative_heights = hits[..., 2] - reference[:, 2:3]
            features, valid = summarize_height_scan(
                relative_heights.reshape(
                    zeros.shape[0],
                    self.height_scan_config.rows,
                    self.height_scan_config.cols,
                ),
                self.height_scan_config,
            )
            self._terrain_valid = valid.to(device=zeros.device)
            return features.to(device=zeros.device, dtype=zeros.dtype)
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            self._terrain_valid = torch.zeros(
                zeros.shape[0], 1, dtype=torch.bool, device=zeros.device
            )
            return zeros

    @property
    def terrain_valid(self) -> torch.Tensor:
        return self._terrain_valid.detach().clone()

    def terrain_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "terrain_feature_source": self.terrain_feature_source,
            "terrain_feature_schema_version": HEIGHT_SCAN_SCHEMA_VERSION,
        }
        if self.terrain_feature_source == "height_scan":
            metadata.update(
                {
                    "height_scanner_name": self.height_scanner_name,
                    "height_scan": asdict(self.height_scan_config),
                }
            )
        return metadata

    def frame(self, previous_action: torch.Tensor) -> torch.Tensor:
        robot = self.robot
        forces = self.contact_sensor.data.net_forces_w[:, self.foot_ids]
        contacts = (torch.linalg.norm(forces, dim=-1) > self.contact_threshold).float()
        command = self.env.command_manager.get_command("base_velocity")[:, :3]
        return compose_proprio_frame(
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            select_joint_order(robot.data.joint_pos - robot.data.default_joint_pos, self.joint_ids),
            select_joint_order(robot.data.joint_vel, self.joint_ids),
            contacts,
            previous_action,
            command,
            self._terrain_features(),
        )

    def safety_state(self, control_timed_out: torch.Tensor) -> SafetyState:
        limits = select_joint_order(self.robot.data.soft_joint_pos_limits, self.joint_ids)
        nonfoot_forces = self.contact_sensor.data.net_forces_w[:, self.nonfoot_ids]
        return SafetyState(
            projected_gravity_xy=self.robot.data.projected_gravity_b[:, :2],
            base_height=self.robot.data.root_pos_w[:, 2:3],
            joint_pos=select_joint_order(self.robot.data.joint_pos, self.joint_ids),
            joint_vel=select_joint_order(self.robot.data.joint_vel, self.joint_ids),
            joint_lower=limits[..., 0],
            joint_upper=limits[..., 1],
            default_joint_pos=select_joint_order(self.robot.data.default_joint_pos, self.joint_ids),
            action_scale=self.action_scale.expand(-1, len(self.joint_ids)),
            nonfoot_contact=(torch.linalg.norm(nonfoot_forces, dim=-1) > self.contact_threshold).any(
                dim=1, keepdim=True
            ),
            control_timed_out=control_timed_out,
        )

    def reward_signals(self) -> RewardSignals:
        foot_forces = torch.linalg.norm(
            self.contact_sensor.data.net_forces_w[:, self.reward_foot_ids], dim=-1
        )
        contacts = foot_forces > self.contact_threshold
        foot_speed = torch.linalg.norm(
            self.robot.data.body_lin_vel_w[:, self.robot_foot_ids, :2], dim=-1
        )
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
            joint_power=torch.abs(
                select_joint_order(self.robot.data.joint_vel, self.joint_ids)
                * select_joint_order(self.robot.data.applied_torque, self.joint_ids)
            ).mean(dim=1, keepdim=True),
            nonfoot_contact=(torch.linalg.norm(nonfoot_forces, dim=-1) > self.contact_threshold)
            .any(dim=1, keepdim=True)
            .float(),
            foot_height=self.robot.data.body_pos_w[:, self.robot_foot_ids, 2],
            foot_contact=contacts.float(),
            foot_edge_offset=(
                self.robot.data.body_pos_w[:, self.robot_foot_ids, 0] - self.track_boundary_x
            ),
        )

    def heldout_mask(self) -> torch.Tensor:
        return heldout_mask_from_base_x(self.base_x(), self.track_boundary_x)

    def base_x(self) -> torch.Tensor:
        return base_x_position(self.robot.data.root_pos_w)
