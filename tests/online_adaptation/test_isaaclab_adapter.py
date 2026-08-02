from types import SimpleNamespace

import torch

import pytest

from rl_training.online_adaptation.isaaclab_adapter import (
    Lite3IsaacLabAdapter,
    compose_proprio_frame,
)

pytestmark = pytest.mark.isaaclab


def test_compose_proprio_frame_has_fixed_order_and_49_dimensions():
    terms = [
        torch.full((2, 3), 1.0),
        torch.full((2, 3), 2.0),
        torch.full((2, 12), 3.0),
        torch.full((2, 12), 4.0),
        torch.full((2, 4), 5.0),
        torch.full((2, 12), 6.0),
        torch.full((2, 3), 7.0),
    ]
    frame = compose_proprio_frame(*terms)
    assert frame.shape == (2, 49)
    assert torch.equal(frame[:, :3], terms[0])
    assert torch.equal(frame[:, -3:], terms[-1])


def test_reward_signals_use_articulation_foot_indices_for_foot_speed():
    sensor_body_names = [
        "TORSO",
        "FL_HIP",
        "FL_THIGH",
        "FL_SHANK",
        "FL_FOOT",
        "FR_HIP",
        "FR_THIGH",
        "FR_SHANK",
        "FR_FOOT",
        "HL_HIP",
        "HL_THIGH",
        "HL_SHANK",
        "HL_FOOT",
        "HR_HIP",
        "HR_THIGH",
        "HR_SHANK",
        "HR_FOOT",
    ]
    robot_body_names = [
        "TORSO",
        "FL_HIP",
        "FR_HIP",
        "HL_HIP",
        "HR_HIP",
        "FL_THIGH",
        "FR_THIGH",
        "HL_THIGH",
        "HR_THIGH",
        "FL_SHANK",
        "FR_SHANK",
        "HL_SHANK",
        "HR_SHANK",
        "FL_FOOT",
        "FR_FOOT",
        "HL_FOOT",
        "HR_FOOT",
    ]
    body_lin_vel_w = torch.zeros(1, len(robot_body_names), 3)
    for index, speed in zip((13, 14, 15, 16), (1.0, 2.0, 3.0, 4.0), strict=True):
        body_lin_vel_w[0, index, 0] = speed
    for index, speed in zip((4, 8, 12), (10.0, 20.0, 30.0), strict=True):
        body_lin_vel_w[0, index, 0] = speed
    sensor_forces = torch.zeros(1, len(sensor_body_names), 3)
    sensor_forces[0, [4, 8, 12, 16], 2] = 2.0
    robot = SimpleNamespace(
        body_names=robot_body_names,
        data=SimpleNamespace(
            joint_pos=torch.zeros(1, 12),
            default_joint_pos=torch.zeros(1, 12),
            soft_joint_pos_limits=torch.stack(
                (torch.full((1, 12), -1.0), torch.full((1, 12), 1.0)), dim=-1
            ),
            body_lin_vel_w=body_lin_vel_w,
            root_lin_vel_b=torch.zeros(1, 3),
            root_ang_vel_b=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            root_pos_w=torch.zeros(1, 3),
            joint_vel=torch.zeros(1, 12),
            applied_torque=torch.zeros(1, 12),
        ),
    )
    robot.find_joints = lambda names, preserve_order: (list(range(12)), names)
    robot.find_bodies = lambda pattern: (
        ([13, 14, 15, 16], ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"])
        if pattern == ".*_FOOT"
        else (list(range(13)), robot_body_names[:13])
    )
    sensor = SimpleNamespace(
        body_names=sensor_body_names,
        data=SimpleNamespace(net_forces_w=sensor_forces),
        find_bodies=lambda pattern: (
            ([4, 8, 12, 16], ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"])
            if pattern == ".*_FOOT"
            else (list(range(13)), sensor_body_names[:13])
        ),
    )
    class Scene:
        sensors = {"contact_forces": sensor}

        def __getitem__(self, name):
            assert name == "robot"
            return robot

    env = SimpleNamespace(
        scene=Scene(),
        command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(1, 3)),
        cfg=SimpleNamespace(track_boundary_x=-7.0),
    )

    adapter = Lite3IsaacLabAdapter(env)

    assert adapter.reward_signals().foot_slip.item() == pytest.approx(2.5)
