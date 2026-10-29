"""CPU-only explicit link/COM buffers for reference-point integration tests."""

from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def reference_data():
    def make(positions=None, velocities=None, *, env_count=1, dtype=torch.float64):
        pos = torch.zeros((env_count, 17, 3), dtype=dtype) if positions is None else positions
        vel = torch.zeros_like(pos) if velocities is None else velocities
        quat = torch.zeros((*pos.shape[:2], 4), dtype=pos.dtype)
        quat[..., 0] = 1
        data = NS(body_link_pos_w=pos, body_link_lin_vel_w=vel, body_link_quat_w=quat,
                  body_com_pos_b=torch.zeros_like(pos), body_com_pos_w=pos.clone(),
                  body_com_lin_vel_w=vel.clone(), body_link_ang_vel_w=torch.zeros_like(pos))
        for root, body in (
            ("root_link_pos_w", "body_link_pos_w"), ("root_link_lin_vel_w", "body_link_lin_vel_w"),
            ("root_link_quat_w", "body_link_quat_w"), ("root_link_ang_vel_w", "body_link_ang_vel_w"),
            ("root_com_pos_w", "body_com_pos_w"), ("root_com_lin_vel_w", "body_com_lin_vel_w"),
        ):
            setattr(data, root, getattr(data, body)[:, 0])
        data.body_pos_w = data.body_link_pos_w
        data.body_lin_vel_w = data.body_com_lin_vel_w
        data.root_quat_w = data.root_link_quat_w
        data.root_lin_vel_w = data.root_com_lin_vel_w
        data.root_lin_vel_b = data.root_com_lin_vel_w.clone()
        data.projected_gravity_b = torch.tensor([[0., 0., -1.]], dtype=pos.dtype).repeat(pos.shape[0], 1)
        return data
    return make
