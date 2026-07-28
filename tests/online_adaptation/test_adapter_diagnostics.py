import torch

from rl_training.online_adaptation.isaaclab_adapter import base_x_position


def test_base_x_position_extracts_world_x_column_without_squeezing():
    root_pos_w = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])

    x = base_x_position(root_pos_w)

    assert torch.equal(x, torch.tensor([[1.0], [-4.0]]))
