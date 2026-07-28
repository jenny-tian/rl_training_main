import torch

from rl_training.online_adaptation.isaaclab_adapter import base_x_position, heldout_mask_from_base_x


def test_base_x_position_extracts_world_x_column_without_squeezing():
    root_pos_w = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])

    x = base_x_position(root_pos_w)

    assert torch.equal(x, torch.tensor([[1.0], [-4.0]]))


def test_heldout_mask_from_base_x_uses_configured_metric_boundary():
    base_x = torch.tensor([[-7.36], [-7.35], [-7.34]])

    mask = heldout_mask_from_base_x(base_x, track_boundary_x=-7.35)

    assert torch.equal(mask, torch.tensor([[False], [True], [True]]))
