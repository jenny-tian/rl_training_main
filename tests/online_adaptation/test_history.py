import torch

from rl_training.online_adaptation.history import HistoryBuffer


def test_history_append_and_preview_do_not_double_append():
    history = HistoryBuffer(num_envs=2, history_length=3, frame_dim=2, device="cpu")
    history.append(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    preview = history.preview(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    assert torch.equal(history.snapshot()[:, -1], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(preview[:, -1], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))


def test_preview_resets_done_environment_before_appending():
    history = HistoryBuffer(num_envs=2, history_length=3, frame_dim=1, device="cpu")
    history.append(torch.tensor([[1.0], [2.0]]))
    preview = history.preview(torch.tensor([[3.0], [4.0]]), reset_mask=torch.tensor([True, False]))
    assert torch.equal(preview[0, :, 0], torch.tensor([0.0, 0.0, 3.0]))
    assert torch.equal(preview[1, :, 0], torch.tensor([0.0, 2.0, 4.0]))
