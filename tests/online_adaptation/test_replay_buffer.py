import torch

from rl_training.online_adaptation.replay_buffer import ReplayBuffer


def make_transition(value: float, batch: int = 2):
    return dict(
        history=torch.full((batch, 3, 2), value),
        current_frame=torch.full((batch, 2), value),
        base_action=torch.full((batch, 1), value),
        residual_action=torch.full((batch, 1), value),
        reward=torch.full((batch, 1), value),
        next_history=torch.full((batch, 3, 2), value + 1),
        next_frame=torch.full((batch, 2), value + 1),
        next_base_action=torch.full((batch, 1), value + 1),
        done=torch.zeros(batch, 1),
    )


def test_replay_overwrites_oldest_and_samples_float32():
    replay = ReplayBuffer(capacity=3, history_length=3, frame_dim=2, action_dim=1, storage_dtype=torch.float16)
    replay.add_batch(**make_transition(1.0))
    replay.add_batch(**make_transition(2.0))
    assert len(replay) == 3
    batch = replay.sample(batch_size=3, device="cpu", generator=torch.Generator().manual_seed(0))
    assert batch.history.dtype == torch.float32
    assert batch.history.shape == (3, 3, 2)


def test_replay_add_batch_filters_paused_environments():
    replay = ReplayBuffer(capacity=3, history_length=3, frame_dim=2, action_dim=1)

    replay.add_batch(**make_transition(4.0), valid_mask=torch.tensor([True, False]))

    assert len(replay) == 1
    batch = replay.sample(batch_size=1, device="cpu")
    assert batch.reward.item() == 4.0
