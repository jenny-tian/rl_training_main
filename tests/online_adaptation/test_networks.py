import torch

from rl_training.online_adaptation.networks import ResidualActor, TwinCritic

LIMITS = torch.tensor([0.24, 0.20, 0.20] * 4)


def test_actor_is_near_zero_and_bounded():
    actor = ResidualActor(frame_dim=49, action_dim=12, history_dim=64, action_limits=LIMITS)
    history = torch.zeros(8, 25, 49)
    frame = torch.zeros(8, 49)
    base = torch.zeros(8, 12)
    action, log_prob, mean = actor.sample(history, frame, base, deterministic=False)
    assert action.shape == (8, 12)
    assert log_prob.shape == (8, 1)
    assert torch.all(action.abs() <= LIMITS + 1e-6)
    assert mean.abs().max().item() < 1e-3


def test_twin_critic_returns_two_columns():
    critic = TwinCritic(frame_dim=49, action_dim=12, history_dim=64)
    q1, q2 = critic(
        torch.zeros(8, 25, 49),
        torch.zeros(8, 49),
        torch.zeros(8, 12),
        torch.zeros(8, 12),
    )
    assert q1.shape == q2.shape == (8, 1)
    assert q1.data_ptr() != q2.data_ptr()
