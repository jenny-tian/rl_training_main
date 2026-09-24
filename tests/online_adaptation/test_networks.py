import torch

from rl_training.online_adaptation.networks import LatentResidualActor, ResidualActor, TwinCritic

LIMITS = torch.tensor([0.24, 0.20, 0.20] * 4)


def test_actor_mean_head_is_exactly_zero_without_zeroing_log_std_head():
    actor = ResidualActor(frame_dim=49, action_dim=12, history_dim=64, action_limits=LIMITS)

    assert torch.equal(actor.mean_layer.weight, torch.zeros_like(actor.mean_layer.weight))
    assert torch.equal(actor.mean_layer.bias, torch.zeros_like(actor.mean_layer.bias))
    assert torch.count_nonzero(actor.log_std_layer.weight).item() > 0


def test_actor_deterministic_mean_is_exactly_zero_for_finite_inputs():
    actor = ResidualActor(frame_dim=49, action_dim=12, history_dim=64, action_limits=LIMITS)
    generator = torch.Generator().manual_seed(42)
    history = torch.randn(8, 25, 49, generator=generator)
    frame = torch.randn(8, 49, generator=generator)
    base = torch.randn(8, 12, generator=generator)

    action, _, mean = actor.sample(history, frame, base, deterministic=True)

    assert torch.equal(mean, torch.zeros_like(mean))
    assert torch.equal(action, torch.zeros_like(action))


def test_actor_stochastic_sample_is_bounded():
    actor = ResidualActor(frame_dim=49, action_dim=12, history_dim=64, action_limits=LIMITS)
    history = torch.zeros(8, 25, 49)
    frame = torch.zeros(8, 49)
    base = torch.zeros(8, 12)
    action, log_prob, mean = actor.sample(history, frame, base, deterministic=False)
    assert action.shape == (8, 12)
    assert log_prob.shape == (8, 1)
    assert torch.all(action.abs() <= LIMITS + 1e-6)
    assert torch.equal(mean, torch.zeros_like(mean))


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


def test_latent_actor_decodes_bounded_action_and_keeps_decoder_frozen():
    actor = LatentResidualActor(
        frame_dim=49,
        action_dim=12,
        code_dim=4,
        history_dim=16,
        action_limits=LIMITS,
    )
    history = torch.zeros(8, 25, 49)
    frame = torch.zeros(8, 49)
    base = torch.zeros(8, 12)

    action, log_prob, mean = actor.sample(history, frame, base, deterministic=True)

    assert action.shape == mean.shape == (8, 12)
    assert log_prob.shape == (8, 1)
    assert torch.equal(action, torch.zeros_like(action))
    assert torch.equal(mean, torch.zeros_like(mean))
    assert not actor.decoder_matrix.requires_grad

    stochastic, stochastic_log_prob, _ = actor.sample(history, frame, base)
    assert stochastic.shape == (8, 12)
    assert stochastic_log_prob.shape == (8, 1)
    assert torch.all(stochastic.abs() <= LIMITS + 1e-6)


def test_latent_actor_critic_gradient_updates_code_policy_not_decoder():
    actor = LatentResidualActor(
        frame_dim=49,
        action_dim=12,
        code_dim=4,
        history_dim=16,
        action_limits=LIMITS,
    )
    history = torch.randn(5, 25, 49)
    frame = torch.randn(5, 49)
    base = torch.randn(5, 12)
    residual, log_prob, _ = actor.sample(history, frame, base)
    loss = residual.square().mean() + log_prob.mean()
    loss.backward()

    assert any(parameter.grad is not None for name, parameter in actor.named_parameters() if name != "decoder_matrix")
    assert actor.decoder_matrix.grad is None
