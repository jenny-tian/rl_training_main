import torch
from torch import nn
from torch.distributions import Normal

from rl_training.online_adaptation.rsl_rl_compat import RslMLPPolicyAdapter


class FakeDistribution:
    def __init__(self):
        self._distribution = None

    def update(self, output):
        self._distribution = Normal(output, torch.ones_like(output))

    def log_prob(self, action):
        return self._distribution.log_prob(action).sum(-1)

    def entropy(self):
        return self._distribution.entropy().sum(-1)


class FakeRslActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_groups = ["policy"]
        self.mlp = nn.Linear(3, 2)
        self.distribution = FakeDistribution()

    def forward(self, obs, stochastic_output=False):
        return self.mlp(obs["policy"])


def test_rsl_adapter_converts_tensor_observation_and_evaluates_requested_actions():
    adapter = RslMLPPolicyAdapter(FakeRslActor())
    observations = torch.randn(6, 3)
    actions = torch.randn(6, 2)
    mean = adapter.mean_action(observations)
    log_prob, entropy = adapter.evaluate_actions(observations, actions)
    assert mean.shape == actions.shape
    assert log_prob.shape == entropy.shape == (6,)


def test_rsl_adapter_requires_explicit_group_for_multiple_groups():
    actor = FakeRslActor()
    actor.obs_groups = ["policy", "privileged"]
    try:
        RslMLPPolicyAdapter(actor)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("multiple observation groups should require an explicit group")


def test_rsl_adapter_applies_action_transform_to_mean_and_distribution():
    adapter = RslMLPPolicyAdapter(FakeRslActor(), action_transform=lambda action: action.clamp(-0.1, 0.1))
    observations = torch.full((2, 3), 100.0)
    mean = adapter.mean_action(observations)
    distribution = adapter.get_distribution(observations)
    assert bool((mean.abs() <= 0.1).all())
    torch.testing.assert_close(distribution.mean, mean)
