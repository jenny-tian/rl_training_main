import copy
import math

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from rl_training.online_adaptation.online_session import OnlinePPOSession
from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater, PPOConfig
from rl_training.online_adaptation.ppo_rollout import PPORollout
from rl_training.online_adaptation.ppo_runtime_config import apply_ppo_runtime_config, read_ppo_runtime_config
from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher


class Policy(nn.Module):
    def __init__(self, *, std_field="log_std_param", action_std=0.01):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.distribution = nn.Module()
        encoded = math.log(action_std) if std_field == "log_std_param" else action_std
        self.distribution.register_parameter(
            std_field, nn.Parameter(torch.full((2,), encoded), requires_grad=False)
        )

    def get_distribution(self, observations):
        if hasattr(self.distribution, "log_std_param"):
            std = self.distribution.log_std_param.exp()
        else:
            std = self.distribution.std_param
        return Normal(self.linear(observations), std)


def make_session(*, std_field="log_std_param", action_std=0.01, actor_lr=5e-5, critic_lr=5e-5):
    policy = Policy(std_field=std_field, action_std=action_std)
    critic = nn.Linear(2, 1)
    updater = OnlinePPOUpdater(
        policy, critic,
        PPOConfig(actor_lr=actor_lr, critic_lr=critic_lr, epochs=1, minibatch_size=8, target_kl=None),
    )
    return OnlinePPOSession(updater, rollout_capacity=8, publisher=DoubleBufferedActorPublisher(policy))


def append_rollout(session, length=8):
    updater = session.updater
    observations = torch.randn(length, 2)
    distribution = updater.policy.get_distribution(observations)
    actions = distribution.sample()
    values = updater.critic(observations).detach()
    session.append(PPORollout(
        observations, actions, distribution.log_prob(actions).sum(-1).detach(), values,
        torch.linspace(-1.0, 1.0, length).reshape(-1, 1),
        torch.zeros(length, 1), torch.zeros(length, 1), values,
        policy_version=updater.policy_version,
    ))


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


def assert_settings(session, *, action_std, actor_lr, critic_lr):
    effective = read_ppo_runtime_config(session)
    assert effective["action_std"] == pytest.approx(action_std)
    assert effective["actor_lr"] == actor_lr
    assert effective["critic_lr"] == critic_lr
    published, version = session.publisher.current()
    assert version == effective["policy_version"]
    std = published.get_distribution(torch.zeros(1, 2)).stddev
    torch.testing.assert_close(std, torch.full_like(std, action_std))
    assert_tree_equal(published.state_dict(), session.updater.policy.state_dict())


@pytest.mark.parametrize("std_field", ["log_std_param", "std_param"])
def test_resume_override_reaches_controller_and_survives_rollback_without_promoting_candidate(std_field):
    torch.manual_seed(50)
    source = make_session(std_field=std_field)
    for _ in range(2):
        append_rollout(source)
        assert source.update().accepted
    checkpoint = source.state_dict()
    resumed = make_session(std_field=std_field, action_std=0.03, actor_lr=3e-4, critic_lr=7e-4)
    resumed.load_state_dict(checkpoint)
    assert read_ppo_runtime_config(resumed)["action_std"] == pytest.approx(0.01)
    assert read_ppo_runtime_config(resumed)["actor_lr"] == 5e-5
    before = resumed.state_dict()

    effective = apply_ppo_runtime_config(resumed, action_std=0.03, actor_lr=3e-4, critic_lr=7e-4)

    assert effective == read_ppo_runtime_config(resumed)
    assert_settings(resumed, action_std=0.03, actor_lr=3e-4, critic_lr=7e-4)
    after = resumed.state_dict()
    assert after["updater"]["policy_version"] == before["updater"]["policy_version"] + 1
    assert after["stable_state"]["policy_version"] == before["stable_state"]["policy_version"]
    assert after["has_unverified_candidate"] is before["has_unverified_candidate"] is True
    assert after["counters"] == before["counters"]
    std_key = f"distribution.{std_field}"
    for branch in ("updater", "stable_state"):
        assert_tree_equal(after[branch]["critic"], before[branch]["critic"])
        for key in before[branch]["policy"]:
            if key != std_key:
                assert_tree_equal(after[branch]["policy"][key], before[branch]["policy"][key])
        for optimizer in ("actor_optimizer", "critic_optimizer"):
            assert before[branch][optimizer]["state"]
            assert_tree_equal(after[branch][optimizer]["state"], before[branch][optimizer]["state"])
            for group_after, group_before in zip(
                after[branch][optimizer]["param_groups"], before[branch][optimizer]["param_groups"]
            ):
                assert_tree_equal(
                    {key: value for key, value in group_after.items() if key != "lr"},
                    {key: value for key, value in group_before.items() if key != "lr"},
                )
    assert resumed.updater.config.actor_lr == 3e-4
    assert resumed.updater.config.critic_lr == 7e-4

    rollback = resumed.rollback("terminated")

    assert rollback.restored_stable_version == before["stable_state"]["policy_version"]
    assert_settings(resumed, action_std=0.03, actor_lr=3e-4, critic_lr=7e-4)
    for key in before["stable_state"]["policy"]:
        if key != std_key:
            assert_tree_equal(resumed.updater.policy.state_dict()[key], before["stable_state"]["policy"][key])


def test_read_frozen_checkpoint_preserves_nonuniform_std_and_does_not_change_state():
    session = make_session(std_field="std_param")
    state = session.state_dict()
    state["updater"]["policy"]["distribution.std_param"] = torch.tensor([0.01, 0.025])
    session.load_state_dict(state)
    before = session.state_dict()
    controller, version = session.publisher.current()
    controller_state = copy.deepcopy(controller.state_dict())

    effective = read_ppo_runtime_config(session)

    assert effective["action_std"] == pytest.approx([0.01, 0.025])
    assert effective["policy_version"] == version
    assert_tree_equal(session.state_dict(), before)
    assert session.publisher.current()[0] is controller
    assert_tree_equal(controller.state_dict(), controller_state)


def test_learning_rate_only_override_does_not_change_policy_version_or_verification_status():
    session = make_session()
    before = session.state_dict()

    apply_ppo_runtime_config(session, action_std=0.01, actor_lr=4e-4, critic_lr=8e-4)

    assert_settings(session, action_std=0.01, actor_lr=4e-4, critic_lr=8e-4)
    assert session.updater.policy_version == before["updater"]["policy_version"]
    assert session.state_dict()["has_unverified_candidate"] is False
    assert_tree_equal(session.updater.policy.state_dict(), before["updater"]["policy"])
    assert session.updater.actor_optimizer.defaults["lr"] == 4e-4
    assert session.updater.critic_optimizer.defaults["lr"] == 8e-4


def test_pending_rollout_rejects_override_without_discarding_or_relabeling_samples():
    session = make_session()
    append_rollout(session, length=1)
    before = session.state_dict()
    controller, version = session.publisher.current()

    with pytest.raises(ValueError, match="pending rollout"):
        apply_ppo_runtime_config(session, action_std=0.03, actor_lr=3e-4, critic_lr=7e-4)

    assert_tree_equal(session.state_dict(), before)
    assert session.publisher.current() == (controller, version)


@pytest.mark.parametrize("field", ["action_std", "actor_lr", "critic_lr"])
@pytest.mark.parametrize("invalid", [0.0, -0.1, float("nan"), float("inf"), True])
def test_invalid_settings_are_rejected_before_mutation(field, invalid):
    session = make_session()
    before = session.state_dict()
    settings = {"action_std": 0.03, "actor_lr": 3e-4, "critic_lr": 7e-4, field: invalid}

    with pytest.raises(ValueError, match=field):
        apply_ppo_runtime_config(session, **settings)

    assert_tree_equal(session.state_dict(), before)
    assert_settings(session, action_std=0.01, actor_lr=5e-5, critic_lr=5e-5)


@pytest.mark.parametrize("std_field", ["log_std_param", "std_param"])
@pytest.mark.parametrize("action_std", [1e-100, 1e100])
def test_unrepresentable_std_is_rejected_atomically(std_field, action_std):
    session = make_session(std_field=std_field)
    before = session.state_dict()

    with pytest.raises(ValueError, match="representable"):
        apply_ppo_runtime_config(session, action_std=action_std, actor_lr=3e-4, critic_lr=7e-4)

    assert_tree_equal(session.state_dict(), before)
    assert_settings(session, action_std=0.01, actor_lr=5e-5, critic_lr=5e-5)


def test_episode_decay_reaches_restored_controller_and_rollback_without_changing_mean(tmp_path):
    from scripts.reinforcement_learning.online_adaptation.run_lora_ppo_multiepisode import (
        build_parser, build_training_command,
    )
    args = build_parser().parse_args(['--base-checkpoint', 'base.pt', '--output-dir', str(tmp_path),
        '--action-std', '0.03', '--final-action-std', '0.01', '--max-episodes', '20',
        '--actor-lr', '0.00005', '--critic-lr', '0.0003'])
    previous = make_session(action_std=0.03, actor_lr=0.00015)
    append_rollout(previous)
    assert previous.update().accepted
    checkpoint = previous.state_dict()
    observations = torch.randn(4, 2)
    before_mean = previous.updater.policy.get_distribution(observations).mean.detach()
    resumed = make_session()
    resumed.load_state_dict(checkpoint)
    command = build_training_command(args, tmp_path / 'e20', tmp_path / 'e19.pt', episode=20)
    applied = apply_ppo_runtime_config(resumed,
        action_std=float(command[command.index('--action-std') + 1]),
        actor_lr=float(command[command.index('--actor-lr') + 1]),
        critic_lr=float(command[command.index('--critic-lr') + 1]))
    assert_settings(resumed, action_std=0.01, actor_lr=5e-5, critic_lr=0.0003)
    torch.testing.assert_close(resumed.updater.policy.get_distribution(observations).mean, before_mean)
    assert applied['policy_version'] == checkpoint['updater']['policy_version'] + 1
    assert_tree_equal(resumed.updater.actor_optimizer.state_dict()['state'],
                      checkpoint['updater']['actor_optimizer']['state'])
    resumed.rollback('test')
    assert_settings(resumed, action_std=0.01, actor_lr=5e-5, critic_lr=0.0003)
