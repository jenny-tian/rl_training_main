from types import SimpleNamespace

import pytest
import torch

from rl_training.policy_recovery_anchor import prepare_recovery_anchor, recovery_anchor_mask, recovery_anchor_reward


def test_ordinary_rough_and_flat_stay_protected_while_recovery_tail_is_free():
    cases = torch.tensor([-1, -1, 26, 26, 26])
    age = torch.tensor([400, 1700, 0, 11, 12])
    assert recovery_anchor_mask(cases, age, 12).tolist() == [True, True, True, True, False]
    age[-1] = 0  # Automatic reset must re-enable the same slot's early protection.
    assert recovery_anchor_mask(cases, age, 12).all()


def test_anchor_uses_pre_action_target_and_rejects_a_stale_target():
    env = SimpleNamespace(_recovery_case=torch.tensor([-1, 26, 26]),
                          episode_length_buf=torch.tensor([600, 11, 12]), common_step_counter=20)
    obs = torch.zeros(3, 48)
    obs[:, 0] = torch.tensor([.1, .2, .3])
    def teacher(x):
        return x[:, :1].expand(-1, 12)
    target, mask = prepare_recovery_anchor(env, teacher, obs, 12)
    obs[:, 0] = 9.  # Mutation after prepare must not turn this into a post-action target.
    env.action_manager = SimpleNamespace(action=torch.zeros(3, 12))
    env.common_step_counter += 1
    torch.testing.assert_close(recovery_anchor_reward(env), torch.tensor([.01, .04, 0.]))
    assert mask.tolist() == [True, True, False]
    env.common_step_counter += 1
    with pytest.raises(RuntimeError, match='immediately before'):
        recovery_anchor_reward(env)


@pytest.mark.parametrize('age,early', [(torch.tensor([-1]), 12), (torch.tensor([0.5]), 12), (torch.tensor([0]), 0)])
def test_invalid_episode_age_cannot_silently_disable_protection(age, early):
    with pytest.raises(ValueError):
        recovery_anchor_mask(torch.tensor([26]), age, early)


def test_fixed_source_subset_keeps_ordinary_slots_on_same_terrain():
    from rl_training.recovery_teacher import single_training_case
    terrain = torch.arange(8).repeat_interleave(8)
    bank = torch.tensor([101, 107])
    validation = torch.tensor([False, False])
    full = single_training_case(terrain, bank, validation, 1)
    subset = single_training_case(terrain, bank, validation, 1, count=12)
    assert (full == 1).sum() == 24
    assert (subset[40:52] == 1).all() and (subset[52:] == -1).all()
    assert (subset[:40] == -1).all()
    with pytest.raises(ValueError, match='slot count'):
        single_training_case(terrain, bank, validation, 1, count=25)
