import pytest
import torch

from rl_training.online_adaptation.recovery_perturbation import scheduled_recovery_impulse


def test_impulses_are_paired_bounded_rough_only_and_exclude_ended_slots():
    active = torch.ones(12, dtype=torch.bool)
    terrain = torch.tensor([0,0,1,1,2,2,3,3,5,5,6,6])
    config = dict(steps=[600,900], roll_velocity_rad_s=1.25, lateral_velocity_m_s=.12)
    before = active.clone()
    first = scheduled_recovery_impulse(600, active, terrain, **config)
    assert torch.where(first.abs().sum(-1)>0)[0].tolist() == [4,6,8,10]
    assert not first[:4].any()
    assert not first[:,[0,2,4,5]].any()
    torch.testing.assert_close(first[4], torch.tensor([0.,-.12,0.,1.25,0.,0.]))
    torch.testing.assert_close(first[6], -first[4])
    assert first[:,3].sum() == 0
    torch.testing.assert_close(first, scheduled_recovery_impulse(600, active, terrain, **config))
    torch.testing.assert_close(-first, scheduled_recovery_impulse(900, active, terrain, **config))
    assert not scheduled_recovery_impulse(601, active, terrain, **config).any()
    torch.testing.assert_close(active, before)
    active[4] = False
    assert not scheduled_recovery_impulse(900, active, terrain, **config)[4].any()


def test_invalid_schedule_is_rejected():
    active = torch.ones(4, dtype=torch.bool)
    with pytest.raises(ValueError, match="increasing"):
        scheduled_recovery_impulse(1, active, torch.zeros(4), steps=[1,1],
            roll_velocity_rad_s=1., lateral_velocity_m_s=.1)
