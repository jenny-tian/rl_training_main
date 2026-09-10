from dataclasses import replace

import pytest

from rl_training.online_adaptation.reward import RewardWeights
from rl_training.online_adaptation.sac import SACConfig
from rl_training.online_adaptation.training_contract import training_contract, validate_training_contract


@pytest.mark.parametrize("change", ["reward", "temperature", "target", "missing"])
def test_resume_rejects_changed_reward_or_sac_objective(change):
    sac = SACConfig(initial_alpha=.02, alpha_max=.05)
    reward = RewardWeights(rough_swing_lift_weight=1.0)
    current = training_contract(sac, reward, target_semantics="timeout-aware")
    validate_training_contract(current, current)
    saved = training_contract(
        replace(sac, initial_alpha=.01) if change == "temperature" else sac,
        replace(reward, rough_swing_lift_weight=0) if change == "reward" else reward,
        target_semantics="legacy-done" if change == "target" else "timeout-aware",
    )
    with pytest.raises(ValueError, match="training contract"):
        validate_training_contract(None if change == "missing" else saved, current)
