from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_collector_clips_the_validated_deterministic_policy_action():
    source = _source("scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py")
    assert "return clip_lite3_policy_action(action).detach().clone()" in source


def test_qualification_clips_policy_action_before_environment_step():
    source = _source("scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py")
    assert "action = clip_lite3_policy_action(policy(observation))" in source


def test_residual_trainer_clips_current_and_next_base_actions():
    source = _source("scripts/reinforcement_learning/online_adaptation/train_residual_sac.py")
    assert source.count("clip_lite3_policy_action(policy(") == 2


def test_play_worker_clips_warmup_current_and_next_base_actions():
    source = _source("scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py")
    assert source.count("clip_lite3_policy_action(base_policy(") == 3
