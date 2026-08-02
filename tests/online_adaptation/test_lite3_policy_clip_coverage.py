from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_collector_clips_the_validated_deterministic_policy_action():
    source = _source("scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py")
    assert "return clip_lite3_policy_action(action).detach().clone()" in source


def test_qualification_clips_policy_action_before_environment_step():
    source = _source("scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py")
    assert "raw_action = policy(observation)" in source
    assert "action = clip_lite3_policy_action(raw_action)" in source
    assert source.index("action = clip_lite3_policy_action(raw_action)") < source.index(
        "wrapped_env.step(decision.action)"
    )


def test_residual_trainer_clips_current_and_next_base_actions():
    source = _source("scripts/reinforcement_learning/online_adaptation/train_residual_sac.py")
    assert source.count("clip_lite3_policy_action(policy(") == 2


def test_play_worker_clips_warmup_current_and_next_base_actions():
    source = _source("scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py")
    assert source.count("clip_lite3_policy_action(base_policy(") == 3


def test_flat_deployment_qualification_uses_shared_startup_factory():
    source = _source(
        "scripts/reinforcement_learning/online_adaptation/qualify_flat_deployment.py"
    )
    assert "make_lite3_deployment_env" in source
    assert "RslRlVecEnvWrapper" not in source
    assert "consume_startup_generation" in source


def test_flat_online_play_uses_shared_startup_factory_and_handoff_reset():
    source = _source(
        "scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py"
    )
    assert "make_lite3_deployment_env" in source
    assert "RslRlVecEnvWrapper" not in source
    assert "consume_startup_generation" in source
    assert "reset_for_deployment_handoff" in source
