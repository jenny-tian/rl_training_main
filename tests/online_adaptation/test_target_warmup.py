from types import SimpleNamespace
import pytest
from rl_training.online_adaptation.target_warmup import TargetWarmup, restore_target_warmup
from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import TransitionUpdateBudget


def test_collection_and_fitting_are_separate_resumable_stages():
    state = TargetWarmup("sac", 4000, 1000)
    for _ in range(2000):
        state.record_transition()
    resumed = TargetWarmup("sac", 4000, 1000)
    resumed.restore(state.state_dict())
    for _ in range(2000):
        resumed.record_transition()
    assert not resumed.collecting and not resumed.ready
    resumed.critic_updates = 1000
    assert resumed.ready
    resumed.record_transition()
    assert resumed.collected_steps == 4000


@pytest.mark.parametrize("state", [
    dict(required_steps=-1), dict(required_steps=True),
    dict(required_steps=10, required_critic_updates=1001),
    dict(required_steps=10, required_critic_updates=3, collected_steps=11),
    dict(required_steps=10, required_critic_updates=3, critic_updates=1),
])
def test_invalid_progress_is_rejected(state):
    with pytest.raises(ValueError):
        TargetWarmup("sac", **state)


def test_resume_refuses_legacy_or_changed_warmup():
    args = SimpleNamespace(target_warmup_steps=4000, target_critic_warmup_updates=1000)
    with pytest.raises(ValueError, match="no target warmup"):
        restore_target_warmup(args, "sac", {})
    state = TargetWarmup("sac", 5000, 1000).state_dict()
    with pytest.raises(ValueError, match="changed"):
        restore_target_warmup(args, "sac", {"target_warmup": state})


def test_actor_update_debits_its_critic_step_atomically():
    budget = TransitionUpdateBudget(critic_updates_per_transition=1., actor_updates_per_transition=1.)
    budget.add(critic_transitions=0, actor_transitions=1)
    assert not budget.take_actor(consume_critic=True)
    assert budget.snapshot() == {"critic": 0., "actor": 1.}
    budget.add(critic_transitions=1, actor_transitions=0)
    assert budget.take_actor(consume_critic=True)
    assert budget.snapshot() == {"critic": 0., "actor": 0.}


@pytest.mark.parametrize("algorithm", ["lora", "sac"])
def test_supervisor_warmup_options_parse_in_actual_worker(tmp_path, algorithm):
    import json
    from scripts.reinforcement_learning.online_adaptation.run_adaptation_session import parser_for, worker_arguments
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"target-critic-warmup-updates": 1000, "critic-huber-delta": 1.} if algorithm == "sac" else {}))
    args = SimpleNamespace(algorithm=algorithm, phase="adaptation", base_checkpoint="base.pt",
                           terrain_seed=101, seed=7, deadline_steps=2000, target_warmup_steps=4000,
                           device="cuda:0", worker_options=str(options))
    parsed = parser_for(algorithm).parse_args(worker_arguments(args, output=tmp_path, steps=2000))
    assert parsed.target_warmup_steps == 4000
    if algorithm == "sac":
        assert parsed.target_critic_warmup_updates == 1000
        assert parsed.critic_huber_delta == 1.
