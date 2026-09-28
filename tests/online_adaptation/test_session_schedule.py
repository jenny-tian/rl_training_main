from types import SimpleNamespace

import pytest

from rl_training.online_adaptation.session_schedule import build_schedule, next_trial
from rl_training.online_adaptation.interaction_budget import InteractionBudget
from test_interaction_budget import manifest


def test_default_probe_reservation_fits_exactly_inside_main_budget():
    schedule = build_schedule(trial_steps=2000, evaluation_trials=10)
    assert sum(p["max_steps"] for p in schedule["probes"]) == 8000
    assert schedule["probes"][-1]["at"] + schedule["probes"][-1]["max_steps"] == 40000


def test_paid_probes_do_not_advance_candidate_and_resume_reaches_total_cap(tmp_path):
    m = manifest(tmp_path)
    m.update(schema="lite3_session_v2", schedule=build_schedule(
        trial_steps=4, evaluation_trials=2, retention_steps=2, retention_trials=2,
        probe_milestones=[12, 24], budget_steps=24))
    m["limits"].update(adaptation=24, selection=10, evaluation=8, retention=4)
    phases = []
    directory = tmp_path / "ledger"
    with InteractionBudget(directory, manifest=m):
        pass
    while True:
        with InteractionBudget(directory) as budget:
            spec = next_trial(budget, "adaptation")
            if spec is None:
                assert budget.remaining("adaptation") == 0
                assert budget.state["spent"]["selection"] == 8
                assert budget.state["spent"]["flat_selection"] == 2
                assert budget.state["latest_checkpoint"] is None
                break
            phases.append(spec["phase"])
            trial = budget.prepare_trial(**spec, output=tmp_path / str(len(phases)), input_checkpoint=m["base"])
            budget.start(trial["id"])
            for step in range(1, spec["max_steps"] + 1):
                budget.reserve(trial["id"], step)
                budget.commit(trial["id"], step)
            budget.finish(trial["id"], steps=spec["max_steps"], outcome={})
    assert phases.count("selection") == 2
    assert phases.count("flat_selection") == 1


def test_short_probe_returns_unused_steps_to_collection_without_extra_starts(tmp_path):
    m = manifest(tmp_path)
    m.update(schema="lite3_session_v2", schedule=build_schedule(
        trial_steps=2, evaluation_trials=1, retention_steps=1,
        probe_milestones=[6], budget_steps=6))
    with InteractionBudget(tmp_path / "ledger", manifest=m) as budget:
        for phase, steps in [("adaptation", 2), ("adaptation", 2), ("selection", 1)]:
            spec = next_trial(budget, "adaptation")
            assert spec["phase"] == phase
            trial = budget.prepare_trial(**spec, output=tmp_path / str(len(budget.state["trials"])), input_checkpoint=m["base"])
            budget.start(trial["id"])
            for step in range(1, steps + 1):
                budget.reserve(trial["id"], step)
                budget.commit(trial["id"], step)
            budget.finish(trial["id"], steps=steps, outcome={"fell": phase == "selection"})
        assert next_trial(budget, "adaptation") == {"phase": "adaptation", "max_steps": 1}


def test_overlapping_probe_schedule_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        build_schedule(trial_steps=2000, evaluation_trials=10, probe_milestones=[1000])


def test_short_fall_failure_is_explicit_in_exported_summary(tmp_path):
    import json
    from scripts.reinforcement_learning.online_adaptation.run_adaptation_session import write_session_summary

    budget = SimpleNamespace(state={"spent": {"adaptation": 99}}, short_fall_stop=lambda: True)
    write_session_summary(tmp_path, budget)
    report = json.loads((tmp_path / "session_summary.json").read_text())
    assert report["adaptation_early_stopped"] is True
    assert report["adaptation_failure_reason"] == "five_consecutive_short_falls"
    assert report["spent"]["adaptation"] == 99
