import copy
import json
import subprocess
import sys

import pytest

from rl_training.online_adaptation.interaction_budget import InteractionBudget, digest
from rl_training.online_adaptation.provenance import sha256_file


def manifest(tmp_path, *, split="development"):
    base = tmp_path / "base.pt"
    base.write_bytes(b"test-base")
    result = {"schema": "lite3_session_v1", "session_id": "test-session", "algorithm": "sac",
              "base": {"path": str(base), "sha256": sha256_file(base)}, "task_objective_sha256": "a" * 64,
              "terrain_split": split, "terrain_seed": 101, "learner_seed": 7,
              "worker_configuration": {"actor_lr": .001},
              "limits": {"adaptation": 6, "selection": 2, "evaluation": 4, "retention": 2}, "seal": None}
    return result


def start(budget, tmp_path, *, steps=2, phase="adaptation"):
    candidate = budget.state["evaluation_checkpoint"] or budget.state["latest_checkpoint"] or budget.manifest["base"]
    trial = budget.prepare_trial(phase=phase, max_steps=steps, output=tmp_path / str(len(budget.state["trials"])), input_checkpoint=candidate)
    budget.start(trial["id"])
    return trial["id"]


def test_steps_are_charged_before_physics_and_commits_are_idempotent(tmp_path):
    with InteractionBudget(tmp_path / "ledger", manifest=manifest(tmp_path)) as budget:
        trial = start(budget, tmp_path, steps=3)
        budget.reserve(trial, 1, num_envs=2)
        assert budget.remaining("adaptation") == 4
        with pytest.raises(ValueError, match="pending"):
            budget.reserve(trial, 2)
        budget.commit(trial, 1)
        budget.commit(trial, 1)
        assert budget.state["trials"][trial]["confirmed_steps"] == 2
        with pytest.raises(ValueError, match="used or skipped"):
            budget.reserve(trial, 1)
        budget.reserve(trial, 2)
        budget.commit(trial, 2)
        with pytest.raises(ValueError, match="exceed budget"):
            budget.reserve(trial, 3)
        budget.finish(trial, steps=3, outcome={})


def test_crash_after_reservation_keeps_uncertain_cost_and_never_reuses_trial(tmp_path):
    directory = tmp_path / "ledger"
    with InteractionBudget(directory, manifest=manifest(tmp_path)) as budget:
        trial = start(budget, tmp_path)
        budget.reserve(trial, 1)
    with InteractionBudget(directory) as resumed:
        assert resumed.remaining("adaptation") == 5
        with pytest.raises(ValueError, match="already started"):
            resumed.start(trial)
        resumed.abort_active("worker died with unknown physics completion")
        assert resumed.state["uncertain_steps"] == 1
        assert resumed.remaining("adaptation") == 5
        retry = start(resumed, tmp_path)
        assert retry != trial
        assert resumed.state["trials"][retry]["reset_seed"] != resumed.state["trials"][trial]["reset_seed"]


def test_abrupt_process_exit_releases_ownership_but_keeps_reserved_cost(tmp_path):
    directory = tmp_path / "ledger"
    m = manifest(tmp_path)
    with InteractionBudget(directory, manifest=m) as budget:
        trial = budget.prepare_trial(phase="adaptation", max_steps=2, output=tmp_path / "run", input_checkpoint=m["base"])
    command = [sys.executable, "-c", "import os,sys; from rl_training.online_adaptation.interaction_budget import InteractionBudget; b=InteractionBudget(sys.argv[1]); b.start(sys.argv[2]); b.reserve(sys.argv[2],1); os._exit(17)", str(directory), trial["id"]]
    assert subprocess.run(command, check=False).returncode == 17
    with InteractionBudget(directory) as budget:
        assert budget.remaining("adaptation") == 5
        budget.abort_active("worker process exited without cleanup")
        assert budget.state["uncertain_steps"] == 1


def test_completed_checkpoint_and_cost_survive_reload_without_reexecution(tmp_path):
    directory = tmp_path / "ledger"
    m = manifest(tmp_path)
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    reference = {"path": str(candidate), "sha256": sha256_file(candidate)}
    with InteractionBudget(directory, manifest=m) as budget:
        trial = start(budget, tmp_path)
        budget.reserve(trial, 1)
        budget.commit(trial, 1)
        with pytest.raises(ValueError, match="summary"):
            budget.finish(trial, steps=2, outcome={})
        budget.finish(trial, steps=1, outcome={"passed": False}, checkpoint=reference)
    with InteractionBudget(directory) as budget:
        assert budget.remaining("adaptation") == 5
        assert budget.state["latest_checkpoint"] == reference
        assert budget.state["active"] is None
        with pytest.raises(ValueError, match="active trial"):
            budget.start(trial)


def test_selection_shares_adaptation_budget_and_has_its_own_cap(tmp_path):
    with InteractionBudget(tmp_path / "ledger", manifest=manifest(tmp_path)) as budget:
        trial = start(budget, tmp_path, phase="selection")
        for step in (1, 2):
            budget.reserve(trial, step)
            budget.commit(trial, step)
        budget.finish(trial, steps=2, outcome={})
        assert budget.remaining("adaptation") == 4
        assert budget.remaining("selection") == 0
        assert budget.remaining("flat_selection") == 0
        with pytest.raises(ValueError, match="exceeds"):
            start(budget, tmp_path, steps=1, phase="flat_selection")


def test_frozen_evaluation_locks_checkpoint_and_disallows_feedback_to_training(tmp_path):
    m = manifest(tmp_path)
    with InteractionBudget(tmp_path / "ledger", manifest=m) as budget:
        with pytest.raises(ValueError, match="frozen evaluation"):
            start(budget, tmp_path, phase="evaluation")
        budget.lock_evaluation()
        for phase in ("adaptation", "selection", "warmup"):
            with pytest.raises(ValueError, match="frozen evaluation"):
                start(budget, tmp_path, phase=phase)
        trial = start(budget, tmp_path, phase="evaluation")
        budget.reserve(trial, 1)
        budget.commit(trial, 1)
        budget.finish(trial, steps=1, outcome={"passed": True})
        assert budget.remaining("adaptation") == 6
        assert budget.remaining("evaluation") == 3
        assert budget.state["evaluation_checkpoint"] == m["base"]


def test_manifest_edits_and_concurrent_owners_are_rejected(tmp_path):
    directory = tmp_path / "ledger"
    with InteractionBudget(directory, manifest=manifest(tmp_path)):
        with pytest.raises(BlockingIOError):
            InteractionBudget(directory)
    m = json.loads((directory / "manifest.json").read_text())
    m["limits"]["adaptation"] = 100
    (directory / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="journal"):
        InteractionBudget(directory)


def test_torn_final_write_is_preserved_and_pending_physics_remains_uncertain(tmp_path):
    directory = tmp_path / "ledger"
    with InteractionBudget(directory, manifest=manifest(tmp_path)) as budget:
        trial = start(budget, tmp_path)
        budget.reserve(trial, 1)
    with (directory / "events.jsonl").open("ab") as stream:
        stream.write(b'{"incomplete_commit":')
    with InteractionBudget(directory) as budget:
        assert budget.state["pending"] is not None
        budget.abort_active("interrupted journal write")
        assert budget.state["uncertain_steps"] == 1
    assert list(directory.glob("torn-tail-*.bin"))


def test_sealed_requires_locked_protocol_and_rejects_configuration_changes(tmp_path):
    m = manifest(tmp_path, split="sealed")
    with pytest.raises(ValueError, match="locked final protocol"):
        InteractionBudget(tmp_path / "missing-seal", manifest=m)
    m["seal"] = {"status": "locked", "protocol_sha256": "b" * 64,
                 "configuration_sha256": digest(m["worker_configuration"]), "terrain_seeds": [101],
                 "selection_rule_sha256": "c" * 64}
    changed = copy.deepcopy(m)
    changed["worker_configuration"]["actor_lr"] *= 2
    with pytest.raises(ValueError, match="configuration"):
        InteractionBudget(tmp_path / "changed", manifest=changed)
    with InteractionBudget(tmp_path / "sealed", manifest=m) as budget:
        # Locked in-session adaptation is allowed; feedback after evaluation is not.
        trial = start(budget, tmp_path)
        budget.abort_active("unit-test boundary")
        budget.lock_evaluation()
        with pytest.raises(ValueError, match="frozen evaluation"):
            start(budget, tmp_path)
