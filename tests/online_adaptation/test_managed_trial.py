from pathlib import Path

import pytest

from rl_training.online_adaptation.interaction_budget import InteractionBudget
from rl_training.online_adaptation.managed_trial import attach_session, worker_configuration
from rl_training.online_adaptation.provenance import sha256_file
from rl_training.online_adaptation.task_objective import FiniteHorizonObjective
from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser


def setup_trial(tmp_path, *, declared_seed=101, phase="evaluation"):
    base = tmp_path / "base.pt"
    base.write_bytes(b"base fixture")
    args = build_play_parser().parse_args([
        "--base-checkpoint", str(base), "--mode", "base", "--evaluation-only",
        "--task-objective", "finite-horizon-v1", "--require-track-completion",
        "--terrain-split", "development", "--terrain-seed", "101",
        "--update-schedule", "control-step", "--max-steps", "4", "--run-dir", str(tmp_path / "run"),
    ])
    m = {"schema": "lite3_session_v1", "session_id": "test", "algorithm": "sac",
         "base": {"path": str(base), "sha256": sha256_file(base)},
         "task_objective_sha256": FiniteHorizonObjective().sha256(),
         "terrain_split": "development", "terrain_seed": declared_seed, "learner_seed": 7,
         "worker_configuration": worker_configuration(args),
         "limits": {"adaptation": 8, "selection": 2, "evaluation": 8, "retention": 2}, "seal": None}
    with InteractionBudget(tmp_path / "ledger", manifest=m) as budget:
        if phase == "evaluation":
            budget.lock_evaluation()
        trial = budget.prepare_trial(phase=phase, max_steps=4, output=tmp_path / "run", input_checkpoint=m["base"])
    args.session_ledger, args.session_run_id = str(tmp_path / "ledger"), trial["id"]
    return args


def test_worker_scope_cannot_disagree_with_manifest_seed(tmp_path):
    args = setup_trial(tmp_path, declared_seed=103)
    with pytest.raises(ValueError, match="declared session scope"):
        attach_session(args, "sac")


def test_managed_checkpoint_cannot_overwrite_base_or_another_trial(tmp_path):
    args = setup_trial(tmp_path, phase="adaptation")
    args.evaluation_only, args.mode = False, "online"
    args.experiment_checkpoint_out = args.base_checkpoint
    with pytest.raises(ValueError, match="checkpoint output"):
        attach_session(args, "sac")
    assert Path(args.base_checkpoint).read_bytes() == b"base fixture"


@pytest.mark.parametrize("change", ["learning", "budget", "configuration", "base"])
def test_worker_rejects_session_contract_changes_before_physics(tmp_path, change):
    args = setup_trial(tmp_path)
    if change == "learning":
        args.evaluation_only = False
    elif change == "budget":
        args.max_steps += 1
    elif change == "configuration":
        args.forward_speed += .1
    else:
        Path(args.base_checkpoint).write_bytes(b"changed")
    with pytest.raises(ValueError):
        attach_session(args, "sac")
    with InteractionBudget(args.session_ledger) as budget:
        assert budget.state["spent"]["evaluation"] == 0
        assert budget.state["trials"][args.session_run_id]["status"] == "ready"


def test_frozen_trial_rejects_replay_mutation_and_completes_exactly_once(tmp_path):
    args = setup_trial(tmp_path)
    managed = attach_session(args, "sac")
    try:
        managed.reserve(1)
        managed.commit(1)
        with pytest.raises(ValueError, match="mutated"):
            managed.finish({"steps": 1, "replay_transitions_written": 1})
        managed.finish({"steps": 1, "updates": 0})
    finally:
        managed.close()
    with pytest.raises(ValueError, match="duplicate"):
        attach_session(args, "sac")


def test_bare_sealed_common_objective_is_rejected_before_loading_environment():
    args = build_play_parser().parse_args([
        "--base-checkpoint", "not-loaded.pt", "--mode", "base", "--evaluation-only",
        "--task-objective", "finite-horizon-v1", "--require-track-completion", "--terrain-seed", "1009",
    ])
    with pytest.raises(ValueError, match="locked session"):
        attach_session(args, "sac")


@pytest.mark.parametrize("flat,completion", [(True, False), (False, False), (True, True)])
def test_terrain_phase_cannot_be_relabelled_as_flat_or_unqualified(tmp_path, flat, completion):
    args = setup_trial(tmp_path)
    args.flat_retention_probe, args.require_track_completion = flat, completion
    with pytest.raises(ValueError, match="terrain mode"):
        attach_session(args, "sac")
