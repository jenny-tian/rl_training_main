from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from rl_training.online_adaptation.provenance import sha256_file
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from scripts.reinforcement_learning.online_adaptation.freeze_flat_online_configuration import (
    freeze_configuration,
    verify_formal_configuration,
)
from scripts.reinforcement_learning.online_adaptation.select_flat_online_configuration import (
    QCalibrationRequired,
    q_calibration,
    select_best_candidate,
)


def candidate(name, **overrides):
    value = {
        "name": name,
        "development_seeds": list(default_flat_online_protocol().development_seeds),
        "base_manifest_sha256": "a" * 64,
        "budgets": {"episodes": 100, "steps": 200_000},
        "pass_rate": 0.8,
        "falls": 0,
        "safety_aborts": 0,
        "flat_retention_cost": 0.1,
        "interaction_steps": 1000,
    }
    value.update(overrides)
    return value


def test_selection_uses_documented_lexicographic_order():
    selected = select_best_candidate(
        [candidate("slow", interaction_steps=2000), candidate("fast", interaction_steps=1000)]
    )
    assert selected["name"] == "fast"


def test_selection_rejects_incomparable_development_inputs():
    changed = candidate("changed", development_seeds=[101])
    with pytest.raises(ValueError, match="development_seeds"):
        select_best_candidate([candidate("valid"), changed])


def test_q_calibration_requires_finite_comparable_scales():
    with pytest.raises(QCalibrationRequired, match="q_calibration_required"):
        q_calibration(
            {
                "q_distributions": {
                    "flat_q": [1.0],
                    "online_q": [float("nan")],
                    "target_q": [1.0],
                    "td_error": [1.0],
                }
            }
        )


def test_freeze_is_read_only_and_detects_changed_inputs(tmp_path):
    base = tmp_path / "base.json"
    replay = tmp_path / "replay.json"
    checkpoint = tmp_path / "experiment.pt"
    for path, content in ((base, "{}"), (replay, "{}"), (checkpoint, "checkpoint")):
        path.write_text(content, encoding="utf-8")
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            {
                "protocol_sha256": default_flat_online_protocol().sha256(),
                "formal_input_hashes": {
                    "base_manifest": sha256_file(base),
                    "flat_replay_manifest": sha256_file(replay),
                    "experiment_checkpoint": sha256_file(checkpoint),
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "formal.json"

    freeze_configuration(
        selected_configuration=selected,
        base_manifest=base,
        flat_replay_manifest=replay,
        experiment_checkpoint=checkpoint,
        output=output,
        git_commit="1" * 40,
    )

    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    verify_formal_configuration(output)
    with pytest.raises(FileExistsError):
        freeze_configuration(
            selected_configuration=selected,
            base_manifest=base,
            flat_replay_manifest=replay,
            experiment_checkpoint=checkpoint,
            output=output,
            git_commit="1" * 40,
        )
    checkpoint.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_formal_configuration(output)


def test_formal_verification_rejects_writable_file(tmp_path):
    path = tmp_path / "formal.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="read-only"):
        verify_formal_configuration(path)
