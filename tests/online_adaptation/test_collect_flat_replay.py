import argparse
import hashlib
import json
import os
import stat
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from rl_training.lite3_joint_conventions import (
    LITE3_SAFE_ACTION_LOWER,
    LITE3_SAFE_ACTION_UPPER,
)
from rl_training.online_adaptation.provenance import sha256_file
from rl_training.online_adaptation.replay_buffer import ReplayBuffer
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from scripts.reinforcement_learning.online_adaptation import collect_flat_replay
from scripts.reinforcement_learning.online_adaptation import cli as cli_module
from scripts.reinforcement_learning.online_adaptation.cli import build_collect_flat_replay_parser


STRICT_FLAT_TASK = "Flat-Deeprobotics-Lite3-v0"
MANIFEST_KEYS = {
    "baseline_manifest_sha256",
    "protocol_sha256",
    "task",
    "seed",
    "transition_count",
    "replay_sha256",
    "falls",
    "safety_invalid_rows",
    "frozen",
}


def _baseline_inputs(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"frozen-flat-ppo\x00\x01")
    checkpoint_bytes = checkpoint.read_bytes()
    baseline_manifest = tmp_path / "flat_baseline_manifest.json"
    baseline_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "terrain_assertion": "plane-only",
                "task": STRICT_FLAT_TASK,
                "files": {
                    str(checkpoint.resolve()): {
                        "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                        "bytes": len(checkpoint_bytes),
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, baseline_manifest


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    checkpoint, baseline_manifest = _baseline_inputs(tmp_path)
    values = {
        "base_checkpoint": str(checkpoint),
        "base_manifest": str(baseline_manifest),
        "output_dir": str(tmp_path / "replay"),
        "task": STRICT_FLAT_TASK,
        "seed": 42,
        "transitions": 100_000,
        "device": "cpu",
        "headless": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _formal_manifest(**overrides):
    manifest = {
        "baseline_manifest_sha256": "a" * 64,
        "protocol_sha256": default_flat_online_protocol().sha256(),
        "task": STRICT_FLAT_TASK,
        "seed": 42,
        "transition_count": 100_000,
        "replay_sha256": "b" * 64,
        "falls": 0,
        "safety_invalid_rows": 0,
        "frozen": True,
    }
    manifest.update(overrides)
    return manifest


def _single_transition(value: float = 1.0):
    return dict(
        history=torch.full((1, 3, 2), value),
        current_frame=torch.full((1, 2), value),
        base_action=torch.full((1, 1), value),
        residual_action=torch.zeros(1, 1),
        reward=torch.full((1, 1), value),
        next_history=torch.full((1, 3, 2), value + 1),
        next_frame=torch.full((1, 2), value + 1),
        next_base_action=torch.full((1, 1), value + 1),
        terminated=torch.zeros(1, 1),
        truncated=torch.zeros(1, 1),
        bootstrap_mask=torch.ones(1, 1),
        discount=torch.full((1, 1), 0.99),
        safety_mask=torch.ones(1, 1),
        curriculum_stage_id=torch.zeros(1, 1, dtype=torch.int64),
        episode_id=torch.full((1, 1), 9, dtype=torch.int64),
    )


def _full_frozen_replay() -> ReplayBuffer:
    replay = ReplayBuffer(
        capacity=100_000,
        history_length=3,
        frame_dim=2,
        action_dim=1,
    )
    full_state = replay.state_dict()
    full_state.update(size=100_000, cursor=0, frozen=True)
    replay.load_state_dict(full_state)
    return replay


def _leave_ready_stage_without_finals(tmp_path, monkeypatch):
    output_dir = tmp_path / "formal"
    replay = _full_frozen_replay()
    real_link = collect_flat_replay.os.link

    class SimulatedCrash(BaseException):
        pass

    def crash_before_first_final_link(_source, _destination):
        raise SimulatedCrash("process stopped before final publication")

    monkeypatch.setattr(collect_flat_replay.os, "link", crash_before_first_final_link)
    with pytest.raises(SimulatedCrash):
        collect_flat_replay.publish_flat_replay(
            replay=replay,
            output_dir=output_dir,
            baseline_manifest_sha256="a" * 64,
            task=STRICT_FLAT_TASK,
            seed=42,
            transition_count=100_000,
            falls=0,
            safety_invalid_rows=0,
        )
    monkeypatch.setattr(collect_flat_replay.os, "link", real_link)
    stage = next(path for path in output_dir.iterdir() if path.name.startswith(".flat_replay.stage-"))
    return output_dir, stage


def _inject_app_launcher_args(monkeypatch):
    calls = []

    def add_launcher_args(parser):
        calls.append(parser)
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--headless", action="store_true")
        return parser

    monkeypatch.setattr(cli_module, "_add_app_launcher_args", add_launcher_args)
    return calls


def test_collect_parser_uses_official_launcher_args_once_without_conflicts(monkeypatch):
    calls = _inject_app_launcher_args(monkeypatch)
    parser = build_collect_flat_replay_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args(
        [
            "--base-checkpoint",
            "base.pt",
            "--base-manifest",
            "baseline.json",
            "--output-dir",
            "output",
        ]
    )

    assert args.task == STRICT_FLAT_TASK
    assert args.seed == 42
    assert args.transitions == 100_000
    assert args.device == "cuda:0"
    assert args.headless is False
    assert calls == [parser]


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--seed", "0"),
        ("--seed", "-1"),
        ("--seed", "1.5"),
        ("--transitions", "99999"),
        ("--transitions", "0"),
        ("--transitions", "100000.0"),
    ],
)
def test_collect_parser_rejects_nonformal_exact_integer_values(flag, value, monkeypatch):
    _inject_app_launcher_args(monkeypatch)
    parser = build_collect_flat_replay_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--base-checkpoint",
                "base.pt",
                "--base-manifest",
                "baseline.json",
                "--output-dir",
                "output",
                flag,
                value,
            ]
        )


def test_nonflat_task_is_rejected_before_runtime_or_app_launcher(tmp_path, monkeypatch):
    args = _args(tmp_path, task="OnlineAdapt-Deeprobotics-Lite3-v0")
    parser = argparse.Namespace(parse_args=lambda argv=None: args)
    runtime = lambda *_args, **_kwargs: pytest.fail("runtime must not be reached")
    monkeypatch.setattr(collect_flat_replay, "build_collect_flat_replay_parser", lambda: parser)
    monkeypatch.setattr(collect_flat_replay, "_run_collection", runtime)

    with pytest.raises(ValueError, match="Flat-Deeprobotics-Lite3-v0"):
        collect_flat_replay.main([])


def test_collect_validation_hashes_exact_manifest_bytes_and_verifies_checkpoint(tmp_path):
    args = _args(tmp_path)

    validated = collect_flat_replay.validate_collect_args(args)

    assert validated.baseline_manifest_sha256 == sha256_file(args.base_manifest)
    assert validated.checkpoint_bytes == Path(args.base_checkpoint).read_bytes()
    Path(args.base_checkpoint).write_bytes(b"changed-policy")
    with pytest.raises(ValueError, match="checkpoint.*manifest"):
        collect_flat_replay.validate_collect_args(args)


def test_private_checkpoint_uses_verified_bytes_after_original_is_replaced(tmp_path):
    args = _args(tmp_path)
    validated = collect_flat_replay.validate_collect_args(args)
    original_bytes = validated.checkpoint_bytes
    Path(args.base_checkpoint).write_bytes(b"replacement-after-validation")

    with collect_flat_replay.materialize_verified_checkpoint(validated) as private_path:
        retained_path = private_path
        assert private_path.read_bytes() == original_bytes
        assert stat.S_IMODE(private_path.stat().st_mode) == 0o600

    assert not retained_path.exists()


def test_collect_validation_rejects_unsafe_or_existing_outputs_without_modifying_them(tmp_path):
    args = _args(tmp_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir()
    existing = output_dir / "flat_replay.pt"
    existing.write_bytes(b"sealed")

    with pytest.raises(FileExistsError, match="flat_replay.pt"):
        collect_flat_replay.validate_collect_args(args)

    assert existing.read_bytes() == b"sealed"


def test_formal_manifest_validation_requires_exact_keys_and_success_constraints():
    manifest = _formal_manifest()
    collect_flat_replay.validate_flat_replay_manifest(manifest)
    assert set(manifest) == MANIFEST_KEYS

    invalid_manifests = [
        {key: value for key, value in manifest.items() if key != "falls"},
        dict(manifest, extra=True),
        _formal_manifest(task="OnlineAdapt-Deeprobotics-Lite3-v0"),
        _formal_manifest(seed=True),
        _formal_manifest(transition_count=99_999),
        _formal_manifest(falls=1),
        _formal_manifest(safety_invalid_rows=1),
        _formal_manifest(frozen=False),
        _formal_manifest(protocol_sha256="c" * 64),
        _formal_manifest(replay_sha256="not-a-sha"),
    ]
    for invalid in invalid_manifests:
        with pytest.raises(ValueError):
            collect_flat_replay.validate_flat_replay_manifest(invalid)


def test_deterministic_base_action_uses_frozen_policy_intent_without_gradients():
    calls = []

    def policy(observation):
        calls.append((observation.clone(), torch.is_grad_enabled()))
        return observation[:, :12] + 0.25

    observation = torch.arange(12, dtype=torch.float32).unsqueeze(0).requires_grad_()
    action = collect_flat_replay.deterministic_base_action(policy, observation)

    expected = torch.minimum(
        observation.detach() + 0.25,
        observation.new_tensor(LITE3_SAFE_ACTION_UPPER),
    )
    torch.testing.assert_close(action, expected, rtol=0, atol=0)
    assert action.requires_grad is False
    assert len(calls) == 1
    assert calls[0][1] is False


def test_deterministic_base_action_clips_to_shared_lite3_bounds():
    action = collect_flat_replay.deterministic_base_action(
        lambda _observation: torch.tensor([[100.0, -100.0, 100.0] * 4]),
        torch.zeros(1, 3),
    )

    assert action.max().item() == pytest.approx(max(LITE3_SAFE_ACTION_UPPER))
    assert action.min().item() == pytest.approx(min(LITE3_SAFE_ACTION_LOWER))


def test_one_step_flat_transition_has_zero_residual_and_explicit_endings():
    values = _single_transition()
    values.pop("residual_action")
    values.pop("discount")

    transition = collect_flat_replay.build_one_step_flat_transition(**values, discount=0.97)

    assert {field.name for field in fields(transition)} == set(values) | {
        "residual_action",
        "requested_residual_action",
        "discount",
    }
    assert torch.equal(transition.residual_action, torch.zeros_like(transition.base_action))
    assert torch.equal(transition.discount, torch.full_like(transition.reward, 0.97))
    assert torch.equal(transition.terminated, values["terminated"])
    assert torch.equal(transition.truncated, values["truncated"])
    assert torch.equal(transition.bootstrap_mask, values["bootstrap_mask"])


def test_safety_diagnostics_identifies_projection_pause_reasons_and_nonfinite_fields():
    decision = argparse.Namespace(
        paused=torch.tensor([[True]]),
        projected=torch.tensor([[True]]),
        reasons=("joint_velocity",),
    )

    diagnostics = collect_flat_replay.safety_diagnostics(
        decision,
        {
            "base_action": torch.tensor([[1.0, float("nan")]]),
            "reward": torch.tensor([[0.0]]),
        },
    )

    assert diagnostics == {
        "safety_paused_rows": 1,
        "safety_projected_rows": 1,
        "safety_reasons": ["joint_velocity"],
        "nonfinite_fields": ["base_action"],
    }


def test_truncated_post_reset_next_state_is_dropped_and_collection_continues():
    replay = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)
    truncated_values = _single_transition(1.0)
    truncated_values.update(
        next_frame=torch.full((1, 2), 999.0),
        truncated=torch.ones(1, 1),
        bootstrap_mask=torch.ones(1, 1),
    )
    truncated = collect_flat_replay.store_flat_transition(
        replay,
        collect_flat_replay.build_one_step_flat_transition(
            **{
                key: value
                for key, value in truncated_values.items()
                if key not in {"residual_action", "discount"}
            },
            discount=0.99,
        ),
    )

    assert truncated.valid_rows == 0
    assert truncated.truncated_rows == 1
    assert truncated.falls == 0
    assert truncated.safety_invalid_rows == 0
    assert len(replay) == 0

    valid_values = _single_transition(2.0)
    valid = collect_flat_replay.store_flat_transition(
        replay,
        collect_flat_replay.build_one_step_flat_transition(
            **{
                key: value
                for key, value in valid_values.items()
                if key not in {"residual_action", "discount"}
            },
            discount=0.99,
        ),
    )

    assert valid.valid_rows == 1
    assert len(replay) == 1
    assert replay.sample(1, "cpu").next_frame[0, 0].item() == 3.0


def test_publish_flat_replay_saves_frozen_state_and_exact_hash_metadata(tmp_path):
    replay = _full_frozen_replay()
    output_dir = tmp_path / "formal"
    with collect_flat_replay.open_metrics_stream(output_dir) as stream:
        collect_flat_replay.write_metric(stream, {"transition_count": 100_000})

    manifest = collect_flat_replay.publish_flat_replay(
        replay=replay,
        output_dir=output_dir,
        baseline_manifest_sha256="a" * 64,
        task=STRICT_FLAT_TASK,
        seed=42,
        transition_count=100_000,
        falls=0,
        safety_invalid_rows=0,
    )

    replay_path = output_dir / "flat_replay.pt"
    manifest_path = output_dir / "flat_replay_manifest.json"
    saved_state = torch.load(replay_path, map_location="cpu", weights_only=True)
    assert set(manifest) == MANIFEST_KEYS
    assert manifest["replay_sha256"] == sha256_file(replay_path)
    assert manifest["baseline_manifest_sha256"] == "a" * 64
    assert manifest["protocol_sha256"] == default_flat_online_protocol().sha256()
    assert manifest["transition_count"] == 100_000
    assert manifest["frozen"] is True
    assert saved_state["frozen"] is True
    assert saved_state["size"] == 100_000
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert {path.name for path in output_dir.iterdir()} == {
        "flat_replay.pt",
        "flat_replay_manifest.json",
        "metrics.jsonl",
        "READY.json",
    }

    replay_bytes = replay_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        collect_flat_replay.publish_flat_replay(
            replay=replay,
            output_dir=output_dir,
            baseline_manifest_sha256="a" * 64,
            task=STRICT_FLAT_TASK,
            seed=42,
            transition_count=100_000,
            falls=0,
            safety_invalid_rows=0,
        )
    assert replay_path.read_bytes() == replay_bytes
    assert manifest_path.read_bytes() == manifest_bytes
    assert collect_flat_replay.recover_incomplete_publication(output_dir) is False
    assert replay_path.read_bytes() == replay_bytes
    assert manifest_path.read_bytes() == manifest_bytes


def test_recovery_completes_ready_stage_after_crash_between_final_links(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path)
    validated = collect_flat_replay.validate_collect_args(args)
    replay = _full_frozen_replay()
    output_dir = Path(args.output_dir)
    with collect_flat_replay.open_metrics_stream(output_dir) as stream:
        collect_flat_replay.write_metric(stream, {"transition_count": 100_000})
    real_link = collect_flat_replay.os.link

    class SimulatedCrash(BaseException):
        pass

    def crash_before_manifest(source, destination):
        if Path(destination).name == "flat_replay_manifest.json":
            raise SimulatedCrash("process stopped after replay publication")
        return real_link(source, destination)

    monkeypatch.setattr(collect_flat_replay.os, "link", crash_before_manifest)
    with pytest.raises(SimulatedCrash):
        collect_flat_replay.publish_flat_replay(
            replay=replay,
            output_dir=output_dir,
            baseline_manifest_sha256=validated.baseline_manifest_sha256,
            task=STRICT_FLAT_TASK,
            seed=42,
            transition_count=100_000,
            falls=0,
            safety_invalid_rows=0,
        )

    assert (output_dir / "flat_replay.pt").is_file()
    assert not (output_dir / "flat_replay_manifest.json").exists()
    assert any(path.name.startswith(".flat_replay.stage-") for path in output_dir.iterdir())

    monkeypatch.setattr(collect_flat_replay.os, "link", real_link)
    with pytest.raises(FileExistsError):
        collect_flat_replay.validate_collect_args(args)

    manifest_path = output_dir / "flat_replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["replay_sha256"] == sha256_file(output_dir / "flat_replay.pt")
    assert not any(path.name.startswith(".flat_replay.stage-") for path in output_dir.iterdir())


def test_recovery_refuses_ambiguous_incomplete_stage_without_overwriting(tmp_path):
    output_dir = tmp_path / "formal"
    output_dir.mkdir()
    stage = output_dir / ".flat_replay.stage-ambiguous"
    stage.mkdir()
    (stage / "flat_replay.pt").write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="ambiguous|incomplete"):
        collect_flat_replay.recover_incomplete_publication(output_dir)

    assert (stage / "flat_replay.pt").read_bytes() == b"partial"
    assert not (output_dir / "flat_replay_manifest.json").exists()


@pytest.mark.parametrize(
    "entry_kind",
    [
        "replay_symlink_to_stage",
        "manifest_symlink_to_stage",
        "dangling_replay_symlink",
        "replay_directory",
        pytest.param(
            "replay_fifo",
            marks=pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unsupported"),
        ),
    ],
)
def test_recovery_refuses_nonregular_final_entries_without_mutation(
    tmp_path,
    monkeypatch,
    entry_kind,
):
    output_dir, stage = _leave_ready_stage_without_finals(tmp_path, monkeypatch)
    final_replay = output_dir / "flat_replay.pt"
    final_manifest = output_dir / "flat_replay_manifest.json"
    if entry_kind == "replay_symlink_to_stage":
        final_replay.symlink_to(stage / "flat_replay.pt")
        guarded_final = final_replay
    elif entry_kind == "manifest_symlink_to_stage":
        os.link(stage / "flat_replay.pt", final_replay)
        final_manifest.symlink_to(stage / "flat_replay_manifest.json")
        guarded_final = final_manifest
    elif entry_kind == "dangling_replay_symlink":
        final_replay.symlink_to(output_dir / "missing-replay.pt")
        guarded_final = final_replay
    elif entry_kind == "replay_directory":
        final_replay.mkdir()
        guarded_final = final_replay
    else:
        os.mkfifo(final_replay)
        guarded_final = final_replay

    before_mode = guarded_final.lstat().st_mode
    real_sha256_file = collect_flat_replay.sha256_file

    def reject_nonregular_hash(path, *args, **kwargs):
        if Path(path) == guarded_final:
            raise AssertionError("nonregular final entry was hashed")
        return real_sha256_file(path, *args, **kwargs)

    monkeypatch.setattr(collect_flat_replay, "sha256_file", reject_nonregular_hash)
    with pytest.raises(RuntimeError, match="ambiguous.*regular|regular.*ambiguous"):
        collect_flat_replay.recover_incomplete_publication(output_dir)

    assert stage.is_dir()
    assert guarded_final.lstat().st_mode == before_mode
    if entry_kind != "manifest_symlink_to_stage":
        with pytest.raises(FileNotFoundError):
            final_manifest.lstat()


@pytest.mark.parametrize("final_name", ["flat_replay.pt", "flat_replay_manifest.json"])
def test_validate_final_pair_rejects_symlink_before_reading_or_hashing(
    tmp_path,
    monkeypatch,
    final_name,
):
    replay = _full_frozen_replay()
    output_dir = tmp_path / "formal"
    collect_flat_replay.publish_flat_replay(
        replay=replay,
        output_dir=output_dir,
        baseline_manifest_sha256="a" * 64,
        task=STRICT_FLAT_TASK,
        seed=42,
        transition_count=100_000,
        falls=0,
        safety_invalid_rows=0,
    )
    final_path = output_dir / final_name
    original_path = output_dir / f"original-{final_name}"
    final_path.rename(original_path)
    final_path.symlink_to(original_path)
    real_sha256_file = collect_flat_replay.sha256_file

    def reject_symlink_hash(path, *args, **kwargs):
        if Path(path) == final_path:
            raise AssertionError("final symlink was hashed")
        return real_sha256_file(path, *args, **kwargs)

    monkeypatch.setattr(collect_flat_replay, "sha256_file", reject_symlink_hash)
    with pytest.raises(RuntimeError, match="regular non-symlink"):
        collect_flat_replay._validate_final_pair(
            output_dir / "flat_replay.pt",
            output_dir / "flat_replay_manifest.json",
        )


def test_publication_cleans_staging_and_finals_on_ordinary_exception(tmp_path, monkeypatch):
    replay = _full_frozen_replay()
    output_dir = tmp_path / "formal"
    output_dir.mkdir()

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(collect_flat_replay.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="serialization failed"):
        collect_flat_replay.publish_flat_replay(
            replay=replay,
            output_dir=output_dir,
            baseline_manifest_sha256="a" * 64,
            task=STRICT_FLAT_TASK,
            seed=42,
            transition_count=100_000,
            falls=0,
            safety_invalid_rows=0,
        )

    assert list(output_dir.iterdir()) == []


def test_ready_stage_is_durable_in_parent_before_first_final_link(tmp_path, monkeypatch):
    replay = _full_frozen_replay()
    output_dir = tmp_path / "formal"
    synced_directories = []
    real_fsync_directory = collect_flat_replay._fsync_directory
    real_link = collect_flat_replay.os.link

    def track_directory_sync(path):
        synced_directories.append(Path(path))
        real_fsync_directory(path)

    def verify_parent_was_synced(source, destination):
        if Path(destination).name == "flat_replay.pt":
            assert output_dir in synced_directories
        return real_link(source, destination)

    monkeypatch.setattr(collect_flat_replay, "_fsync_directory", track_directory_sync)
    monkeypatch.setattr(collect_flat_replay.os, "link", verify_parent_was_synced)

    collect_flat_replay.publish_flat_replay(
        replay=replay,
        output_dir=output_dir,
        baseline_manifest_sha256="a" * 64,
        task=STRICT_FLAT_TASK,
        seed=42,
        transition_count=100_000,
        falls=0,
        safety_invalid_rows=0,
    )


def test_metrics_stream_fsyncs_each_record_and_is_exclusive(tmp_path, monkeypatch):
    output_dir = tmp_path / "formal"
    fsynced = []
    monkeypatch.setattr(collect_flat_replay.os, "fsync", fsynced.append)
    with collect_flat_replay.open_metrics_stream(output_dir) as stream:
        stream_fd = stream.fileno()
        collect_flat_replay.write_metric(stream, {"step": 1, "falls": 1})

    metrics_path = output_dir / "metrics.jsonl"
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {"step": 1, "falls": 1}
    assert stream_fd in fsynced
    with pytest.raises(FileExistsError, match="metrics.jsonl"):
        collect_flat_replay.open_metrics_stream(output_dir)
