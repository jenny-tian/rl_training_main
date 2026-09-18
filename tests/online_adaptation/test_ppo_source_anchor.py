import hashlib
import json

import pytest
import torch

from rl_training.online_adaptation.ppo_source_anchor import (
    ASSET_FILENAME,
    MANIFEST_FILENAME,
    SOURCE_ANCHOR_SCHEMA,
    load_source_anchor,
    prepare_source_anchor,
    reconstruct_policy_observations,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def source(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"registered frozen teacher fixture")
    base_manifest = tmp_path / "base_manifest.json"
    write_json(base_manifest, {
        "schema_version": 1, "task": "Flat-Deeprobotics-Lite3-v0", "terrain_assertion": "plane-only",
        "files": {str(checkpoint): {"bytes": checkpoint.stat().st_size, "sha256": sha(checkpoint)}},
    })
    replay = tmp_path / "flat_replay.pt"
    torch.save({
        "schema_version": 1, "frozen": True, "size": 8, "frame_dim": 49, "action_dim": 12,
        "current_frame": torch.arange(8 * 49).reshape(8, 49).half() / 100,
        "safety_mask": torch.ones(8, 1), "terminated": torch.zeros(8, 1), "truncated": torch.zeros(8, 1),
        "residual_action": torch.zeros(8, 12),
        "base_action": torch.full((8, 12), 900.0), "reward": torch.full((8, 1), 777.0),
    }, replay)
    replay_manifest = tmp_path / "flat_replay_manifest.json"
    write_json(replay_manifest, {
        "task": "Flat-Deeprobotics-Lite3-v0", "frozen": True, "falls": 0, "safety_invalid_rows": 0,
        "transition_count": 8, "replay_sha256": sha(replay), "baseline_manifest_sha256": sha(base_manifest),
        "protocol_sha256": "a" * 64,
    })
    return {
        "replay_path": replay, "replay_manifest_path": replay_manifest, "base_manifest_path": base_manifest,
        "base_checkpoint_path": checkpoint, "output_dir": tmp_path / "anchor", "sample_count": 4,
    }


def test_frame_mapping_preserves_joint_order_scales_and_omits_contacts():
    frame = torch.zeros(1, 49)
    frame[:, :3] = torch.tensor([4.0, -8.0, 12.0])
    frame[:, 3:6] = torch.tensor([0.1, 0.2, -0.9])
    frame[:, 6:18] = torch.arange(12)
    frame[:, 18:30] = 20.0
    frame[:, 30:34] = 99.0  # Foot contacts have no slot in the 45D actor input.
    frame[:, 34:46] = -torch.arange(12)
    frame[:, 46:49] = torch.tensor([0.3, 0.0, 0.2])
    before = frame.clone()
    obs = reconstruct_policy_observations(frame)
    expected = torch.cat((
        torch.tensor([1.0, -2.0, 3.0, 0.1, 0.2, -0.9, 0.3, 0.0, 0.2]),
        torch.arange(12), torch.ones(12), -torch.arange(12),
    )).unsqueeze(0)
    torch.testing.assert_close(obs, expected)
    torch.testing.assert_close(frame, before, rtol=0, atol=0)
    frame[:, 0] = 400
    frame[:, 18] = 200
    clipped = reconstruct_policy_observations(frame)
    assert clipped[0, 0].item() == 25.0
    assert clipped[0, 21].item() == 5.0


def test_asset_round_trip_binds_fixed_indices_source_hashes_and_recomputed_teacher_contract(source):
    manifest = prepare_source_anchor(**source)
    directory = source["output_dir"]
    asset = torch.load(directory / ASSET_FILENAME, weights_only=True)
    torch.testing.assert_close(asset["sample_indices"], torch.tensor([0, 2, 4, 7]))
    assert set(asset) == {"schema_version", "schema", "observations", "sample_indices", "metadata"}
    assert "base_action" not in asset and "reward" not in asset
    obs, metadata = load_source_anchor(directory, base_checkpoint_path=source["base_checkpoint_path"])
    assert obs.shape == (4, 45) and obs.dtype == torch.float32 and obs.device.type == "cpu"
    assert torch.isfinite(obs).all()
    assert metadata["schema"] == SOURCE_ANCHOR_SCHEMA
    assert metadata["asset_sha256"] == sha(directory / ASSET_FILENAME)
    assert metadata["manifest_sha256"] == sha(directory / MANIFEST_FILENAME)
    assert metadata["source_files"]["replay"]["sha256"] == sha(source["replay_path"])
    assert metadata["source_files"]["base_checkpoint"]["sha256"] == sha(source["base_checkpoint_path"])
    assert metadata["stored_base_actions_used_as_targets"] is False
    assert "not exact replay" in metadata["reconstruction_caveat"]
    assert manifest["metadata"]["sample_count"] == 4
    obs.zero_()
    again, _ = load_source_anchor(directory / ASSET_FILENAME, base_checkpoint_path=source["base_checkpoint_path"])
    assert torch.count_nonzero(again) > 0
    with pytest.raises(FileExistsError, match="immutable"):
        prepare_source_anchor(**source)


@pytest.mark.parametrize("field", ["replay_path", "base_checkpoint_path", "base_manifest_path"])
def test_preparation_rejects_modified_source_or_provenance(source, field):
    with source[field].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        prepare_source_anchor(**source)
    assert not source["output_dir"].exists()


@pytest.mark.parametrize("change", ["nonfinite", "not_frozen", "unsafe"])
def test_preparation_validates_replay_contents_after_matching_file_hash(source, change):
    data = torch.load(source["replay_path"], weights_only=True)
    if change == "nonfinite":
        data["current_frame"][0, 0] = float("nan")
    elif change == "not_frozen":
        data["frozen"] = False
    else:
        data["safety_mask"][0] = 0
    torch.save(data, source["replay_path"])
    manifest = json.loads(source["replay_manifest_path"].read_text())
    manifest["replay_sha256"] = sha(source["replay_path"])
    write_json(source["replay_manifest_path"], manifest)
    with pytest.raises(ValueError):
        prepare_source_anchor(**source)


def test_loader_rejects_a_different_teacher_checkpoint(source, tmp_path):
    prepare_source_anchor(**source)
    wrong_teacher = tmp_path / "other_model.pt"
    wrong_teacher.write_bytes(b"different teacher")
    with pytest.raises(ValueError, match="base checkpoint SHA256"):
        load_source_anchor(source["output_dir"], base_checkpoint_path=wrong_teacher)


@pytest.mark.parametrize("change", ["asset_bytes", "schema", "nonfinite", "wrong_shape", "wrong_indices", "reward_payload"])
def test_loader_rejects_corrupt_or_incompatible_assets(source, change):
    prepare_source_anchor(**source)
    directory = source["output_dir"]
    asset_path, manifest_path = directory / ASSET_FILENAME, directory / MANIFEST_FILENAME
    if change == "asset_bytes":
        with asset_path.open("ab") as stream:
            stream.write(b"changed")
    else:
        payload = torch.load(asset_path, weights_only=True)
        if change == "schema":
            payload["schema"] = "not_reconstructed_policy45"
        elif change == "nonfinite":
            payload["observations"][0, 0] = float("nan")
        elif change == "wrong_shape":
            payload["observations"] = torch.zeros(4, 48)
        elif change == "wrong_indices":
            payload["sample_indices"][1] = 3
        else:
            payload["reward"] = torch.ones(4, 1)
        torch.save(payload, asset_path)
        # Re-seal the outer file hash to exercise the inner schema/tensor
        # validation, rather than having every case fail at the same SHA check.
        manifest = json.loads(manifest_path.read_text())
        manifest["asset_sha256"] = sha(asset_path)
        write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        load_source_anchor(directory, base_checkpoint_path=source["base_checkpoint_path"])
