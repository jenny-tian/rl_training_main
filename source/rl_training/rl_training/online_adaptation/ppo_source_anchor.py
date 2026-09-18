"""Verified, reconstructed flat observations for a separate PPO teacher anchor.

The legacy replay stores raw 49D proprioceptive frames, not the original
noisy 45D policy observations. Only reconstructed inputs are exported. A
caller must evaluate its frozen teacher on these same inputs; stored replay
actions, rewards, and transitions are never supplied to PPO by this module.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


SOURCE_ANCHOR_SCHEMA = "reconstructed_clean_flat_policy45_v1"
ASSET_FILENAME = "source_anchor.pt"
MANIFEST_FILENAME = "source_anchor_manifest.json"
INDEX_STRATEGY = "torch.linspace(0, source_frame_count - 1, sample_count, dtype=torch.float32).to(torch.int64)"
RECONSTRUCTION_CAVEAT = (
    "The source contains float16 raw 49D frames and does not retain the original actor observation noise. "
    "These are reconstructed clean flat 45D policy inputs, not exact replay of the original noisy observations. "
    "Recompute frozen teacher means on these same inputs; do not pair them with the stored base_action."
)
OBSERVATION_MAPPING = [
    {"field": "base_angular_velocity", "source_slice": [0, 3], "output_slice": [0, 3], "clip": [-100.0, 100.0], "scale": 0.25},
    {"field": "projected_gravity", "source_slice": [3, 6], "output_slice": [3, 6], "clip": [-100.0, 100.0], "scale": 1.0},
    {"field": "velocity_commands", "source_slice": [46, 49], "output_slice": [6, 9], "clip": [-100.0, 100.0], "scale": 1.0},
    {"field": "relative_joint_position", "source_slice": [6, 18], "output_slice": [9, 21], "clip": [-100.0, 100.0], "scale": 1.0},
    {"field": "joint_velocity", "source_slice": [18, 30], "output_slice": [21, 33], "clip": [-100.0, 100.0], "scale": 0.05},
    {"field": "previous_final_action", "source_slice": [34, 46], "output_slice": [33, 45], "clip": [-100.0, 100.0], "scale": 1.0},
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _tensor_sha256(value: Tensor) -> str:
    dtype = {torch.float32: "<f4", torch.int64: "<i8"}.get(value.dtype)
    if dtype is None:
        raise ValueError("tensor hashes require float32 or int64")
    array = value.detach().cpu().contiguous().numpy().astype(dtype, copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _uniform_indices(source_count: int, sample_count: int) -> Tensor:
    _positive_integer(source_count, "source_frame_count", 2)
    _positive_integer(sample_count, "sample_count", 2)
    if sample_count > source_count:
        raise ValueError("sample_count must not exceed source_frame_count")
    indices = torch.linspace(0, source_count - 1, sample_count, dtype=torch.float32, device="cpu").long()
    if not bool((indices[1:] > indices[:-1]).all().item()):
        raise ValueError("uniform sampling must select distinct source frames")
    return indices


def reconstruct_policy_observations(frames49: Tensor) -> Tensor:
    """Map raw leg-major 49D frames to clean 45D actor inputs on the CPU."""
    if not isinstance(frames49, Tensor) or frames49.ndim != 2 or frames49.shape[1] != 49:
        raise ValueError("source frames must have shape [N, 49]")
    if not frames49.is_floating_point() or not bool(torch.isfinite(frames49).all().item()):
        raise ValueError("source frames must contain finite real floating-point values")
    frames = frames49.detach().to(device="cpu", dtype=torch.float32)
    terms = []
    for field in OBSERVATION_MAPPING:
        start, stop = field["source_slice"]
        terms.append(frames[:, start:stop].clamp(*field["clip"]) * field["scale"])
    observations = torch.cat(terms, dim=-1).contiguous()
    if not bool(torch.isfinite(observations).all().item()):
        raise ValueError("reconstructed policy observations must be finite")
    return observations


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def prepare_source_anchor(
    *,
    replay_path: str | Path,
    replay_manifest_path: str | Path,
    base_manifest_path: str | Path,
    base_checkpoint_path: str | Path,
    output_dir: str | Path,
    sample_count: int = 4096,
) -> dict[str, Any]:
    """Verify source provenance and write a new immutable observation asset.

    No existing asset or manifest is overwritten. The manifest is published
    after the asset has been completely written, so the loader cannot accept
    a partial preparation.
    """
    replay_path, replay_manifest_path, base_manifest_path, base_checkpoint_path, output_dir = (
        Path(path).resolve() for path in (
            replay_path, replay_manifest_path, base_manifest_path, base_checkpoint_path, output_dir,
        )
    )
    asset_path, manifest_path = output_dir / ASSET_FILENAME, output_dir / MANIFEST_FILENAME
    if asset_path.exists() or manifest_path.exists():
        raise FileExistsError("source anchor assets are immutable; use a new output directory")
    replay_manifest = _read_json(replay_manifest_path)
    base_manifest = _read_json(base_manifest_path)
    source_files = {
        "replay": _file_record(replay_path),
        "replay_manifest": _file_record(replay_manifest_path),
        "base_manifest": _file_record(base_manifest_path),
        "base_checkpoint": _file_record(base_checkpoint_path),
    }
    if source_files["replay"]["sha256"] != replay_manifest.get("replay_sha256"):
        raise ValueError("source replay SHA256 does not match its manifest")
    if source_files["base_manifest"]["sha256"] != replay_manifest.get("baseline_manifest_sha256"):
        raise ValueError("baseline manifest SHA256 does not match the replay manifest")
    if (
        replay_manifest.get("task") != "Flat-Deeprobotics-Lite3-v0"
        or replay_manifest.get("frozen") is not True
        or replay_manifest.get("falls") != 0
        or replay_manifest.get("safety_invalid_rows") != 0
        or base_manifest.get("schema_version") != 1
        or base_manifest.get("task") != "Flat-Deeprobotics-Lite3-v0"
        or base_manifest.get("terrain_assertion") != "plane-only"
    ):
        raise ValueError("source manifests must describe a frozen, safe, plane-only Lite3 replay")
    declared_files = base_manifest.get("files")
    if not isinstance(declared_files, dict) or str(base_checkpoint_path) not in declared_files:
        raise ValueError("base checkpoint is not registered in the baseline manifest")
    verified_baseline_files = []
    for name, expected in declared_files.items():
        actual = _file_record(Path(name))
        if not isinstance(expected, dict) or actual["sha256"] != expected.get("sha256") or actual["bytes"] != expected.get("bytes"):
            raise ValueError(f"baseline source file does not match its manifest: {name}")
        verified_baseline_files.append(actual)

    replay = torch.load(replay_path, map_location="cpu", weights_only=True)
    if not isinstance(replay, dict) or replay.get("schema_version") != 1 or replay.get("frozen") is not True:
        raise ValueError("source replay must be a frozen schema-1 replay dictionary")
    source_count = _positive_integer(replay.get("size"), "source_frame_count", 2)
    if replay.get("frame_dim") != 49 or replay.get("action_dim") != 12 or replay_manifest.get("transition_count") != source_count:
        raise ValueError("source replay dimensions/count do not match the declared 49D/12D schema")
    frames = replay.get("current_frame")
    if not isinstance(frames, Tensor) or frames.ndim != 2 or frames.shape[1] != 49 or frames.shape[0] < source_count:
        raise ValueError("source current_frame must contain all declared [N, 49] rows")
    frames = frames[:source_count]
    if not frames.is_floating_point() or not bool(torch.isfinite(frames).all().item()):
        raise ValueError("source frames must contain finite floating-point values")
    for name, width, expected in (("safety_mask", 1, 1), ("terminated", 1, 0), ("truncated", 1, 0), ("residual_action", 12, 0)):
        value = replay.get(name)
        if (
            not isinstance(value, Tensor) or value.ndim != 2
            or value.shape[0] < source_count or value.shape[1] != width
            or not bool((value[:source_count] == expected).all().item())
        ):
            raise ValueError(f"source replay {name} does not satisfy the frozen flat-source contract")
    indices = _uniform_indices(source_count, sample_count)
    selected_frames = frames[indices].float()
    observations = reconstruct_policy_observations(selected_frames)
    metadata = {
        "source_files": source_files,
        "verified_baseline_files": verified_baseline_files,
        "source_task": replay_manifest["task"],
        "source_protocol_sha256": replay_manifest.get("protocol_sha256"),
        "source_frame_count": source_count,
        "source_frame_dim": 49,
        "source_frame_storage_dtype": str(frames.dtype).removeprefix("torch."),
        "source_selected_frames_float32_sha256": _tensor_sha256(selected_frames),
        "sample_count": sample_count,
        "index_strategy": INDEX_STRATEGY,
        "observation_mapping": copy.deepcopy(OBSERVATION_MAPPING),
        "dropped_source_fields": {"foot_contact": [30, 34]},
        "default_joint_velocity_assumption": 0.0,
        "observation_noise_added": False,
        "stored_base_actions_used_as_targets": False,
        "teacher_target_contract": "Recompute frozen base-checkpoint means on these exact reconstructed observations at runtime.",
        "reconstruction_caveat": RECONSTRUCTION_CAVEAT,
        "tensor_hash_encoding": "SHA256 of contiguous CPU little-endian C-order bytes",
        "tensor_schema": {
            "observations": {"dtype": "float32", "shape": [sample_count, 45], "sha256": _tensor_sha256(observations)},
            "sample_indices": {"dtype": "int64", "shape": [sample_count], "sha256": _tensor_sha256(indices)},
        },
    }
    payload = {
        "schema_version": 1, "schema": SOURCE_ANCHOR_SCHEMA,
        "observations": observations, "sample_indices": indices, "metadata": metadata,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    asset_bytes = buffer.getvalue()
    manifest = {
        "schema_version": 1, "schema": SOURCE_ANCHOR_SCHEMA,
        "asset_file": ASSET_FILENAME, "asset_sha256": hashlib.sha256(asset_bytes).hexdigest(),
        "metadata": metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with asset_path.open("xb") as stream:
        stream.write(asset_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return copy.deepcopy(manifest)


def load_source_anchor(
    asset_path_or_dir: str | Path,
    *,
    base_checkpoint_path: str | Path,
) -> tuple[Tensor, dict[str, Any]]:
    """Read only verified actor inputs and provenance, never PPO transitions."""
    path = Path(asset_path_or_dir).resolve()
    asset_path = path / ASSET_FILENAME if path.is_dir() else path
    if asset_path.name != ASSET_FILENAME:
        raise ValueError(f"source anchor file must be named {ASSET_FILENAME}")
    manifest_path = asset_path.parent / MANIFEST_FILENAME
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1 or manifest.get("schema") != SOURCE_ANCHOR_SCHEMA
        or manifest.get("asset_file") != ASSET_FILENAME
    ):
        raise ValueError("incompatible source anchor manifest schema")
    asset_sha = _sha256(asset_path)
    if manifest.get("asset_sha256") != asset_sha:
        raise ValueError("source anchor asset SHA256 does not match its manifest")
    payload = torch.load(asset_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "schema", "observations", "sample_indices", "metadata"}
        or payload.get("schema_version") != 1 or payload.get("schema") != SOURCE_ANCHOR_SCHEMA
    ):
        raise ValueError("incompatible source anchor tensor schema")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata != manifest.get("metadata"):
        raise ValueError("source anchor metadata does not match its manifest")
    if (
        metadata.get("source_frame_dim") != 49
        or metadata.get("source_task") != "Flat-Deeprobotics-Lite3-v0"
        or metadata.get("index_strategy") != INDEX_STRATEGY
        or metadata.get("observation_mapping") != OBSERVATION_MAPPING
        or metadata.get("observation_noise_added") is not False
        or metadata.get("stored_base_actions_used_as_targets") is not False
        or metadata.get("reconstruction_caveat") != RECONSTRUCTION_CAVEAT
    ):
        raise ValueError("source anchor reconstruction contract is incompatible")
    source_files = metadata.get("source_files")
    if not isinstance(source_files, dict) or not isinstance(source_files.get("base_checkpoint"), dict):
        raise ValueError("source anchor does not declare its teacher checkpoint")
    if _sha256(Path(base_checkpoint_path)) != source_files["base_checkpoint"].get("sha256"):
        raise ValueError("base checkpoint SHA256 does not match the source anchor teacher")
    expected_indices = _uniform_indices(metadata.get("source_frame_count"), metadata.get("sample_count"))
    observations, indices = payload["observations"], payload["sample_indices"]
    sample_count = metadata["sample_count"]
    if (
        not isinstance(observations, Tensor) or observations.dtype != torch.float32
        or observations.shape != (sample_count, 45)
        or not bool(torch.isfinite(observations).all().item())
    ):
        raise ValueError("source anchor observations must be finite float32 [N, 45]")
    if not isinstance(indices, Tensor) or indices.dtype != torch.int64 or not torch.equal(indices, expected_indices):
        raise ValueError("source anchor sample indices do not match the fixed uniform selection")
    tensor_schema = metadata.get("tensor_schema")
    expected_schema = {
        "observations": {"dtype": "float32", "shape": [sample_count, 45], "sha256": _tensor_sha256(observations)},
        "sample_indices": {"dtype": "int64", "shape": [sample_count], "sha256": _tensor_sha256(indices)},
    }
    if tensor_schema != expected_schema:
        raise ValueError("source anchor tensor shape/dtype/SHA256 metadata is inconsistent")
    returned_metadata = copy.deepcopy(metadata)
    returned_metadata.update({
        "schema": SOURCE_ANCHOR_SCHEMA, "asset_path": str(asset_path),
        "asset_sha256": asset_sha, "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    })
    return observations.detach().clone(), returned_metadata


__all__ = [
    "SOURCE_ANCHOR_SCHEMA", "ASSET_FILENAME", "MANIFEST_FILENAME",
    "reconstruct_policy_observations", "prepare_source_anchor", "load_source_anchor",
]
