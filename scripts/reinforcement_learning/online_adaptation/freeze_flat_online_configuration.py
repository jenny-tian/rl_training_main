from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def freeze_configuration(
    *,
    selected_configuration: str | Path,
    base_manifest: str | Path,
    flat_replay_manifest: str | Path,
    experiment_checkpoint: str | Path,
    output: str | Path,
    git_commit: str | None = None,
) -> dict[str, Any]:
    paths = {
        "selected_configuration": Path(selected_configuration).resolve(),
        "base_manifest": Path(base_manifest).resolve(),
        "flat_replay_manifest": Path(flat_replay_manifest).resolve(),
        "experiment_checkpoint": Path(experiment_checkpoint).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("every formal configuration input must be a file")
    target = Path(output)
    if target.exists():
        raise FileExistsError("formal configuration already exists")
    selected = _object(paths["selected_configuration"])
    protocol = default_flat_online_protocol()
    if set(protocol.development_seeds) & set(protocol.sealed_seeds):
        raise ValueError("development and sealed seeds overlap")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    expected_hashes = selected.get("formal_input_hashes")
    if expected_hashes is not None and expected_hashes != {
        name: hashes[name] for name in ("base_manifest", "flat_replay_manifest", "experiment_checkpoint")
    }:
        raise ValueError("formal inputs changed after development selection")
    if selected.get("protocol_sha256") != protocol.sha256():
        raise ValueError("selected configuration protocol hash mismatch")
    if git_commit is None:
        git_commit = subprocess.run(
            ("git", "rev-parse", "HEAD"), check=True, text=True, capture_output=True
        ).stdout.strip()
    payload = {
        "schema_version": 1,
        "selected_configuration": selected,
        "protocol": asdict(protocol),
        "protocol_sha256": protocol.sha256(),
        "git_commit": git_commit,
        "inputs": {
            name: {"path": str(paths[name]), "sha256": digest}
            for name, digest in hashes.items()
        },
    }
    write_json_atomic(target, payload)
    normalized_payload = json.loads(json.dumps(payload, allow_nan=False))
    if _object(target) != normalized_payload:
        raise RuntimeError("formal configuration read-back verification failed")
    os.chmod(target, 0o444)
    return payload


def verify_formal_configuration(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.stat().st_mode & 0o222:
        raise PermissionError("formal configuration must be read-only")
    payload = _object(target)
    if payload.get("protocol_sha256") != default_flat_online_protocol().sha256():
        raise ValueError("formal configuration protocol hash mismatch")
    expected_protocol = json.loads(
        json.dumps(asdict(default_flat_online_protocol()), allow_nan=False)
    )
    if payload.get("protocol") != expected_protocol:
        raise ValueError("formal configuration canonical protocol mismatch")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("formal configuration inputs are missing")
    for record in inputs.values():
        if not isinstance(record, dict) or sha256_file(record["path"]) != record.get("sha256"):
            raise ValueError("formal configuration input hash mismatch")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--selected-configuration", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--flat-replay-manifest", required=True)
    parser.add_argument("--experiment-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    freeze_configuration(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
