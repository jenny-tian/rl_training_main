from __future__ import annotations

import hashlib
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


_SCHEMA_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda_device_count": cuda_device_count,
        "torch_cuda": (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if cuda_device_count
            else ()
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    expected_keys = {
        "python",
        "numpy",
        "torch_cpu",
        "cuda_device_count",
        "torch_cuda",
    }
    if not isinstance(state, Mapping) or set(state) != expected_keys:
        raise ValueError("RNG state has an unsupported schema")
    saved_cuda_count = state["cuda_device_count"]
    current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if saved_cuda_count != current_cuda_count:
        raise RuntimeError(
            "CUDA device count differs from checkpoint: "
            f"saved={saved_cuda_count}, current={current_cuda_count}"
        )
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (tuple, list)) or len(cuda_states) != current_cuda_count:
        raise ValueError("checkpoint CUDA RNG state count is inconsistent")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].to(device="cpu"))
    if current_cuda_count:
        normalized_cuda_states = []
        for cuda_state in cuda_states:
            try:
                normalized = cuda_state.detach().to(device="cpu")
            except (AttributeError, RuntimeError, TypeError) as error:
                raise ValueError("checkpoint CUDA RNG state is not a tensor") from error
            if not isinstance(normalized, torch.Tensor) or normalized.dtype != torch.uint8:
                raise ValueError("checkpoint CUDA RNG state must be a torch.uint8 tensor")
            normalized_cuda_states.append(normalized)
        torch.cuda.set_rng_state_all(normalized_cuda_states)


def save_experiment_checkpoint(path: str | Path, state: Mapping[str, Any]) -> None:
    target = Path(path)
    if not isinstance(state, Mapping):
        raise TypeError("experiment state must be a mapping")
    protocol_sha256 = state.get("protocol_sha256")
    if not isinstance(protocol_sha256, str) or not protocol_sha256:
        raise ValueError("experiment state requires a non-empty protocol_sha256")
    target.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": _SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "experiment_state": dict(state),
        "rng_state": capture_rng_state(),
    }
    checkpoint_temp: Path | None = None
    checksum_temp: Path | None = None
    checkpoint_backup: Path | None = None
    checksum_backup: Path | None = None
    checkpoint_installed = False
    checksum_installed = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            checkpoint_temp = Path(stream.name)
            torch.save(envelope, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256_file(checkpoint_temp)
        checksum_path = _checksum_path(target)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=target.parent,
            prefix=f".{checksum_path.name}.", delete=False
        ) as stream:
            checksum_temp = Path(stream.name)
            stream.write(digest + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            checkpoint_backup = _copy_to_temporary(target)
        if checksum_path.exists():
            checksum_backup = _copy_to_temporary(checksum_path)
        os.replace(checkpoint_temp, target)
        checkpoint_temp = None
        checkpoint_installed = True
        os.replace(checksum_temp, checksum_path)
        checksum_temp = None
        checksum_installed = True
        _fsync_directory(target.parent)
    except BaseException:
        if checkpoint_installed:
            if checkpoint_backup is not None:
                os.replace(checkpoint_backup, target)
                checkpoint_backup = None
            else:
                target.unlink(missing_ok=True)
        if checksum_installed:
            if checksum_backup is not None:
                os.replace(checksum_backup, checksum_path)
                checksum_backup = None
            else:
                checksum_path.unlink(missing_ok=True)
        _fsync_directory(target.parent)
        raise
    finally:
        for temporary in (
            checkpoint_temp, checksum_temp, checkpoint_backup, checksum_backup
        ):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def load_experiment_checkpoint(
    path: str | Path,
    *,
    expected_protocol_sha256: str,
    map_location: str | torch.device | None = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    target = Path(path)
    expected_checksum = _checksum_path(target).read_text(encoding="ascii").strip()
    actual_checksum = _sha256_file(target)
    if not expected_checksum or actual_checksum != expected_checksum:
        raise ValueError("experiment checkpoint checksum mismatch")
    envelope = torch.load(target, map_location=map_location, weights_only=False)
    if not isinstance(envelope, Mapping) or envelope.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("experiment checkpoint schema version mismatch")
    if envelope.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError("experiment checkpoint protocol hash mismatch")
    experiment_state = envelope.get("experiment_state")
    if not isinstance(experiment_state, Mapping):
        raise ValueError("experiment checkpoint state is invalid")
    if restore_rng:
        restore_rng_state(envelope.get("rng_state"))
    return dict(experiment_state)


def _checksum_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_to_temporary(source: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=source.parent, prefix=f".{source.name}.backup."
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
