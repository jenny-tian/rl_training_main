#!/usr/bin/env python3
"""Compare a Lite3 RSL-RL checkpoint with its TorchScript and ONNX exports.

This is an offline CPU-only audit. It never starts Isaac Sim, runs an
environment, or modifies the checkpoint/export files. The ONNX export used by
the current Lite3 runner has a fixed batch size of one, so inputs are evaluated
one at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _build_actor(state: dict[str, Any]) -> tuple[nn.Sequential, list[int]]:
    layer_indices = sorted(
        int(key.split(".")[1])
        for key in state
        if key.startswith("mlp.") and key.endswith(".weight")
    )
    if not layer_indices or layer_indices != list(range(0, layer_indices[-1] + 1, 2)):
        raise ValueError("checkpoint actor must contain sequential mlp layers at even indices")

    layers: list[nn.Module] = []
    dimensions: list[int] = []
    for position, layer_index in enumerate(layer_indices):
        weight = state[f"mlp.{layer_index}.weight"]
        bias = state[f"mlp.{layer_index}.bias"]
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(f"invalid actor parameter shapes at mlp.{layer_index}")
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if position + 1 < len(layer_indices):
            layers.append(nn.ELU())
        if not dimensions:
            dimensions.append(int(weight.shape[1]))
        dimensions.append(int(weight.shape[0]))

    actor = nn.Sequential(*layers).cpu().eval()
    return actor, dimensions


def _sample_inputs(observation_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.rand((256, observation_dim), generator=generator) * 2.0 - 1.0
    identity = torch.eye(observation_dim)
    return torch.cat(
        (
            torch.zeros((1, observation_dim)),
            torch.ones((1, observation_dim)),
            -torch.ones((1, observation_dim)),
            torch.full((1, observation_dim), 0.5),
            torch.full((1, observation_dim), -0.5),
            random,
            identity,
            -identity,
        ),
        dim=0,
    ).contiguous()


def _error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.abs(reference - candidate)
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()),
    }


def audit(
    checkpoint_path: Path,
    torchscript_path: Path,
    onnx_path: Path,
    *,
    context_files: tuple[Path, ...] = (),
    seed: int = 1998,
    atol: float = 1e-5,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    torchscript_path = torchscript_path.resolve()
    onnx_path = onnx_path.resolve()
    for path in (checkpoint_path, torchscript_path, onnx_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    context_files = tuple(path.resolve() for path in context_files)
    for path in context_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "actor_state_dict" not in checkpoint:
        raise ValueError("expected an RSL-RL checkpoint with actor_state_dict")
    actor_state = checkpoint["actor_state_dict"]
    actor, dimensions = _build_actor(actor_state)
    if "distribution.log_std_param" not in actor_state:
        raise ValueError("checkpoint has no Gaussian actor standard deviation parameters")
    log_std_shape = list(actor_state["distribution.log_std_param"].shape)
    if log_std_shape != [dimensions[-1]]:
        raise ValueError("actor output width does not match distribution parameter width")

    torchscript = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_inputs = session.get_inputs()
    onnx_outputs = session.get_outputs()
    expected_input_shape = [1, dimensions[0]]
    expected_output_shape = [1, dimensions[-1]]
    if len(onnx_inputs) != 1 or onnx_inputs[0].shape != expected_input_shape:
        raise ValueError(f"unexpected ONNX input signature: {[item.shape for item in onnx_inputs]}")
    if len(onnx_outputs) != 1 or onnx_outputs[0].shape != expected_output_shape:
        raise ValueError(f"unexpected ONNX output signature: {[item.shape for item in onnx_outputs]}")

    observations = _sample_inputs(dimensions[0], seed)
    with torch.inference_mode():
        expected = actor(observations).cpu().numpy()
        actual_torchscript = torchscript(observations).cpu().numpy()
        actual_onnx = np.concatenate(
            [
                session.run(
                    [onnx_outputs[0].name],
                    {onnx_inputs[0].name: row[None, :].numpy().astype(np.float32)},
                )[0]
                for row in observations
            ],
            axis=0,
        )

    for name, output in (
        ("checkpoint_actor", expected),
        ("torchscript", actual_torchscript),
        ("onnx", actual_onnx),
    ):
        if output.shape != expected.shape or not np.isfinite(output).all():
            raise ValueError(f"{name} output has an invalid shape or non-finite values")

    torchscript_error = _error_metrics(expected, actual_torchscript)
    onnx_error = _error_metrics(expected, actual_onnx)
    torchscript_passed = torchscript_error["max_abs"] <= atol
    onnx_passed = onnx_error["max_abs"] <= atol
    report = {
        "schema_version": 1,
        "audit": "lite3_checkpoint_export_parity_cpu",
        "checkpoint_iteration": checkpoint.get("iter"),
        "device": "cpu",
        "seed": seed,
        "atol": atol,
        "inputs": {
            "count": int(observations.shape[0]),
            "shape": list(observations.shape),
            "construction": "zeros, +/- ones, +/- 0.5, 256 uniform[-1,1], +/- identity basis",
            "sha256_float32_c_order": hashlib.sha256(
                observations.numpy().astype("<f4", copy=False).tobytes(order="C")
            ).hexdigest(),
        },
        "actor": {
            "layer_dimensions": dimensions,
            "activation": "ELU between linear layers",
            "distribution_log_std_shape": log_std_shape,
            "output_shape": list(expected.shape),
            "output_min": float(expected.min()),
            "output_max": float(expected.max()),
        },
        "exports": {
            "torchscript": {
                "error_vs_checkpoint": torchscript_error,
                "passed": torchscript_passed,
            },
            "onnx": {
                "input_name": onnx_inputs[0].name,
                "input_shape": onnx_inputs[0].shape,
                "output_name": onnx_outputs[0].name,
                "output_shape": onnx_outputs[0].shape,
                "providers": session.get_providers(),
                "error_vs_checkpoint": onnx_error,
                "error_vs_torchscript": _error_metrics(actual_torchscript, actual_onnx),
                "passed": onnx_passed,
            },
        },
        "files": {
            "checkpoint": {
                "path": _display_path(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            },
            "torchscript": {
                "path": _display_path(torchscript_path),
                "bytes": torchscript_path.stat().st_size,
                "sha256": _sha256(torchscript_path),
            },
            "onnx": {
                "path": _display_path(onnx_path),
                "bytes": onnx_path.stat().st_size,
                "sha256": _sha256(onnx_path),
            },
            "context": [
                {
                    "path": _display_path(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in context_files
            ],
        },
        "passed": torchscript_passed and onnx_passed,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--context-file",
        action="append",
        type=Path,
        default=[],
        help="Add an environment/config/source file hash to the report (repeatable)",
    )
    parser.add_argument("--report", type=Path, help="Write the JSON report to a new file")
    args = parser.parse_args()
    if args.atol <= 0:
        parser.error("--atol must be positive")

    report = audit(
        args.checkpoint,
        args.torchscript,
        args.onnx,
        context_files=tuple(args.context_file),
        seed=args.seed,
        atol=args.atol,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(serialized, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
