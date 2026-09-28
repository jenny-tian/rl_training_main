"""Chunked diagnostic trajectories, independent of training replay/checkpoints."""

from pathlib import Path

import numpy as np
import torch

from .provenance import write_json_atomic


class StepTrajectory:
    """Persist detached transitions; preserve reward/pose precision for audits.

    Observations and actions use float16; scalar rewards, poses and timings
    retain their original precision. A manifest is updated after each chunk,
    and an interrupted run is distinguishable from a completed trajectory.
    """

    def __init__(self, directory, *, metadata, chunk_steps=256):
        if type(chunk_steps) is not int or chunk_steps < 1:
            raise ValueError("chunk_steps must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.chunk_steps = chunk_steps
        self.pending = []
        self.layout = None
        self.closed = False
        self.manifest = {
            "schema": "lite3_step_trajectory_v1", "status": "recording",
            "metadata": metadata, "chunks": [], "steps": 0, "bytes": 0,
            "pose_semantics": "post_action_pre_reset",
            "observation_semantics": "pre_action",
        }
        self._write_manifest()

    def _write_manifest(self):
        write_json_atomic(self.directory / "manifest.json", self.manifest)

    @staticmethod
    def snapshot(**fields):
        """Copy inputs before env.step can mutate the simulator's tensor views."""
        result = {}
        for key, value in fields.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.dtype.kind not in "biuf":
                raise ValueError(f"{key} must be numeric")
            compact = "observation" in key or "action" in key or key == "history"
            with np.errstate(over="ignore"):
                array = array.astype(np.float16 if compact else array.dtype, copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"{key} is non-finite or overflows storage dtype")
            result[key] = array
        return result

    def append(self, **fields):
        if self.closed:
            raise RuntimeError("trajectory is closed")
        row = self.snapshot(**fields)
        layout = {key: (value.shape, str(value.dtype)) for key, value in row.items()}
        if self.layout is None:
            self.layout = layout
            self.manifest["fields"] = {
                key: {"shape": list(shape), "dtype": dtype}
                for key, (shape, dtype) in layout.items()
            }
        elif layout != self.layout:
            raise ValueError("trajectory fields, shapes and dtypes must remain constant")
        self.pending.append(row)
        if len(self.pending) >= self.chunk_steps:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        filename = f"chunk-{len(self.manifest['chunks']):05d}.npz"
        path = self.directory / filename
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **{
                key: np.stack([row[key] for row in self.pending]) for key in self.pending[0]
            })
        temporary.replace(path)
        size = path.stat().st_size
        self.manifest["chunks"].append({"file": filename, "steps": len(self.pending), "bytes": size})
        self.manifest["steps"] += len(self.pending)
        self.manifest["bytes"] += size
        self.pending.clear()
        self._write_manifest()

    def close(self, *, complete=False):
        if not self.closed:
            self.flush()
            self.manifest["status"] = "complete" if complete else "interrupted"
            self._write_manifest()
            self.closed = True
        return dict(self.manifest)
