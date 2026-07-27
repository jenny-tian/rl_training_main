from __future__ import annotations

import json
import torch
import traceback
from pathlib import Path


def as_torch_device(device: str | torch.device) -> torch.device:
    """Normalize device values returned by Isaac Lab before inspecting their properties."""
    return torch.device(device)


def write_failure_record(run_dir: str | Path, error: Exception) -> Path:
    """Persist an exception before Isaac Sim shutdown can obscure it."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "failure.json"
    record = {
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
