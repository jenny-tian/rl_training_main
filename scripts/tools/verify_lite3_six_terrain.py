"""Verify the released six-terrain checkpoints on CPU, without Isaac Sim."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "models/lite3_six_terrain"


def main():
    torch.set_num_threads(1)
    for line in (PACKAGE / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        with (PACKAGE / relative).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"hash mismatch: {relative}")
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    generator = torch.Generator().manual_seed(20261124)
    observations = torch.randn(256, 45, generator=generator)
    reports = []
    for name, model in manifest["models"].items():
        state = torch.load(PACKAGE / model["path"], map_location="cpu", weights_only=True)
        for component in ("actor_state_dict", "critic_state_dict"):
            if not all(torch.isfinite(value).all() for value in state[component].values()):
                raise ValueError(f"nonfinite {component}: {name}")
        if not state["optimizer_state_dict"]["state"]:
            raise ValueError(f"missing trained optimizer: {name}")
        if state["critic_state_dict"]["mlp.0.weight"].shape != (512, 393):
            raise ValueError(f"unexpected critic input: {name}")
        layers = []
        for index, (inputs, outputs) in enumerate(zip((45, 512, 256, 128), (512, 256, 128, 12))):
            layer = torch.nn.Linear(inputs, outputs)
            layer.load_state_dict({key: state["actor_state_dict"][f"mlp.{index * 2}.{key}"]
                                   for key in ("weight", "bias")})
            layers.append(layer)
            if index < 3:
                layers.append(torch.nn.ELU())
        actor = torch.nn.Sequential(*layers).eval()
        with torch.inference_mode():
            actions = actor(observations)
            single = torch.cat([actor(observation[None]) for observation in observations])
        if actions.shape != (256, 12) or not torch.isfinite(actions).all():
            raise ValueError(f"invalid actor output: {name}")
        torch.testing.assert_close(actions, single, atol=1e-4, rtol=1e-5)
        updates = {"lowstairs10000": 10000, "zero10100": 10100, "B10200": 10200}[name]
        if state["iter"] != updates - 1:
            raise ValueError(f"wrong update count: {name}")
        reports.append(dict(model=name, completed_updates=updates, finite_outputs=True,
                            batch_single_atol=1e-4, batch_single_rtol=1e-5,
                            batch_single_max_error=float((actions - single).abs().max())))
    print(json.dumps(dict(status="passed", checkpoint_tests=reports), indent=2))


if __name__ == "__main__":
    main()
