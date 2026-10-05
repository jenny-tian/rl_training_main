"""Count recovery coverage using training replay and reference traces only."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.run)
    checkpoint = root / "round-01/adapter.pt"
    replay = torch.load(checkpoint, map_location="cpu", weights_only=False)["buffer"]
    thresholds = (10, 15, 20, 30)
    result = dict(new_physics=0, source_sha256=sha256_file(checkpoint),
                  scope="Training data only; no evaluation observations or fitted weights.", buffers={}, references=[])
    for role, rows in replay.items():
        obs = torch.cat([o for o, _ in rows])
        result["buffers"][role] = dict(rows=len(obs), tilted_rows={
            str(t): int((-obs[:, 5] < math.cos(math.radians(t))).sum()) for t in thresholds})
    for folder in sorted(root.glob("reference-*")):
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text())
        success = np.array([e["success"] for e in report["episodes"]])
        counts = {str(t): 0 for t in thresholds}
        episodes = {str(t): set() for t in thresholds}
        hashes = {}
        for file in sorted((folder / "trajectory").glob("chunk-*.npz")):
            hashes[file.name] = sha256_file(file)
            with np.load(file, allow_pickle=False) as data:
                valid = data["active"] & success[None, :]
                for t in thresholds:
                    tilted = valid & (-data["policy_observation"][..., 5] < np.float32(math.cos(math.radians(t))))
                    counts[str(t)] += int(tilted.sum())
                    episodes[str(t)].update(np.where(tilted)[1].tolist())
        result["references"].append(dict(phase=folder.name, report_sha256=sha256_file(report_path),
            successful_tilted_rows=counts, successful_tilted_episodes={k: len(v) for k, v in episodes.items()},
            trace_sha256=hashes))
    write_json_atomic(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != "references"}, indent=2))
    for reference in result["references"]:
        print(reference["phase"], reference["successful_tilted_rows"], reference["successful_tilted_episodes"])


if __name__ == "__main__":
    main()
