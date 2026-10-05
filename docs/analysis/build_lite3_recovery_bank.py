"""Extract kinematic reset states from old TRAINING failures, not action labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows, provenance = [], {}
    for run in ("pilot-v11", "pilot-v12"):
        for phase in ("reference-00", "reference-01", "round-00", "round-01"):
            folder = Path("artifacts/flowdagger_20261005") / run / phase
            report = json.loads((folder / "report.json").read_text())
            assert report["status"] == "complete"
            eligible = {e["env_id"]: e for e in report["episodes"] if e["failure"]}
            initial = np.load(folder / "initial_state.npz", allow_pickle=False)
            seen, offset = set(), 0
            provenance[str(folder / "report.json")] = sha256_file(folder / "report.json")
            for file in sorted((folder / "trajectory").glob("chunk-*.npz")):
                with np.load(file, allow_pickle=False) as data:
                    chunk_length = len(data["active"])
                    for i, episode in eligible.items():
                        if i in seen:
                            continue
                        obs = data["policy_observation"][:, i]
                        tilt = np.degrees(np.arccos(np.clip(-obs[:, 5], -1, 1)))
                        valid = data["active"][:, i] & (tilt >= 20) & (tilt <= 30)
                        indices = np.flatnonzero(valid)
                        if not len(indices):
                            continue
                        j = int(indices[0]); seen.add(i)
                        root = data["policy_root_state"][j, i].copy()
                        root[:3] -= initial["origins"][i]
                        rows.append(dict(id=f"{run}/{phase}/{i}", terrain=int(episode["terrain"]),
                            step=offset+j, tilt_deg=float(tilt[j]), root=root,
                            joint_pos=obs[j, 12:24] + np.array([0., -.65, 1.3]*4, np.float32),
                            joint_vel=obs[j, 24:36]/np.float32(.05), source=str(file), sha256=sha256_file(file)))
                offset += chunk_length
    # One state per source episode. Hold out every fourth episode within each
    # terrain; no adjacent states from the same episode can cross the split.
    counts = {101: 0, 107: 0}
    for row in rows:
        t = row["terrain"]
        row["split"] = "validation" if counts[t] % 4 == 0 else "train"
        counts[t] += 1
    assert all(sum(r["terrain"] == t and r["split"] == s for r in rows) >= 2
               for t in (101, 107) for s in ("train", "validation"))
    np.savez_compressed(out / "bank.npz", root=np.stack([r["root"] for r in rows]),
        joint_pos=np.stack([r["joint_pos"] for r in rows]), joint_vel=np.stack([r["joint_vel"] for r in rows]),
        terrain=np.array([r["terrain"] for r in rows]),
        validation=np.array([r["split"] == "validation" for r in rows]))
    report = dict(status="complete", new_physics=0, bank_sha256=sha256_file(out / "bank.npz"),
        scope="Kinematic recovery resets from training failures only; action history and solver/actuator memory reset, not exact simulator snapshots. No action labels.",
        episodes=[{k:v for k,v in row.items() if k not in ("root", "joint_pos", "joint_vel")} for row in rows],
        provenance=provenance, split_counts={f"{t}:{s}": sum(r["terrain"] == t and r["split"] == s for r in rows)
            for t in (101,107) for s in ("train","validation")})
    write_json_atomic(out / "report.json", report)
    print(json.dumps(report["split_counts"]))


if __name__ == "__main__":
    main()
