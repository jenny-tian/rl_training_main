"""Analyze the completed Flat trace on CPU without replaying the simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def cumulative_trapezoid(values, time):
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time)
    return np.concatenate(([0.0], np.cumsum(increments)))


def analyze(archive):
    samples = [json.loads(line) for line in (archive / "samples.jsonl").read_text().splitlines()]
    record = json.loads((archive / "result.json").read_text())["record"]
    if [r["step"] for r in samples] != list(range(1, record["steps"] + 1)):
        raise ValueError("missing or reordered control samples")
    values = [r["sample"] for r in samples]
    time = np.arange(1, len(values) + 1, dtype=float) * 0.02
    position = np.array([s["root_position_w_m"] for s in values])
    yaw = np.unwrap([s["root_yaw_rad"] for s in values])
    velocity = np.array([s["base_lin_vel_b_mps"] for s in values])
    wz = np.array([s["root_ang_vel_b_rps"][2] for s in values])
    if not all(np.isfinite(x).all() for x in (position, yaw, velocity, wz)):
        raise ValueError("non-finite archived motion state")
    heading = cumulative_trapezoid(np.sin(yaw) * velocity[:, 0], time)
    lateral = cumulative_trapezoid(np.cos(yaw) * velocity[:, 1], time)
    observed = position[:, 1] - position[0, 1]
    residual = observed - heading - lateral
    windows = []
    for start, stop in zip([0, 5, 10, 15, 20, 25], [5, 10, 15, 20, 25, time[-1]]):
        mask = (time > start) & (time <= stop)
        windows.append({"start_s": start, "end_s": float(stop), "samples": int(mask.sum()),
                        "yaw_start_rad": float(yaw[mask][0]), "yaw_end_rad": float(yaw[mask][-1]),
                        "mean_body_yaw_rate_rad_s": float(wz[mask].mean()),
                        "lateral_end_m": float(position[mask][-1, 1])})
    report = {
        "schema": "lite3_flat_drift_offline_audit_v1", "simulator_started": False,
        "source_sha256": {name: hashlib.sha256((archive / name).read_bytes()).hexdigest()
                          for name in ("samples.jsonl", "result.json", "runtime_readback.json")},
        "control_samples": len(values), "sample_interval_s": 0.02,
        "integration_interval_s": [float(time[0]), float(time[-1])],
        "observed_lateral_displacement_m": float(observed[-1]),
        "heading_rotated_forward_contribution_m": float(heading[-1]),
        "body_lateral_contribution_m": float(lateral[-1]),
        "planar_reconstruction_residual_m": float(residual[-1]),
        "planar_reconstruction_max_abs_residual_m": float(np.abs(residual).max()),
        "mean_body_vx_mps": float(velocity[:, 0].mean()), "mean_body_vy_mps": float(velocity[:, 1].mean()),
        "yaw_change_rad": float(yaw[-1] - yaw[0]), "yaw_change_degrees": float(np.rad2deg(yaw[-1] - yaw[0])),
        "mean_body_yaw_rate_rad_s": float(wz.mean()),
        "integral_body_wz_rad": float(cumulative_trapezoid(wz, time)[-1]),
        "windows": windows,
        "interpretation": "Progressive heading drift and positive body-frame lateral velocity largely cancel; their planar integral explains the observed world-frame lateral displacement.",
        "limitations": [
            "Yaw-only planar rotation and 50 Hz trapezoidal integration; root vertical velocity and full quaternion were not logged for ordinary non-contact samples.",
            "Body angular velocity z is not Euler yaw rate; its integral differs from the recorded yaw change and is not used as a heading estimator.",
            "This is a kinematic accounting, not a causal test of policy, initial conditions, material asymmetry or contact dynamics.",
            "The frozen command has heading_command=False and the policy has no absolute yaw/lateral-position observation; no heading controller or policy update is introduced.",
            "The historical single-condition safety pass remains unchanged; this analysis does not qualify straight-line motion or G3.",
        ],
    }
    arrays = dict(time=time, progress=position[:, 0] - record["terrain_origin_w_m"][0],
                  world_y=position[:, 1], yaw=yaw, observed=observed, heading=heading, lateral=lateral,
                  predicted=heading + lateral, velocity=velocity)
    return report, arrays


def plot(arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(arrays["progress"], arrays["world_y"], color="#2364aa")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set(xlabel="Forward progress (m)", ylabel="World lateral offset (m)", title="Recorded path")
    ax = axes[0, 1]
    ax.plot(arrays["time"], np.rad2deg(arrays["yaw"]), color="#b53d3d")
    ax.set(xlabel="Control time (s)", ylabel="Heading (degrees)", title="Heading drift with zero yaw-rate command")
    ax = axes[1, 0]
    for key, label, style in (("observed", "Recorded displacement", "-"),
                              ("predicted", "Planar reconstruction", "--"),
                              ("heading", "Forward velocity rotated by yaw", "-"),
                              ("lateral", "Body lateral velocity contribution", "-")):
        ax.plot(arrays["time"], arrays[key], style, label=label, linewidth=1.4)
    ax.set(xlabel="Control time (s)", ylabel="Lateral displacement (m)", title="Two opposing contributions")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    # Half-second bins display the body velocity trend without gait-scale clutter.
    bins = np.arange(0, arrays["time"][-1] + 0.5, 0.5)
    for i, label in ((0, "Body vx"), (1, "Body vy")):
        means = [arrays["velocity"][(arrays["time"] > a) & (arrays["time"] <= b), i].mean()
                 for a, b in zip(bins, bins[1:])]
        ax.plot((bins[:-1] + bins[1:]) / 2, means, label=label)
    ax.axhline(0.3, color="0.5", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Control time (s)", ylabel="Body velocity (m/s)", title="0.5 s averages; commands: vx=0.3, vy=0")
    ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Lite3 Flat L1: archived trace analysis, no simulation replay", fontsize=14)
    fig.savefig(output / "drift_analysis.png", dpi=180)
    fig.savefig(output / "drift_analysis.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, arrays = analyze(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "drift_analysis.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    plot(arrays, args.output)
    print(json.dumps({key: report[key] for key in (
        "observed_lateral_displacement_m", "heading_rotated_forward_contribution_m",
        "body_lateral_contribution_m", "planar_reconstruction_residual_m")}, indent=2))


if __name__ == "__main__":
    main()
