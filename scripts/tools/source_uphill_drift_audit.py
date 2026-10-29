"""Analyze immutable uphill traces on CPU; no policy or simulator execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.tools.source_flat_drift_audit import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[2]
UPHILL = ROOT / "docs/lite3-source-policy-prequalification-uphill-l1-20261029"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace(archive):
    rows = [json.loads(line) for line in (archive / "samples.jsonl").read_text().splitlines()]
    record = json.loads((archive / "result.json").read_text())["record"]
    if [r["step"] for r in rows] != list(range(1, record["steps"] + 1)):
        raise ValueError("missing/reordered samples")
    samples = [r["sample"] for r in rows]
    position = np.array([s["root_position_w_m"] for s in samples])
    yaw = np.unwrap([s["root_yaw_rad"] for s in samples])
    body_xy = np.array([s["base_lin_vel_b_mps"] for s in samples])
    time = np.arange(1, len(rows) + 1) * 0.02
    progress = position[:, 0] - record["terrain_origin_w_m"][0]
    if not all(np.isfinite(v).all() for v in (position, yaw, body_xy)):
        raise ValueError("nonfinite motion state")
    if not np.all(np.diff(progress) > 0):
        raise ValueError("progress interpolation requires a strictly forward trace")
    return rows, record, dict(time=time, progress=progress, position=position, yaw=yaw, body_xy=body_xy)


def lateral_contributions(quaternions, body_velocity):
    """World-Y contributions from body x/y/z using the full wxyz quaternion."""
    w, x, y, z = np.asarray(quaternions).T
    coefficients = np.column_stack((2 * (x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)))
    return coefficients * body_velocity


def analyze(archive=UPHILL):
    rows, record, arrays = load_trace(archive)
    manifest = json.loads((archive / "manifest.json").read_text())
    q = np.array([r["kinematics"]["root_quat_w_wxyz"] for r in rows])
    body = np.array([r["kinematics"]["root_lin_vel_b_mps"] for r in rows])
    world = np.array([r["kinematics"]["root_lin_vel_w_mps"] for r in rows])
    if not all(np.isfinite(v).all() for v in (q, body, world)):
        raise ValueError("nonfinite full kinematics")
    time, progress, yaw = (arrays[k] for k in ("time", "progress", "yaw"))
    contributions = lateral_contributions(q, body)
    rotation_error = float(np.abs(contributions.sum(axis=1) - world[:, 1]).max())
    if rotation_error > 1e-5:
        raise ValueError("body/world rotation mismatch")
    integrals = np.column_stack([cumulative_trapezoid(v, time) for v in contributions.T])
    observed = arrays["position"][:, 1] - arrays["position"][0, 1]
    reconstructed = integrals.sum(axis=1)
    residual = observed - reconstructed
    world_integral = cumulative_trapezoid(world[:, 1], time)
    geometry = manifest["cell"]["resolved_parameters"]["geometry"]
    start = manifest["cell"]["resolved_parameters"]["track"]["start_x_m"]
    entry = geometry["flat_length_m"] - start
    full_slope = entry + geometry["transition_length_m"]
    exit_flat = geometry["size"][0] - geometry["exit_flat_length_m"] - start
    exit_transition = exit_flat - geometry["transition_length_m"]
    cuts = [progress[0], entry, full_slope, exit_transition, exit_flat, geometry["size"][0] - start]
    segments = []
    def at(progress_value, values):
        return float(np.interp(progress_value, progress, values))
    for label, lo, hi in zip(("entry_flat", "entry_transition", "slope", "exit_transition", "exit_flat"), cuts, cuts[1:]):
        lo, hi = max(lo, progress[0]), min(hi, progress[-1])
        if lo >= hi:
            continue
        mask = (progress >= lo) & (progress <= hi)
        segments.append({
            "section": label, "progress_interval_m": [float(lo), float(hi)],
            "interpolated_time_interval_s": [at(lo, time), at(hi, time)],
            "yaw_start_end_deg": [float(np.rad2deg(at(lo, yaw))), float(np.rad2deg(at(hi, yaw)))],
            "lateral_start_end_m": [at(lo, arrays["position"][:, 1]), at(hi, arrays["position"][:, 1])],
            "mean_body_vx_vy_mps": body[mask, :2].mean(axis=0).tolist(),
            "mean_world_vx_vy_mps": world[mask, :2].mean(axis=0).tolist(),
            "lateral_displacement_m": at(hi, observed) - at(lo, observed),
            "body_xyz_contribution_m": [at(hi, col) - at(lo, col) for col in integrals.T],
            "integration_residual_m": at(hi, residual) - at(lo, residual),
        })
    crossings = []
    for name, values, thresholds in (("abs_yaw_deg", np.abs(np.rad2deg(yaw)), (5, 10, 20)),
                                     ("abs_lateral_m", np.abs(arrays["position"][:, 1]), (0.5, 1.0))):
        for threshold in thresholds:
            hit = np.flatnonzero(values >= threshold)
            index = int(hit[0]) if len(hit) else None
            crossings.append({"metric": name, "descriptive_threshold": threshold,
                              "step": index + 1 if index is not None else None,
                              "time_s": float(time[index]) if index is not None else None,
                              "progress_m": float(progress[index]) if index is not None else None})
    arrays.update(body=body, world=world, observed=observed, integrals=integrals,
                  reconstructed=reconstructed, residual=residual, section_progress=[entry, full_slope])
    report = {
        "schema": "lite3_uphill_drift_offline_audit_v1", "simulation_run": False,
        "runtime_revision": manifest["revision"],
        "source_sha256": {str(archive.relative_to(ROOT) / n): sha(archive / n)
                          for n in ("samples.jsonl", "result.json", "manifest.json", "runtime_readback.json")},
        "control_samples": len(rows), "integration_interval_s": [float(time[0]), float(time[-1])],
        "sample_interval_s": 0.02, "observed_lateral_displacement_m": float(observed[-1]),
        "body_xyz_world_y_contribution_m": integrals[-1].tolist(),
        "world_velocity_integral_m": float(world_integral[-1]),
        "integration_residual_m": float(residual[-1]),
        "max_abs_integration_residual_m": float(np.abs(residual).max()),
        "max_rotation_error_mps": rotation_error,
        "quaternion_norm_max_error": float(np.abs(np.linalg.norm(q, axis=1) - 1).max()),
        "segments": segments, "descriptive_crossings": crossings,
        "terminal_phase": rows[-1]["kinematics_phase"],
        "termination_terms": record["termination_terms"],
        "limitations": [
            "Kinematic accounting cannot identify policy, initialization or contact-dynamics causes.",
            "Segments use root position and registered geometry, not measured first foot contact with the slope.",
            "Trapezoidal integration starts at the first 0.02 s sample; the reset-to-first-sample interval is not reconstructed.",
            "Section crossings use linear interpolation between 50 Hz samples; reported descriptive thresholds are not new safety criteria.",
            "The exact historical 17-body boundary predicate inputs and current-versus-prediction branch remain unrecorded.",
            "A heading contribution is a coordinate rotation of velocity, not evidence for a particular controller defect.",
        ],
    }
    return report, arrays


def comparisons():
    result = []
    for label in ("flat", "rough-l1", "uphill-l1"):
        archive = ROOT / f"docs/lite3-source-policy-prequalification-{label}-20261029"
        _, record, arrays = load_trace(archive)
        manifest = json.loads((archive / "manifest.json").read_text())
        result.append({
            "label": label, "runtime_revision": manifest["revision"],
            "geometry_seed": record["geometry_seed"], "initial_state_seed": record["initial_state_seed"],
            "steps": record["steps"], "completed": record["completed"],
            "forward_progress_m": record["forward_progress_m"],
            "lateral_offset_peak_m": record["lateral_offset_peak_m"],
            "yaw_end_deg": float(np.rad2deg(arrays["yaw"][-1])),
            "mean_body_vx_mps": record["mean_base_velocity_x_mps"],
            "mean_velocity_error_mps": record["mean_velocity_error_mps"],
            "source_sha256": {str(archive.relative_to(ROOT) / name): sha(archive / name)
                              for name in ("manifest.json", "result.json", "samples.jsonl")},
        })
    old = ROOT / "docs/lite3-source-scan-preparation-v0.2-2026-10-22/family_runtime_20261023/slope_L1_uphill.json"
    legacy = json.loads(old.read_text())
    r = legacy["record"]
    return {"schema": "lite3_uphill_historical_context_v1", "current_archives": result,
            "older_uphill": {"source": str(old.relative_to(ROOT)), "sha256": sha(old),
                              "source_code_sha256": legacy["source_sha256"],
                              "runtime_revision": "not_recorded_in_report",
                              "checkpoint_sha256": legacy["checkpoint_sha256"],
                              "geometry": legacy["cell"]["resolved_parameters"]["geometry"],
                              "record": {k: r[k] for k in ("geometry_seed", "initial_state_seed", "steps", "completed", "forward_progress_m", "lateral_offset_peak_m", "root_yaw_end_rad", "mean_base_velocity_x_mps", "termination_reason")}},
            "limitations": ["Different terrain/state seeds and runtime revisions; no matched causal comparison or pooled success rate.",
                            "Older uphill used a 10 m track, a different geometry seed and root lateral termination; the current run uses a 12 m track and the full-body predictive guard.",
                            "Flat lacks ordinary-step full quaternion/world-velocity telemetry; only recorded summary metrics are compared here."]}


def plot(arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    t, p = arrays["time"], arrays["progress"]
    ax = axes[0, 0]
    ax.plot(p, arrays["position"][:, 1], label="Root lateral position")
    ax.set(xlabel="Forward progress from reset origin (m)", ylabel="Lateral offset (m)", title="Path before boundary termination")
    ax = axes[0, 1]
    ax.plot(p, np.rad2deg(arrays["yaw"]), color="#b53d3d")
    ax.set(xlabel="Forward progress from reset origin (m)", ylabel="Yaw (degrees)", title="Heading change; commanded yaw rate = 0")
    for ax in axes[0]:
        for boundary, label in zip(arrays["section_progress"], ("Entry transition", "Full slope")):
            ax.axvline(boundary, color="gray", linestyle=":", linewidth=1, label=label)
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(t, arrays["observed"], color="black", label="Recorded displacement")
    ax.plot(t, arrays["reconstructed"], "--", label="Full quaternion reconstruction")
    for index, label in enumerate(("Body vx contribution", "Body vy contribution", "Body vz contribution")):
        ax.plot(t, arrays["integrals"][:, index], label=label)
    ax.set(xlabel="Control time (s)", ylabel="World-Y displacement (m)", title="Velocity contribution integrals")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    bins = np.arange(0, t[-1] + 0.5, 0.5)
    for values, label in ((arrays["body"][:, 0], "Body vx"), (arrays["body"][:, 1], "Body vy"),
                          (arrays["world"][:, 1], "World vy")):
        means = [values[(t > a) & (t <= b)].mean() for a, b in zip(bins, bins[1:])]
        ax.plot((bins[1:] + bins[:-1]) / 2, means, label=label)
    ax.axhline(0.3, color="gray", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Control time (s)", ylabel="Velocity (m/s)", title="0.5 s means; commanded body vx = 0.3")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Lite3 5-degree uphill: archived failed episode, offline analysis only")
    fig.savefig(output / "drift_analysis.png", dpi=180)
    fig.savefig(output / "drift_analysis.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, arrays = analyze()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("drift_analysis.json", report), ("historical_comparison.json", comparisons())):
        with (args.output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    plot(arrays, args.output)
    print(json.dumps({k: report[k] for k in ("observed_lateral_displacement_m", "body_xyz_world_y_contribution_m", "integration_residual_m", "segments")}, indent=2))


if __name__ == "__main__":
    main()
