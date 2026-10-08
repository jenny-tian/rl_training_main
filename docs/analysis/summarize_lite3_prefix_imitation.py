"""Summarize both fixed imitation fits without new physics or model selection."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    evidence = {}

    def read(path):
        path = Path(path)
        evidence[str(path)] = sha256_file(path)
        return json.loads(path.read_text())

    results = []
    reports = []
    roots = []
    for suffix, learner in (("", "learner2247"), ("_preserve10", "learner2247-preserve10")):
        result = read(f"docs/lite3_prefix_imitation{suffix}_pilot_result_20261008.json")
        assert result["status"] == "verified"
        for path, digest in result["evidence"].items():
            assert sha256_file(path) == digest
        root = Path("artifacts/prefix_imitation_20261008") / learner / "pilot-evaluation"
        roots.append(root)
        reports.append(read(root / "report.json"))
        results.append(result)

    phases = [(roots[0], reports[0], "base"), (roots[0], reports[0], "trial"),
              (roots[1], reports[1], "trial")]
    names = ["Original actor", "Imitation (weight 1)", "Imitation (weight 10)"]
    diagnostics = {}
    for label, (root, report, model) in zip(names, phases):
        diagnostics[label] = {}
        for condition in ("course", "recovery"):
            phase = report["phases"][condition + "-" + model]
            episodes = [e for e in phase["episodes"] if
                        (e["terrain"] != "flat" if condition == "course" else e["case"] == 26)]
            worst = max(episodes, key=lambda e: e["max_body_lateral_m"])
            folder = root / (condition + "-" + model)
            with np.load(folder / "initial.npz", allow_pickle=False) as initial:
                origins = initial["origins"].copy()
                initial_y = initial["root"][:, 1] - origins[:, 1]
            ids = [e["env_id"] for e in episodes]
            extra = 0.
            worst_peak = {"absolute_lateral_m": -1.}
            last_y = np.zeros(len(origins))
            offset = 0
            for file in sorted((folder / "trajectory").glob("chunk-*.npz")):
                evidence[str(file)] = sha256_file(file)
                with np.load(file, allow_pickle=False) as trace:
                    active = trace["active"]
                    y = trace["post_root_state"][..., 1] - origins[:, 1]
                    displacement = np.abs(y[:, ids] - initial_y[ids])
                    extra = max(extra, float(np.where(active[:, ids], displacement, 0).max()))
                    for j in range(len(y)):
                        last_y[active[j]] = y[j, active[j]]
                    i = worst["env_id"]
                    values = np.where(active[:, i], np.abs(y[:, i]), -1.)
                    j = int(values.argmax())
                    if values[j] > worst_peak["absolute_lateral_m"]:
                        worst_peak = dict(absolute_lateral_m=float(values[j]),
                                          time_s=(offset + j + 1) * .02,
                                          x_from_terrain_origin_m=float(trace["post_root_state"][j, i, 0] - origins[i, 0]))
                    offset += len(y)
            np.testing.assert_allclose(worst_peak["absolute_lateral_m"], worst["max_body_lateral_m"], atol=1e-6)
            diagnostics[label][condition] = dict(
                by_terrain={terrain: dict(
                    success=sum(e["success"] for e in episodes if e["terrain"] == terrain),
                    total=sum(e["terrain"] == terrain for e in episodes),
                    mean_rms_m=statistics.mean(e["body_lateral_rms_m"] for e in episodes if e["terrain"] == terrain),
                    peak_m=max(e["max_body_lateral_m"] for e in episodes if e["terrain"] == terrain))
                    for terrain in sorted({e["terrain"] for e in episodes})},
                worst_episode=worst, worst_peak=worst_peak,
                initial_absolute_y_max_m=float(np.abs(initial_y[ids]).max()),
                maximum_displacement_from_own_initial_y_m=extra,
                mean_absolute_y_at_episode_end_m=float(np.abs(last_y[ids]).mean()),
                failures=[e for e in episodes if not e["success"]])

    collection_controls = 0
    for kind in ("course", "prefix"):
        audit = read(f"docs/lite3_prefix_imitation_collect_{kind}_audit_20261008.json")
        assert audit["status"] == "verified"
        for path, digest in audit["evidence"].items():
            assert sha256_file(path) == digest
        collection_controls += audit["controls"]
    assert not any(result["pilot_passed"] for result in results)
    costs = dict(collection=collection_controls, first_pilot=results[0]["controls"],
                 preserve10_pilot=results[1]["controls"], supervised_physics=0,
                 conditional_validation=0, new_uncertain=0)
    stage = costs["collection"] + costs["first_pilot"] + costs["preserve10_pilot"]
    costs.update(stage_controls=stage, cumulative_charged=25876352 + stage,
                 cumulative_confirmed=25876288 + stage, historical_uncertain=64,
                 separate_full_ppo_controls=11900884)
    summary = dict(status="verified", diagnostics=diagnostics, costs=costs,
                   first_pilot_passed=results[0]["pilot_passed"],
                   preserve10_pilot_passed=results[1]["pilot_passed"],
                   evidence=evidence, new_physics=0,
                   scope="One TRAIN recovery source; repeated delay slots, no original7-source qualification, independent learner replication, new terrain, or hardware test in this stage.")
    write_json_atomic("docs/lite3_prefix_imitation_comparison_20261008.json", summary)

    metrics = [results[0]["results"]["base"], results[0]["results"]["trial"], results[1]["results"]["trial"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    colors = ["#65758B", "#D08A46", "#2A8C82"]
    ticks = ["Original", "Weight 1", "Weight 10"]
    def bars(ax, key, multiplier, title, threshold=None, denominator=None):
        values = [m[key] * multiplier for m in metrics]
        ax.bar(ticks, values, color=colors, width=.6)
        for i, value in enumerate(values):
            ax.text(i, value, f"{int(value)}/{denominator}" if denominator else f"{value:.2f}", ha="center", va="bottom")
        if threshold is not None:
            ax.axhline(threshold, color="#B44747", ls="--", lw=1.3, label=f"Gate: {threshold:.2f}")
            ax.legend(frameon=False)
        ax.set_ylim(0, max(max(values), threshold or 0) * 1.22)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    bars(axes[0, 0], "course_total", 1, "Ordinary full-course completion", denominator=64)
    bars(axes[0, 1], "course_rms_m", 100, "Ordinary rough mean episode RMS (cm)", metrics[0]["course_rms_m"] * 105)
    bars(axes[1, 0], "course_peak_m", 100, "Ordinary rough peak lateral error (cm)", metrics[0]["course_peak_m"] * 100)
    bars(axes[1, 1], "case26_recovery", 1, "TRAIN case26 autonomous recovery + completion", denominator=24)
    fig.suptitle("Recovery learned; ordinary walking restored, precision gates still fail\nSame evaluation seed 2251; no scripted prefix or teacher takeover", fontsize=14)
    target = Path("docs/figures/lite3-prefix-imitation-20261008")
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(target.with_suffix(suffix), dpi=160)
    svg = target.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
