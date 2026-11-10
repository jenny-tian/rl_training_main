"""Plot six tile-family curriculum curves from two explicit full-run TensorBoard paths (CPU only)."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training")]
from rl_training.ea_lpl.stage1 import FAMILIES  # noqa: E402

COLORS = {42: "#2a78d6", 43: "#eb6834"}
PREFIX = "Curriculum/source_curriculum_levels"


def load_curves(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    curves = {}
    for family in FAMILIES:
        curves[family] = {}
        for metric in ("mean_level", "completion_ema"):
            tag = f"{PREFIX}/{family}/{metric}"
            if tag not in accumulator.Tags()["scalars"]:
                raise ValueError(f"missing required TensorBoard scalar: {tag} in {path}")
            events = accumulator.Scalars(tag)
            # On resume, the latest event for an iteration supersedes an earlier one.
            points = {event.step: event.value for event in events}
            maximum = 9 if metric == "mean_level" else 1
            if not points or any(not math.isfinite(value) or not 0 <= value <= maximum for value in points.values()):
                raise ValueError(f"empty, nonfinite or out-of-range TensorBoard scalar: {tag}")
            curves[family][metric] = sorted(points.items())
    return curves


def plot_curves(curves_by_seed, out):
    # Avoid writing a Matplotlib cache outside the workspace in headless CPU jobs.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stage1-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, constrained_layout=True)
    for family, axis in zip(FAMILIES, axes.flat):
        completion_axis = axis.twinx()
        for seed, curves in sorted(curves_by_seed.items()):
            for metric, target, style in (("mean_level", axis, "-"), ("completion_ema", completion_axis, "--")):
                points = curves[family][metric]
                target.plot([p[0] for p in points], [p[1] for p in points], color=COLORS[seed], linestyle=style,
                            linewidth=1.4, alpha=.9)
        axis.set(title=family.replace("_", " "), xlabel="Iteration", ylabel="Mean level", ylim=(0, 9))
        completion_axis.set(ylabel="Completion EMA", ylim=(0, 1))
        axis.grid(alpha=.2)
    handles = [Line2D([0], [0], color=color, label=f"Seed {seed}") for seed, color in COLORS.items()]
    handles += [Line2D([0], [0], color="black", linestyle=style, label=label)
                for style, label in (("-", "Mean level"), ("--", "Completion EMA"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Stage-1 full-run source curriculum", fontsize=15)
    out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{extension}"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42", type=Path, required=True, help="explicit full-run event file or directory")
    parser.add_argument("--seed43", type=Path, required=True, help="explicit full-run event file or directory")
    parser.add_argument("--out", type=Path, required=True, help="output stem for PNG and SVG")
    args = parser.parse_args()
    plot_curves({42: load_curves(args.seed42), 43: load_curves(args.seed43)}, args.out)


if __name__ == "__main__":
    main()
