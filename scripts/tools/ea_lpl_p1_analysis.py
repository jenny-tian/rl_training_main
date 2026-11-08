"""P1 screening, confirmation specifications, EA diagnostics and final basis choice."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training")]
from rl_training.ea_lpl.basis import BASIS_NAMES, basis  # noqa: E402
from rl_training.ea_lpl.headroom import (CONFIRM_NAMESPACE, GRID_VALUES, choose_basis, collect_rows,  # noqa: E402
                                         confirmation_candidates, confirmation_result, confirmation_rows,
                                         ea_decision, pack_specs, screen_table)
from rl_training.ea_lpl.replay import fixed_code_execution_stats, replay_fidelity, switching_diagnostics  # noqa: E402
from rl_training.ea_lpl.specs import validate_spec  # noqa: E402
from rl_training.online_adaptation.safety import SafetyLimits  # noqa: E402

LIMITS = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4))


def load_results(evidence, prefix):
    results = []
    for path in sorted(evidence.glob(f"{prefix}*/result.json")):
        report = json.loads(path.read_text())
        if report["status"] != "passed":
            continue
        results.append(report)
    return results


def write_csv(path, table):
    keys = list(table[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in table:
            w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()})


def subset(rec, idx):
    out = {}
    for k in rec.files:
        a = rec[k]
        if k in ("joint_lower", "joint_upper", "default_joint_pos", "request", "basis_index", "u", "episode_end"):
            out[k] = a[idx]
        elif k == "action_scale":
            out[k] = a
        else:
            out[k] = a[:, idx]
    return out


def ea_diagnostics(evidence):
    files = sorted(evidence.glob("p1_grid_*/p1_grid_*_replay.npz"))
    per_file, active = [], []
    for path in files:
        rec = np.load(path)
        stats = fixed_code_execution_stats(rec)
        entry = dict(file=path.name, **stats)
        if stats["p_proj"] > 0 or stats["p_guard"] > 0:
            active.append(path.name)
        switch = {}
        for i, name in enumerate(BASIS_NAMES):
            idx = np.flatnonzero(np.asarray(rec["basis_index"]) == i)
            if len(idx):
                switch[name] = switching_diagnostics(subset(rec, idx), basis(name), LIMITS, seed=0)
        entry["switching"] = switch
        per_file.append(entry)
        print(path.name, json.dumps(stats), flush=True)
    # fidelity: first file plus every file with projection or guard activity
    fidelity = {}
    for path in files:
        if path == files[0] or path.name in active:
            fidelity[path.name] = replay_fidelity(np.load(path), LIMITS)
            print("fidelity", path.name, fidelity[path.name], flush=True)
    if max(fidelity.values()) > 1e-6:
        raise SystemExit(f"replay fidelity failure: {fidelity}")
    total = sum(e["samples"] for e in per_file)
    c1a = dict(p_proj=sum(e["p_proj"] * e["samples"] for e in per_file) / total,
               p_guard=sum(e["p_guard"] * e["samples"] for e in per_file) / total, samples=total)
    c1b = {}
    for name in BASIS_NAMES:
        items = [e["switching"][name] for e in per_file if name in e["switching"]]
        d = sum(s["decisions"] for s in items)
        c1b[name] = dict(
            decisions=d,
            informative_fraction_single=sum(s["informative_fraction_single"] * s["decisions"] for s in items) / max(d, 1),
            informative_fraction_macro=sum(s["informative_fraction_macro"] * s["decisions"] for s in items) / max(d, 1),
            median_prediction_error=float(np.median([s["median_prediction_error"] for s in items])),
            projection_activation=float(np.mean([s["projection_activation"] for s in items])))
    return dict(schema="ea_lpl_p1_ea_diagnostics_v1", c1a=c1a, c1b=c1b, replay_fidelity=fidelity,
                active_files=active, per_file=per_file)


def heatmaps(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = sorted({(r["level"], r["column"], r["terrain"]) for r in table})
    fig, axes = plt.subplots(len(conds), 3, figsize=(10, 3.2 * len(conds)), squeeze=False)
    for i, (level, column, terrain) in enumerate(conds):
        zero = next(r["completed"] for r in table if (r["level"], r["column"]) == (level, column) and r["code_name"] == "zero")
        for j, name in enumerate(BASIS_NAMES):
            grid = np.zeros((5, 5))
            for r in table:
                if (r["level"], r["column"]) == (level, column) and (r["basis"] == name or r["code_name"] == "zero"):
                    a, b = r["u"]
                    grid[GRID_VALUES.index(a), GRID_VALUES.index(b)] = r["completed"]
            ax = axes[i][j]
            ax.imshow(grid, vmin=0, vmax=8, cmap="viridis")
            for (a, b), v in np.ndenumerate(grid):
                ax.text(b, a, int(v), ha="center", va="center", color="w" if v < 5 else "k", fontsize=8)
            ax.set_xticks(range(5), [f"{g:+.0f}" for g in GRID_VALUES])
            ax.set_yticks(range(5), [f"{g:+.0f}" for g in GRID_VALUES])
            ax.set_xlabel("u2"); ax.set_ylabel("u1")
            ax.set_title(f"{level} c{column} {terrain}\n{name} (zero={zero}/8)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def screen(args):
    rows = collect_rows(load_results(args.evidence, "p1_grid_"))
    table = screen_table(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "screen.csv", table)
    heatmaps(table, args.out / "headroom_heatmaps.png")
    candidates = confirmation_candidates(table)
    targets = json.loads(args.inputs.read_text())["targets"]
    specs = pack_specs(confirmation_rows(candidates, targets), label_prefix="p1_confirm", namespace=CONFIRM_NAMESPACE)
    (args.out / "specs").mkdir(exist_ok=True)
    for spec in specs:
        validate_spec(spec)
        (args.out / "specs" / f"{spec['label']}.json").write_text(json.dumps(spec, indent=2) + "\n")
    (args.out / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    print(f"{len(rows)} rows, {len(candidates)} candidates, {len(specs)} confirmation specs")
    (args.out / "ea_diagnostics.json").write_text(json.dumps(ea_diagnostics(args.evidence), indent=2) + "\n")


def final(args):
    rows = collect_rows(load_results(args.evidence, "p1_confirm_"))
    confirmed = confirmation_result(screen_table(rows))
    if confirmed:
        write_csv(args.out / "confirm.csv", confirmed)
    else:
        write_csv(args.out / "confirm.csv", screen_table(rows))
    choice = choose_basis(confirmed) if confirmed else None
    ea = json.loads((args.out / "ea_diagnostics.json").read_text())
    basis_name = choice["basis"] if choice else "sag"
    decision = ea_decision(ea["c1a"], ea["c1b"], basis_name=basis_name)
    out = dict(schema="ea_lpl_p2_inputs_v1", basis=choice["basis"] if choice else None,
               scores=choice["scores"] if choice else None, targets=choice["targets"] if choice else [],
               ea=dict(decision, c1a=ea["c1a"], c1b=ea["c1b"][basis_name], c1b_basis=basis_name))
    (args.out / "p2_inputs.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if choice is None:
        print("DECISION POINT D-1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--evidence", type=Path, required=True)
    s.add_argument("--inputs", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("final")
    f.add_argument("--evidence", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    screen(args) if args.cmd == "screen" else final(args)


if __name__ == "__main__":
    main()
