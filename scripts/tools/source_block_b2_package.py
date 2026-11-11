"""Package completed B2 baseline and first-segment evidence without checkpoints/logs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from source_block_b2_protocol import EVIDENCE, ROOT, Q4_LIMITS, save_json

OUT = ROOT / "docs/lite3-block-b2-initialization-20261111"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    records, copied, original = [], [], []

    def archive(source, name):
        source = Path(source)
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(target)
        original.append(source)
        return json.loads(source.read_text()) if source.suffix == ".json" else None

    checks = {}
    for size in (120, 4096):
        checks[str(size)] = archive(EVIDENCE / f"check{size}/checks.json", f"validation/check{size}.json")
        archive(EVIDENCE / f"check{size}/execution.json", f"validation/execution{size}.json")
        original.append(EVIDENCE / f"check{size}/runtime.log")
    archive(EVIDENCE / "cpu.xml", "validation/cpu.xml")
    archive(EVIDENCE / "cpu_importer.xml", "validation/cpu_importer.xml")
    baselines = {}
    for start in ("flat1998", "terrain3497"):
        folder = EVIDENCE / "baseline" / start
        block = archive(folder / "blocks/evaluation.json", f"baseline/{start}_blocks.json")
        flat = archive(folder / "flat.json", f"baseline/{start}_flat.json")
        gate = archive(folder / "decision.json", f"baseline/{start}_decision.json")
        baselines[start] = block
        original.extend([folder / "blocks/runtime.log", folder / "blocks/execution.json", folder / "flat.sim.log"])
        original.append(Path(block["checkpoint"]))
        records.append(row(start, "baseline", 0, gate, block))
    paired = baselines["flat1998"]["initial_state_hashes_by_column"] == baselines["terrain3497"]["initial_state_hashes_by_column"]
    if not paired:
        raise ValueError("baseline initial states do not match")
    runs, audits, training_heads = [], [], set()
    for start in ("flat1998", "terrain3497"):
        for seed in (42, 43):
            name = f"{start}_seed{seed}"
            folder = EVIDENCE / "training" / name
            run = archive(folder / "training_report.json", f"training/{name}.json")
            execution = archive(folder / "execution.json", f"training/{name}_execution.json")
            training_heads.add(execution["source_commit"])
            if execution["status"] != "completed" or run["status"] not in ("budget_completed", "stopped_by_development_rule"):
                raise ValueError(f"incomplete run: {name}")
            original.append(folder / "runtime.log")
            original.extend((folder / "training/params").glob("*.yaml"))
            for gate in run["checks"]:
                updates = gate["updates"]
                check = folder / f"update_{updates:04d}"
                block = archive(check / "evaluation/blocks/evaluation.json", f"evaluations/{name}_{updates}_blocks.json")
                archive(check / "evaluation/flat.json", f"evaluations/{name}_{updates}_flat.json")
                stats = archive(check / "statistics.json", f"statistics/{name}_{updates}.json")
                original.extend([check / "decision.json", check / "model.pt", check / "evaluation/flat.sim.log",
                                 check / "evaluation/blocks/runtime.log", check / "evaluation/blocks/execution.json"])
                if digest(check / "model.pt") != gate["sha256"]:
                    raise ValueError("trained checkpoint identity mismatch")
                if block["initial_state_hashes_by_column"] != baselines[start]["initial_state_hashes_by_column"]:
                    raise ValueError("checkpoint evaluation is not paired to its baseline")
                if sum(map(sum, stats["transitions_by_column_direction"])) != updates * 4096 * 24:
                    raise ValueError("training transition count differs from update budget")
                expected = [n * updates * 24 for n in stats["environment_column_counts"]]
                if list(map(sum, stats["transitions_by_column_direction"])) != expected:
                    raise ValueError("per-column interaction count differs from allocation")
                audits.append(audit_model(run, check / "model.pt", updates))
                records.append(row(start, seed, updates, gate, block))
            runs.append(run)
    with (OUT / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    copied.append(OUT / "comparison.csv")
    tests = [t for path in (EVIDENCE / "cpu.xml", EVIDENCE / "cpu_importer.xml")
             for t in ET.parse(path).getroot().findall("testsuite")]
    summary = dict(paired_initial_states=paired, rows=records, runs=[{
        k: run[k] for k in ("start", "seed", "status", "completed_updates", "training_transitions")}
        for run in runs], total_updates=sum(r["completed_updates"] for r in runs),
        total_training_transitions=sum(r["training_transitions"] for r in runs), q4_limits=Q4_LIMITS,
        cpu_tests=sum(int(t.attrib["tests"]) for t in tests),
        validation={size: dict(passed=sum(c["passed"] for c in value["checks"]), total=len(value["checks"]))
                    for size, value in checks.items()}, selected_theta1=None, final_qualification_performed=False)
    save_json(OUT / "summary.json", summary)
    copied.append(OUT / "summary.json")
    save_json(OUT / "model_audit.json", audits)
    copied.append(OUT / "model_audit.json")
    plot(records)
    copied.append(OUT / "comparison.png")
    source_paths = subprocess.check_output(["git", "diff", "--name-only", "99616dd", "HEAD", "--", "scripts", "source", "tests"], cwd=ROOT, text=True).splitlines()
    original.extend(ROOT / p for p in source_paths)
    manifest = dict(schema="b2_initialization_segment_v1", training_source_heads=sorted(training_heads),
        packaging_source_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        raw_evidence_root=str(EVIDENCE), files=[dict(path=str(p), sha256=digest(p)) for p in sorted(set(original))])
    save_json(OUT / "artifact_manifest.json", manifest)
    copied.append(OUT / "artifact_manifest.json")
    copied.extend(path for path in (OUT / "plan.md", OUT / "README.md", OUT / "next-stage-proposal.md") if path.exists())
    (OUT / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.relative_to(OUT)}\n" for p in sorted(copied)))
    print(json.dumps(summary, indent=2), flush=True)


def audit_model(run, path, updates):
    import torch
    original = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
    current = torch.load(path, map_location="cpu", weights_only=True)
    changes = {}
    for component in ("actor_state_dict", "critic_state_dict"):
        before, after = original[component], current[component]
        if before.keys() != after.keys() or not all(torch.isfinite(v).all() for v in after.values()):
            raise ValueError("model schema or finite check failed")
        keys = [k for k in before if k.startswith("mlp.")]
        changed = [k for k in keys if not torch.equal(before[k], after[k])]
        if len(changed) != len(keys):
            raise ValueError("a learned MLP tensor did not change")
        changes[component] = dict(changed_mlp_tensors=len(changed), total_mlp_tensors=len(keys),
            weight_change_l2=sum(float((after[k].double() - before[k].double()).square().sum()) for k in keys) ** .5)
    before = original["optimizer_state_dict"]["state"]
    after = current["optimizer_state_dict"]["state"]
    increments = sorted({int(after[k]["step"]) - int(before[k]["step"]) for k in before})
    if increments != [updates * 20]:
        raise ValueError("Adam step audit failed")
    return dict(start=run["start"], seed=run["seed"], updates=updates, checkpoint=str(path),
                sha256=digest(path), initial_full_state_restore_exact=run["restored_exact"],
                optimizer_step_increments=increments, changes=changes,
                initial_algorithm_learning_rate=run["initial_algorithm_learning_rate"],
                initial_optimizer_group_lrs=run["initial_optimizer_group_lrs"],
                action_std_before=original["actor_state_dict"]["distribution.log_std_param"].exp().tolist(),
                action_std_after=current["actor_state_dict"]["distribution.log_std_param"].exp().tolist())


def row(start, seed, updates, gate, block):
    m = gate["flat_metrics"]
    return dict(start=start, seed=seed, updates=updates, upstairs=block["columns"][4]["complete"],
                downstairs=block["columns"][5]["complete"], rough_a=block["columns"][6]["complete"],
                rough_b=block["columns"][15]["complete"], rms_xy=m["rms_xy_error"], rms_yaw=m["rms_yaw_error"],
                pure_turn_rms_yaw=m["categories"]["only_ang"]["rms_yaw_error"],
                falls=m["fall_episodes"], ended=m["ended_episodes"],
                continue_training=gate["continue_training"], stop_reasons="; ".join(gate["reasons"]))


def plot(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for start, color in (("flat1998", "#3178b8"), ("terrain3497", "#d07026")):
        baseline = next(r for r in records if r["start"] == start and r["updates"] == 0)
        for seed, style in ((42, "-"), (43, "--")):
            rows = [baseline] + [r for r in records if r["start"] == start and r["seed"] == seed]
            for ax, key, label in zip(axes.flat, ("upstairs", "downstairs", "rms_xy", "rms_yaw"),
                                      ("Upstairs completed / 32", "Downstairs completed / 32", "Flat translation RMS (m/s)", "Flat yaw RMS (rad/s)")):
                ax.plot([r["updates"] for r in rows], [r[key] for r in rows], style, color=color,
                        marker="o", label=f"{start}, seed {seed}")
                ax.set_xlabel("New PPO updates")
                ax.set_ylabel(label)
                ax.grid(alpha=.25)
    axes[0, 0].set_ylim(-1, 33)
    axes[0, 1].set_ylim(-1, 33)
    axes[1, 0].axhline(Q4_LIMITS["rms_xy_error"], color="red", linestyle=":")
    axes[1, 1].axhline(Q4_LIMITS["rms_yaw_error"], color="red", linestyle=":")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("B2 initialization comparison: development checks, not final qualification")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
