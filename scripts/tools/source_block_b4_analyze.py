"""Verify B4 pairing and frozen evidence, then package both seeds without selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from source_block_b2_protocol import Q4_LIMITS, ROOT
from source_block_b3_analyze import save, table
from source_block_b3_protocol import CASES, FIELDS, REPEATS, first_episode_masks, schedule, summarize
from source_block_b4_evaluate import completed_models
from source_block_b4_protocol import OUT, RAW, SEEDS, candidate_gate, q4_failures


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_delivery(directory):
    entries = (directory / "SHA256SUMS").read_text().splitlines()
    for line in entries:
        digest, filename = line.split("  ", 1)
        if sha(directory / filename) != digest:
            raise ValueError(f"historical evidence changed: {directory / filename}")
    return len(entries)


def paired_metric(candidate, control, case, dt=.02):
    """Compare each initial condition only on common first-episode samples."""
    a, amask = candidate
    b, bmask = control
    if case.startswith("heading"):
        lo, hi, index, metric = 14, 20, 6, "mean_abs_heading_error"
    elif case.startswith(("stop", "ramp")) or case == "standing":
        lo, hi, index, metric = 10, 14, 5, "mean_abs_yaw_rate"
    elif case.startswith("reverse"):
        lo, hi, index, metric = 10, 14, 5, "yaw_rms_error"
    elif case.startswith("lateral"):
        lo, hi, index, metric = 4, 8, 4, "lateral_rms_error"
    elif case == "forward":
        lo, hi, index, metric = 4, 8, 3, "forward_rms_error"
    else:
        lo, hi, index, metric = 4, 8, 5, "yaw_rms_error"
    case_id = CASES.index(case)
    part = slice(case_id * REPEATS, (case_id + 1) * REPEATS)
    times = np.arange(len(a)) * dt
    window = (times >= lo) & (times < hi)
    common = amask[window, part] & bmask[window, part]
    count = common.sum(0)
    keep = count > 0
    if not keep.any():
        return dict(case=case, metric=metric, window=f"{lo}-{hi}s", paired_starts=0,
                    mean_difference=None, bootstrap_95_low=None, bootstrap_95_high=None)
    if "rms" in metric:
        def score(trace):
            error = trace[window, part, index] - trace[window, part, index - 3]
            return np.sqrt((error[:, keep] ** 2 * common[:, keep]).sum(0) / count[keep])
    else:
        def score(trace):
            return (abs(trace[window, part, index][:, keep]) * common[:, keep]).sum(0) / count[keep]
    delta = score(a) - score(b)
    rng = np.random.default_rng(20261113)
    means = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(-1)
    return dict(case=case, metric=metric, window=f"{lo}-{hi}s", paired_starts=int(keep.sum()),
        mean_difference=float(delta.mean()), bootstrap_95_low=float(np.quantile(means, .025)),
        bootstrap_95_high=float(np.quantile(means, .975)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    b2 = ROOT / "docs/lite3-block-b2-initialization-20261111"
    b3 = ROOT / "docs/lite3-block-b3-diagnosis-20261111"
    historical = {"b2_delivery_files": verify_delivery(b2), "b3_delivery_files": verify_delivery(b3)}
    previous_raw = {v["path"]: v["sha256"] for v in read(b3 / "artifact_manifest.json")}
    manifest, executions, models, rows, episodes, blocks, metrics = [], [], completed_models(), [], [], {}, {}
    training, pairing, responses = {}, {}, {d: {} for d in ("flat", "block")}

    def record(path, historical_raw=False):
        digest = sha(path)
        if historical_raw and previous_raw.get(str(path)) != digest:
            raise ValueError(f"historical raw evidence changed: {path}")
        manifest.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest))

    def copy(directory, target, names):
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = directory / name
            shutil.copyfile(source, target / name)
            record(source)

    def runtime(directory, name, kind):
        execution = read(directory / "execution.json")
        if (execution["status"] != "completed" or execution["exit_code"] != 0
                or execution["fatal_messages"]):
            raise ValueError(f"failed runtime: {directory}")
        executions.append(dict(model=name, kind=kind, **execution,
            error_log_lines=[line for line in (directory / "runtime.log").read_text(errors="replace").splitlines()
                             if "[Error]" in line]))

    baseline_block = read(b2 / "baseline/terrain3497_blocks.json")
    baseline_flat = read(b2 / "baseline/terrain3497_flat.json")
    summary = [dict(model="terrain3497", seed=None, updates=0,
        **{k: baseline_flat["metrics"][k] for k in Q4_LIMITS}, q4_pass=True,
        upstairs=baseline_block["columns"][4]["complete"], downstairs=baseline_block["columns"][5]["complete"],
        rough_a=baseline_block["columns"][6]["complete"], rough_b=baseline_block["columns"][15]["complete"])]
    for spec in models:
        name = spec["name"]
        directory = RAW / "training" / name
        training[name] = read(directory / "training_report.json")
        if (training[name]["training_transitions"] != 4096 * 24 * 25
                or training[name]["environment_steps"] != 600
                or training[name]["optimizer_step_increase"] != 500):
            raise ValueError(f"training budget mismatch: {name}")
        counts = np.array(read(directory / "statistics.json")["transitions_by_column_direction"]).sum(-1)
        if counts.tolist() != [v * 600 for v in training[name]["environment_column_counts"]]:
            raise ValueError(f"recorded terrain exposure differs from allocated steps: {name}")
        runtime(directory, name, "training")
        copy(directory, OUT / "training" / name, ("training_report.json", "statistics.json", "execution.json"))
        for filename in ("model_0025.pt", "runtime.log", "config_audit.json", "training/params/env.yaml", "training/params/agent.yaml"):
            record(directory / filename)
        source = RAW / "evaluations" / name
        flat, block = read(source / "flat.json"), read(source / "blocks/evaluation.json")
        if block["status"] != "passed" or block["initial_state_hashes_by_column"] != baseline_block["initial_state_hashes_by_column"]:
            raise ValueError("unpaired or failed block evaluation")
        if flat["cleanup"] != {"environment": "returned", "application": "returned"}:
            raise ValueError("incomplete Q4 cleanup")
        failure = q4_failures(flat)
        blocks[name], metrics[name] = block, flat
        runtime(source / "blocks", name, "blocks")
        copy(source, OUT / "evaluations" / name, ("flat.json",))
        copy(source / "blocks", OUT / "evaluations" / name / "blocks", ("evaluation.json", "execution.json"))
        for filename in ("flat.sim.log", "flat.phase.json", "blocks/runtime.log"):
            record(source / filename)
        executions.append(dict(model=name, kind="flat_q4", **flat["runtime"],
            error_log_lines=[line for line in (source / "flat.sim.log").read_text(errors="replace").splitlines()
                             if "[Error]" in line]))
        summary.append(dict(model=name, seed=spec["seed"], updates=25,
            **{k: flat["metrics"][k] for k in Q4_LIMITS}, q4_pass=not failure,
            upstairs=block["columns"][4]["complete"], downstairs=block["columns"][5]["complete"],
            rough_a=block["columns"][6]["complete"], rough_b=block["columns"][15]["complete"]))

    for seed in SEEDS:
        control, candidate = (f"{branch}_seed{seed}" for branch in ("control", "turngate"))
        for key in ("initial_hashes", "first_action_sha256", "first_episode_length_sha256"):
            if training[control][key] != training[candidate][key]:
                raise ValueError(f"unpaired training: seed={seed}, field={key}")
        pairing[f"training_seed{seed}"] = dict(initial_hashes=training[control]["initial_hashes"],
            first_action_sha256=training[control]["first_action_sha256"],
            first_episode_length_sha256=training[control]["first_episode_length_sha256"])
    first_q4 = metrics[models[0]["name"]]["initial_state_hashes"]
    if any(v["initial_state_hashes"] != first_q4 for v in metrics.values()):
        raise ValueError("Q4 initial states differ")
    pairing["q4_all_four"] = first_q4
    pairing["block_all_four_and_original"] = baseline_block["initial_state_hashes_by_column"]

    for domain in responses:
        reference = None
        for name in ["terrain3497", *[m["name"] for m in models]]:
            old = name == "terrain3497"
            directory = (ROOT / "logs/stage1/block_b3/matrix" / domain / "controlled" / name if old
                         else RAW / "evaluations" / name / "controlled" / domain)
            report = read(directory / "report.json")
            if (report["status"] != "passed" or report["probe"] or not report["actor_unchanged"]
                    or report["cleanup"] != {"environment": "returned", "application": "returned"}
                    or report["steps"] != 1000 or report["num_envs"] != 512):
                raise ValueError(f"controlled diagnostic incomplete: {directory}")
            if reference is None:
                reference = report["initial_hashes"]
            if reference != report["initial_hashes"]:
                raise ValueError(f"unpaired controlled initial state: {directory}")
            runtime(directory, name, f"controlled_{domain}")
            with np.load(directory / "trace.npz") as loaded:
                trace, ended, terminated = loaded["trace"], loaded["ended"], loaded["terminated"]
                _, valid = first_episode_masks(ended)
                recomputed = summarize(trace, ended, terminated,
                    {case: np.arange(512) // REPEATS == i for i, case in enumerate(CASES)},
                    report["reward_names"], .02, "controlled")
                if recomputed != report["metrics"] or not np.isfinite(trace).all():
                    raise ValueError(f"trace/metrics mismatch or nonfinite data: {directory}")
                responses[domain][name] = (trace[..., :len(FIELDS)].copy(), valid.copy())
            for row in report["metrics"]["rows"]:
                rows.append(dict(model=name, domain=domain,
                    **{k: v for k, v in row.items() if k not in ("reward_rates", "mean_velocity", "mean_command")},
                    **{f"mean_velocity_{axis}": row["mean_velocity"][i] for i, axis in enumerate(("x", "y", "yaw"))}))
            for case, episode in report["metrics"]["episodes"].items():
                episodes.append(dict(model=name, domain=domain, case=case, **episode))
            if not old:
                copy(directory, OUT / "evaluations" / name / "controlled" / domain,
                     ("report.json", "config.json", "execution.json"))
            for filename in ("report.json", "config.json", "execution.json", "trace.npz", "initial.npz", "runtime.log"):
                record(directory / filename, historical_raw=old)
        pairing[f"controlled_{domain}_all_four_and_original"] = reference

    comparisons = []
    for domain, data in responses.items():
        for seed in SEEDS:
            control, candidate = (f"{branch}_seed{seed}" for branch in ("control", "turngate"))
            for a, b in ((candidate, control), (candidate, "terrain3497"), (control, "terrain3497")):
                for case in CASES:
                    comparisons.append(dict(domain=domain, candidate=a, reference=b,
                                            **paired_metric(data[a], data[b], case)))
        figure, axes = plt.subplots(4, 4, figsize=(17, 13), sharex=True)
        colors = dict(terrain3497="#222222", control_seed42="#e0872a", turngate_seed42="#2673b8",
                      control_seed43="#be5b80", turngate_seed43="#338c58")
        times = np.arange(1000) * .02
        for case_id, case in enumerate(CASES):
            ax = axes.flat[case_id]
            part = slice(case_id * REPEATS, (case_id + 1) * REPEATS)
            index = 6 if case.startswith("heading") else 4 if case.startswith("lateral") else 3 if case == "forward" else 5
            for name, (trace, valid) in data.items():
                mask = valid[:, part]
                count = mask.sum(-1)
                mean = np.divide((trace[:, part, index] * mask).sum(-1), count,
                                 out=np.full(len(times), np.nan), where=count > 0)
                ax.plot(times + .02, mean, lw=1.1, color=colors[name], label=name)
            if index != 6:
                ax.plot(times, [schedule(case, t)[index - 3] for t in times], "k--", lw=.8, label="command")
            else:
                ax.axhline(0, color="gray", ls="--", lw=.8)
            ax.set_title(case)
            ax.set_ylabel("heading error (rad)" if index == 6 else "m/s" if index in (3, 4) else "rad/s")
            ax.set_xlabel("time (s)")
            ax.grid(alpha=.2)
        figure.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=6)
        figure.suptitle(f"B4 {domain}: paired starts, means exclude terminal/reset samples; failures listed separately", y=.965)
        figure.tight_layout(rect=(0, 0, 1, .94))
        figure.savefig(OUT / f"{domain}_responses.png", dpi=150)
        figure.savefig(OUT / f"{domain}_responses.pdf")
        plt.close(figure)

    gates = {str(seed): candidate_gate(metrics[f"turngate_seed{seed}"], blocks[f"turngate_seed{seed}"],
                                     blocks[f"control_seed{seed}"]) for seed in SEEDS}
    copy(RAW / "smoke", OUT / "validation/smoke", ("checks.json", "config_audit.json", "execution.json"))
    record(RAW / "smoke/runtime.log")
    table(OUT / "comparison.csv", summary)
    table(OUT / "response_metrics.csv", rows)
    table(OUT / "episodes.csv", episodes)
    table(OUT / "paired_comparisons.csv", comparisons)
    save(OUT / "summary.json", dict(models=models, rows=summary, candidate_gates=gates,
        both_seeds_eligible_for_discussion=all(g["eligible_for_discussion"] for g in gates.values()),
        automatic_extension=False, training_interactions=sum(t["training_transitions"] for t in training.values()),
        new_controlled_first_episodes=4 * 2 * 512, q4_interactions=4 * 1024 * 3000,
        paired_comparison_limitation="Conditional on shared first-episode survival; 32 paired starts per case, not independent training seeds. No multiplicity correction."))
    save(OUT / "pairing.json", pairing)
    save(OUT / "execution_audit.json", executions)
    save(OUT / "artifact_manifest.json", list({v["path"]: v for v in manifest}.values()))
    save(OUT / "validation.json", dict(status="passed", historical=historical,
        raw_artifacts_verified=len({v["path"] for v in manifest}), all_initial_pairing_verified=True,
        controlled_metrics_recomputed=True, stopped_at_25=True, both_seeds_reported=True,
        controlled_graphics_error_jobs=sum(bool(e["error_log_lines"]) for e in executions if e["kind"].startswith("controlled"))))
    print(json.dumps(dict(rows=summary, gates=gates), indent=2))


if __name__ == "__main__":
    main()
