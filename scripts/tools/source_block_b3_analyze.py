"""Package the frozen B3 matrix, verify pairing, and plot actual responses."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from source_block_b3_protocol import CASES, FIELDS, REPEATS, ROOT, first_episode_masks, models, schedule

RAW = ROOT / "logs/stage1/block_b3/matrix"
OUT = ROOT / "docs/lite3-block-b3-diagnosis-20261111"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = models()
    labels = ["Original 1998", "Original 3497", "Flat +100 / 42", "Flat +100 / 43", "Terrain +100 / 42", "Terrain +100 / 43"]
    colors = ["#202020", "#2673b8", "#de6627", "#e0a22b", "#288b54", "#9a55ac"]
    manifests, all_rows, episodes, audits, executions, comparisons = [], [], [], [], [], []
    pairing = {}
    response_data = {}
    for domain in ("flat", "block"):
        response_data[domain] = []
        for mode in ("controlled", "native"):
            reference = None
            for spec in specs:
                directory = RAW / domain / mode / spec["name"]
                report = json.loads((directory / "report.json").read_text())
                execution = json.loads((directory / "execution.json").read_text())
                assert report["status"] == "passed" and not report["probe"] and report["actor_unchanged"]
                assert report["cleanup"] == {"environment": "returned", "application": "returned"}
                assert execution["status"] == "completed" and execution["exit_code"] == 0 and not execution["fatal_messages"]
                if reference is None:
                    reference = report["initial_hashes"]
                assert report["initial_hashes"] == reference, f"unpaired initial state: {directory}"
                pairing[f"{domain}/{mode}"] = dict(models=6, all_initial_fields_exact=True, hashes=reference)
                destination = OUT / "runs" / domain / mode / spec["name"]
                destination.mkdir(parents=True, exist_ok=True)
                for filename in ("report.json", "execution.json", "config.json"):
                    shutil.copyfile(directory / filename, destination / filename)
                executions.append(dict(model=spec["name"], domain=domain, mode=mode,
                    elapsed_s=execution["elapsed_s"], peak_gpu_mib=execution["peak_total_gpu_used_mib"],
                    source_commit=execution["source_commit"], status=execution["status"],
                    error_log_lines=[line for line in (directory / "runtime.log").read_text(errors="replace").splitlines()
                                     if "[Error]" in line]))
                for row in report["metrics"]["rows"]:
                    record = dict(model=spec["name"], domain=domain, mode=mode,
                        **{k: v for k, v in row.items() if k not in ("reward_rates", "mean_command", "mean_velocity")})
                    record.update({f"mean_cmd_{a}": row["mean_command"][i] for i, a in enumerate(("x", "y", "yaw"))})
                    record.update({f"mean_vel_{a}": row["mean_velocity"][i] for i, a in enumerate(("x", "y", "yaw"))})
                    record.update({f"reward_{k}": v for k, v in row["reward_rates"].items()})
                    all_rows.append(record)
                for group, row in report["metrics"]["episodes"].items():
                    episodes.append(dict(model=spec["name"], domain=domain, mode=mode, group=group, **row))
                loaded = np.load(directory / "trace.npz")
                trace = loaded["trace"]
                first, valid = first_episode_masks(loaded["ended"])
                if mode == "controlled":
                    assert report["num_envs"] == 512
                    assert all(e["starts"] == 32 for e in report["metrics"]["episodes"].values())
                    # Retain only the fields needed by response plots and paired comparisons.
                    response_data[domain].append((trace[..., :len(FIELDS)].copy(), valid.copy()))
                else:
                    columns = loaded["columns"]
                    group_masks = {"all": np.ones(report["num_envs"], bool)} if domain == "flat" else {
                        "dedicated_turn_c09": columns == 9, "other_omni_c10_12": np.isin(columns, [10, 11, 12]),
                        "straight_flat_c13": columns == 13}
                    for group, group_mask in group_masks.items():
                        data = trace[valid & group_mask[None]]
                        cmd = data[:, :3]
                        pure = np.linalg.norm(cmd[:, :2], axis=-1) < 1e-6
                        yaw = abs(cmd[pure, 2])
                        old_rate = data[:, len(FIELDS)+report["reward_names"].index("stand_still")]
                        candidate_rate = old_rate * (np.linalg.norm(cmd, axis=-1) < .1)
                        audits.append(dict(domain=domain, model=spec["name"], group=group, valid_samples=len(data),
                            pure_turn_samples=int((pure & (abs(cmd[:, 2]) >= 1e-6)).sum()),
                            pure_low_yaw_samples=int((pure & (abs(cmd[:, 2]) >= 1e-6) & (abs(cmd[:, 2]) < .2)).sum()),
                            heading_fraction=float(data[:, 7].mean()), standing_fraction=float(data[:, 8].mean()),
                            pure_yaw_min=float(yaw.min()) if len(yaw) else None,
                            pure_yaw_max=float(yaw.max()) if len(yaw) else None,
                            positive_yaw_samples=int((cmd[:, 2] > 1e-6).sum()),
                            negative_yaw_samples=int((cmd[:, 2] < -1e-6).sum()),
                            stand_still_rate=float(old_rate.mean()),
                            candidate_rate_change_on_same_states=float((candidate_rate-old_rate).mean()),
                            scan_invalid_fraction=float(data[:, 15].mean())))
                for filename in ("report.json", "config.json", "execution.json", "runtime.log", "initial.npz", "trace.npz"):
                    path = directory / filename
                    manifests.append(dict(path=str(path), bytes=path.stat().st_size,
                        sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest()))
                del trace, loaded

    cross_domain = {}
    for spec in specs:
        a = np.load(RAW / "flat/controlled" / spec["name"] / "initial.npz")
        b = np.load(RAW / "block/controlled" / spec["name"] / "initial.npz")
        delta = float(abs((a["root"][:, :3]-a["origins"])-(b["root"][:, :3]-b["origins"])).max())
        assert delta <= 1e-5
        assert np.array_equal(a["observation"], b["observation"])
        assert np.array_equal(a["materials"], b["materials"])
        cross_domain[spec["name"]] = dict(max_relative_position_delta_m=delta,
            observation_exact=True, material_exact=True)

    for domain, data in response_data.items():
        figure, axes = plt.subplots(4, 4, figsize=(17, 13), sharex=True)
        times = np.arange(len(data[0][0])) * .02
        for case_id, case in enumerate(CASES):
            ax = axes.flat[case_id]
            part = slice(case_id * REPEATS, (case_id+1)*REPEATS)
            index = 6 if case.startswith("heading") else 4 if case.startswith("lateral") else 3 if case == "forward" else 5
            for model_id, (trace, valid) in enumerate(data):
                mask = valid[:, part]
                values = trace[:, part, index]
                count = mask.sum(-1)
                mean = np.divide((values * mask).sum(-1), count, out=np.full(len(times), np.nan), where=count > 0)
                ax.plot(times+.02, mean, color=colors[model_id], lw=1.1, alpha=.9, label=labels[model_id])
            if not case.startswith("heading"):
                command_index = index-3
                ax.plot(times, [schedule(case, t)[command_index] for t in times], "k--", lw=1., label="Command")
            else:
                ax.axhline(0, color="k", ls="--", lw=.8)
            for t in (2, 8, 10):
                ax.axvline(t, color="gray", lw=.5, alpha=.4)
            ax.set_title(case, fontsize=11)
            ax.set_ylabel("Heading error (rad)" if index == 6 else "Velocity (m/s)" if index in (3, 4) else "Yaw rate (rad/s)")
            ax.grid(alpha=.2)
            ax.set_xlabel("Time (s)")
        figure.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=7, fontsize=10)
        figure.suptitle(f"B3 {domain}: 32 first episodes per condition; means exclude terminal/reset samples", y=.965)
        figure.tight_layout(rect=(0, 0, 1, .94))
        figure.savefig(OUT / f"{domain}_responses.png", dpi=150)
        figure.savefig(OUT / f"{domain}_responses.pdf")
        plt.close(figure)
        # Compare each new policy with its own initialization on the same surviving intervals.
        generator = np.random.default_rng(20261113)
        for model_id in range(2, 6):
            baseline_id = 0 if specs[model_id]["name"].startswith("flat") else 1
            new_trace, new_valid = data[model_id]
            old_trace, old_valid = data[baseline_id]
            for case_id, case in enumerate(CASES):
                part = slice(case_id*REPEATS, (case_id+1)*REPEATS)
                window = (times >= 4) & (times < 8)
                common = new_valid[window, part] & old_valid[window, part]
                count = common.sum(0)
                new_error = (new_trace[window, part, 5] - new_trace[window, part, 2])**2
                old_error = (old_trace[window, part, 5] - old_trace[window, part, 2])**2
                has_data = count > 0
                diff = (np.sqrt((new_error[:, has_data]*common[:, has_data]).sum(0)/count[has_data])
                        -np.sqrt((old_error[:, has_data]*common[:, has_data]).sum(0)/count[has_data]))
                if not len(diff):
                    continue
                bootstrap = diff[generator.integers(0, len(diff), (2000, len(diff)))].mean(-1)
                comparisons.append(dict(domain=domain, model=specs[model_id]["name"],
                    baseline=specs[baseline_id]["name"], case=case, window="4-8s", paired_starts=len(diff),
                    mean_per_episode_yaw_rms_difference=float(diff.mean()),
                    bootstrap_95_low=float(np.quantile(bootstrap, .025)), bootstrap_95_high=float(np.quantile(bootstrap, .975)),
                    limitation="conditional on both policies remaining in first episode; failures reported separately"))

    exposure = []
    for path in sorted((ROOT / "docs/lite3-block-b2-initialization-20261111/statistics").glob("*.json")):
        data = json.loads(path.read_text())
        transitions = np.array(data["transitions_by_column_direction"]).sum(-1)
        exposure.append(dict(run=path.stem, total_steps=int(transitions.sum()),
            dedicated_turn_steps=int(transitions[9]), other_omni_steps=int(transitions[10:13].sum()),
            command_categories=data["omni_command_categories"], command_steps=data["omni_command_steps"],
            definition="B2 categories use linear norm .1 and yaw .1 thresholds; categories can overlap; not B3 exact-axis bins"))
    table(OUT / "metrics.csv", all_rows)
    table(OUT / "episodes.csv", episodes)
    table(OUT / "paired_comparisons.csv", comparisons)
    save(OUT / "summary.json", dict(status="passed", frozen_jobs=24,
        controlled_first_episode_starts=6*2*16*32, native_first_episode_starts=6*2*1024,
        requested_interactions=6*(512*1000*2+1024*1000+1024*1500),
        pairing=pairing, cross_domain_initial_checks=cross_domain, executions=executions,
        command_audits=audits, training_exposure=exposure))
    save(OUT / "artifact_manifest.json", manifests)
    print(json.dumps(dict(jobs=len(executions), metric_rows=len(all_rows), episodes=len(episodes), pairing="passed")))


if __name__ == "__main__":
    main()
