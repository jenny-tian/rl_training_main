"""Qualify explicitly declared final checkpoints using paired stage-1 evidence (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
from ea_lpl_p0_summary import load, resolve_run  # noqa: E402
from rl_training.ea_lpl.specs import stage1_specs  # noqa: E402
from rl_training.ea_lpl.stage1 import CONDITIONS, LEVELS, choose_seed, family_table, validate_records, verdicts  # noqa: E402
from rl_training.ea_lpl.summary import condition_table  # noqa: E402


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(evidence, name, actor, *, kind):
    """Resolve exactly one passed attempt per expected label, preserving reset provenance."""
    records = {level: [] for level in LEVELS}
    runs = {}
    for expected in stage1_specs({name: actor}, kind=kind):
        label, level = expected["label"], expected["level"]
        folder = resolve_run(evidence, label)
        if folder is None:
            raise ValueError(f"missing passed attempt: {label}")
        spec, manifest, report = (read_json(folder / f"{part}.json") for part in ("spec", "manifest", "result"))
        if json.dumps(spec, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise ValueError(f"{label}: spec differs from the declared policy/design")
        if (manifest["schema"] != "ea_lpl_run_manifest_v1" or manifest["label"] != label
                or manifest["spec_sha256"] != sha256(folder / "spec.json")
                or manifest["actor_sha256"] != actor["sha256"] or manifest["actor_path"] != actor["path"]):
            raise ValueError(f"{label}: manifest/spec/actor identity mismatch")
        case = load(folder)
        if (report["label"] != label or report["level"] != level or report["slots"] != spec["slots"]
                or case["label"] != label or case["codes"] is not None):
            raise ValueError(f"{label}: result identity mismatch")
        assignments = case["initial"]["assignments"]
        initial_sha = case["initial"]["initial_state_sha256"]
        if not isinstance(initial_sha, str) or re.fullmatch(r"[0-9a-f]{64}", initial_sha) is None:
            raise ValueError(f"{label}: invalid initial-state hash")
        if not isinstance(assignments, list) or len(assignments) != 120:
            raise ValueError(f"{label}: expected 120 initial assignments")
        for env_id, assignment in enumerate(assignments):
            column = spec["columns"][env_id]
            if (assignment["env_id"] != env_id or assignment["column"] != column
                    or assignment["subterrain"] != CONDITIONS[column]):
                raise ValueError(f"{label}: initial terrain assignment mismatch")
        if not isinstance(case["records"], list) or len(case["records"]) != 120:
            raise ValueError(f"{label}: expected exactly 120 first-episode records")
        env_ids = set()
        for record in case["records"]:
            env_id = record["env_id"]
            if type(env_id) is not int or not 0 <= env_id < 120 or env_id in env_ids:
                raise ValueError(f"{label}: duplicate or invalid env_id")
            env_ids.add(env_id)
            if (record["episode_index"] != 0 or record["column"] != spec["columns"][env_id]
                    or record["terrain"] != assignments[env_id]["subterrain"]):
                raise ValueError(f"{label}: record/assignment identity mismatch")
            records[level].append(dict(record, namespace=spec["namespace"], slot=spec["slots"][env_id]))
        runs[label] = dict(folder=str(folder), level=level, slots=spec["slots"],
                           reset_table_seed=spec["reset_table_seed"], initial_state_sha256=initial_sha,
                           spec_sha256=manifest["spec_sha256"])
    for level in LEVELS:
        validate_records(records[level])
    return dict(records=records, runs=runs)


def check_pairing(candidate, baseline):
    for level in LEVELS:
        def keys(policy):
            return {(r["namespace"], r["slot"], r["episode_index"], r["terrain"])
                    for r in policy["records"][level]}
        if keys(candidate) != keys(baseline):
            raise ValueError(f"{level}: baseline and policy reset keys/conditions differ")
    def starts(policy):
        return {(r["level"], r["reset_table_seed"], tuple(r["slots"])): r["initial_state_sha256"]
                for r in policy["runs"].values()}
    if starts(candidate) != starts(baseline):
        raise ValueError("baseline and policy initial-state hashes differ on matched reset keys")


def load_retention(path, actor):
    report = read_json(path)
    if (report["status"] != "passed" or report["checkpoint_sha256"] != actor["sha256"]
            or str(Path(report["checkpoint"]).resolve()) != actor["path"]
            or report["task"] != "Flat-Deeprobotics-Lite3-v0" or report["num_envs"] != 1024
            or report["seed"] != 20261109 or report["requested_steps"] != 3000 or report["completed_steps"] != 3000):
        raise ValueError("retention report is incomplete or differs from the declared actor/Flat evaluation")
    return report["metrics"]


def markdown(report):
    lines = ["# Stage-1 qualification", "", f"Design: {report['kind']}; declared final checkpoints only.", "",
             f"Selected seed: {report['selected_seed'] if report['selected_seed'] is not None else 'none'}.",
             report["selection_reason"], "", "| Policy | Q1 | Q2 | Q3 | Q4 | L2 completion | Total safety |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, policy in report["policies"].items():
        q = ["PASS" if policy["verdicts"][f"Q{i}"]["pass"] else "FAIL" for i in range(1, 5)]
        lines.append(f"| {name} | {' | '.join(q)} | {policy['l2_completion_rate']} | {policy['total_safety_endings']} |")
    for name, policy in report["policies"].items():
        for error in policy["errors"]:
            lines.extend(["", f"{name}: {error}"])
    for name, policy in report["policies"].items():
        lines.extend(["", f"## {name}", "", f"Actor: `{policy['actor']['path']}`", "",
                      f"SHA-256: `{policy['actor']['sha256']}`", "",
                      "| Level | Family | Completed/n | Rate | 95% CI | Safety | Step-weighted projection |",
                      "| --- | --- | --- | --- | --- | --- | --- |"])
        for level, rows in policy["families"].items():
            for row in rows:
                lines.append(f"| {level} | {row['family']} | {row['completed']}/{row['n']} | {row['rate']:.4f} | "
                             f"{row['ci_low']:.4f}–{row['ci_high']:.4f} | {row['safety']} | {row['projected_fraction']:.6f} |")
        lines.extend(["", "| Level | Condition | Completed/n | 95% CI | Safety | Progress m | Peak lateral m | Abs yaw deg |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"])
        for rows in policy["conditions"].values():
            for row in rows:
                lines.append(f"| {row['level']} | {row['terrain']} | {row['completed']}/{row['n']} | "
                             f"{row['ci_low']:.4f}–{row['ci_high']:.4f} | {row['safety']} | {row['progress_mean_m']:.4f} | "
                             f"{row['lateral_peak_mean_m']:.4f} | {row['terminal_abs_yaw_mean_deg']:.4f} |")
        lines.extend(["", "Q4 detail: `" + json.dumps(policy["verdicts"]["Q4"]["detail"], sort_keys=True) + "`"])
    return "\n".join(lines) + "\n"


def summarize(evidence, actors, retention_paths, *, kind, baseline_name="baseline", seed_names=None):
    seed_names = seed_names or {"seed42": 42, "seed43": 43}
    loaded, retention, errors = {}, {}, {name: [] for name in actors}
    for name, actor in actors.items():
        try:
            loaded[name] = load_policy(evidence, name, actor, kind=kind)
        except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
            errors[name].append(f"terrain evidence: {error}")
        try:
            retention[name] = load_retention(retention_paths[name], actor)
        except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
            errors[name].append(f"retention evidence: {error}")
    policies = {}
    for name, actor in actors.items():
        data = loaded.get(name)
        paired = False
        try:
            check_pairing(loaded[name], loaded[baseline_name])
            paired = True
        except (KeyError, ValueError) as error:
            errors[name].append(f"paired baseline evidence: {error}")
        qs = verdicts(data["records"] if data else {}, retention.get(name), retention.get(baseline_name))
        if not paired:
            for i in (1, 2, 3):
                qs[f"Q{i}"] = {"pass": False, "detail": {"error": "matched baseline starts were not verified"}}
        records = data["records"] if data else {}
        policies[name] = dict(actor=actor, seed=seed_names.get(name), verdicts=qs, paired_starts_verified=paired,
                              errors=errors[name], runs=data["runs"] if data else {},
                              families={level: family_table(rows) for level, rows in records.items()},
                              conditions={level: condition_table(rows, level=level) for level, rows in records.items()},
                              l2_completion_rate=sum(r["completed"] for r in records["L2"]) / 480 if data else None,
                              total_safety_endings=sum(r["safety_ending"] for rows in records.values() for r in rows) if data else None)
    selected, reason = choose_seed([policies[name] for name in seed_names])
    selected_name = next((name for name, seed in seed_names.items() if seed == selected), None)
    return dict(schema="ea_lpl_stage1_qualification_v1", kind=kind, baseline=baseline_name, policies=policies,
                selected_seed=selected, selected_policy=selected_name,
                selected_actor=actors[selected_name] if selected_name else None, selection_reason=reason)


def assignments(parser, items, option):
    values = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not value or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None or name in values:
            parser.error(f"{option} requires unique NAME=PATH entries")
        values[name] = Path(value).expanduser().resolve(strict=True)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--kind", choices=("qualify", "extend"), default="qualify")
    parser.add_argument("--policy", action="append", required=True, metavar="NAME=PATH",
                        help="explicit final checkpoint for each policy; no checkpoint search or intermediate selection")
    parser.add_argument("--retention", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--baseline", default="baseline", help="baseline policy name")
    parser.add_argument("--seed42", default="seed42", help="policy name for seed 42's declared final checkpoint")
    parser.add_argument("--seed43", default="seed43", help="policy name for seed 43's declared final checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    policies = assignments(parser, args.policy, "--policy")
    retention_paths = assignments(parser, args.retention, "--retention")
    names = {args.baseline, args.seed42, args.seed43}
    if len(names) != 3 or set(policies) != names or set(retention_paths) != names:
        parser.error("provide exactly the baseline, seed42 and seed43 policy and retention entries")
    actors = {name: dict(path=str(path), sha256=sha256(path)) for name, path in policies.items()}
    report = summarize(args.evidence, actors, retention_paths, kind=args.kind, baseline_name=args.baseline,
                       seed_names={args.seed42: 42, args.seed43: 43})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "qualification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "qualification.md").write_text(markdown(report))
    print(json.dumps({key: report[key] for key in ("selected_seed", "selected_policy", "selection_reason")}, indent=2))


if __name__ == "__main__":
    main()
