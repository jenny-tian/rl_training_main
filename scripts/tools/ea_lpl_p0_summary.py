"""Summarize P0 runs, check the archived reproduction and select P1 targets by the registered rule."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training")]
from rl_training.ea_lpl.summary import condition_table, paired_mcnemar, select_target_conditions  # noqa: E402

ARCHIVE = ROOT / "docs/lite3-source-full-episode-comparison-20261105/raw/runtime/result.json"
RUNS = ("p0_e0n_l1", "p0_e0p_l1", "p0_e1p_l1", "p0_e1p_l2", "p0_e0p_l2")


def resolve_run(evidence, label):
    """Return the first passed attempt folder for a label (<label>, then <label>_retry1), or None."""
    for name in (label, f"{label}_retry1"):
        result = evidence / name / "result.json"
        if result.exists() and json.loads(result.read_text())["status"] == "passed":
            return evidence / name
    return None


def load(folder):
    report = json.loads((folder / "result.json").read_text())
    if report["status"] != "passed":
        raise ValueError(f"{folder} did not pass")
    return report["case"]


def reproduction(case):
    archived = json.loads(ARCHIVE.read_text())["cases"][0]
    if archived["label"] != "original":
        raise ValueError("archived original case not found")
    old = {r["env_id"]: (bool(r["completed"]), r["steps"]) for r in archived["first_episodes"]}
    new = {r["env_id"]: (bool(r["completed"]), r["steps"]) for r in case["records"]}
    mismatched = sorted(i for i in old if old[i] != new.get(i))
    return dict(archived_completed=sum(v[0] for v in old.values()), reproduced_completed=sum(v[0] for v in new.values()),
                identical=not mismatched, mismatched_env_ids=mismatched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folders = {name: resolve_run(args.evidence, name) for name in RUNS}
    cases = {name: load(folder) for name, folder in folders.items() if folder is not None}
    for required in ("p0_e0n_l1", "p0_e0p_l1", "p0_e1p_l1"):
        if required not in cases:
            raise ValueError(f"missing required run {required}")
    tables = {name: condition_table(case["records"], level=name[-2:].upper()) for name, case in cases.items()}
    heading_effect = {}
    for level in ("l1", "l2"):
        off, on = cases.get(f"p0_e0p_{level}"), cases.get(f"p0_e1p_{level}")
        if off and on:
            heading_effect[level] = paired_mcnemar(off["records"], on["records"])
    hold_rows = tables["p0_e1p_l1"] + tables.get("p0_e1p_l2", [])
    eligible, chosen = select_target_conditions(hold_rows)
    summary = dict(run_folders={name: folder.name for name, folder in folders.items() if folder is not None},
                   reproduction=reproduction(cases["p0_e0n_l1"]), tables=tables, heading_effect=heading_effect,
                   eligible=eligible, targets=chosen)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "conditions.csv").open("w", newline="") as stream:
        fields = ["run"] + list(tables["p0_e1p_l1"][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, rows in tables.items():
            for row in rows:
                writer.writerow(dict(row, run=name))
    (args.out / "p1_inputs.json").write_text(json.dumps(dict(schema="ea_lpl_p1_inputs_v1", targets=chosen),
                                                        indent=2) + "\n")
    print(json.dumps(dict(run_folders=summary["run_folders"], reproduction=summary["reproduction"],
                          heading_effect=heading_effect, targets=chosen), indent=2))


if __name__ == "__main__":
    main()
