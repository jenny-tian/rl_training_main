"""Validate the exact B1 pilot budget and summarize per-family/direction exposure."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from rl_training.source_curriculum.allocation import targeted_column_counts
from rl_training.source_curriculum.rows import COLUMN_SPECS, OMNI_COLUMNS


def summarize(run_dir, runtime):
    execution = json.loads((runtime / "execution.json").read_text())
    if execution["status"] != "completed" or execution["exit_code"] != 0 or execution["fatal_messages"]:
        raise ValueError("pilot did not exit healthy")
    start = ROOT / "logs/rsl_rl/deeprobotics_lite3_source_curriculum/2026-11-10_16-00-10_full_seed42/model_3497.pt"
    final = run_dir / "model_3526.pt"
    source = torch.load(start, map_location="cpu", weights_only=True)
    result = torch.load(final, map_location="cpu", weights_only=True)
    assert source["iter"] == 3497 and result["iter"] == 3526
    changes = {}
    for name in ("actor_state_dict", "critic_state_dict"):
        assert all(bool(torch.isfinite(value).all()) for value in result[name].values())
        changes[name] = max(float((source[name][key] - value).abs().max()) for key, value in result[name].items())
        assert changes[name] > 0
    initial_optimizer = source["optimizer_state_dict"]["state"]
    final_optimizer = result["optimizer_state_dict"]["state"]
    assert initial_optimizer.keys() == final_optimizer.keys()
    optimizer_increments = {int(final_optimizer[key]["step"] - initial_optimizer[key]["step"]) for key in initial_optimizer}
    assert optimizer_increments == {600}  # 30 updates * 5 epochs * 4 minibatches
    assert all(bool(torch.isfinite(value).all()) for state in final_optimizer.values()
               for value in state.values() if isinstance(value, torch.Tensor))
    events = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    events.Reload()
    tags = events.Tags()["scalars"]
    assert [e.step for e in events.Scalars("Loss/value")] == list(range(3497, 3527))
    assert all(math.isfinite(e.value) for tag in tags for e in events.Scalars(tag))
    times = [a.value + b.value for a, b in zip(events.Scalars("Perf/collection_time"), events.Scalars("Perf/learning_time"))]
    counts = json.loads((run_dir / "column_statistics.json").read_text())
    assert counts["schema"] == "source_block_l2_counts_v1"
    assert counts["environment_column_counts"] == targeted_column_counts(4096)
    transitions = torch.tensor(counts["transitions_by_column_direction"])
    assert torch.equal(transitions.sum(1), torch.tensor(targeted_column_counts(4096)) * 720)
    assert int(transitions.sum()) == 2_949_120
    starts = torch.tensor(counts["starts_by_column_replica_direction"])
    ends = torch.tensor(counts["endings_by_column_replica_direction_outcome"])
    assert int(starts.sum()) == int(ends.sum()) + 4096
    summaries = []
    groups = [(family, [i for i, (_, spec) in enumerate(COLUMN_SPECS) if spec["family"] == family])
              for family in ("flat", "multi_scale_rough", "slope", "multi_step_stairs", "slope_plus_rough", "friction_variation")]
    groups += [("stairs_ascending", [4]), ("stairs_descending", [5]), ("omni_flat", list(OMNI_COLUMNS))]
    for name, cols in groups:
        outcomes = ends[cols].sum((0, 1, 2)).tolist()
        n = int(transitions[cols].sum())
        summaries.append(dict(name=name, environments=sum(counts["environment_column_counts"][i] for i in cols),
                              transitions=n, physical_l2_fraction=1., transitions_per_training_second=n / sum(times),
                              starts=int(starts[cols].sum()), completion=outcomes[0], failure=outcomes[1], timeout=outcomes[2],
                              completed_fraction=outcomes[0] / sum(outcomes) if sum(outcomes) else None,
                              transitions_by_direction=transitions[cols].sum(0).tolist()))
    return dict(status="passed_numerical_and_budget_checks", task="SourceBlock-L2-Deeprobotics-Lite3-v0",
                policy_qualified=False, updates=30, first_iteration=3497, last_iteration=3526,
                num_envs=4096, steps_per_update=24, transitions=2_949_120, physical_l2_fraction=1.,
                checkpoint=str(final.resolve()), checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                checkpoint_disposition="pilot only; excluded from B2 starting policies and qualification candidates",
                initial_checkpoint_sha256=hashlib.sha256(start.read_bytes()).hexdigest(),
                optimizer_step_increments=sorted(optimizer_increments), maximum_parameter_changes=changes,
                nonfinite_metrics=0, training_seconds=sum(times), mean_update_s=statistics.mean(times),
                transitions_per_training_second=2_949_120 / sum(times), wall_seconds=execution["elapsed_s"],
                peak_total_gpu_used_mib=execution["peak_total_gpu_used_mib"], source_commit=execution["source_commit"],
                groups=summaries, omni_command_categories=counts["omni_command_categories"],
                omni_command_steps=counts["omni_command_steps"], termination_counts=counts["termination_counts"],
                stair_edges_at_end=counts["stair_edges_at_end"],
                final_metrics={tag: events.Scalars(tag)[-1].value for tag in tags})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run, args.runtime)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("final_metrics", "termination_counts")}, indent=2))
