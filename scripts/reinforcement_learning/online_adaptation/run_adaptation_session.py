"""Run/resume finite-horizon physical trials under one immutable step ledger.

This development runner selects the last completed candidate. Final sealed
seed generation and experiment selection remain separate protocol work.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

from rl_training.online_adaptation.interaction_budget import InteractionBudget, canonical
from rl_training.online_adaptation.managed_trial import worker_configuration
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.task_objective import FiniteHorizonObjective
from rl_training.online_adaptation.session_schedule import build_schedule, next_trial


def parser_for(algorithm):
    if algorithm == "lora":
        from scripts.reinforcement_learning.online_adaptation.train_lora_ppo import build_parser
        return build_parser()
    from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser
    return build_play_parser()


def worker_arguments(args, *, output, steps, checkpoint=None, trial_id=None, phase=None):
    phase = args.phase if phase is None else phase
    frozen = phase in {"selection", "flat_selection", "evaluation", "retention"}
    flat = phase in {"flat_selection", "retention"}
    common = ["--base-checkpoint", str(Path(args.base_checkpoint).resolve()),
              "--task", "RockyRoad-Deeprobotics-Lite3-v0", "--terrain-stage", "rough_4cm",
              "--terrain-seed", str(args.terrain_seed), "--terrain-split", "development",
              "--seed", str(args.seed), "--forward-speed", "0.3", "--max-steps", str(steps),
              "--task-objective", "finite-horizon-v1", "--task-deadline-steps", str(args.deadline_steps),
              "--initial-state", "perturbed-v1", "--trajectory-dtype", "float32", "--residual-rate-limit", "0.03",
              "--target-warmup-steps", str(args.target_warmup_steps),
              "--headless", "--device", args.device,
              f"--kit_args=--/renderer/activeGpu={args.device.split(':')[-1]} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"]
    common += ["--flat-retention-probe" if flat else "--require-track-completion"]
    if args.algorithm == "lora":
        common += ["--output", str(output)]
        if checkpoint is not None:
            common += ["--checkpoint-in", checkpoint["path"]]
    else:
        mode = ("base" if checkpoint is None else "residual_frozen") if frozen else "online"
        common += ["--run-dir", str(output), "--mode", mode, "--terrain-feature-source", "proprio",
                   "--update-schedule", "control-step", "--experiment-checkpoint-out", str(output / "session.pt")]
        if checkpoint is not None:
            common += ["--experiment-checkpoint-in", checkpoint["path"]]
    if args.worker_options is not None:
        options = json.loads(Path(args.worker_options).read_text())
        allowed = ({"replay-capacity", "min-replay-size", "n-step", "initial-alpha", "actor-updates-per-transition",
                    "critic-updates-per-transition", "novelty-calibration-steps", "target-critic-warmup-updates",
                    "critic-huber-delta"} if args.algorithm == "sac" else
                   {"rollout-steps", "ppo-epochs", "minibatch-size", "actor-lr", "critic-lr", "rank"})
        if not isinstance(options, dict) or not set(options) <= allowed:
            raise ValueError("worker options contain unsupported session overrides")
        for key, value in options.items():
            common += [f"--{key}", str(value)]
    if frozen:
        common += ["--evaluation-only"]
    if trial_id is not None:
        common += ["--session-ledger", str(Path(args.session_dir).resolve() / "ledger"), "--session-run-id", trial_id]
    return common


def recover_interrupted_trial(budget):
    active = budget.state["active"]
    if active is None:
        return
    trial = budget.state["trials"][active]
    output = Path(trial["output"])
    summary_path = output / ("summary.json" if budget.manifest["algorithm"] == "lora" else "episode_summary.json")
    trace_path = output / "trajectory/manifest.json"
    if budget.state["pending"] is None and summary_path.exists() and trace_path.exists():
        summary, trace = json.loads(summary_path.read_text()), json.loads(trace_path.read_text())
        if trace["status"] == "complete" and summary["steps"] == trace["steps"] == trial["confirmed_steps"]:
            checkpoint = None
            if trial["phase"] == "adaptation":
                path = output / ("latest.pt" if budget.manifest["algorithm"] == "lora" else "session.pt")
                if not path.exists():
                    budget.abort_active("complete trace has no learner checkpoint")
                    return
                checkpoint = {"path": str(path), "sha256": sha256_file(path)}
            budget.finish(active, steps=summary["steps"], outcome={"reconciled_from_summary": True, "fell": summary.get("fell"), "task_stop_reason": summary.get("task_stop_reason")}, checkpoint=checkpoint)
            return
    budget.abort_active("interrupted physical trial; retain cost and resume the last completed learner")



def write_session_summary(root, budget):
    stopped = budget.short_fall_stop()
    write_json_atomic(root / "session_summary.json", {
        **budget.state, "adaptation_early_stopped": stopped,
        "adaptation_failure_reason": "five_consecutive_short_falls" if stopped else None,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--algorithm", choices=("lora", "sac"), required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--terrain-seed", type=int, default=101)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--phase", choices=("adaptation", "evaluation", "retention"), default="adaptation")
    parser.add_argument("--budget-steps", type=int, default=40000)
    parser.add_argument("--target-warmup-steps", type=int, default=4000)
    parser.add_argument("--selection-budget", type=int, default=8000)
    parser.add_argument("--evaluation-steps", type=int, default=20000)
    parser.add_argument("--evaluation-trials", type=int, default=10)
    parser.add_argument("--retention-trials", type=int, default=5)
    parser.add_argument("--retention-trial-steps", type=int, default=1000)
    parser.add_argument("--probe-milestones", type=int, nargs="*", default=[10000, 20000, 40000])
    parser.add_argument("--trial-steps", type=int, default=2000)
    parser.add_argument("--deadline-steps", type=int, default=2000)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-options", default=None)
    args = parser.parse_args()
    if not 0 <= args.target_warmup_steps <= args.budget_steps:
        parser.error("target warmup must fit within the adaptation budget")
    if (args.evaluation_trials < 1 or args.retention_trials < 1
            or not 1 <= args.retention_trial_steps <= args.deadline_steps
            or not 1 <= args.trial_steps <= args.deadline_steps) or (args.max_trials is not None and args.max_trials < 1):
        parser.error("invalid physical trial length/count")
    root = Path(args.session_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prototype = parser_for(args.algorithm).parse_args(worker_arguments(args, output=root / "prototype", steps=args.trial_steps, phase="adaptation"))
    base = {"path": str(Path(args.base_checkpoint).resolve()), "sha256": sha256_file(args.base_checkpoint)}
    expected = {"schema": "lite3_session_v2", "session_id": "", "algorithm": args.algorithm,
                "base": base, "task_objective_sha256": FiniteHorizonObjective(args.deadline_steps).sha256(),
                "terrain_split": "development", "terrain_seed": args.terrain_seed, "learner_seed": args.seed,
                "worker_configuration": worker_configuration(prototype),
                "limits": {"adaptation": args.budget_steps, "selection": args.selection_budget,
                           "evaluation": args.evaluation_steps, "retention": args.retention_trials * args.retention_trial_steps}, "seal": None,
                "schedule": build_schedule(trial_steps=args.trial_steps, evaluation_trials=args.evaluation_trials,
                    retention_trials=args.retention_trials, retention_steps=args.retention_trial_steps,
                    probe_milestones=args.probe_milestones, budget_steps=args.budget_steps)}
    if not (root / "ledger").exists():
        expected["session_id"] = str(uuid.uuid4())
        with InteractionBudget(root / "ledger", manifest=expected):
            pass
    count = 0
    while args.max_trials is None or count < args.max_trials:
        with InteractionBudget(root / "ledger") as budget:
            expected["session_id"] = budget.manifest["session_id"]
            if canonical(expected) != canonical(budget.manifest):
                raise ValueError("session configuration/source changed; preserve this session and start a new version")
            recover_interrupted_trial(budget)
            if args.phase in {"evaluation", "retention"} and budget.state["phase"] == "adaptation":
                budget.lock_evaluation()
            if args.phase == "adaptation" and budget.state["phase"] != "adaptation":
                raise ValueError("frozen evaluation cannot feed back into adaptation")
            spec = next_trial(budget, args.phase)
            if spec is None:
                write_session_summary(root, budget)
                return
            phase, steps = spec["phase"], spec["max_steps"]
            output = root / f"trial-{len(budget.state['trials']):05d}"
            candidate = budget.state["evaluation_checkpoint"] or budget.state["latest_checkpoint"] or base
            trial = budget.prepare_trial(phase=phase, max_steps=steps, output=output, input_checkpoint=candidate)
        output.mkdir(exist_ok=False)
        arguments = worker_arguments(args, output=output, steps=steps, checkpoint=None if candidate == base else candidate, trial_id=trial["id"], phase=phase)
        script = "train_lora_ppo.py" if args.algorithm == "lora" else "play_online_adaptation.py"
        command = [sys.executable, "-u", f"scripts/reinforcement_learning/online_adaptation/{script}", *arguments]
        write_json_atomic(output / "invocation.json", {"command": command, "trial": trial})
        with (output / "runtime.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        with InteractionBudget(root / "ledger") as budget:
            record = budget.state["trials"][trial["id"]]
            if record["status"] != "complete" or result.returncode not in (0, 1):
                recover_interrupted_trial(budget)
                write_session_summary(root, budget)
                raise RuntimeError(f"worker failed; costs retained: {output / 'runtime.log'}")
            write_session_summary(root, budget)
        count += 1


if __name__ == "__main__":
    main()
