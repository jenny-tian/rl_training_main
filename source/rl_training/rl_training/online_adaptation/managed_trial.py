"""Bind actual worker invocations to a durable session trial."""

from pathlib import Path

from .checkpointing import capture_rng_state, restore_rng_state
from .interaction_budget import InteractionBudget, canonical, digest
from .provenance import sha256_file
from .task_objective import resolve_task_objective


VARIABLE_ARGUMENTS = {
    "session_ledger", "session_run_id", "output", "run_dir", "checkpoint_in",
    "experiment_checkpoint_in", "experiment_checkpoint_out", "max_steps", "device", "kit_args",
    "headless", "evaluation_only", "mode", "video", "video_length", "enable_cameras",
}


def worker_configuration(args):
    # Include actual source content, not a HEAD that can hide uncommitted edits.
    package = Path(__file__).resolve().parent
    root = package.parents[3]
    paths = sorted(package.glob("*.py")) + sorted((root / "scripts/reinforcement_learning/online_adaptation").glob("*.py"))
    return {"arguments": {key: value for key, value in vars(args).items() if key not in VARIABLE_ARGUMENTS},
            "source_sha256": digest({str(p.relative_to(root)): sha256_file(p) for p in paths})}


def add_session_args(parser):
    parser.add_argument("--session-ledger", default=None)
    parser.add_argument("--session-run-id", default=None)


class ManagedTrial:
    def __init__(self, args, algorithm):
        self.budget = InteractionBudget(args.session_ledger)
        self.saved_rng = None
        try:
            manifest = self.budget.manifest
            trial = self.budget.state["trials"].get(args.session_run_id)
            if trial is None or trial["status"] != "ready":
                raise ValueError("worker trial is not ready; refuse duplicate execution")
            self.trial = trial
            self.frozen = trial["phase"] in {"selection", "flat_selection", "evaluation", "retention"}
            if (args.terrain_split, args.terrain_seed, args.seed) != (manifest["terrain_split"], manifest["terrain_seed"], manifest["learner_seed"]):
                raise ValueError("worker terrain/reset identity differs from declared session scope")
            if algorithm != manifest["algorithm"] or canonical(worker_configuration(args)) != canonical(manifest["worker_configuration"]):
                raise ValueError("worker/source configuration differs from locked session")
            if args.max_steps != trial["max_steps"]:
                raise ValueError("worker budget differs from reserved trial")
            objective = resolve_task_objective(args)
            if objective is None or objective.sha256() != manifest["task_objective_sha256"]:
                raise ValueError("managed worker requires the locked common task objective")
            if sha256_file(args.base_checkpoint) != manifest["base"]["sha256"]:
                raise ValueError("base checkpoint changed")
            expected = trial["input_checkpoint"]
            checkpoint = args.checkpoint_in if algorithm == "lora" else args.experiment_checkpoint_in
            actual = manifest["base"] if checkpoint is None else {"path": str(Path(checkpoint).resolve()), "sha256": sha256_file(checkpoint)}
            if actual != expected:
                raise ValueError("worker input checkpoint differs from the session chain")
            if args.evaluation_only != self.frozen:
                raise ValueError("worker learning mode differs from the registered phase")
            if algorithm == "sac":
                expected_mode = ("base" if checkpoint is None else "residual_frozen") if self.frozen else "online"
                if args.mode != expected_mode or args.update_schedule != "control-step":
                    raise ValueError("managed SAC requires phase-correct mode and control-step updates")
                if self.frozen and args.stochastic_control:
                    raise ValueError("frozen evaluation cannot explore")
            output = args.output if algorithm == "lora" else args.run_dir
            if str(Path(output).resolve()) != trial["output"]:
                raise ValueError("worker output directory differs from registered trial")
            if algorithm == "sac" and not self.frozen and (
                args.experiment_checkpoint_out is None
                or Path(args.experiment_checkpoint_out).resolve() != Path(output).resolve() / "session.pt"
            ):
                raise ValueError("managed checkpoint output must stay in its registered trial directory")
            self.output = Path(output)
            self.algorithm = algorithm
            self.budget.start(trial["id"])
        except BaseException:
            self.budget.close()
            raise

    @property
    def reset_seed(self):
        return self.trial["reset_seed"]

    def reserve(self, step):
        self.budget.reserve(self.trial["id"], step)

    def commit(self, step):
        self.budget.commit(self.trial["id"], step)

    def remember_rng(self):
        self.saved_rng = capture_rng_state()

    def restore_rng(self):
        if self.saved_rng is not None:
            restore_rng_state(self.saved_rng)

    def snapshot(self):
        current = self.budget.state["trials"][self.trial["id"]]
        return {"session_id": self.budget.manifest["session_id"], "trial_id": current["id"],
                "phase": current["phase"], "reset_seed": current["reset_seed"],
                "confirmed_steps": current["confirmed_steps"], "charged_steps": current["charged_steps"],
                "journal_head_before_finish": self.budget.previous_hash}

    def finish(self, summary, checkpoint=None):
        if self.frozen and (summary.get("updates", 0) or summary.get("critic_updates", 0) or summary.get("replay_transitions_written", 0)):
            raise ValueError("frozen trial mutated learning state")
        artifact = None if checkpoint is None else {"path": str(Path(checkpoint).resolve()), "sha256": sha256_file(checkpoint)}
        self.budget.finish(self.trial["id"], steps=summary["steps"], checkpoint=artifact,
                           outcome={key: summary.get(key) for key in (
                               "passed", "fell", "stop_reason", "task_stop_reason", "reset_count",
                               "updates", "actor_updates", "critic_updates", "update_attempts",
                           )})

    def close(self):
        # Do not append after a failed write. Recovery reopens the durable log.
        self.budget.close()


def attach_session(args, algorithm):
    if bool(args.session_ledger) != bool(args.session_run_id):
        raise ValueError("session ledger and trial ID must be provided together")
    if not args.session_ledger:
        if args.task_objective != "legacy" and args.terrain_split == "sealed":
            raise ValueError("sealed common-objective execution requires a locked session protocol")
        return None
    return ManagedTrial(args, algorithm)
