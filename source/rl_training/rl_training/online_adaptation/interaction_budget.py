"""Durable, single-writer transition accounting and session phase boundaries.

Physics is never replayed to recover a journal entry. A reservation whose
completion cannot be established remains charged as an uncertain transition.
"""

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .provenance import sha256_file


ADAPTATION_PHASES = {"warmup", "adaptation", "selection", "flat_selection"}
PHASES = ADAPTATION_PHASES | {"evaluation", "retention"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_manifest(manifest):
    required = {"schema", "session_id", "algorithm", "base", "task_objective_sha256",
                "terrain_split", "terrain_seed", "learner_seed", "worker_configuration", "limits", "seal"}
    if manifest.get("schema") == "lite3_session_v2":
        required.add("schedule")
    if set(manifest) != required or manifest["schema"] not in {"lite3_session_v1", "lite3_session_v2"}:
        raise ValueError("session manifest schema mismatch")
    if "schedule" in manifest:
        schedule = manifest["schedule"]
        if set(schedule) != {"evaluation_trials", "trial_steps", "retention_trials", "retention_steps", "probes", "short_fall_steps", "consecutive_short_falls"}:
            raise ValueError("session schedule schema mismatch")
        for key, value in schedule.items():
            if key != "probes":
                positive_integer(value, key)
        previous_end = 0
        total = 0
        if not isinstance(schedule["probes"], list):
            raise ValueError("probe schedule must be a list")
        for probe in schedule["probes"]:
            if set(probe) != {"at", "phase", "max_steps"} or probe["phase"] not in {"selection", "flat_selection"}:
                raise ValueError("invalid probe schedule")
            positive_integer(probe["max_steps"], "probe steps")
            if type(probe["at"]) is not int or probe["at"] < previous_end:
                raise ValueError("probe schedule overlaps")
            previous_end = probe["at"] + probe["max_steps"]
            total += probe["max_steps"]
        if total > manifest["limits"]["selection"] or previous_end > manifest["limits"]["adaptation"]:
            raise ValueError("scheduled probes exceed shared budgets")
    if manifest["algorithm"] not in {"lora", "sac"}:
        raise ValueError("unsupported session algorithm")
    if manifest["terrain_split"] not in {"development", "sealed"}:
        raise ValueError("unsupported session terrain split")
    if not isinstance(manifest["worker_configuration"], dict) or not manifest["session_id"]:
        raise ValueError("session requires an identity and locked worker configuration")
    for key in ("terrain_seed", "learner_seed"):
        if type(manifest[key]) is not int or manifest[key] < 0:
            raise ValueError("session seeds must be nonnegative integers")
    if set(manifest["limits"]) != {"adaptation", "selection", "evaluation", "retention"}:
        raise ValueError("session requires all budget limits")
    for key, value in manifest["limits"].items():
        if key == "selection" and type(value) is int and value == 0:
            continue
        positive_integer(value, key)
    if manifest["limits"]["selection"] > manifest["limits"]["adaptation"]:
        raise ValueError("selection is part of the adaptation budget")
    for value in (manifest["base"]["sha256"], manifest["task_objective_sha256"]):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("session hashes must be SHA-256 digests")
    if manifest["terrain_split"] == "sealed":
        seal = manifest["seal"]
        if not isinstance(seal, dict) or set(seal) != {"status", "protocol_sha256", "configuration_sha256", "terrain_seeds", "selection_rule_sha256"}:
            raise ValueError("sealed execution requires an explicit locked final protocol")
        if seal["status"] != "locked" or seal["configuration_sha256"] != digest(manifest["worker_configuration"]):
            raise ValueError("sealed algorithm configuration is not locked")
        if manifest["terrain_seed"] not in seal["terrain_seeds"]:
            raise ValueError("terrain seed is outside the sealed protocol")
        for key in ("protocol_sha256", "selection_rule_sha256"):
            if not isinstance(seal[key], str) or len(seal[key]) != 64 or any(c not in "0123456789abcdef" for c in seal[key]):
                raise ValueError("sealed protocol contains unresolved hashes")
    elif manifest["seal"] is not None:
        raise ValueError("development sessions cannot claim a seal")


class InteractionBudget:
    """Append-only fsynced journal, exclusive ownership until close()."""

    def __init__(self, directory, *, manifest=None):
        self.directory = Path(directory)
        if manifest is not None:
            validate_manifest(manifest)
            self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = (self.directory / "writer.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.directory / "manifest.json"
            if manifest is not None:
                with path.open("x") as stream:
                    stream.write(canonical(manifest) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            self.manifest = json.loads(path.read_text())
            validate_manifest(self.manifest)
            self.state = {"phase": "adaptation", "trials": {}, "active": None, "pending": None,
                          "spent": {key: 0 for key in PHASES}, "latest_checkpoint": None,
                          "evaluation_checkpoint": None, "uncertain_steps": 0}
            self.sequence, self.previous_hash = 0, digest(self.manifest)
            self.journal = (self.directory / "events.jsonl").open("a+b")
            self._load()
            if self.sequence == 0:
                self._append({"type": "created", "manifest_sha256": digest(self.manifest)})
                descriptor = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        except BaseException:
            if hasattr(self, "journal"):
                self.journal.close()
            self.lock.close()
            raise

    def _load(self):
        self.journal.seek(0)
        good_bytes = 0
        while line := self.journal.readline():
            if not line.endswith(b"\n"):
                # A torn final write never authorizes the next physics call.
                evidence = self.directory / f"torn-tail-{good_bytes}.bin"
                evidence.write_bytes(line)
                self.journal.truncate(good_bytes)
                self.journal.flush()
                os.fsync(self.journal.fileno())
                break
            record = json.loads(line)
            signature = record.pop("sha256")
            if record["seq"] != self.sequence or record["previous"] != self.previous_hash or digest(record) != signature:
                raise ValueError("session journal hash/sequence mismatch")
            self._apply(self.state, record["event"])
            self.sequence += 1
            self.previous_hash = signature
            good_bytes += len(line)

    def _append(self, event):
        # Validate before writing. The event is durable before live state moves.
        trial_state = copy.deepcopy(self.state)
        self._apply(trial_state, event)
        record = {"seq": self.sequence, "previous": self.previous_hash, "event": event}
        signature = digest(record)
        self.journal.write((canonical({**record, "sha256": signature}) + "\n").encode())
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.state = trial_state
        self.sequence += 1
        self.previous_hash = signature

    def remaining(self, phase, state=None):
        state = self.state if state is None else state
        if phase not in PHASES:
            raise ValueError("unknown budget phase")
        spent, limits = state["spent"], self.manifest["limits"]
        if phase in ADAPTATION_PHASES:
            remaining = limits["adaptation"] - sum(spent[key] for key in ADAPTATION_PHASES)
            if phase in {"selection", "flat_selection"}:
                remaining = min(remaining, limits["selection"] - spent["selection"] - spent["flat_selection"])
            return remaining
        return limits[phase] - spent[phase]

    def _apply(self, state, event):
        kind = event["type"]
        if kind == "created":
            if event["manifest_sha256"] != digest(self.manifest):
                raise ValueError("manifest changed after session creation")
            return
        if kind == "evaluation_lock":
            if state["active"] is not None or state["phase"] != "adaptation":
                raise ValueError("evaluation can only start at a completed physical trial boundary")
            expected = state["latest_checkpoint"] or self.manifest["base"]
            if event["checkpoint"] != expected:
                raise ValueError("evaluation must bind the session's last committed candidate")
            state["phase"] = "evaluation"
            state["evaluation_checkpoint"] = event["checkpoint"]
            return
        if kind == "prepare":
            trial = event["trial"]
            phase = trial["phase"]
            if state["active"] is not None or trial["id"] in state["trials"]:
                raise ValueError("trial is already active or its ID has been consumed")
            if phase not in PHASES or (phase in ADAPTATION_PHASES) != (state["phase"] == "adaptation"):
                raise ValueError("frozen evaluation cannot return to adaptation or selection")
            schedule = self.manifest.get("schedule")
            if schedule is not None:
                horizon = schedule["retention_steps"] if phase in {"retention", "flat_selection"} else schedule["trial_steps"]
                if trial["max_steps"] > horizon:
                    raise ValueError("trial exceeds the locked physical horizon")
                if phase in {"evaluation", "retention"} and sum(t["phase"] == phase for t in state["trials"].values()) >= schedule[phase + "_trials"]:
                    raise ValueError("fixed evaluation starts exhausted; do not replace short trials")
                if phase in ADAPTATION_PHASES and self.short_fall_stop(state):
                    raise ValueError("consecutive short falls stopped adaptation")
            positive_integer(trial["max_steps"], "max_steps")
            if trial["max_steps"] > self.remaining(phase, state):
                raise ValueError("trial exceeds remaining transition budget")
            expected = (state["evaluation_checkpoint"] if state["phase"] == "evaluation" else
                        state["latest_checkpoint"] or self.manifest["base"])
            if trial["input_checkpoint"] != expected:
                raise ValueError("trial checkpoint is outside the committed session chain")
            state["trials"][trial["id"]] = {**trial, "status": "ready", "confirmed_steps": 0,
                                           "confirmed_calls": 0, "charged_steps": 0}
            state["active"] = trial["id"]
            return
        trial = state["trials"].get(event.get("trial_id"))
        if trial is None or state["active"] != trial["id"]:
            raise ValueError("event does not belong to the active trial")
        if kind == "start":
            if trial["status"] != "ready":
                raise ValueError("trial already started; do not repeat physics")
            trial["status"] = "running"
        elif kind == "reserve":
            if trial["status"] != "running" or state["pending"] is not None:
                raise ValueError("resolve the pending transition before another physics call")
            count = event["num_envs"]
            positive_integer(count, "num_envs")
            if event["call"] != trial["confirmed_calls"] + 1:
                raise ValueError("transition call ID was already used or skipped")
            if count > self.remaining(trial["phase"], state) or trial["charged_steps"] + count > trial["max_steps"]:
                raise ValueError("transition would exceed budget")
            state["pending"] = event
            trial["charged_steps"] += count
            state["spent"][trial["phase"]] += count
        elif kind == "commit":
            pending = state["pending"]
            if pending is None or (pending["trial_id"], pending["call"]) != (trial["id"], event["call"]):
                raise ValueError("no matching transition reservation")
            trial["confirmed_steps"] += pending["num_envs"]
            trial["confirmed_calls"] += 1
            state["pending"] = None
        elif kind == "finish":
            if trial["status"] != "running" or state["pending"] is not None or event["steps"] != trial["confirmed_steps"]:
                raise ValueError("summary does not match committed transitions")
            trial.update(status="complete", outcome=event["outcome"], output_checkpoint=event["checkpoint"])
            if trial["phase"] == "adaptation" and event["checkpoint"] is not None:
                state["latest_checkpoint"] = event["checkpoint"]
            state["active"] = None
        elif kind == "abort":
            if state["pending"] is not None:
                state["uncertain_steps"] += state["pending"]["num_envs"]
                trial["uncertain_steps"] = state["pending"]["num_envs"]
                state["pending"] = None
            trial.update(status="aborted", reason=event["reason"])
            state["active"] = None
        else:
            raise ValueError("unknown journal event")

    def short_fall_stop(self, state=None):
        """Learning-trial outcomes persist across restarts; probes cannot reset the streak."""
        schedule = self.manifest.get("schedule")
        if schedule is None:
            return False
        state = self.state if state is None else state
        trials = [t for t in state["trials"].values() if t["phase"] in {"warmup", "adaptation"}]
        streak = 0
        for trial in trials:
            if (trial["status"] == "complete" and trial["outcome"].get("fell") is True
                    and trial["confirmed_steps"] < schedule["short_fall_steps"]):
                streak += 1
                if streak >= schedule["consecutive_short_falls"]:
                    return True
            else:
                streak = 0
        return False

    def prepare_trial(self, *, phase, max_steps, output, input_checkpoint):
        index = len(self.state["trials"])
        trial_id = f"trial-{index:05d}"
        phase_index = sum(t["phase"] == phase for t in self.state["trials"].values())
        reset_seed = int(digest({"reset_contract": "phase_local_v2", "learner_seed": self.manifest["learner_seed"],
                                 "phase": phase, "trial": phase_index})[:8], 16) % (2**31)
        trial = {"id": trial_id, "phase": phase, "max_steps": max_steps, "output": str(Path(output).resolve()),
                 "input_checkpoint": input_checkpoint, "reset_seed": reset_seed, "phase_index": phase_index}
        self._append({"type": "prepare", "trial": trial})
        return copy.deepcopy(self.state["trials"][trial_id])

    def start(self, trial_id):
        self._append({"type": "start", "trial_id": trial_id})

    def reserve(self, trial_id, call, num_envs=1):
        positive_integer(call, "call")
        self._append({"type": "reserve", "trial_id": trial_id, "call": call, "num_envs": num_envs})

    def commit(self, trial_id, call):
        positive_integer(call, "call")
        trial = self.state["trials"][trial_id]
        if 1 <= call <= trial["confirmed_calls"]:
            return  # idempotent acknowledgement, never permission to repeat step()
        self._append({"type": "commit", "trial_id": trial_id, "call": call})

    def finish(self, trial_id, *, steps, outcome, checkpoint=None):
        if checkpoint is not None and sha256_file(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("output checkpoint checksum mismatch")
        self._append({"type": "finish", "trial_id": trial_id, "steps": steps, "outcome": outcome, "checkpoint": checkpoint})

    def abort_active(self, reason):
        if self.state["active"] is not None:
            self._append({"type": "abort", "trial_id": self.state["active"], "reason": reason})

    def lock_evaluation(self):
        checkpoint = self.state["latest_checkpoint"] or self.manifest["base"]
        if sha256_file(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("selected checkpoint changed before frozen evaluation")
        self._append({"type": "evaluation_lock", "checkpoint": checkpoint})

    def close(self):
        self.journal.close()
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
