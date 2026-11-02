"""Durable short-probe evidence and parent-process acceptance (standard library only)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

from .parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS


SCHEMA = "lite3_source_training_wiring_smoke_v2"


def artifact_paths(output):
    output = Path(output)
    return {"result": output, "steps": output.with_suffix(".steps.jsonl"),
            "cleanup": output.with_suffix(".cleanup.json")}


def _encoded(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")


def _sync_directory(folder):
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    """Serialize first, then fsync a sibling temporary file and atomically replace."""
    path = Path(path)
    payload = _encoded(value)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class SmokeArtifactWriter:
    def __init__(self, output, report):
        self.paths = artifact_paths(output)
        self.attempt_id = report["attempt_id"]
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("attempt_id must be nonempty")
        if any(path.exists() for path in self.paths.values()):
            raise FileExistsError("probe artifact already exists; choose a new approved attempt directory")
        payload = _encoded({**report, "schema": SCHEMA})
        self.paths["result"].parent.mkdir(parents=True, exist_ok=True)
        with self.paths["result"].open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.stream = self.paths["steps"].open("xb")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        _sync_directory(self.paths["result"].parent)
        self.count = 0

    def append_step(self, row):
        if type(row.get("step")) is not int or row["step"] != self.count + 1:
            raise ValueError("step journal must be consecutive and one-based")
        payload = _encoded({**row, "attempt_id": self.attempt_id})
        self.stream.write(payload)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.count += 1

    def persist(self, report):
        if report.get("attempt_id") != self.attempt_id:
            raise ValueError("attempt identity changed")
        atomic_json(self.paths["result"], {
            **report, "schema": SCHEMA, "steps_written": self.count,
            "steps_file": self.paths["steps"].name,
            "steps_sha256": hashlib.sha256(self.paths["steps"].read_bytes()).hexdigest(),
        })

    def close(self):
        self.stream.close()


def finalize_and_close(writer, report, *, environment_close=None, application_close=None):
    """Persist the result before either close callback, which may end the process.

    Cleanup failures change the result to failed before the next callback.
    Evidence I/O failures never prevent attempted resource cleanup. The parent
    validator rejects absent/inconsistent files even if native shutdown exits 0.
    """
    errors = []

    def save(function, *args):
        try:
            function(*args)
        except Exception:
            errors.append(traceback.format_exc())
            report.update(status="failed", persistence_errors=list(errors))

    report["phase"] = "result_before_shutdown"
    save(writer.persist, report)
    save(writer.close)
    cleanup = {"attempt_id": writer.attempt_id, "environment": "not_created", "application": "not_created",
               "persistence_errors": errors}
    for name, callback in (("environment", environment_close), ("application", application_close)):
        if callback is not None:
            cleanup[name] = "closing"
        save(atomic_json, writer.paths["cleanup"], cleanup)
        if callback is None:
            continue
        if errors:
            save(writer.persist, report)
        try:
            callback()
        except Exception:
            cleanup[name] = "failed"
            cleanup[name + "_error"] = traceback.format_exc()
            report.update(status="failed", **{name + "_cleanup_error": cleanup[name + "_error"]})
            save(writer.persist, report)
        else:
            cleanup[name] = "returned"
        save(atomic_json, writer.paths["cleanup"], cleanup)
    if errors:
        save(writer.persist, report)
        raise OSError("probe persistence failed; parent must reject incomplete evidence")


def _strict_json(text):
    def invalid(value):
        raise ValueError("non-finite JSON constant: " + value)
    return json.loads(text, parse_constant=invalid)


def _matrix(value, rows, columns, *, boolean=False):
    if not isinstance(value, list) or len(value) != rows:
        return False
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            return False
        for item in row:
            if boolean:
                if type(item) is not bool:
                    return False
            elif type(item) not in (int, float) or not math.isfinite(item):
                return False
    return True


def validate_smoke_artifacts(output, *, returncode, expected_attempt_id, expected_revision,
                             expected_checkpoint_sha256):
    """Reject stale/partial output independently of subprocess exit status."""
    paths = artifact_paths(output)
    reasons = []
    result = {"schema": "lite3_source_smoke_acceptance_v1", "accepted": False,
              "returncode": returncode, "reasons": reasons, "record_count": 0}
    if returncode != 0:
        reasons.append("child_exit_nonzero")
    try:
        report = _strict_json(paths["result"].read_text())
        rows = [_strict_json(line) for line in paths["steps"].read_text().splitlines()]
        cleanup = _strict_json(paths["cleanup"].read_text())
        result["record_count"] = len(rows)
        expected = {"schema": SCHEMA, "status": "passed", "attempt_id": expected_attempt_id,
                    "base_revision": expected_revision, "checkpoint_sha256": expected_checkpoint_sha256,
                    "num_envs": 2, "simulation_control_steps": 24, "total_transitions": 48,
                    "steps_written": 24, "ppo_updates": 0, "training_ready": False,
                    "g3_g4_qualified": False, "diagnostic_only": True,
                    "injected_timeout": True, "phase": "result_before_shutdown",
                    "ppo_seed": 130001, "geometry_seed": 30000, "sampler_seed": 131001,
                    "noise_rng_seed": 132001, "reset_state_id": 40000, "reset_randomized": False,
                    "shared_bank_rows": 1, "stochastic_std": 1.0}
        for key, value in expected.items():
            if type(report.get(key)) is not type(value) or report[key] != value:
                reasons.append("report_field:" + key)
        if report.get("steps_file") != paths["steps"].name:
            reasons.append("journal_filename_mismatch")
        if report.get("steps_sha256") != hashlib.sha256(paths["steps"].read_bytes()).hexdigest():
            reasons.append("journal_digest_mismatch")
        if len(rows) != 24:
            reasons.append("journal_length_not_24")
        if (cleanup.get("attempt_id") != expected_attempt_id or cleanup.get("environment") != "returned"
                or cleanup.get("application") not in ("closing", "returned")
                or cleanup.get("persistence_errors") != []):
            reasons.append("cleanup_incomplete_or_failed")
        result["application_close_returned"] = cleanup.get("application") == "returned"
        labels = tuple(SOURCE_RESET_BLOCK_COUNTS)
        counted = {name: {'episode_assignments': 0, 'transitions': 0} for name in labels}
        previous_columns = None
        for index, row in enumerate(rows, 1):
            prefix = "step_" + str(index) + ":"
            if row.get("step") != index or row.get("attempt_id") != expected_attempt_id:
                reasons.append(prefix + "identity_or_sequence")
            for field in ("raw_action", "base_action", "requested_action", "executed_action"):
                if not _matrix(row.get(field), 2, 12):
                    reasons.append(prefix + field)
            for field in ("projected", "paused"):
                if not _matrix(row.get(field), 2, 1, boolean=True):
                    reasons.append(prefix + field)
            if row.get("paused") != [[False], [False]]:
                reasons.append(prefix + "safety_pause")
            last = index == 24
            if row.get("done") != ([1, 0] if last else [0, 0]):
                reasons.append(prefix + "done")
            if row.get("time_outs") != ([True, False] if last else [False, False]):
                reasons.append(prefix + "timeout_mask")
            if row.get("episode_counters") != ([0, 24] if last else [index, index]):
                reasons.append(prefix + "episode_counters")
            before, after = row.get('terrain_columns_before'), row.get('terrain_columns_after')
            columns_valid = all(isinstance(columns, list) and len(columns) == 2
                                and all(type(col) is int and 0 <= col < len(labels) for col in columns)
                                for columns in (before, after))
            if not columns_valid:
                reasons.append(prefix + 'terrain_columns')
            else:
                if previous_columns is not None and previous_columns != before:
                    reasons.append(prefix + 'column_sequence')
                for env_id in range(2):
                    if index == 1:
                        counted[labels[before[env_id]]]['episode_assignments'] += 1
                    counted[labels[before[env_id]]]['transitions'] += 1
                    if last and env_id == 0:
                        counted[labels[after[env_id]]]['episode_assignments'] += 1
                    elif before[env_id] != after[env_id]:
                        reasons.append(prefix + 'unreset_column_changed')
                previous_columns = after
            terminal = row.get("terminal")
            if last:
                if (not isinstance(terminal, dict) or terminal.get("ids") != [0]
                        or terminal.get("steps") != [2000] or terminal.get("hold") != [0]
                        or terminal.get("completed") != [False]
                        or terminal.get("terminated") != [False, False]
                        or terminal.get("truncated") != [True, False]):
                    reasons.append(prefix + "terminal_snapshot")
                else:
                    observations = terminal.get("observations", {})
                    critic_width = report.get("critic_width")
                    if (not _matrix(observations.get("policy"), 2, 45)
                            or type(critic_width) is not int or critic_width < 1
                            or not _matrix(observations.get("critic"), 2, critic_width)):
                        reasons.append(prefix + "terminal_observations")
                    terms = terminal.get("terms", {})
                    required = {"source_training_timeout", "source_training_completion", "source_training_safety",
                                "source_terrain_tile_bounds", "illegal_contact", "bad_orientation_2"}
                    if (not required.issubset(terms) or any(
                            values != ([True] if name == "source_training_timeout" else [False])
                            for name, values in terms.items())):
                        reasons.append(prefix + "terminal_terms")
            elif terminal is not None:
                reasons.append(prefix + "unexpected_terminal")
        exposure = report.get("exposure", {})
        if (set(exposure) != set(SOURCE_RESET_BLOCK_COUNTS)
                or any(set(value) != {"episode_assignments", "transitions"} for value in exposure.values())
                or any(type(count) is not int or count < 0 for value in exposure.values() for count in value.values())
                or sum(value['transitions'] for value in exposure.values()) != 48
                or sum(value['episode_assignments'] for value in exposure.values()) != 3):
            reasons.append("exposure_missing_or_inconsistent")
        if exposure != counted:
            reasons.append('exposure_differs_from_step_attribution')
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
        reasons.append(type(error).__name__ + ":" + str(error))
    result["accepted"] = not reasons
    return result
