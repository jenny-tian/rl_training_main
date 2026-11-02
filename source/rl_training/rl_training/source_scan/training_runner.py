"""Budgeted, non-resumable Source PPO updates and durable run accounting."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import torch

from .source_training import TransitionBudget
from .training_lifecycle import SourceRunSeeds
from .parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS


TRAINING_NODES = {
    "numerics": (91, 262_144),
    "midpoint": (364, 1_048_576),
    "pilot": (728, 2_097_152),
}
_POLICY_NODES = (91, 364, 728)


def _receipt(reference, label):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"missing hashed {label} receipt")
    payload = Path(reference["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError(f"changed {label} receipt")
    return json.loads(payload)


def validate_readiness(manifest, input_sha256):
    """Require actual scale evidence and a separately locked selection protocol."""
    runtime = _receipt(manifest.get("runtime_qualification"), "runtime qualification")
    required = {"status": "parallel_runtime_qualified", "num_envs": 120,
                "device": "cuda:0", "g3_passed": True, "g4_passed": True,
                "input_sha256": input_sha256}
    for key, value in required.items():
        if type(runtime.get(key)) is not type(value) or runtime[key] != value:
            raise ValueError(f"runtime qualification mismatch: {key}")
    selection = _receipt(manifest.get("source_selection_lock"), "source selection lock")
    if (selection.get("status") != "locked" or selection.get("scope") != "source_only"
            or selection.get("registered_cells") != list(SOURCE_RESET_BLOCK_COUNTS)
            or selection.get("development_or_sealed_feedback_allowed") is not False):
        raise ValueError("source selection protocol is not locked for all 15 cells")
    # Thresholds and tie-breaking remain explicit protocol content, never inferred
    # from the diagnostic's results or silently invented by this entrypoint.
    for key in ("quality_gates", "selection_rule", "seed_splits", "evaluation_nodes"):
        if not selection.get(key):
            raise ValueError(f"source selection lock is incomplete: {key}")
    dependencies = manifest.get("dependency_sha256")
    if not isinstance(dependencies, dict) or len(dependencies) < 35:
        raise ValueError("runtime dependency hashes are incomplete")
    for name, digest in dependencies.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"changed runtime dependency: {name}")


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite Source {label}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_training_authorization(
    manifest: dict[str, Any],
    *,
    run: int,
    budget_node: str,
    output_dir: Path,
    checkpoint_sha256: str,
    revision: str,
    input_sha256: dict[str, str],
) -> dict[str, Any]:
    """Validate the complete, explicit lock before launching Isaac Sim."""
    if not isinstance(manifest, dict) or manifest.get("schema") != "lite3_source_ppo_authorization_v1":
        raise ValueError("unsupported Source PPO authorization schema")
    if manifest.get("user_approved") is not True or manifest.get("status") != "approved":
        raise PermissionError("Source PPO requires an approved authorization manifest")
    if budget_node != "numerics":
        raise NotImplementedError("midpoint/pilot continuation is not implemented; fresh restarts would double-count the seed budget")
    seeds = SourceRunSeeds(run)
    try:
        expected_updates, expected_transitions = TRAINING_NODES[budget_node]
    except KeyError as error:
        raise ValueError("budget_node must be numerics, midpoint or pilot") from error
    expected = {
        "action": "source_ppo_training",
        "run": run,
        "ppo_seed": seeds.ppo,
        "geometry_seed": seeds.geometry,
        "sampler_seed": seeds.sampler,
        "reset_state_id": seeds.reset_state_id,
        "num_envs": 120,
        "rollout_length": 24,
        "level": "L1",
        "registered_cells": 15,
        "six_families_retained": True,
        "budget_node": budget_node,
        "max_updates": expected_updates,
        "max_transitions": expected_transitions,
        "device": "cuda:0",
        "minimum_free_vram_mib": 12_288,
        "retry": False,
        "resume": False,
        "automatic_scale_increase": False,
        "development_or_sealed_feedback_allowed": False,
        "base_revision": revision,
        "remote_revision_verified": revision,
        "checkpoint_sha256": checkpoint_sha256,
        "input_sha256": input_sha256,
    }
    for key, value in expected.items():
        if type(manifest.get(key)) is not type(value) or manifest[key] != value:
            raise ValueError(f"authorization contract mismatch: {key}")
    if manifest.get("output_dir") != str(output_dir.resolve()):
        raise ValueError("authorization output directory differs from the requested path")
    validate_readiness(manifest, input_sha256)
    receipts = manifest.get("prior_gate_receipts", {})
    required_receipts = {"midpoint": ("numerics",), "pilot": ("numerics", "midpoint")}.get(budget_node, ())
    for node in required_receipts:
        receipt = receipts.get(node) if isinstance(receipts, dict) else None
        if not isinstance(receipt, dict) or receipt.get("status") != "passed" or not receipt.get("sha256"):
            raise ValueError(f"{budget_node} authorization requires a passed {node} gate receipt")
    return {key: manifest[key] for key in expected} | {
        "schema": manifest["schema"], "status": manifest["status"],
        "user_approved": True, "output_dir": str(output_dir.resolve()),
        "prior_gate_receipts": receipts,
        "runtime_qualification": manifest["runtime_qualification"],
        "source_selection_lock": manifest["source_selection_lock"],
        "dependency_sha256": manifest["dependency_sha256"],
    }


class SourceTrainingRunner:
    """Run only complete, pre-budgeted PPO rollouts; never restore old state."""

    def __init__(
        self,
        runner,
        *,
        run: int,
        budget: TransitionBudget,
        output_dir: Path,
        checkpoint_sha256: str,
        checkpoint_nodes: tuple[int, ...] | None = None,
    ) -> None:
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError("Source training output already exists; interrupted runs are not resumed")
        if budget.max_updates < 1:
            raise ValueError("transition budget cannot fund one whole rollout")
        seeds = SourceRunSeeds(run)
        env = runner.env
        config = getattr(getattr(env, "unwrapped", env), "cfg", None)
        if config is None or getattr(config, "source_training_run", None) != run:
            raise ValueError("runner environment is not configured for the registered Source run")
        if (getattr(config, "seed", None) != seeds.ppo
                or getattr(config, "source_parallel_sampler_seed", None) != seeds.sampler
                or getattr(config, "source_terrain_level", None) != "L1"):
            raise ValueError("Source run seed or level differs from the registered contract")
        if env.num_envs != budget.num_envs or runner.cfg.get("num_steps_per_env") != budget.rollout_length:
            raise ValueError("runner dimensions differ from the transition budget")
        if getattr(runner, "is_distributed", False):
            raise ValueError("Source training runner does not support distributed execution")
        if getattr(runner, "current_learning_iteration", None) != 0:
            raise ValueError("Source training must start from iteration zero")
        algorithm = runner.alg
        if algorithm.__class__.__module__ != "rl_training.source_scan.training_ppo":
            raise ValueError("Source training requires the audited SourcePPO adapter")
        if getattr(algorithm.storage, "step", None) != 0 or algorithm.optimizer.state:
            raise ValueError("Source PPO storage and optimizer must be fresh")
        if algorithm.rnd is not None or algorithm.symmetry is not None:
            raise ValueError("Source PPO contract does not enable RND or symmetry extensions")
        if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        if checkpoint_nodes is None:
            checkpoint_nodes = tuple(node for node in _POLICY_NODES if node <= budget.max_updates)
        if len(set(checkpoint_nodes)) != len(checkpoint_nodes) or any(
            type(node) is not int or node <= 0 or node > budget.max_updates for node in checkpoint_nodes
        ):
            raise ValueError("policy checkpoint nodes must be unique completed updates within the budget")

        self.runner = runner
        self.env = env
        self.algorithm = algorithm
        self.run_id = run
        self.budget = budget
        self.output_dir = output_dir
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_nodes = tuple(sorted(checkpoint_nodes))
        self.started = time.monotonic()
        self.synthetic = getattr(env, "source_synthetic_fixture", False) is True
        self.report: dict[str, Any] = {
            "schema": "lite3_source_ppo_run_report_v1",
            "status": "starting",
            "run": run,
            "ppo_seed": seeds.ppo,
            "geometry_seed": seeds.geometry,
            "sampler_seed": seeds.sampler,
            "reset_state_id": seeds.reset_state_id,
            "num_envs": budget.num_envs,
            "rollout_length": budget.rollout_length,
            "transitions_per_update": budget.transitions_per_update,
            "transition_ceiling": budget.max_transitions,
            "max_updates": budget.max_updates,
            "committed_transition_ceiling": budget.committed_transitions,
            "unused_transition_ceiling": budget.unused_transitions,
            "checkpoint_sha256": checkpoint_sha256,
            "data_kind": "synthetic" if self.synthetic else "source_simulation",
            "attempted_transition_upper_bound": 0,
            "returned_transitions": 0,
            "stored_transitions": 0,
            "completed_updates": 0,
            "rollout_steps_started": 0,
            "partial_rollout_stored_steps": 0,
            "automatic_resume": False,
            "training_selection_performed": False,
            "training_ready": False,
            "g3_g4_qualified": False,
            "exposure": self._exposure(),
            "update_receipts": [],
        }

    def _exposure(self):
        exposure = getattr(self.env, "exposure", None)
        return exposure.report() if exposure is not None else None

    def _persist(self) -> None:
        self.report["elapsed_seconds"] = time.monotonic() - self.started
        self.report["exposure"] = self._exposure()
        self.report["real_source_training_transitions"] = 0 if self.synthetic else self.report["returned_transitions"]
        self.report["synthetic_transitions"] = self.report["returned_transitions"] if self.synthetic else 0
        self.report["real_source_ppo_updates"] = 0 if self.synthetic else self.report["completed_updates"]
        for name in ("source_backward_checks", "source_optimizer_steps_started", "source_optimizer_steps_completed"):
            self.report[name] = getattr(self.algorithm, name, 0)
        _atomic_json(self.output_dir / "run_report.json", self.report)

    def _append_receipt(self, receipt: dict[str, Any]) -> None:
        path = self.output_dir / "update_receipts.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _validate_step(self, actions, observations, rewards, dones) -> None:
        _assert_finite(actions, "sampled actions")
        _assert_finite(observations, "observations")
        _assert_finite(rewards, "rewards")
        _assert_finite(dones, "ending flags")
        self._validate_observations(observations)
        if actions.shape != (self.budget.num_envs, 12):
            raise ValueError("Source PPO action batch has an invalid shape")
        if rewards.shape not in ((self.budget.num_envs,), (self.budget.num_envs, 1)):
            raise ValueError("Source PPO reward batch has an invalid shape")
        if dones.shape != (self.budget.num_envs,) or bool(((dones != 0) & (dones != 1)).any()):
            raise ValueError("Source PPO ending flags must be one binary value per environment")

    def _validate_observations(self, observations) -> None:
        _assert_finite(observations, "observations")
        for group, width in (("policy", 45), ("critic", 393)):
            if group not in observations or observations[group].shape != (self.budget.num_envs, width):
                raise ValueError(f"Source {group} observations require width {width}")

    def _storage_is_finite(self) -> None:
        storage = self.algorithm.storage
        if storage.step != self.budget.rollout_length:
            raise RuntimeError("Source PPO rollout storage is incomplete")
        for name in ("observations", "actions", "actions_log_prob", "values", "rewards", "dones",
                     "returns", "advantages", "distribution_params"):
            _assert_finite(getattr(storage, name, None), f"rollout {name}")

    def _save_policy(self, update: int) -> None:
        state = {key: value.detach().cpu().contiguous()
                 for key, value in self.algorithm.actor.state_dict().items()}
        _assert_finite(state, "policy checkpoint")
        _atomic_torch_save(self.output_dir / f"actor_update_{update}.pt", state)

    def run(self, *, before_step=None) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.report["status"] = "running"
        self._persist()
        try:
            observations = self.env.get_observations().to(self.runner.device)
            self._validate_observations(observations)
            self.algorithm.train_mode()
            while self.report["completed_updates"] < self.budget.max_updates:
                completed = self.report["completed_updates"]
                self.budget.assert_update_allowed(completed)
                if (self.report["attempted_transition_upper_bound"] + self.budget.transitions_per_update
                        > self.budget.max_transitions):
                    raise RuntimeError("next complete Source rollout would exceed the hard transition ceiling")
                rollout_started = time.monotonic()
                start_returned = self.report["returned_transitions"]
                start_stored = self.report["stored_transitions"]
                self.report["rollout_steps_started"] = 0
                self.report["partial_rollout_stored_steps"] = 0
                self._persist()
                with torch.inference_mode():
                    for step in range(self.budget.rollout_length):
                        if before_step is not None:
                            before_step()
                        actions = self.algorithm.act(observations)
                        _assert_finite(actions, "sampled actions")
                        self.report["attempted_transition_upper_bound"] += self.budget.num_envs
                        self.report["rollout_steps_started"] = step + 1
                        self._persist()
                        observations, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        self.report["returned_transitions"] += self.budget.num_envs
                        self._validate_step(actions, observations, rewards, dones)
                        observations = observations.to(self.runner.device)
                        rewards = rewards.to(self.runner.device)
                        dones = dones.to(self.runner.device)
                        self.algorithm.process_env_step(observations, rewards, dones, extras)
                        self.report["stored_transitions"] += self.budget.num_envs
                        self.report["partial_rollout_stored_steps"] = step + 1
                        self._persist()
                    self.algorithm.compute_returns(observations)
                self._storage_is_finite()
                losses = self.algorithm.update()
                for name, value in losses.items():
                    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        raise FloatingPointError(f"non-finite Source update loss: {name}")
                completed += 1
                self.report["completed_updates"] = completed
                self.report["partial_rollout_stored_steps"] = 0
                self.runner.current_learning_iteration = completed
                receipt = {
                    "update": completed,
                    "rollout_steps": self.budget.rollout_length,
                    "returned_transitions": self.report["returned_transitions"] - start_returned,
                    "stored_transitions": self.report["stored_transitions"] - start_stored,
                    "losses": {name: float(value) for name, value in losses.items()},
                    "actor_sha256": _state_digest(self.algorithm.actor),
                    "exposure": self._exposure(),
                    "elapsed_seconds": time.monotonic() - rollout_started,
                }
                self._append_receipt(receipt)
                self.report["update_receipts"].append(receipt)
                if completed in self.checkpoint_nodes:
                    self._save_policy(completed)
                self._persist()
            self.report["status"] = "completed"
            self.report["training_transitions"] = self.report["stored_transitions"]
            self._persist()
            return dict(self.report)
        except KeyboardInterrupt:
            self.report.update(status="interrupted", training_transitions=self.report["stored_transitions"],
                               interruption="KeyboardInterrupt", automatic_resume=False)
            self._persist()
            raise
        except Exception as error:
            self.report.update(status="failed", training_transitions=self.report["stored_transitions"],
                               error_type=type(error).__name__, error=str(error), automatic_resume=False)
            self._persist()
            raise

    def record_cleanup(self, *, environment_close_returned: bool, application_close_returned: bool) -> None:
        self.report["cleanup"] = {
            "environment_close_returned": bool(environment_close_returned),
            "application_close_returned": bool(application_close_returned),
        }
        self._persist()


__all__ = ["SourceTrainingRunner", "TRAINING_NODES", "validate_training_authorization"]
