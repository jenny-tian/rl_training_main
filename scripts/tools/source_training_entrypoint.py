"""Run one explicitly approved, budgeted Source PPO candidate on cuda:0."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rl_training"))

from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
from rl_training.source_scan.source_training import TransitionBudget, load_source_actor_only
from rl_training.source_scan.training_lifecycle import SourceRunSeeds
from rl_training.source_scan.training_runner import (
    TRAINING_NODES,
    SourceTrainingRunner,
    validate_training_authorization,
    _atomic_json,
)


def _git_state() -> tuple[str, str, str]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch],
                                    cwd=ROOT, text=True, timeout=30).split()
    if len(remote) != 2 or remote[1] != "refs/heads/" + branch:
        raise ValueError("Source branch is missing from the configured remote")
    upstream = remote[0]
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return revision, upstream, dirty


def _input_hashes() -> dict[str, str]:
    paths = {str(path.relative_to(ROOT)) for parent in (ROOT / "source", ROOT / "scripts")
             for path in parent.rglob("*.py")}
    paths.update(str(path.relative_to(ROOT)) for path in
                 (ROOT / "deep_robotics_model/Lite3/Lite3_usd").rglob("*.usd"))
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(paths)}


def validate_before_simulation(args, manifest: dict) -> tuple[dict, str, dict[str, str]]:
    revision, upstream, dirty = _git_state()
    if dirty:
        raise ValueError("Source PPO requires a clean committed worktree")
    if revision != upstream:
        raise ValueError("Source PPO requires the current commit to match its tracked remote revision")
    hashes = _input_hashes()
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("checkpoint differs from the registered Source policy")
    if importlib.metadata.version("rsl-rl-lib") != "5.0.1":
        raise RuntimeError("Source training entrypoint is audited only for RSL-RL 5.0.1")
    lock = validate_training_authorization(
        manifest,
        run=args.run,
        budget_node=args.budget_node,
        output_dir=args.output_dir,
        checkpoint_sha256=checkpoint_sha256,
        revision=revision,
        input_sha256=hashes,
    )
    if args.output_dir.exists():
        raise FileExistsError("Source training output exists; retries and resume are disabled")
    if args.device != "cuda:0":
        raise ValueError("Source PPO candidate is locked to cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise ValueError("Source GPU0 visibility mapping differs from the contract")
    return lock, revision, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--budget-node", choices=tuple(TRAINING_NODES), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true", help="Validate without importing Isaac or starting CUDA")
    parser.add_argument("--supervised-ticket", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    manifest = json.loads(args.authorization.read_text(encoding="utf-8"))
    if args.supervised_ticket is None:
        lock, revision, hashes = validate_before_simulation(args, manifest)
    else:
        from rl_training.source_scan.cumulative_training import validate_supervised_child
        from rl_training.source_scan.training_budget_ledger import canonical_budget_path
        revision, remote, dirty = _git_state()
        if args.device != 'cuda:0' or os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '0'):
            raise ValueError('supervised Source GPU mapping changed')
        hashes = _input_hashes()
        if dirty or revision != remote or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError('supervised Source worktree, remote or checkpoint changed')
        if args.supervised_ticket.resolve() != args.output_dir / 'supervision.json':
            raise ValueError('supervision ticket is outside the approved attempt')
        lock = validate_supervised_child(manifest, json.loads(args.supervised_ticket.read_text()),
            output_dir=args.output_dir, revision=revision, input_sha256=hashes,
            ledger_path=canonical_budget_path(ROOT))
        if args.run != lock['run'] or args.budget_node != lock['budget_node']:
            raise ValueError('supervised Source run or budget node changed')
        if importlib.metadata.version('rsl-rl-lib') != '5.0.1':
            raise RuntimeError('supervised Source requires RSL-RL 5.0.1')
    if args.validate_only:
        print(json.dumps({"status": "authorization_validated", "simulation_started": False,
                          "training_started": False, "base_revision": revision}))
        return 0
    cumulative = lock['schema'] in ('lite3_source_ppo_cumulative_authorization_v2','lite3_source_ppo_bounded_numerics_authorization_v3')
    if cumulative and args.supervised_ticket is None:
        raise ValueError('cumulative numeric execution requires source_training_supervisor.py')

    import numpy as np
    import torch
    seeds = SourceRunSeeds(args.run)
    app = None
    wrapped_env = None
    base_env = None
    training = None
    environment_closed = False
    application_closed = False
    error = None
    if args.supervised_ticket is None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    elif (args.output_dir / 'attempt.json').exists():
        raise FileExistsError('supervised Source child cannot restart an attempt')
    attempt = {"status": "starting", "base_revision": revision, "training_started": False,
               "authorization": lock, "environment_close_returned": False,
               "application_close_state": "not_started"}
    _atomic_json(args.output_dir / "attempt.json", attempt)
    reservation = None
    try:
        if not cumulative:
            from rl_training.source_scan.training_budget_ledger import BudgetLedger
            ledger = BudgetLedger(lock['budget_ledger']['path'])
            reservation = ledger.reserve(run=args.run, attempt_id='source-numerics-' + hashlib.sha256(
                str(args.output_dir).encode()).hexdigest(), transitions=lock['max_updates'] * 120 * 24,
                purpose='source_numerics', output_dir=args.output_dir, revision=revision,
                expected_reference=lock['budget_ledger'])
            attempt['budget_reservation'] = reservation
            _atomic_json(args.output_dir / "attempt.json", attempt)
        import csv

        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader"
        ], text=True, timeout=10)
        free_mib = None
        for row in csv.reader(gpu.splitlines()):
            if row and row[0].strip() == "0":
                free_mib = int(row[1].strip().split()[0])
                break
        if free_mib is None or free_mib < lock["minimum_free_vram_mib"]:
            raise RuntimeError("GPU0 free memory is below the approved Source PPO threshold")

        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device=args.device).app
        attempt["application_close_state"] = "open"
        import gymnasium as gym
        import rl_training.tasks  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from rl_training.source_scan.training_env import SourceTrainingVecEnv
        from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_training_env_cfg import DeeproboticsLite3SourceTrainingEnvCfg
        sys.path.insert(0, str(ROOT))
        from scripts.tools.source_training_preflight import runner_config

        cfg = DeeproboticsLite3SourceTrainingEnvCfg(
            source_training_run=args.run,
            source_parallel_sampler_enabled=True,
            source_parallel_sampler_num_envs=lock["num_envs"],
            source_shared_bank_rows=1,
        )
        cfg.sim.device = args.device
        base_env = gym.make("SourceTerrain-Deeprobotics-Lite3-v0", cfg=cfg)
        wrapped_env = SourceTrainingVecEnv(RslRlVecEnvWrapper(base_env, clip_actions=None))

        torch.manual_seed(seeds.ppo)
        np.random.seed(seeds.ppo)
        random.seed(seeds.ppo)
        cfg_runner = runner_config()
        cfg_runner["num_steps_per_env"] = lock["rollout_length"]
        cfg_runner["max_iterations"] = lock["max_updates"]
        cfg_runner["save_interval"] = lock["max_updates"] + 1
        cfg_runner["check_for_nan"] = True
        cfg_runner["algorithm"]["class_name"] = "rl_training.source_scan.training_ppo:SourcePPO"
        runner = OnPolicyRunner(wrapped_env, cfg_runner, log_dir=None, device=args.device)
        actor_init = load_source_actor_only(runner.alg.actor, args.checkpoint, action_std=1.0)
        training = SourceTrainingRunner(
            runner,
            run=args.run,
            budget=TransitionBudget(lock["num_envs"], lock["rollout_length"], lock["max_transitions"]),
            output_dir=args.output_dir / "training",
            checkpoint_sha256=lock["checkpoint_sha256"],
            checkpoint_nodes=tuple(lock['checkpoint_nodes']) if cumulative else None,
        )
        training.report.update(
            training_ready=not cumulative,
            g3_g4_qualified=not cumulative,
            bounded_numerics_ready=cumulative,
            g4_end_to_end_qualified=False if cumulative else True,
            base_revision=revision,
            remote_revision_verified=revision,
            input_sha256=hashes,
            authorization_sha256=hashlib.sha256(args.authorization.read_bytes()).hexdigest(),
            authorization_status="approved",
            authorization_budget_node=args.budget_node,
            actor_initialization=actor_init,
            startup_gpu0_free_mib=free_mib,
            rsl_rl_version=importlib.metadata.version("rsl-rl-lib"),
        )
        from rl_training.source_scan.training_runner import _state_digest
        training.report.update(initial_actor_sha256=_state_digest(runner.alg.actor),
            initial_critic_sha256=_state_digest(runner.alg.critic),
            cumulative_budget=lock.get('cumulative_budget'))
        attempt.update(status="running", training_started=True)
        _atomic_json(args.output_dir / "attempt.json", attempt)
        print('[source-numerics] initial-ready', flush=True)
        curves=None
        if lock['schema']=='lite3_source_ppo_bounded_numerics_authorization_v3':
            from rl_training.source_scan.delivery_metrics import TrainingCurves
            curves=TrainingCurves(args.output_dir/'training/curves.jsonl')
        training.run(observer=curves,before_step=lambda: (
            None if app.is_running() else (_ for _ in ()).throw(RuntimeError("Isaac application stopped"))
        ))
        attempt["status"] = "completed"
    except BaseException as exc:
        error = exc
        attempt.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        closing_env = wrapped_env if wrapped_env is not None else base_env
        if closing_env is not None:
            try:
                closing_env.close()
                environment_closed = True
            except Exception as exc:
                attempt.update(status="failed", cleanup_error=f"environment_close: {type(exc).__name__}: {exc}")
                if training is not None:
                    training.report["cleanup_error"] = f"environment_close: {type(exc).__name__}: {exc}"
        if training is not None:
            training.record_cleanup(
                environment_close_returned=environment_closed,
                application_close_returned=False,
            )
        attempt.update(environment_close_returned=environment_closed,
                       application_close_state="closing" if app is not None else "not_started")
        _atomic_json(args.output_dir / "attempt.json", attempt)
        if app is not None:
            try:
                app.close()
                application_closed = True
                attempt["application_close_state"] = "returned"
            except Exception as exc:
                attempt["application_close_state"] = "failed"
                attempt["status"] = "failed"
                attempt["cleanup_error"] = f"application_close: {type(exc).__name__}: {exc}"
        if training is not None:
            training.record_cleanup(
                environment_close_returned=environment_closed,
                application_close_returned=application_closed,
            )
        if error is not None and training is not None:
            training.report.setdefault("entrypoint_error", f"{type(error).__name__}: {error}")
            training._persist()
        _atomic_json(args.output_dir / "attempt.json", attempt)
        if reservation is not None:
            counts = training.report if training is not None else {}
            ledger.finish(reservation['attempt_id'], accepted=attempt['status'] == 'completed',
                attempted=counts.get('attempted_transition_upper_bound', 0),
                returned=counts.get('returned_transitions', 0), stored=counts.get('stored_transitions', 0),
                updates=counts.get('completed_updates', 0))
    if attempt["status"] != "completed":
        raise RuntimeError("Source PPO attempt did not complete with successful cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
