"""Audit Source PPO initialization on CPU, without simulator steps or updates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rl_training"))

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from rl_training.source_scan.frozen_actor import load_frozen_source_actor
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.source_training import TransitionBudget, load_source_actor_only


def runner_config():
    """Resolved RSL-RL 5.x equivalent of the registered Lite3 PPO candidate."""
    model = {"class_name": "rsl_rl.models:MLPModel", "hidden_dims": [512, 256, 128],
             "activation": "elu", "obs_normalization": False}
    return {
        "num_steps_per_env": 24, "save_interval": 50,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {**model, "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": 1.0, "std_type": "log"}},
        "critic": deepcopy(model),
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO", "value_loss_coef": 1.0,
            "use_clipped_value_loss": True, "clip_param": 0.2, "entropy_coef": 0.01,
            "num_learning_epochs": 5, "num_mini_batches": 4,
            "learning_rate": 0.001, "schedule": "adaptive", "gamma": 0.99,
            "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0,
            "rnd_cfg": None, "symmetry_cfg": None,
        },
    }


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validation_cells():
    from rl_training.source_scan.source_selection import registered_validation_cells
    return registered_validation_cells()


def audit(checkpoint):
    torch.set_num_threads(1)
    version = importlib.metadata.version("rsl-rl-lib")
    if version != "5.0.1":
        raise RuntimeError(f"RSL-RL version has not been audited: {version}")
    frozen = load_frozen_source_actor(checkpoint)
    probe_rng = torch.Generator().manual_seed(130000)
    probes = TensorDict({"policy": torch.cat([
        torch.zeros(1, 45), torch.randn(256, 45, generator=probe_rng)
    ])}, batch_size=[257])
    reference = frozen(probes)
    runs = []
    for index, (ppo_seed, terrain_seed, state_id) in enumerate(
        [(130001, 30000, 40000), (130002, 33000, 43000), (130003, 36000, 46000)], 1
    ):
        torch.manual_seed(ppo_seed)
        # Synthetic critic width intentionally differs from checkpoint input.
        # This audit proves initialization isolation, not the live observation contract.
        obs = TensorDict({"policy": torch.zeros(2, 45), "critic": torch.zeros(2, 48)}, batch_size=[2])
        env = SimpleNamespace(num_envs=2, num_actions=12, cfg={}, get_observations=lambda: obs)
        runner = OnPolicyRunner(env, runner_config(), log_dir=None, device="cpu")
        critic_before = state_digest(runner.alg.critic.state_dict())
        optimizer_before = deepcopy(runner.alg.optimizer.state_dict())
        rng_before = torch.random.get_rng_state().clone()
        loaded = load_source_actor_only(runner.alg.actor, checkpoint, action_std=1.0)
        with torch.no_grad():
            output = runner.alg.actor(probes)
        checks = {
            "mean_exact_parity": torch.equal(reference, output),
            "critic_unchanged": critic_before == state_digest(runner.alg.critic.state_dict()),
            "optimizer_fresh_and_unchanged": not runner.alg.optimizer.state
                and optimizer_before == runner.alg.optimizer.state_dict(),
            "iteration_zero": runner.current_learning_iteration == 0,
            "rollout_storage_empty": runner.alg.storage.step == 0,
            "rng_unchanged_by_loading": torch.equal(rng_before, torch.random.get_rng_state()),
            "std_reset_to_one": bool(torch.equal(
                runner.alg.actor.distribution.log_std_param, torch.zeros(12))),
            "actor_trainable": all(p.requires_grad for p in runner.alg.actor.parameters()),
        }
        if not all(checks.values()):
            raise RuntimeError(f"initialization audit failed: {checks}")
        runs.append({"run": index, "ppo_seed": ppo_seed, "terrain_generator_seed": terrain_seed,
                     "reset_state_id": state_id, "sampler_seed_candidate": ppo_seed + 1000,
                     "initialization": loaded, "checks": checks, "critic_sha256": critic_before,
                     "probe_count": 257, "max_mean_error": float((reference - output).abs().max())})
    if len({row["critic_sha256"] for row in runs}) != 3:
        raise RuntimeError("fresh critic seeds did not produce distinct initializations")
    budgets = {}
    for name, ceiling in [("numerics", 262144), ("midpoint", 1048576), ("pilot", 2097152)]:
        budget = TransitionBudget(120, 24, ceiling)
        budgets[name] = {"ceiling": ceiling, "updates": budget.max_updates,
                         "transitions": budget.committed_transitions, "unused": budget.unused_transitions}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return {
        "schema": "lite3_source_training_preflight_v1",
        "status": "cpu_initialization_passed_runtime_pending",
        "simulation_steps": 0, "ppo_updates": 0, "training_ready": False,
        "runtime": {"python": sys.version, "torch": torch.__version__, "rsl_rl": version, "device": "cpu"},
        "checkpoint": str(checkpoint),
        "checkpoint_original_std": payload["actor_state_dict"]["distribution.log_std_param"].exp().tolist(),
        "cpu_fixture": {"num_envs": 2, "policy_width": 45, "critic_width": 48, "synthetic": True},
        "candidate": {"num_envs": 120, "rollout_length": 24, "level": "L1_only",
                      "reset_block_counts": SOURCE_RESET_BLOCK_COUNTS,
                      "budget_nodes": budgets, "runner_config": runner_config(), "runs": runs,
                      "validation_cells": validation_cells(),
                      "validation_seed_reservation": "candidate_no_simulation_or_selection_performed",
                      "state_seed_semantics": "bookkeeping_only_current_reset_is_deterministic"},
        "remaining_runtime_work": [
            "Implement PPO sampled-action safety wrapper without rewriting stored raw actions/log-probabilities.",
            "Implement vector completion/50-step hold, bounded episodes and pre-reset terminal observation handling.",
            "Wire explicit training generator seeds; default Source config still uses scan seed 10001.",
            "Wire independent sampler RNG and reset/transition ledgers; define interrupted-run handling.",
            "Complete moving-environment isolation and GPU throughput checks at the selected scale.",
            "Lock six-family quality/selection rules and verify seed reservations before validation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; audit artifacts must not be overwritten")
    result = audit(args.checkpoint)
    result["base_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    inputs = [Path(__file__), ROOT / "source/rl_training/rl_training/source_scan/source_training.py",
              ROOT / "source/rl_training/rl_training/source_scan/frozen_actor.py",
              ROOT / "source/rl_training/rl_training/source_scan/parallel_terrain_allocation.py"]
    result["input_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in inputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"CPU initialization passed for three seeds; simulation=0, PPO updates=0: {args.output}")


if __name__ == "__main__":
    main()
