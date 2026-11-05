"""CPU profile/real-checkpoint audit. Does not import Isaac, simulate, or train."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rl_training"))

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from rl_training.source_scan.adaptation_profile import (
    SOURCE_ADAPTATION_PROFILE, build_source_adaptation_rewards,
    initialize_source_adaptation_actor, validate_source_adaptation_config,
)
from rl_training.source_scan.frozen_actor import FrozenSourceActor
from rl_training.source_scan.training_budget_ledger import BudgetLedger, canonical_budget_path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(checkpoint, observation_paths):
    torch.set_num_threads(1)
    ledger = BudgetLedger(canonical_budget_path(ROOT))
    budget_before = ledger.reference()
    checkpoint_before = digest(checkpoint)
    cfg = SimpleNamespace(source_reward_profile=SOURCE_ADAPTATION_PROFILE, reward_gait_level=1.0,
        rewards=build_source_adaptation_rewards(SimpleNamespace, SimpleNamespace))
    configured = validate_source_adaptation_config(cfg)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"]
    observations = []
    for path in observation_paths:
        snapshots = torch.load(path, map_location="cpu", weights_only=True)
        observations.extend(snapshot["observation"] for snapshot in snapshots.values())
    values = torch.cat(observations)
    if values.ndim != 2 or values.shape[1] != 45 or not bool(torch.isfinite(values).all()):
        raise ValueError("invalid archived policy observations")
    obs = TensorDict({"policy": values}, batch_size=[len(values)])
    with torch.random.fork_rng(devices=[]):
        actor = MLPModel(obs, {"actor": ["policy"]}, "actor", 12, hidden_dims=[512,256,128],
            activation="elu", obs_normalization=False,
            distribution_cfg={"class_name":"rsl_rl.modules.distribution:GaussianDistribution",
                              "std_type":"log", "init_std":1.0})
    rng = torch.random.get_rng_state().clone()
    initialization = initialize_source_adaptation_actor(actor, checkpoint, cfg)
    rng_unchanged = torch.equal(rng, torch.random.get_rng_state())
    with torch.inference_mode():
        error = float((actor(obs) - FrozenSourceActor(saved)(obs)).abs().max())
    all_tensors_equal = all(torch.equal(v, saved[k]) for k, v in actor.state_dict().items())
    if error != 0.0 or not all_tensors_equal or not rng_unchanged:
        raise RuntimeError("checkpoint-preserving initialization changed the policy")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    budget_after = ledger.reference()
    if budget_after != budget_before or digest(checkpoint) != checkpoint_before:
        raise RuntimeError("preflight changed checkpoint or training budget")
    std = saved["distribution.log_std_param"].exp()
    source_paths = [Path(__file__), ROOT / "source/rl_training/rl_training/source_scan/adaptation_profile.py",
        ROOT / "source/rl_training/rl_training/source_scan/source_training.py",
        ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py",
        ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_adaptation_env_cfg.py"]
    return dict(status="passed_cpu_preflight", evidence_kind="specification_and_real_actor_audit",
        profile=configured, initialization=initialization, actor_observation_count=len(values),
        actor_output_max_abs_difference=error, all_actor_tensors_equal=all_tensors_equal,
        rng_unchanged=rng_unchanged, action_std_per_joint=std.tolist(),action_std_mean=float(std.mean()),
        checkpoint=dict(path=str(checkpoint),sha256=checkpoint_before),
        observations=[dict(path=str(p),sha256=digest(p)) for p in observation_paths],
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_paths},
        base_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        budget_before=budget_before,budget_after=budget_after,
        training_transitions_charged=ledger.charged(1,ledger.read()),
        numerical_budget_remaining=262144-ledger.charged(1,ledger.read()),
        simulation_steps=0,training_transitions=0,ppo_updates=0,cuda_initialized=False,
        live_manager_qualification=False,
        kernel_check=dict(command_mps=0.3,perfect_tracking_reward=4.0,
            stationary_reward=4.0*math.exp(-0.3**2/0.25**2),
            historical_stationary_reward=4.0*math.exp(-0.3**2/0.5)),
        limits=["SDK scene construction and resolved runtime rewards are not exercised by this CPU audit",
                "archived trajectories cannot reconstruct all newly enabled reward components",
                "locomotion improvement after retraining remains unmeasured"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--observations",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint,args.observations)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("status","actor_observation_count","actor_output_max_abs_difference",
                                          "action_std_mean","numerical_budget_remaining","cuda_initialized")}))
