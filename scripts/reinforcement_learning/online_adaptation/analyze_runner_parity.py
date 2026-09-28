"""Compare recorded frozen trajectories and replay real actor paths offline."""

import argparse
import copy
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.lora import inject_lora
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.rsl_rl_compat import RslMLPPolicyAdapter
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState


def load_trace(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["vector_dtype"] != "float32":
        raise ValueError("parity requires completed float32 traces")
    chunks = []
    for item in manifest["chunks"]:
        with np.load(directory / item["file"], allow_pickle=False) as data:
            chunks.append(dict(data))
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def configuration_diff(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [row for key in sorted(a.keys() | b.keys())
                for row in configuration_diff(a.get(key), b.get(key), f"{prefix}.{key}")]
    return [] if a == b else [{"field": prefix, "lora": a, "sac": b}]


def physical_diff(a, b):
    result = {}
    for key in ("observation", "command", "root_state", "joint_pos", "joint_vel",
                "raw_base_action", "base_action", "requested_action", "executed_action", "base_x", "base_y"):
        count = min(len(a[key]), len(b[key]))
        differences = np.abs(a[key][:count] - b[key][:count]).reshape(count, -1).max(axis=1)
        unequal = np.flatnonzero(differences > 1e-6)
        result[key] = {"initial_max_abs": float(differences[0]), "max_abs": float(differences.max()),
                       "first_step_above_1e_6": int(unequal[0] + 1) if len(unequal) else None}
    result["steps"] = {"lora": len(a["step"]), "sac": len(b["step"])}
    return result


def task_diff(a, b):
    result = {"same_steps": len(a["step"]) == len(b["step"])}
    for key in ("reward", "ordinary_reward", "terminal_reward_override", "learning_terminated",
                "learning_truncated", "collector_cutoff", "bootstrap_mask"):
        left, right = a[key].reshape(len(a["step"]), -1), b[key].reshape(len(b["step"]), -1)
        result[key] = bool(left.shape == right.shape and np.allclose(left, right, rtol=0., atol=1e-6))
    for name, trace in (("lora", a), ("sac", b)):
        result[f"{name}_bootstrap_contract"] = bool(np.array_equal(
            trace["bootstrap_mask"].reshape(-1), ~trace["learning_terminated"].reshape(-1)))
        result[f"{name}_reward_reconstruction"] = bool(np.allclose(
            trace["reward"].reshape(-1), (trace["ordinary_reward"] + trace["terminal_reward_override"]).reshape(-1), rtol=0., atol=2e-5))
    return {"checks": result, "passed": all(result.values())}


@torch.no_grad()
def offline_actions(trace, checkpoint, device):
    observations = torch.as_tensor(trace["observation"], device=device).squeeze(1)
    initial = TensorDict({"policy": observations[:1]}, batch_size=[1])
    model = MLPModel(initial, {"actor": ["policy"]}, "actor", 12,
                     hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False,
                     distribution_cfg={"class_name": "rsl_rl.modules.distribution.GaussianDistribution", "std_type": "log"}).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["actor_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    adapted = copy.deepcopy(model)
    inject_lora(adapted, rank=2)
    lora = RslMLPPolicyAdapter(adapted, action_transform=clip_lite3_policy_action).eval()
    reference = RslMLPPolicyAdapter(model, action_transform=clip_lite3_policy_action).eval()
    limits = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4, device=device), residual_rate_limit=0.03)
    filters = [SafetyFilter(limits), SafetyFilter(limits)]
    maximum = {key: 0. for key in ("base_path", "zero_lora", "zero_safety", "recorded_raw")}
    for index, observation in enumerate(observations):
        obs = observation.unsqueeze(0)
        grouped = TensorDict({"policy": obs}, batch_size=[1])
        raw = model(grouped)
        sac_base = clip_lite3_policy_action(raw)
        lora_base = reference.mean_action(obs)
        mean = lora.mean_action(obs)
        state = SafetyState(**{field.name: torch.as_tensor(trace[f"safety_{field.name}"][index], device=device)
                               for field in fields(SafetyState)})
        a = filters[0].project(lora_base, mean - lora_base, state)
        b = filters[1].project(sac_base, torch.zeros_like(sac_base), state, force_zero_residual=True)
        for key, delta in {
            "base_path": lora_base - sac_base, "zero_lora": mean - sac_base,
            "zero_safety": a.action - b.action,
            "recorded_raw": raw - torch.as_tensor(trace["raw_base_action"][index], device=device),
        }.items():
            maximum[key] = max(maximum[key], float(delta.abs().max()))
    # CPU/GPU kernels may differ; recorded_raw is diagnostic, not the same-device path gate.
    return {"device": device, "observations": len(observations), "max_abs": maximum,
            "same_device_paths_passed": all(maximum[key] <= 1e-6 for key in ("base_path", "zero_lora", "zero_safety"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(args.input)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete" or sha256_file(args.base_checkpoint) != manifest["base_sha256"]:
        raise ValueError("incomplete diagnostic or mismatched base checkpoint")
    report = {"base_sha256": manifest["base_sha256"], "runs": []}
    for seed in sorted({run["seed"] for run in manifest["runs"]}):
        a, b = root / f"lora-{seed}", root / f"sac-{seed}"
        lora_trace, sac_trace = load_trace(a / "trajectory"), load_trace(b / "trajectory")
        common_steps = min(len(lora_trace["reward"]), len(sac_trace["reward"]))
        report["runs"].append({
            "seed": seed, "trajectory_difference": physical_diff(lora_trace, sac_trace),
            "reward_max_abs_difference": float(np.abs(lora_trace["reward"][:common_steps] - sac_trace["reward"][:common_steps]).max()),
            "environment_configuration_difference": configuration_diff(
                json.loads((a / "resolved_env.json").read_text()), json.loads((b / "resolved_env.json").read_text())),
            "offline_action_paths": offline_actions(sac_trace, args.base_checkpoint, args.device),
            "task_objective_parity": None if manifest.get("task_objective") is None else task_diff(lora_trace, sac_trace),
        })
    write_json_atomic(root / "runner_parity_report.json", report)
    print(json.dumps(report, indent=2))
    if any(row["task_objective_parity"] is not None and not row["task_objective_parity"]["passed"] for row in report["runs"]):
        raise SystemExit("common task reward/endings parity failed; see runner_parity_report.json")


if __name__ == "__main__":
    main()
