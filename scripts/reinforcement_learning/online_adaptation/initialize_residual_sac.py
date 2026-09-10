"""Create a clean Lite3 residual SAC initializer from the frozen flat replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from rl_training.online_adaptation.agent import AgentConfig, OnlineAdaptationAgent
from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.replay_buffer import ReplayBuffer, adapt_replay_state_frame_dim
from rl_training.online_adaptation.sac import SACConfig
from rl_training.online_adaptation.safety import SafetyLimits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_replay(path: Path, target_frame_dim: int) -> ReplayBuffer:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not bool(state.get("frozen")):
        raise ValueError("flat replay must be a frozen replay state")
    state = adapt_replay_state_frame_dim(state, target_frame_dim)
    replay = ReplayBuffer(
        capacity=int(state["capacity"]),
        history_length=int(state["history_length"]),
        frame_dim=int(state["frame_dim"]),
        action_dim=int(state["action_dim"]),
        storage_dtype=getattr(torch, state["storage_dtype"]),
    )
    replay.load_state_dict(state)
    if int(state["frame_dim"]) != target_frame_dim or int(state["action_dim"]) != 12:
        raise ValueError("clean initializer requires the verified frame/action replay schema")
    if len(replay) <= 0:
        raise ValueError("flat replay is empty")
    return replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-replay", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--warmup-updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--frame-dim", type=int, choices=(49, 59), default=49)
    parser.add_argument("--initial-alpha", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup_updates < 0 or args.batch_size <= 0:
        raise ValueError("warmup-updates must be nonnegative and batch-size must be positive")
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    replay_path = Path(args.flat_replay).resolve()
    replay = _load_frozen_replay(replay_path, args.frame_dim)
    device = torch.device(args.device)
    limits = torch.tensor([0.24, 0.20, 0.20] * 4, device=device)
    actor = ResidualActor(args.frame_dim, 12, 64, limits).to(device)
    critic = TwinCritic(args.frame_dim, 12, 64).to(device)
    config = AgentConfig(
        frame_dim=args.frame_dim,
        action_dim=12,
        batch_size=args.batch_size,
        min_replay_size=256,
        replay_capacity=10_000,
        flat_critic_warmup_updates=args.warmup_updates,
        residual_gain=0.0,
        n_step=3,
        critic_replay_mode="dual",
    )
    agent = OnlineAdaptationAgent(
        num_envs=1,
        actor=actor,
        critic=critic,
        safety_limits=SafetyLimits(residual_limits=limits),
        config=config,
        sac_config=SACConfig(flat_anchor_weight=0.5, initial_alpha=args.initial_alpha, alpha_max=args.alpha_max),
        device=device,
    )
    agent.flat_replay = replay
    # Replay indices are stored on CPU; keep its sampling generator on CPU even
    # when the networks and sampled tensors are moved to CUDA.
    stats = agent.prewarm_critics_from_flat(torch.Generator().manual_seed(args.seed))
    updater_path = output / "residual_sac_initial.pt"
    agent.updater.save(updater_path)
    torch.save(agent.state_dict(), output / "agent_initial_state.pt")
    deterministic = torch.zeros(1, 25, args.frame_dim, device=device)
    frame = torch.zeros(1, args.frame_dim, device=device)
    base = torch.zeros(1, 12, device=device)
    with torch.inference_mode():
        residual, _, _ = agent.updater.actor.sample(deterministic, frame, base, deterministic=True)
    metadata = {
        "schema_version": 1,
        "initializer": "clean_residual_sac_v1",
        "critic_action_semantics": agent.updater.CRITIC_ACTION_SEMANTICS,
        "updater_schema_version": agent.updater.CHECKPOINT_SCHEMA_VERSION,
        "replay_schema_version": 2,
        "seed": args.seed,
        "device": str(device),
        "frame_dim": args.frame_dim,
        "action_dim": 12,
        "history_length": 25,
        "actor_source": "fresh_zero_mean_actor",
        "critic_source": "fresh_critic_then_flat_replay_prewarm",
        "optimizer_source": "fresh_optimizers",
        "online_replay_size": len(agent.replay),
        "adaptation_replay_size": len(agent.adaptation_replay),
        "flat_replay_size": len(replay),
        "flat_replay_sha256": _sha256(replay_path),
        "warmup_updates_requested": args.warmup_updates,
        "warmup_updates_completed": len(stats),
        "actor_version": agent.updater.version,
        "deterministic_zero_input_residual_max_abs": float(residual.abs().max().cpu()),
        "forbidden_sources": ["v10_step2cm_safe_updater.pt", "v24_fall_softreset"],
        "updater_sha256": _sha256(updater_path),
        "agent_config": asdict(config),
        "sac_config": asdict(agent.updater.config),
    }
    (output / "initializer.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
