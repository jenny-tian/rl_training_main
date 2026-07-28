from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

SOURCE_TASK = "OnlineAdapt-Deeprobotics-Lite3-v0"
HELDOUT_TASK = "HeldoutBoxes-Deeprobotics-Lite3-v0"


def _add_app_launcher_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parser(description: str) -> argparse.ArgumentParser:
    return _add_app_launcher_args(argparse.ArgumentParser(description=description))


def build_train_parser() -> argparse.ArgumentParser:
    parser = _parser("Pretrain the Lite3 residual SAC policy on source terrains.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--task", default=SOURCE_TASK)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--log-dir", default="logs/online_adaptation/pretrain")
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--min-replay-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    return parser


def build_qualify_parser() -> argparse.ArgumentParser:
    parser = _parser("Qualify a frozen Lite3 base policy on source terrains.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--task", default=SOURCE_TASK)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 31])
    parser.add_argument("--steps-per-seed", type=int, default=1_000)
    parser.add_argument("--max-mean-velocity-error", type=float, default=0.35)
    parser.add_argument("--output", default="logs/online_adaptation/base_qualification.json")
    return parser


def build_play_parser() -> argparse.ArgumentParser:
    parser = _parser("Run the single-pass Lite3 online-adaptation demonstration.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint")
    parser.add_argument("--mode", choices=("base", "residual_frozen", "online"), default="online")
    parser.add_argument("--task", default=HELDOUT_TASK)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--forward-speed", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--run-dir", default="logs/online_adaptation/demo")
    parser.add_argument("--min-replay-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--updates-per-second", type=float, default=5.0)
    parser.add_argument("--actor-updates-per-second", type=float, default=1.0)
    parser.add_argument("--novelty-calibration-steps", type=int, default=256)
    parser.add_argument("--novelty-enter-threshold", type=float, default=2.5)
    parser.add_argument("--novelty-exit-threshold", type=float, default=1.0)
    parser.add_argument("--novelty-enter-count", type=int, default=10)
    parser.add_argument("--novelty-enter-window", type=int, default=20)
    parser.add_argument("--novelty-recovery-steps", type=int, default=100)
    parser.add_argument("--adaptation-min-replay-size", type=int, default=128)
    parser.add_argument("--rollback-window", type=int, default=50)
    parser.add_argument("--rollback-patience", type=int, default=2)
    parser.add_argument("--rollback-degradation-ratio", type=float, default=0.10)
    parser.add_argument("--stable-improvement-ratio", type=float, default=0.05)
    parser.add_argument("--adaptation-cooldown-steps", type=int, default=100)
    parser.add_argument("--control-timeout-ms", type=float, default=20.0)
    parser.add_argument("--stochastic-control", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=int, default=0, help="0 records the complete requested pass")
    return parser
