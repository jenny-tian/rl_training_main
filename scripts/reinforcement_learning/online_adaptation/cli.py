from __future__ import annotations

import argparse

from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol

SOURCE_TASK = "OnlineAdapt-Deeprobotics-Lite3-v0"
STRICT_FLAT_BASELINE_TASK = "Flat-Deeprobotics-Lite3-v0"
HELDOUT_TASK = "HeldoutBoxes-Deeprobotics-Lite3-v0"
FLAT_ONLINE_TASK = "FlatOnline-Deeprobotics-Lite3-v0"
FORMAL_FLAT_REPLAY_TRANSITIONS = 100_000


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an exact positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be an exact positive integer")
    return parsed


def _formal_transition_count(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < FORMAL_FLAT_REPLAY_TRANSITIONS:
        raise argparse.ArgumentTypeError(
            f"formal collection requires at least {FORMAL_FLAT_REPLAY_TRANSITIONS} transitions"
        )
    return parsed


def _unit_interval_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite scalar in [0, 1]") from error
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite scalar in [0, 1]")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be finite and nonnegative") from error
    if parsed != parsed or parsed == float("inf") or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


class _PlayArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        protocol = default_flat_online_protocol()
        if parsed.mode == "online" and parsed.experiment_checkpoint_out is None:
            self.error("--experiment-checkpoint-out is required for online mode")
        allowed_seeds = (
            protocol.sealed_seeds if parsed.evaluation_only else protocol.development_seeds
        )
        if parsed.terrain_seed not in allowed_seeds:
            kind = "sealed" if parsed.evaluation_only else "development"
            self.error(f"--terrain-seed must be a {kind} seed")
        return parsed


def _add_app_launcher_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parser(
    description: str,
    *,
    allow_abbrev: bool = True,
) -> argparse.ArgumentParser:
    return _add_app_launcher_args(
        argparse.ArgumentParser(description=description, allow_abbrev=allow_abbrev)
    )


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
    parser = _parser("Qualify a frozen Lite3 base policy on the strict-flat baseline task.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--task", default=STRICT_FLAT_BASELINE_TASK)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 31])
    parser.add_argument("--steps-per-seed", type=int, default=1_000)
    parser.add_argument("--max-mean-velocity-error", type=float, default=0.35)
    parser.add_argument("--output", default="logs/online_adaptation/base_qualification.json")
    return parser


def build_flat_baseline_manifest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record strict-flat Lite3 PPO baseline provenance.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--training-command-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default=STRICT_FLAT_BASELINE_TASK)
    return parser


def build_collect_flat_replay_parser() -> argparse.ArgumentParser:
    parser = _parser(
        description="Collect immutable strict-flat Lite3 transitions from a frozen PPO.",
        allow_abbrev=False,
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default=STRICT_FLAT_BASELINE_TASK)
    parser.add_argument("--seed", type=_positive_int, default=42)
    parser.add_argument(
        "--transitions",
        type=_formal_transition_count,
        default=FORMAL_FLAT_REPLAY_TRANSITIONS,
    )
    return parser


def build_play_parser() -> argparse.ArgumentParser:
    protocol = default_flat_online_protocol()
    parser = _add_app_launcher_args(
        _PlayArgumentParser(
            description="Run the single-pass Lite3 online-adaptation demonstration."
        )
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint")
    parser.add_argument("--mode", choices=("base", "residual_frozen", "online"), default="online")
    parser.add_argument("--task", default=FLAT_ONLINE_TASK)
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
    parser.add_argument(
        "--terrain-stage",
        choices=tuple(stage.name for stage in protocol.stages),
        default=protocol.stages[0].name,
    )
    parser.add_argument("--terrain-seed", type=int, default=protocol.development_seeds[0])
    parser.add_argument("--protocol-sha256", default=protocol.sha256())
    parser.add_argument("--experiment-checkpoint-in")
    parser.add_argument("--experiment-checkpoint-out")
    parser.add_argument("--flat-replay")
    parser.add_argument("--flat-anchor-weight", type=_nonnegative_float, default=0.0)
    parser.add_argument("--n-step", type=int, choices=(1, 3, 5), default=1)
    parser.add_argument("--residual-gain", type=_unit_interval_float, default=0.0)
    parser.add_argument("--evaluation-only", action="store_true")
    return parser
