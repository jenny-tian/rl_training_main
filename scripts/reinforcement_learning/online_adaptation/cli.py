from __future__ import annotations

import argparse
import math

from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from rl_training.lite3_stability_contract import LITE3_MAX_PROJECTED_GRAVITY_XY
from rl_training.online_adaptation.safety import SafetyLimits, risk_gate_kwargs

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


def _positive_float(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be finite") from error
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _fraction_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]") from error
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]")
    return parsed


class _PlayArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # Keep the validation text available on SystemExit for callers/tests that
        # need to distinguish a semantic CLI rejection from argparse syntax help.
        error = SystemExit(message)
        raise error

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        validate_risk_gate_args(self, parsed)
        protocol = default_flat_online_protocol()
        if parsed.terrain_split is None:
            parsed.terrain_split = "sealed" if parsed.evaluation_only else "development"
        if parsed.mode == "online" and parsed.experiment_checkpoint_out is None:
            self.error("--experiment-checkpoint-out is required for online mode")
        if parsed.latent_code_dim and parsed.terrain_feature_source != "height_scan":
            self.error("--latent-code-dim requires --terrain-feature-source height_scan")
        if parsed.latent_code_dim and not parsed.risk_gate_enabled:
            self.error("--latent-code-dim requires --risk-gate")
        if parsed.latent_code_dim and parsed.mode == "base":
            self.error("--latent-code-dim cannot be used with --mode base")
        if parsed.latent_code_dim and parsed.latent_residual_fraction <= 0.0:
            self.error("--latent-residual-fraction must be positive when latent SAC is enabled")
        if parsed.latent_code_dim > 12:
            self.error("--latent-code-dim cannot exceed the 12-dimensional action space")
        if parsed.latent_decoder is not None and not parsed.latent_code_dim:
            self.error("--latent-decoder requires --latent-code-dim")
        if parsed.flat_retention_probe and parsed.require_track_completion:
            self.error("flat-retention probes cannot require track completion")
        if parsed.novelty_baseline_out and not (
            parsed.mode == "base" and parsed.flat_retention_probe
        ):
            self.error("--novelty-baseline-out requires a base flat-retention probe")
        if parsed.novelty_baseline_in and parsed.experiment_checkpoint_in:
            self.error("novelty baseline and experiment checkpoint are mutually exclusive")
        if bool(parsed.novelty_baseline_in) != bool(parsed.novelty_baseline_sha256):
            self.error("novelty baseline path and SHA-256 must be supplied together")
        if (
            parsed.mode == "online"
            and parsed.require_track_completion
            and parsed.experiment_checkpoint_in is None
            and parsed.novelty_baseline_in is None
        ):
            self.error("the first formal online worker requires --novelty-baseline-in")
        allowed_seeds = (
            (7, 11, 19, 23, 31)
            if parsed.flat_retention_probe
            else protocol.sealed_seeds
            if parsed.terrain_split == "sealed"
            else protocol.development_seeds
        )
        if parsed.terrain_seed not in allowed_seeds:
            self.error(f"--terrain-seed must be a {parsed.terrain_split} seed")
        return parsed


def add_risk_gate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--risk-gate", dest="risk_gate_enabled", action=argparse.BooleanOptionalAction,
        default=False, help="enable state-risk residual gating for a new experiment",
    )
    for name, default in risk_gate_kwargs(None).items():
        if name == "risk_gate_enabled":
            continue
        parser.add_argument(
            "--" + name.replace("_", "-"), default=default,
            type=_unit_interval_float if name.endswith("scale") else _nonnegative_float,
        )


def validate_risk_gate_args(parser: argparse.ArgumentParser, args) -> None:
    try:
        SafetyLimits(residual_limits=None, **risk_gate_kwargs(args))
    except ValueError as error:
        parser.error(str(error))


def append_risk_gate_options(command: list[str], args) -> None:
    # Explicitly bind even the disabled mode across episode subprocesses.
    config = risk_gate_kwargs(args)
    command.append("--risk-gate" if config.pop("risk_gate_enabled") else "--no-risk-gate")
    for name, value in config.items():
        command.extend(("--" + name.replace("_", "-"), str(value)))


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
    parser.add_argument("--flat-replay")
    parser.add_argument("--forward-speed", type=float, default=0.3)
    parser.add_argument("--residual-gain", type=_unit_interval_float, default=0.1)
    parser.add_argument("--n-step", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--flat-anchor-weight", type=_nonnegative_float, default=0.5)
    parser.add_argument("--flat-fraction-initial", type=_unit_interval_float, default=0.75)
    parser.add_argument("--flat-fraction-ready", type=_unit_interval_float, default=0.25)
    parser.add_argument("--flat-critic-warmup-updates", type=int, default=1_000)
    parser.add_argument("--terrain-step-height", type=float, default=None)
    parser.add_argument(
        "--terrain-feature-source", choices=("proprio", "geometry", "height_scan"), default="geometry"
    )
    parser.add_argument("--terrain-min-valid-fraction", type=_fraction_float, default=0.90)
    parser.add_argument("--latent-code-dim", type=_positive_int, default=0)
    parser.add_argument("--latent-residual-fraction", type=_unit_interval_float, default=0.25)
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


def build_deployment_qualify_parser() -> argparse.ArgumentParser:
    parser = _parser(
        "Qualify a frozen Lite3 base policy on the deployment-matched flat-online task."
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--task", default=FLAT_ONLINE_TASK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-episodes", "--expected_episodes", dest="expected_episodes", type=_positive_int, default=100)
    parser.add_argument("--expected-steps", "--expected_steps", dest="expected_steps", type=_positive_int, default=100_000)
    parser.add_argument("--max-project-gravity-xy", "--max_projected_gravity_xy", dest="max_projected_gravity_xy", type=float, default=LITE3_MAX_PROJECTED_GRAVITY_XY)
    parser.add_argument("--max-mean-velocity-error", type=float, default=0.35)
    parser.add_argument("--output", default="logs/online_adaptation/base_qualification_deployment.json")
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


def build_flat_online_curriculum_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervise the cross-reset Lite3 flat-online curriculum.",
        allow_abbrev=False,
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--flat-replay", required=True)
    parser.add_argument(
        "--initial-residual-checkpoint",
        help="clean residual SAC checkpoint used for the first online episode",
    )
    parser.add_argument("--forward-speed", type=float, default=0.3)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--control-timeout-ms", type=float, default=50.0)
    parser.add_argument("--configuration")
    parser.add_argument(
        "--target-semantics", choices=("legacy-done", "timeout-aware"), default=None
    )
    parser.add_argument("--n-step", type=int, choices=(1, 3, 5), default=None)
    parser.add_argument(
        "--critic-replay-mode", choices=("online-only", "dual"), default=None
    )
    parser.add_argument("--flat-fraction-initial", type=_unit_interval_float, default=None)
    parser.add_argument("--flat-fraction-ready", type=_unit_interval_float, default=None)
    parser.add_argument("--flat-anchor-weight", type=_nonnegative_float, default=None)
    parser.add_argument("--max-policy-residual-mse", type=_nonnegative_float, default=None)
    parser.add_argument(
        "--terrain-feature-source", choices=("proprio", "geometry", "height_scan"), default=None
    )
    parser.add_argument("--terrain-min-valid-fraction", type=_fraction_float, default=None)
    parser.add_argument("--flat-critic-warmup-updates", type=int, default=None)
    parser.add_argument("--max-episodes", type=_positive_int, default=None)
    parser.add_argument("--max-total-steps", type=_positive_int, default=None)
    parser.add_argument("--max-stage-episodes", type=_positive_int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def build_play_parser() -> argparse.ArgumentParser:
    protocol = default_flat_online_protocol()
    parser = _add_app_launcher_args(
        _PlayArgumentParser(
            description="Run the single-pass Lite3 online-adaptation demonstration."
        )
    )
    add_risk_gate_args(parser)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint")
    parser.add_argument("--mode", choices=("base", "residual_frozen", "online"), default="online")
    parser.add_argument("--task", default=FLAT_ONLINE_TASK)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--forward-speed", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--run-dir", default="logs/online_adaptation/demo")
    parser.add_argument("--trajectory-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--min-replay-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--updates-per-second", type=float, default=5.0)
    parser.add_argument("--actor-updates-per-second", type=float, default=2.0)
    parser.add_argument(
        "--update-schedule",
        choices=("wall-clock", "control-step"),
        default="wall-clock",
        help=(
            "use real-time background updates or synchronize at most one "
            "eligible optimizer update to each control step"
        ),
    )
    parser.add_argument(
        "--critic-updates-per-transition",
        type=_nonnegative_float,
        default=0.5,
        help="optimizer updates charged per committed critic transition",
    )
    parser.add_argument(
        "--actor-updates-per-transition",
        type=_nonnegative_float,
        default=0.005,
        help="optimizer updates charged per committed actor transition",
    )
    parser.add_argument("--novelty-calibration-steps", type=int, default=256)
    parser.add_argument("--novelty-enter-threshold", type=float, default=2.5)
    parser.add_argument("--novelty-exit-threshold", type=float, default=1.0)
    parser.add_argument("--novelty-enter-count", type=int, default=10)
    parser.add_argument("--novelty-enter-window", type=int, default=20)
    parser.add_argument("--novelty-recovery-steps", type=int, default=100)
    parser.add_argument("--adaptation-min-replay-size", type=int, default=128)
    parser.add_argument(
        "--stalled-actor-replay-stride",
        type=_positive_int,
        default=16,
        help=(
            "retain one out of every N stalled transitions after actor replay "
            "warm-up instead of freezing the replay on its first samples"
        ),
    )
    parser.add_argument("--rollback-window", type=int, default=50)
    parser.add_argument("--rollback-patience", type=int, default=4)
    parser.add_argument("--rollback-degradation-ratio", type=float, default=0.10)
    parser.add_argument("--stable-improvement-ratio", type=float, default=0.05)
    parser.add_argument("--adaptation-cooldown-steps", type=int, default=100)
    parser.add_argument("--control-timeout-ms", type=float, default=20.0)
    parser.add_argument("--stochastic-control", action="store_true")
    parser.add_argument(
        "--adapting-stochastic-control",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-critic-loss", type=_positive_float, default=800.0)
    parser.add_argument("--max-mean-td-error", type=_positive_float, default=10.0)
    parser.add_argument("--max-critic-grad-norm", type=_positive_float, default=3_000.0)
    parser.add_argument(
        "--critic-huber-delta",
        type=_positive_float,
        default=None,
        help="use Huber critic loss with this delta instead of squared TD loss",
    )
    parser.add_argument("--target-q-min", type=_finite_float, default=None)
    parser.add_argument("--target-q-max", type=_finite_float, default=None)
    parser.add_argument("--initial-alpha", type=_positive_float, default=1.0)
    parser.add_argument("--alpha-max", type=_positive_float, default=None)
    parser.add_argument(
        "--deterministic-actor-weight", type=_nonnegative_float, default=0.0
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=int, default=0, help="0 records the complete requested pass")
    parser.add_argument(
        "--terrain-stage",
        choices=tuple(stage.name for stage in protocol.stages),
        default=protocol.stages[0].name,
    )
    parser.add_argument(
        "--terrain-rough-amplitude",
        type=_nonnegative_float,
        default=None,
        help="override the selected stage roughness bound in metres for terrain scans",
    )
    parser.add_argument("--terrain-seed", type=int, default=protocol.development_seeds[0])
    parser.add_argument(
        "--terrain-split",
        choices=("development", "sealed"),
        default=None,
        help=(
            "select the terrain seed partition independently of whether online "
            "updates are enabled; defaults to sealed for --evaluation-only and "
            "development otherwise"
        ),
    )
    parser.add_argument("--protocol-sha256", default=protocol.sha256())
    parser.add_argument("--protocol-max-episodes", type=_positive_int, default=100)
    parser.add_argument("--protocol-max-total-steps", type=_positive_int, default=200_000)
    parser.add_argument("--protocol-max-stage-episodes", type=_positive_int, default=20)
    parser.add_argument("--experiment-checkpoint-in")
    parser.add_argument("--experiment-checkpoint-out")
    parser.add_argument("--novelty-baseline-in")
    parser.add_argument("--novelty-baseline-sha256")
    parser.add_argument("--novelty-baseline-out")
    parser.add_argument("--flat-replay")
    parser.add_argument("--flat-anchor-weight", type=_nonnegative_float, default=0.5)
    parser.add_argument("--max-policy-residual-mse", type=_nonnegative_float, default=None)
    parser.add_argument(
        "--terrain-feature-source", choices=("proprio", "geometry", "height_scan"), default="geometry"
    )
    parser.add_argument(
        "--residual-frame-dim",
        type=int,
        choices=(49, 59),
        default=None,
        help="require an explicit residual observation width at runtime",
    )
    parser.add_argument("--terrain-min-valid-fraction", type=_fraction_float, default=0.90)
    parser.add_argument(
        "--latent-code-dim", type=_positive_int, default=0,
        help="enable latent-space SAC with this compact steering-code dimension; requires height_scan",
    )
    parser.add_argument(
        "--latent-residual-fraction", type=_unit_interval_float, default=0.25,
        help="per-joint residual bound as a fraction of the standard action limits",
    )
    parser.add_argument(
        "--latent-decoder", default=None,
        help="optional immutable teacher decoder artifact produced by fit_low_rank_teacher_decoder",
    )
    parser.add_argument("--n-step", type=int, choices=(1, 3, 5), default=1)
    parser.add_argument(
        "--target-semantics",
        choices=("legacy-done", "timeout-aware"),
        default="timeout-aware",
    )
    parser.add_argument(
        "--critic-replay-mode", choices=("online-only", "dual"), default="dual"
    )
    parser.add_argument("--flat-fraction-initial", type=_unit_interval_float, default=0.75)
    parser.add_argument("--flat-fraction-ready", type=_unit_interval_float, default=0.25)
    parser.add_argument("--flat-critic-warmup-updates", type=int, default=1_000)
    parser.add_argument("--residual-gain", type=_unit_interval_float, default=0.0)
    parser.add_argument(
        "--adapting-residual-gain", type=_unit_interval_float, default=0.5
    )
    parser.add_argument(
        "--post-step-residual-gain", type=_unit_interval_float, default=0.25
    )
    parser.add_argument(
        "--post-step-recovery-steps", type=_positive_int, default=25,
        help="control steps used to ramp residual gain after crossing the step",
    )
    parser.add_argument(
        "--post-step-stability-weight", type=_nonnegative_float, default=3.0
    )
    parser.add_argument(
        "--post-step-progress-weight", type=_nonnegative_float, default=2.0
    )
    parser.add_argument(
        "--stall-penalty-weight", type=_nonnegative_float, default=2.0
    )
    parser.add_argument(
        "--completion-bonus", type=_nonnegative_float, default=5.0
    )
    parser.add_argument("--rough-swing-lift-weight", type=_nonnegative_float, default=0.0)
    parser.add_argument("--rough-swing-lift-target", type=_positive_float, default=0.045)
    parser.add_argument(
        "--pre-step-clearance-weight", type=_nonnegative_float, default=4.0
    )
    parser.add_argument(
        "--pre-step-clearance-target", type=_positive_float, default=0.045
    )
    parser.add_argument(
        "--pre-step-foot-forward-weight", type=_nonnegative_float, default=2.0
    )
    parser.add_argument(
        "--pre-step-foot-forward-target", type=_positive_float, default=0.04
    )
    parser.add_argument(
        "--pre-step-approach-distance", type=_positive_float, default=0.50
    )
    parser.add_argument("--residual-rate-limit", type=_nonnegative_float, default=0.03)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--flat-retention-probe", action="store_true")
    parser.add_argument(
        "--require-track-completion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="require reset-relative traversal of the complete generated track",
    )
    return parser
