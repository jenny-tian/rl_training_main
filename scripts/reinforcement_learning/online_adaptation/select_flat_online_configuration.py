from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol


class QCalibrationRequired(RuntimeError):
    pass


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_development_candidate(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir)
    summary = _json_object(directory / "development_summary.json")
    if "sealed_seeds" in summary:
        raise ValueError("development selection must not read sealed results")
    summary["source_run_dir"] = str(directory.resolve())
    summary["source_summary_sha256"] = sha256_file(directory / "development_summary.json")
    return summary


def _validate_comparable(candidates: Sequence[Mapping[str, object]]) -> None:
    if not candidates:
        raise ValueError("at least one development candidate is required")
    reference = candidates[0]
    for name in ("development_seeds", "base_manifest_sha256", "budgets"):
        if name not in reference or any(candidate.get(name) != reference[name] for candidate in candidates[1:]):
            raise ValueError(f"development candidates differ in {name}")
    protocol = default_flat_online_protocol()
    if tuple(reference["development_seeds"]) != protocol.development_seeds:
        raise ValueError("development seeds differ from the protocol")


def _finite_metric(candidate: Mapping[str, object], name: str) -> float:
    value = candidate.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate metric {name} must be finite")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"candidate metric {name} must be finite")
    return value


def select_best_candidate(candidates: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    _validate_comparable(candidates)
    return dict(
        min(
            candidates,
            key=lambda candidate: (
                -_finite_metric(candidate, "pass_rate"),
                _finite_metric(candidate, "falls"),
                _finite_metric(candidate, "safety_aborts"),
                _finite_metric(candidate, "flat_retention_cost"),
                _finite_metric(candidate, "interaction_steps"),
            ),
        )
    )


def q_calibration(candidate: Mapping[str, object]) -> dict[str, object]:
    distributions = candidate.get("q_distributions")
    if not isinstance(distributions, Mapping):
        raise QCalibrationRequired("q_calibration_required")
    output: dict[str, object] = {"status": "passed", "distributions": {}}
    scales = []
    for name in ("flat_q", "online_q", "target_q", "td_error"):
        values = distributions.get(name)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or abs(float(value)) > 1.0e6
                for value in values
            )
        ):
            raise QCalibrationRequired("q_calibration_required")
        normalized = [float(value) for value in values]
        scale = max(abs(value) for value in normalized)
        scales.append(max(scale, 1.0e-12))
        output["distributions"][name] = {
            "minimum": min(normalized),
            "maximum": max(normalized),
            "mean": sum(normalized) / len(normalized),
            "count": len(normalized),
        }
    if max(scales) / min(scales) > 100.0:
        raise QCalibrationRequired("q_calibration_required")
    return output


def write_selection(candidates: Sequence[Mapping[str, object]], output: str | Path) -> dict[str, Any]:
    selected = select_best_candidate(candidates)
    calibration = q_calibration(selected)
    selected["protocol_sha256"] = default_flat_online_protocol().sha256()
    selected["source_hashes"] = {
        candidate["source_run_dir"]: candidate["source_summary_sha256"]
        for candidate in candidates
    }
    target = Path(output)
    write_json_atomic(target, selected)
    write_json_atomic(target.with_name("q_calibration.json"), calibration)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    target = subparsers.add_parser("select-target")
    target.add_argument("--n-step-1", required=True)
    target.add_argument("--n-step-3", required=True)
    target.add_argument("--n-step-5", required=True)
    target.add_argument("--output", default="selected_target_configuration.json")
    final = subparsers.add_parser("select-final")
    final.add_argument("--target-configuration", required=True)
    final.add_argument("--online-only", required=True)
    final.add_argument("--dual-without-anchor", required=True)
    final.add_argument("--dual-with-anchor", required=True)
    final.add_argument("--output", default="selected_development_configuration.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "select-target":
        directories = (args.n_step_1, args.n_step_3, args.n_step_5)
        candidates = [load_development_candidate(path) for path in directories]
        for expected_n_step, candidate in zip((1, 3, 5), candidates, strict=True):
            configuration = candidate.get("target_configuration", candidate)
            if (
                not isinstance(configuration, Mapping)
                or configuration.get("target_semantics") != "timeout-aware"
                or configuration.get("n_step") != expected_n_step
            ):
                raise ValueError("target candidates must be timeout-aware 1-, 3-, and 5-step runs")
        write_selection(candidates, args.output)
        return 0
    directories = (args.online_only, args.dual_without_anchor, args.dual_with_anchor)
    candidates = [load_development_candidate(path) for path in directories]
    target_hash = sha256_file(args.target_configuration)
    if any(candidate.get("target_configuration_sha256") != target_hash for candidate in candidates):
        raise ValueError("final candidates differ from selected target configuration")
    configurations = [candidate.get("target_configuration", candidate) for candidate in candidates]
    expected = (("online-only", 0.0), ("dual", 0.0), ("dual", None))
    for configuration, (replay_mode, anchor_weight) in zip(configurations, expected, strict=True):
        if not isinstance(configuration, Mapping) or configuration.get("critic_replay_mode") != replay_mode:
            raise ValueError("final candidates are not the required replay/anchor ablations")
        weight = configuration.get("flat_anchor_weight")
        if (anchor_weight is None and not isinstance(weight, (int, float))) or (
            anchor_weight is None and float(weight) <= 0.0
        ) or (anchor_weight is not None and weight != anchor_weight):
            raise ValueError("final candidates are not the required replay/anchor ablations")
    write_selection(candidates, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
