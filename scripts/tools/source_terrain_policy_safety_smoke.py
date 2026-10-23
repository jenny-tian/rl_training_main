"""Run a bounded, non-qualifying frozen-policy/SafetyFilter Flat smoke.

This checks the checkpoint, observation/action wiring and one capped simulator
segment. It is not a complete episode qualification or a Source capability scan.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/rl_training"))

from rl_training.source_scan.manifest import load_manifest  # noqa: E402
from scripts.reinforcement_learning.online_adaptation.run_source_qualification_scan import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    _make_wrapped_env,
    _run_episode,
    _sha256,
    _tensor_sha256,
)

DEFAULT_MANIFEST = (
    ROOT
    / "docs/lite3-source-scan-preparation-v0.2-2026-10-22/"
      "source_qualification_scan_manifest_v0.2_preparation.json"
)
MAX_SMOKE_STEPS = 200
EXPECTED_MANIFEST_SHA256 = "dba281ad8a2fe31adae410c4df65947b51886ceb597c5472f3ca38bcfbaa6c54"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cell-id", default="flat_L1_forward_flat")
    parser.add_argument("--max-steps", type=int, default=MAX_SMOKE_STEPS)
    parser.add_argument("--output", type=Path)
    return parser


def _select_flat_smoke_cell(manifest_path: Path, cell_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if manifest_path.resolve() != DEFAULT_MANIFEST.resolve():
        raise ValueError("bounded smoke is restricted to the registered preparation manifest")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("preparation manifest differs from the registered smoke contract")
    manifest = load_manifest(manifest_path)
    if manifest.get("manifest_id") != "lite3_source_qualification_scan_20261022_preparation_v0_2":
        raise ValueError("unexpected Source scan preparation manifest")
    if manifest.get("status") != "draft_not_for_run" or manifest.get("execution_authorized") is not False:
        raise ValueError("bounded smoke expects the unapproved preparation manifest")
    if manifest.get("split") != "source_qualification_scan":
        raise ValueError("bounded smoke requires the Source-only preparation manifest")
    cells = manifest.get("seed_plan", {}).get("cells", [])
    matches = [cell for cell in cells if cell.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest cell named {cell_id!r}")
    cell = matches[0]
    if cell.get("terrain_family") != "flat" or cell.get("level") != 1:
        raise ValueError("policy safety smoke is restricted to the Flat L1 cell")
    return manifest, cell


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.output is None:
        parser.error("--output is required")
    if type(args.max_steps) is not int or not 1 <= args.max_steps <= MAX_SMOKE_STEPS:
        parser.error(f"--max-steps must be in [1, {MAX_SMOKE_STEPS}]")
    try:
        manifest, cell = _select_flat_smoke_cell(args.manifest, args.cell_id)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output_path = args.output.resolve()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() or temporary.exists():
        parser.error(f"refusing to overwrite smoke output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(manifest["base_checkpoint"]["path"])
    if checkpoint.resolve() != DEFAULT_CHECKPOINT.resolve():
        parser.error("smoke is restricted to the registered model_1998.pt checkpoint")
    expected_digest = manifest["base_checkpoint"].get("sha256")
    try:
        checkpoint_digest_before = _sha256(checkpoint)
    except OSError as error:
        parser.error(str(error))
    if checkpoint_digest_before != expected_digest:
        parser.error("model_1998.pt digest differs from the preparation manifest")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    started_at = _utc_now()
    try:
        from rsl_rl.runners import OnPolicyRunner

        from rl_training.online_adaptation.rsl_rl_compat import (
            freeze_rsl_rl_models,
            load_frozen_rsl_rl_checkpoint,
        )

        geometry_seed = cell["geometry_seed_range"][0]
        initial_state_seed = cell["initial_state_seed_range"][0]
        wrapped_env, agent_cfg = _make_wrapped_env(
            cell, geometry_seed, initial_state_seed, device=args.device
        )
        try:
            runner = OnPolicyRunner(
                wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device
            )
            load_frozen_rsl_rl_checkpoint(runner, str(checkpoint))
            frozen_modules = freeze_rsl_rl_models(runner)
            actor = getattr(runner.alg, "actor", None) or getattr(runner.alg, "policy", None)
            if actor is None:
                actor = frozen_modules[0]
            actor_digest_before = _tensor_sha256(actor)
            policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
            episode = _run_episode(
                wrapped_env,
                policy,
                cell,
                geometry_seed,
                initial_state_seed,
                episode_index=1,
                max_steps=args.max_steps,
                bounded_smoke=True,
            )
            actor_digest_after = _tensor_sha256(actor)
            if actor_digest_after != actor_digest_before:
                raise RuntimeError("frozen model parameters changed during bounded smoke")
        finally:
            wrapped_env.close()

        checkpoint_digest_after = _sha256(checkpoint)
        if checkpoint_digest_after != checkpoint_digest_before:
            raise RuntimeError("checkpoint changed during bounded smoke")
        report = {
            "schema_version": "lite3_source_policy_safety_smoke_v1",
            "status": "bounded_policy_safety_smoke",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "cell_id": cell["cell_id"],
            "terrain_family": cell["terrain_family"],
            "level": cell["level"],
            "geometry_seed": geometry_seed,
            "initial_state_seed": initial_state_seed,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_digest_after,
            "actor_state_sha256_before": actor_digest_before,
            "actor_state_sha256_after": actor_digest_after,
            "policy_updated": False,
            "sealed_accessed": False,
            "bounded_by_steps": args.max_steps,
            "steps_executed": episode["steps"],
            "complete_episode": False,
            "qualification_claim": False,
            "episode_metrics_are_diagnostic_only": True,
            "record": episode,
        }
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
        print(json.dumps({"output": str(output_path), "steps": episode["steps"],
                          "qualification_claim": False}, indent=2))
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
