"""Verify pre-reset terminal capture with one injected manager timeout.

This is a bounded wiring test.  It uses the frozen checkpoint only to produce
valid actions, disables normal failure terms, injects a four-step manager
timeout, and records both the terminal state seen by the callback and the
post-reset state.  The result is never a policy qualification or terrain
failure outcome.
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
from scripts.tools.source_terrain_policy_safety_smoke import (  # noqa: E402
    DEFAULT_MANIFEST,
    EXPECTED_MANIFEST_SHA256,
    _select_flat_smoke_cell,
)


PROBE_STEPS = 4
MAX_PROBE_STEPS = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    if manifest_path != DEFAULT_MANIFEST.resolve():
        parser.error("termination capture smoke is restricted to the registered preparation manifest")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
        parser.error("preparation manifest differs from the registered smoke contract")
    try:
        manifest, cell = _select_flat_smoke_cell(manifest_path, "flat_L1_forward_flat")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output_path = args.output.resolve()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() or temporary.exists():
        parser.error(f"refusing to overwrite probe output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(manifest["base_checkpoint"]["path"])
    if checkpoint.resolve() != DEFAULT_CHECKPOINT.resolve():
        parser.error("termination capture smoke is restricted to model_1998.pt")
    checkpoint_digest_before = _sha256(checkpoint)
    if checkpoint_digest_before != manifest["base_checkpoint"]["sha256"]:
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
            cell,
            geometry_seed,
            initial_state_seed,
            device=args.device,
            termination_probe_steps=PROBE_STEPS,
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
            record = _run_episode(
                wrapped_env,
                policy,
                cell,
                geometry_seed,
                initial_state_seed,
                episode_index=1,
                max_steps=MAX_PROBE_STEPS,
                bounded_smoke=True,
                termination_capture_test=True,
            )
            actor_digest_after = _tensor_sha256(actor)
            if actor_digest_after != actor_digest_before:
                raise RuntimeError("frozen model parameters changed during termination capture smoke")
        finally:
            wrapped_env.close()

        checkpoint_digest_after = _sha256(checkpoint)
        if checkpoint_digest_after != checkpoint_digest_before:
            raise RuntimeError("checkpoint changed during termination capture smoke")
        if record["steps"] != PROBE_STEPS:
            raise RuntimeError(f"injected timeout did not terminate at {PROBE_STEPS} steps")
        if not record["pre_reset_capture_observed"]:
            raise RuntimeError("pre-reset terminal callback did not observe a terminal sample")
        if not record["pre_reset_state_differs_from_post_reset"]:
            raise RuntimeError("captured terminal state did not differ from the post-reset state")
        report: dict[str, Any] = {
            "schema_version": "lite3_source_termination_capture_smoke_v1",
            "status": "termination_capture_test_only",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "cell_id": cell["cell_id"],
            "geometry_seed": geometry_seed,
            "initial_state_seed": initial_state_seed,
            "checkpoint_sha256": checkpoint_digest_after,
            "actor_state_sha256_before": actor_digest_before,
            "actor_state_sha256_after": actor_digest_after,
            "policy_updated": False,
            "sealed_accessed": False,
            "injected_termination": {
                "term": "time_out",
                "steps": PROBE_STEPS,
                "normal_failure_terms_disabled": [
                    "illegal_contact", "bad_orientation_2", "terrain_out_of_bounds"
                ],
            },
            "qualification_claim": False,
            "record": record,
        }
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
        print(json.dumps({"output": str(output_path), "steps": record["steps"],
                          "pre_reset_capture_observed": record["pre_reset_capture_observed"],
                          "qualification_claim": False}, indent=2))
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
