"""Bounded Source L1 runtime diagnostics, never a qualification scan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/rl_training"))

from rl_training.source_scan.manifest import load_manifest
from scripts.reinforcement_learning.online_adaptation.run_source_qualification_scan import (
    DEFAULT_CHECKPOINT, _make_wrapped_env, _run_episode, _seeded_wrapper_reset,
    _sha256, _tensor_sha256,
)

MANIFEST = ROOT / "docs/lite3-source-scan-preparation-v0.2-2026-10-22/source_qualification_scan_manifest_v0.2_preparation.json"
MANIFEST_SHA256 = "dba281ad8a2fe31adae410c4df65947b51886ceb597c5472f3ca38bcfbaa6c54"
CELL_IDS = (
    "slope_L1_uphill", "slope_L1_downhill",
    "multi_step_stairs_L1_ascending", "multi_step_stairs_L1_descending",
)
MAX_STEPS = 1600


def select_cell(cell_id: str):
    if cell_id not in CELL_IDS:
        raise ValueError("only the four registered L1 diagnostic cells are allowed")
    if _sha256(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("preparation manifest changed")
    manifest = load_manifest(MANIFEST)
    if manifest["status"] != "draft_not_for_run" or manifest["execution_authorized"] is not False:
        raise ValueError("diagnostic requires the non-executable preparation draft")
    if manifest["split"] != "source_qualification_scan":
        raise ValueError("Source-only manifest required")
    for record in manifest["execution_contract_draft"]["artifacts"].values():
        if "path" in record and _sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"registered artifact changed: {record['path']}")
    checkpoint = Path(manifest["base_checkpoint"]["path"])
    if checkpoint.resolve() != DEFAULT_CHECKPOINT.resolve() or _sha256(checkpoint) != manifest["base_checkpoint"]["sha256"]:
        raise ValueError("frozen checkpoint differs from registration")
    cells = [cell for cell in manifest["seed_plan"]["cells"] if cell["cell_id"] == cell_id]
    if len(cells) != 1:
        raise ValueError("expected exactly one registered cell")
    return manifest, cells[0]


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True, choices=CELL_IDS)
    parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.headless:
        parser.error("this diagnostic requires --headless")
    manifest, cell = select_cell(args.cell_id)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the output before starting Isaac so interrupted attempts remain visible.
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"status": "started_not_completed", "cell_id": args.cell_id}, stream)
    checkpoint = Path(manifest["base_checkpoint"]["path"])
    source_files = [Path(__file__).resolve(),
                    ROOT / "scripts/reinforcement_learning/online_adaptation/run_source_qualification_scan.py",
                    ROOT / "source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py",
                    ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_isaac.py"]
    source_hashes = {str(path): _sha256(path) for path in source_files}
    started = datetime.now(timezone.utc).isoformat()
    app = AppLauncher(args).app
    try:
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from rl_training.online_adaptation.rsl_rl_compat import freeze_rsl_rl_models, load_frozen_rsl_rl_checkpoint

        geometry_seed = cell["geometry_seed_range"][0]
        state_seed = cell["initial_state_seed_range"][0]
        env, cfg = _make_wrapped_env(cell, geometry_seed, state_seed, device=args.device)
        try:
            runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=None, device=args.device)
            load_frozen_rsl_rl_checkpoint(runner, str(checkpoint))
            modules = freeze_rsl_rl_models(runner)
            before = [_tensor_sha256(module) for module in modules]
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            record = _run_episode(env, policy, cell, geometry_seed, state_seed,
                                  episode_index=1, max_steps=MAX_STEPS, bounded_smoke=True)
            observation, _ = _seeded_wrapper_reset(env, state_seed)
            reset_root = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().tolist()
            observations = list(observation.values()) if hasattr(observation, "values") else [observation]
            reset_check = {
                "root_position_w_m": reset_root,
                "finite_observation": all(bool(torch.isfinite(value).all()) for value in observations),
                "episode_length_zero": bool((env.unwrapped.episode_length_buf == 0).all()),
                "root_xy_matches_origin": bool(torch.allclose(
                    env.unwrapped.scene["robot"].data.root_pos_w[:, :2],
                    env.unwrapped.scene.terrain.env_origins[:, :2], atol=1e-6)),
            }
            after = [_tensor_sha256(module) for module in modules]
            if before != after:
                raise RuntimeError("frozen model state changed")
            if not all(reset_check[key] for key in ("finite_observation", "episode_length_zero", "root_xy_matches_origin")):
                raise RuntimeError(f"explicit reset check failed: {reset_check}")
        finally:
            env.close()
        if _sha256(checkpoint) != manifest["base_checkpoint"]["sha256"]:
            raise RuntimeError("checkpoint bytes changed")
        if any(_sha256(Path(path)) != digest for path, digest in source_hashes.items()):
            raise RuntimeError("runtime source changed during diagnostic")
        report = {
            "schema_version": "lite3_source_family_runtime_smoke_v1",
            "status": "bounded_runtime_diagnostic_completed",
            "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": MANIFEST_SHA256, "checkpoint_sha256": manifest["base_checkpoint"]["sha256"],
            "source_sha256": source_hashes, "cell": cell,
            "headless": True, "max_steps": MAX_STEPS,
            "termination_probe_injected": False,
            "policy_updated": False, "sealed_accessed": False, "qualification_claim": False,
            "frozen_module_sha256_before": before, "frozen_module_sha256_after": after,
            "explicit_reset_check": reset_check, "record": record,
        }
        temporary = output.with_suffix(".json.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(output)
        print(json.dumps({"output": str(output), "steps": record["steps"],
                          "termination_reason": record["termination_reason"]}), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
