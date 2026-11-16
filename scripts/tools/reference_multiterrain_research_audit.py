"""核验官方源码快照，汇总既有训练覆盖，并离线计算净空裕量敏感性。

只读输入，JSON 写到标准输出。不导入官方下载代码，不调用仿真或优化器。
候选计算固定原状态和轨迹，只能说明奖励变化，不能预测训练后的行为。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/lite3-multiterrain-research-20261116"
TRAIN = ROOT / "logs/reference_six/low_stairs_zero_preservation_20261115/train_001_100"
SHADOW = ROOT / "logs/reference_six/low_stairs_reward_state_20261116"
CHECKPOINT_SHA256 = "290d94746c467b5cd21ef74c47a908011521c1f145da35ba7b94f1b92d9b7497"
FAMILY = {"flat": "flat", "rough": "rough", "slope_up": "slope", "slope_down": "slope",
          "stairs_up": "stairs", "stairs_down": "stairs", "friction_lateral": "friction",
          "friction_longitudinal": "friction", "slope_rough_up": "slope_rough",
          "slope_rough_down": "slope_rough"}


def read(path):
    return json.loads(path.read_text())


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size}


def aggregate_coverage(report):
    coverage = np.asarray(report["coverage_by_column_level"], dtype=np.int64)
    charged = np.asarray(report["clearance_by_column_level"], dtype=np.int64)
    assert coverage.shape == charged.shape == (20, 10)
    assert (coverage >= charged).all() and (charged >= 0).all()
    assert int(coverage.sum()) == report["interactions"]
    names = report["terrain_column_names"]
    frames = report["environment_steps"]
    columns = []
    for index, name in enumerate(names):
        count = int(coverage[index].sum())
        assert count % frames == 0
        columns.append(dict(column=index, terrain=name, environments=count // frames,
                            environment_frames=count, positive_clearance_cost_frames=int(charged[index].sum())))
    by_type = []
    for name in dict.fromkeys(names):
        selected = [item for item in columns if item["terrain"] == name]
        count = sum(item["environment_frames"] for item in selected)
        positive = sum(item["positive_clearance_cost_frames"] for item in selected)
        by_type.append(dict(terrain=name, columns=len(selected),
                            environments=sum(item["environments"] for item in selected),
                            environment_frames=count, share=count / report["interactions"],
                            positive_clearance_cost_frames=positive,
                            positive_clearance_cost_fraction=positive / count))
    family_frames = Counter()
    for item in by_type:
        family_frames[FAMILY[item["terrain"]]] += item["environment_frames"]
    return dict(total_environment_frames=report["interactions"], control_steps=frames,
                by_type=by_type, by_family=dict(family_frames),
                near_zero_command_frames=report["standing_fixed_left_fixed_right_frames"][0],
                near_zero_command_fraction=report["standing_fixed_left_fixed_right_frames"][0] / report["interactions"],
                interpretation="环境帧不等于跨阶次数；净空帧为至少一足 cost>0，非所有摆脚帧；近零命令为平移模长<0.1 且偏航绝对值<0.1。")


def cost(bottom, target, margin):
    deficit = np.maximum(target + margin - bottom, 0.0)
    squared = deficit * deficit
    return squared / (0.05 ** 2 + squared)


def sensitivity(profile, inputs, historical):
    folder = SHADOW / profile
    for filename in ("report.json", "trace.npz", "reward_state.npz"):
        path = folder / filename
        info = fingerprint(path)
        assert info["sha256"] == historical[str(path)], path
        inputs.append(info)
    report = read(folder / "report.json")
    assert report["checkpoint_sha256"] == CHECKPOINT_SHA256 and report["step_dt"] == 0.02
    with np.load(folder / "trace.npz", allow_pickle=False) as archive:
        first = archive["active"]
    with np.load(folder / "reward_state.npz", allow_pickle=False) as archive:
        bottom, target = archive["bottom"], archive["clear_after_target"]
        scored, baseline = archive["clear_after_scored"], archive["clear_after_cost"]
        recorded_rate = archive["clearance_rate"]
    assert bottom.shape == target.shape == scored.shape == baseline.shape == (*first.shape, 4)
    assert all(np.isfinite(value).all() for value in (bottom, target, baseline, recorded_rate))
    recomputed = cost(bottom, target, 0.025) * scored
    max_error = float(np.max(np.abs(recomputed - baseline)))
    assert max_error < 1e-6
    assert np.max(np.abs(-baseline.mean(axis=-1) - recorded_rate)) < 1e-6
    candidate = cost(bottom, target, 0.040) * scored
    assert (candidate >= baseline - 1e-6).all()
    assert not np.any(candidate[~scored])

    def summarize(ids):
        valid = first[:, ids]
        original = baseline[:, ids][valid]
        proposed = candidate[:, ids][valid]
        return dict(episodes=len(ids), environment_frames=int(valid.sum()),
                    original_positive_foot_frames=int((original > 0).sum()),
                    candidate_positive_foot_frames=int((proposed > 0).sum()),
                    newly_charged_foot_frames=int(((original == 0) & (proposed > 0)).sum()),
                    original_clearance_return=float(-original.mean(axis=-1, dtype=np.float64).sum() * 0.02),
                    candidate_clearance_return=float(-proposed.mean(axis=-1, dtype=np.float64).sum() * 0.02))

    records = report["records"]
    ids = [record["env_id"] for record in records]
    assert sorted(ids) == list(range(240))
    by_outcome = {}
    for name in sorted({record["outcome"] for record in records}):
        by_outcome[name] = summarize([record["env_id"] for record in records if record["outcome"] == name])
    focus = []
    for record in records:
        if record["outcome"] != "success":
            focus.append({key: record[key] for key in ("env_id", "height", "speed", "outcome")} |
                         summarize([record["env_id"]]))
    return dict(profile=profile, original_formula_max_error=max_error,
                all_first_episodes=summarize(ids), by_outcome=by_outcome, failed_episodes=focus)


def main():
    manifest = read(DOCS / "upstream_manifest.json")
    for item in manifest["files"]:
        path = Path(item["snapshot_path"])
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"], path
        assert len(data) == item["bytes"] and len(data.decode().splitlines()) == item["lines"]
    inputs = [fingerprint(path) for path in (TRAIN / "configuration.json", TRAIN / "report.json",
              ROOT / "scripts/tools/reference_low_stairs_clearance.py",
              ROOT / "scripts/tools/reference_low_stairs_rewards.py",
              ROOT / "scripts/tools/reference_low_stairs_zero_preservation_train.py")]
    checkpoint = fingerprint(TRAIN / "model_updates_10100.pt")
    assert checkpoint["sha256"] == CHECKPOINT_SHA256
    inputs.append(checkpoint)
    report, config = read(TRAIN / "report.json"), read(TRAIN / "configuration.json")
    historical_path = ROOT / "docs/lite3-low-stairs-reward-state-20261116/raw_manifest.json"
    inputs.append(fingerprint(historical_path))
    historical = {item["path"]: item["sha256"] for item in read(historical_path)}
    results = [sensitivity(profile, inputs, historical) for profile in ("nominal", "training_randomized")]
    env = config["environment"]
    policy = env["observations"]["policy"]
    data = dict(status="passed", date="2026-11-16", new_simulation_episodes=0, new_optimizer_updates=0,
                upstream_files_verified=len(manifest["files"]),
                upstream_manifest=fingerprint(DOCS / "upstream_manifest.json"),
                coverage=aggregate_coverage(report),
                policy_active_terms=[key for key, value in policy.items() if isinstance(value, dict) and "func" in value],
                policy_history_length=policy["history_length"], agent=config["agent"],
                clearance_sensitivity=dict(original_margin_m=0.025, candidate_margin_m=0.040,
                    unchanged_trajectory_and_state=True, behavior_effect_measured=False, profiles=results),
                inputs=inputs, audit_source=fingerprint(Path(__file__)))
    print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
