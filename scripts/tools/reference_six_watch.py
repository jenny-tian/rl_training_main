"""Read-only training observer; writes progress reports without changing a run.

The training supervisor owns numerical-error shutdown and the update budget.
This observer never starts/restarts training or launches the next phase.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import subprocess
import time
from zoneinfo import ZoneInfo


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def save(path, value):
    write(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def log_metrics(path):
    # Only read the tail; the reference asset emits many startup visual warnings.
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        text = stream.read().decode(errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    markers = list(re.finditer(r"REFERENCE SIX train: (\d+)/(\d+)", text))
    if not markers:
        return {}
    last = markers[-1]
    start = markers[-2].end() if len(markers) > 1 else 0
    block = text[start:last.start()]
    result = {"completed_updates": int(last.group(1))}
    names = {"Mean reward": "mean_reward", "Mean episode length": "mean_episode_length_steps",
             "Metrics/base_velocity/error_vel_xy": "logged_velocity_xy_error",
             "Metrics/base_velocity/error_vel_yaw": "logged_velocity_yaw_error",
             "Curriculum/terrain_levels": "mean_terrain_level",
             "Episode_Termination/bad_orientation_2": "logged_orientation_termination",
             "Episode_Termination/time_out": "logged_time_out"}
    for label, name in names.items():
        match = re.search(re.escape(label) + r":\s*([-+0-9.eE]+)", block)
        if match:
            result[name] = float(match.group(1))
    return result


def snapshot(run):
    execution, report = read(run / "execution.json"), read(run / "report.json")
    if execution is None or report is None:
        return None
    completed = report["completed_updates"]
    overrides = read(run / "runtime_overrides.json")
    if overrides is None:
        return None
    budget = overrides["max_iterations"]
    rows = report["updates"][-50:]
    seconds = statistics.mean(r["collect_s"] + r["learn_s"] for r in rows) if rows else None
    checkpoints = sorted((run / "training").glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    status = "running"
    if execution["status"] == "failed":
        status = "failed"
    elif execution["status"] == "completed":
        status = "completed" if report["status"] == "passed" and completed == budget else "incomplete"
    result = dict(observed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                  status=status, source_commit=execution["source_commit"], physical_gpu=execution["physical_gpu"],
                  worker_pid=execution["pid"], completed_updates=completed, budget_updates=budget,
                  progress_percent=100 * completed / budget, interactions=report["environment_steps"] * overrides["num_envs"],
                  from_scratch=report["from_scratch"], checkpoint_loaded=report["checkpoint_loaded"],
                  average_seconds_per_update_last_50=seconds,
                  estimated_remaining_hours=(budget - completed) * seconds / 3600 if seconds is not None else None,
                  latest_update=rows[-1] if rows else None, logged_metrics=log_metrics(run / "runtime.log"),
                  report_age_seconds=time.time() - (run / "report.json").stat().st_mtime,
                  latest_checkpoint=str(checkpoints[-1]) if checkpoints else None,
                  checkpoint_iteration_is_zero_based=True,
                  resets=report["resets"], terminated=report["terminated"], truncated=report["truncated"],
                  final_checkpoint=report.get("output_checkpoint"), final_checkpoint_sha256=report.get("output_sha256"),
                  failure=execution.get("failure"), next_phase_started=False)
    error = read(run / "error.json")
    if error:
        result["worker_error"] = error
    if execution["status"] == "running" and not Path(f"/proc/{execution['pid']}").exists():
        # The supervisor may still be writing its terminal state. Recheck after
        # its normal two-second polling interval instead of declaring success.
        result["status"] = "awaiting_supervisor_result"
    try:
        memory = subprocess.check_output(["nvidia-smi", f"--id={execution['physical_gpu']}",
                  "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        used, free, utilization = [int(value.strip()) for value in memory.strip().split(",")]
        result["gpu"] = dict(total_used_mib=used, free_mib=free, utilization_percent=utilization)
    except (subprocess.CalledProcessError, ValueError):
        result["gpu"] = None
    return result


def markdown(value):
    last = value["latest_update"] or {}
    metrics = value["logged_metrics"]
    eta = value["estimated_remaining_hours"]
    lines = ["# 六类地形从头训练进度", "", f"记录时间：{value['observed_at']}", "",
             f"状态：{value['status']}；已完成 {value['completed_updates']} / {value['budget_updates']} 轮。",
             f"累计训练交互：{value['interactions']:,}。", "",
             f"最近最多 50 轮平均耗时：{value['average_seconds_per_update_last_50']} 秒/轮。",
             f"预计剩余：{eta:.2f} 小时。" if eta is not None else "尚无足够更新记录估算剩余时间。", "",
             f"当前学习率：{last.get('learning_rate')}。",
             f"当前平均地形等级：{last.get('mean_terrain_level')}。",
             f"最新损失：{last.get('losses')}。", "",
             f"训练日志指标对应第 {metrics.get('completed_updates')} 轮：",
             f"平均奖励 {metrics.get('mean_reward')}；平均回合步数 {metrics.get('mean_episode_length_steps')}；",
             f"日志速度误差 xy={metrics.get('logged_velocity_xy_error')}，yaw={metrics.get('logged_velocity_yaw_error')}。",
             "这些是参考训练日志指标，不等同于独立冻结策略评估结果。", "",
             f"显存与利用率：{value['gpu']}。",
             f"最近检查点：{value['latest_checkpoint']}。",
             "检查点文件名中的迭代号从 0 开始；更新完成轮数以本报告为准。", "",
             f"累计 reset={value['resets']}，终止={value['terminated']}，超时={value['truncated']}。",
             "终止和超时可能重叠，不能相加作为总回合数。", "",
             "训练从随机参数和空优化器状态开始，没有加载历史检查点。",
             "达到授权预算后停止；独立评估或追加训练等待用户确认。", ""]
    if value["status"] == "completed":
        lines += [f"最终模型：{value['final_checkpoint']}", f"SHA256：{value['final_checkpoint_sha256']}", ""]
    if value["failure"] or value.get("worker_error"):
        lines += ["失败信息：", "```", str(value["failure"] or value["worker_error"]), "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / "execution.json").exists():
        parser.error("the training supervisor has not created this run")
    milestones = run / "milestones"
    observed_thresholds = {int(path.stem.split("_")[1]) for path in milestones.glob("milestone_*.json")}
    missing_supervisor = 0
    while True:
        value = snapshot(run)
        if value is not None:
            missing_supervisor = missing_supervisor + 1 if value["status"] == "awaiting_supervisor_result" else 0
            if missing_supervisor >= 3:
                value.update(status="incomplete", failure="worker exited without a completed supervisor report")
            save(run / "progress.json", value)
            write(run / "progress.md", markdown(value))
            threshold = value["completed_updates"] // 1000 * 1000
            if threshold and threshold not in observed_thresholds:
                observed_thresholds.add(threshold)
                stem = f"milestone_{threshold:05d}_observed_{value['completed_updates']:05d}"
                save(milestones / f"{stem}.json", value)
                write(milestones / f"{stem}.md", markdown(value))
            print(json.dumps({key: value[key] for key in ("observed_at", "status", "completed_updates", "estimated_remaining_hours")}), flush=True)
            if value["status"] in ("completed", "failed", "incomplete"):
                save(run / "phase_result.json", value)
                write(run / "phase_result.md", markdown(value))
                return 0 if value["status"] == "completed" else 1
        if args.once:
            return 0 if value is not None else 1
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
