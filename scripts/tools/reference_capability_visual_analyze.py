"""Build side-by-side videos and a local gallery from verified frozen replays."""
from __future__ import annotations

import html
from pathlib import Path
import shutil
import subprocess

import numpy as np

from reference_capability_visual_run import OUTPUT, DOCS
from reference_capability_visual_replay import read, load_npz
from reference_stairs_eval import save, sha
from reference_core500_analyze import write_csv

LABELS = {
    "rough_randomized": ("随机化粗糙地面", "第 1 行，起点 22，前进 0.4 m/s；比较方向保持。"),
    "startup_randomized": ("随机化平地起步", "起点 5；前 0.5 s 零命令，随后加速。0.4 倍慢放。"),
    "flat": ("普通平地", "起点 0，前进 0.4 m/s。"),
    "rough_nominal": ("普通粗糙地面", "第 8 行，起点 30，前进 0.4 m/s。"),
    "slope_up": ("普通上坡", "第 8 行，起点 50，前进 0.4 m/s。"),
    "slope_down": ("普通下坡", "第 8 行，起点 70，前进 0.4 m/s。"),
    "friction_lateral": ("横向摩擦分区", "第 8 行，起点 90，前进 0.4 m/s；画面颜色不代表摩擦系数。"),
    "friction_longitudinal": ("纵向摩擦分区", "第 8 行，起点 110，前进 0.4 m/s。"),
    "slope_rough_up": ("粗糙上坡", "普通参数，第 8 行，起点 130，前进 0.4 m/s。"),
    "slope_rough_down": ("粗糙下坡", "普通参数，第 8 行，起点 156；关注 B 超时前的推进。"),
    "turn_right": ("中速右转", "普通参数，起点 100；2 s 后给定 -0.5 rad/s，前进速度为零。"),
    "heading_right": ("右侧航向目标", "普通参数，起点 260；2 s 后目标为向右转 90°，查看角度残差。"),
    "arc_left": ("边走边左转", "普通参数，起点 280；2 s 后前进 0.3 m/s、左转 0.5 rad/s。"),
    "stairs_up_5cm": ("5 cm 上台阶", "普通参数，起点 200，前进 0.4 m/s，带既有航向反馈。"),
    "stairs_down_5cm": ("5 cm 下台阶", "普通参数，起点 200，前进 0.4 m/s，带既有航向反馈。"),
}
OUTCOMES = {"success": "完成", "survived": "存活至结束", "lateral_exit": "横向越界",
            "orientation_failure": "姿态异常终止", "timeout": "超时", "stuck": "卡住"}


def metrics(job_path, clip):
    data = load_npz(job_path/"trace.npz")
    i = clip["env_id"]
    n = int(data["active"][:, i].sum())
    t = np.arange(n)*.02
    mask = (t >= 4.) & (t < 8.)
    velocity = data["velocity"][:n, i]
    command = data["command"][:n, i]
    result = dict(duration_s=clip["record"]["duration_s"], outcome=clip["record"]["outcome"],
        complete_4_8s=bool(mask.sum() == 200),
        yaw_actual_4_8s=float(velocity[mask, 5].mean()) if mask.any() else None,
        yaw_mae_4_8s=float(np.abs(velocity[mask, 5]-command[mask, 2]).mean()) if mask.any() else None,
        forward_actual_4_8s=float(velocity[mask, 0].mean()) if mask.any() else None,
        lateral_actual_4_8s=float(velocity[mask, 1].mean()) if mask.any() else None,
        terminal_heading_error_deg=None,
        max_tilt_deg=float(np.degrees(np.arccos(np.clip(-data["gravity"][:n, i, 2], -1, 1))).max()))
    if clip["name"] == "heading_right":
        result["terminal_heading_error_deg"] = float(np.degrees(abs(data["heading_error"][n-1, i])))
    del data
    return result


def pair_clips(old, new, dest):
    import imageio.v2 as imageio
    from PIL import Image
    assert old["fps"] == new["fps"]
    old_path, new_path = Path(old["path"]), Path(new["path"])
    readers = [imageio.get_reader(p/"rollout.mp4") for p in (old_path, new_path)]
    expected = [old["total_frames"], new["total_frames"]]
    last = [None, None]
    writer = imageio.get_writer(dest/"comparison.mp4", fps=old["fps"], codec="libx264",
        quality=7, macro_block_size=2, ffmpeg_params=["-threads", "2"])
    try:
        for index in range(max(expected)):
            for side in range(2):
                if index < expected[side]:
                    last[side] = readers[side].get_next_data()
                assert last[side].shape == (540, 960, 3)
            writer.append_data(np.concatenate(last, axis=1))
        for side in range(2):
            try:
                readers[side].get_next_data()
            except (IndexError, StopIteration):
                pass
            else:
                raise AssertionError("video contains unaccounted frames")
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    for name in ("first", "middle", "terminal"):
        images = [np.asarray(Image.open(p/f"{name}.png").convert("RGB")) for p in (old_path, new_path)]
        imageio.imwrite(dest/f"{name}.png", np.concatenate(images, axis=1))
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(dest/"comparison.mp4"), "-f", "null", "-"], check=True)
    probe = read_probe(dest/"comparison.mp4")
    assert int(probe["nb_frames"]) == max(expected)
    assert (probe["width"], probe["height"]) == (1920, 540)
    return dict(frames=max(expected), fps=old["fps"], video_duration_s=max(expected)/old["fps"],
        shorter_clip_holds_terminal_state=True, decode="passed",
        paired_videos_share_first_sample_time_s=.02,
        terminal_sample_timing_note="The last interval can be shorter than the regular sampling stride; on-screen time is authoritative.")


def read_probe(path):
    import json
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)], text=True))["streams"][0]


def gallery(rows):
    cards = []
    for row in rows:
        scene = row["scene"]
        title, detail = LABELS[scene]
        old, new = row["metrics"]["zero10100"], row["metrics"]["B"]
        outcome = f"旧模型：{OUTCOMES[old['outcome']]}，{old['duration_s']:.2f} s；B：{OUTCOMES[new['outcome']]}，{new['duration_s']:.2f} s"
        status = "两模型均逐帧复现原评测" if row["both_replays_exact"] else "有回放差异，见复现记录"
        cards.append(f'<article id="{scene}"><h2>{html.escape(title)}</h2><p>{html.escape(detail)}</p>'
            f'<video controls preload="none" poster="comparisons/{scene}/middle.png" src="comparisons/{scene}/comparison.mp4"></video>'
            f'<p>{html.escape(outcome)}。{status}。</p>'
            f'<a href="comparisons/{scene}/comparison.mp4">打开视频</a> · '
            f'<a href="comparisons/{scene}/terminal.png">最终状态截图</a></article>')
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lite3 旧模型与 B：多地形动作对比</title>
<style>body{margin:0;background:#101722;color:#e5edf7;font:16px/1.65 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:30px}
h1{font-size:30px}h2{font-size:22px;margin-bottom:4px}article{margin:28px 0;padding:22px;background:#182333;border-radius:12px}
video{width:100%;background:#080d14;border-radius:6px}a{color:#8cc8ff}p{color:#c4d1e1}aside{padding:16px;border-left:4px solid #66b8a5;background:#182333}</style>
<main><h1>Lite3 旧模型与 B：多地形动作对比</h1><p>左：zero10100；右：B10200。2026-11-16 · 冻结模型 · Isaac Lab</p>
<aside>先按原协议补录根部和关节状态，再与正式轨迹核对；画面由这些状态在原仿真场景中渲染。
渲染不推进物理，不修改原成功率。提前结束的一侧停留在终止姿态，不能把停留画面理解为继续站稳。
起步视频为 0.4 倍速，其余约为 1 倍速。每段只代表一个固定起点，不代表该地形总体成功率。</aside>
""" + "\n".join(cards) + "</main></html>\n"


def main():
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    complete = read(OUTPUT/"complete.json")
    plan = read(OUTPUT/"plan_v2.json")
    assert complete["status"] == "completed" and complete["replay_episodes"] == 2240
    assert all(sha(r["path"]) == r["sha256"] for r in plan["sources"]+plan["formal_manifests"])
    sources, scenes, raw, display_repairs = [], {}, set(), []
    for entry in complete["matrix"]:
        path = Path(entry["path"])
        result = read(path/"visual_complete.json")
        policy = result["job"]["policy"]
        execution = read(path/"execution.json")
        assert execution["status"] == "completed" and execution["physical_gpu"] == 1
        for filename in ("execution.json", "visual_complete.json", "capture_validation.json", "report.json", "config.json"):
            dest = DOCS/"evidence"/entry["job_id"]/filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path/filename, dest)
        for clip in result["clips"]:
            scene = clip["name"]
            assert scene in LABELS
            if scene == "flat":
                repair_path = OUTPUT/f"{policy}_flat_display_repair"
                repair = read(repair_path/"repair_complete.json")
                execution_repair = read(repair_path/"execution.json")
                assert repair["status"] == execution_repair["status"] == "completed"
                assert repair["policy_rollout_steps"] == repair["new_optimizer_updates"] == 0
                assert repair["source_root_states_sha256"] == sha(path/"render_states.npz")
                assert repair["clip"]["record"] == clip["record"] and repair["clip"]["frame_steps"] == clip["frame_steps"]
                clip = repair["clip"]
                evidence = DOCS/"evidence"/f"{policy}_flat_display_repair"
                evidence.mkdir(parents=True, exist_ok=True)
                for filename in ("execution.json", "repair_complete.json"):
                    shutil.copy2(repair_path/filename, evidence/filename)
                raw.update(p for p in repair_path.rglob("*") if p.is_file())
                display_repairs.append(dict(policy=policy, elapsed_s=execution_repair["elapsed_s"],
                    source_sha256=repair["source_sha256"], policy_rollout_steps=0,
                    original_env_id=0, render_display_env_id=1))
            scenes.setdefault(scene, {})[policy] = (clip, path)
            target = DOCS/"clips"/policy/scene
            target.mkdir(parents=True, exist_ok=True)
            for filename in ("rollout.mp4", "first.png", "middle.png", "terminal.png", "video.json"):
                shutil.copy2(Path(clip["path"])/filename, target/filename)
        raw.update(p for p in path.rglob("*") if p.is_file())
        if result["job"].get("recover_capture"):
            raw.update(p for p in Path(result["job"]["recover_capture"]).rglob("*") if p.is_file())
        sources.append(dict(job_id=entry["job_id"], elapsed_s=execution["elapsed_s"],
            peak_total_gpu_used_mib=execution["peak_total_gpu_used_mib"],
            reproduction=result["reproduction"], replay_episodes=result["replay_episodes"]))
    assert set(scenes) == set(LABELS)
    rows, csv_rows = [], []
    for scene in LABELS:
        entries = scenes[scene]
        assert set(entries) == {"zero10100", "B"}
        old, new = entries["zero10100"][0], entries["B"][0]
        dest = DOCS/"comparisons"/scene
        dest.mkdir(parents=True, exist_ok=True)
        video = pair_clips(old, new, dest)
        values = {policy: metrics(path, clip) for policy, (clip, path) in entries.items()}
        row = dict(scene=scene, env_id=old["env_id"], title=LABELS[scene][0],
            both_replays_exact=old["replay_exact"] and new["replay_exact"],
            metrics=values, video=video, rendering_is_saved_state_replay=True)
        rows.append(row)
        csv_rows.extend(dict(scene=scene, policy=policy, **metric) for policy, metric in values.items())
        save(dest/"comparison.json", row)
    write_csv(DOCS/"selected_metrics.csv", csv_rows)
    # This is generated output, not a source-file edit.
    (DOCS/"index.html").write_text(gallery(rows), encoding="utf-8")
    focus = ["rough_randomized", "slope_rough_down", "startup_randomized",
             "stairs_up_5cm", "turn_right", "arc_left"]
    sheet = Image.new("RGB", (1280, 6*230), (17, 24, 36))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    for index, scene in enumerate(focus):
        draw.text((12, index*230+3), f"{scene} | zero10100 (left) / B (right)", font=font, fill="white")
        poster = Image.open(DOCS/"comparisons"/scene/"terminal.png").convert("RGB").resize((1280, 180))
        sheet.paste(poster, (0, index*230+30))
    sheet.save(DOCS/"overview.png")
    shutil.copy2(OUTPUT/"complete.json", DOCS/"completion.json")
    raw.update(p for p in OUTPUT.iterdir() if p.is_file())
    save(DOCS/"raw_manifest.json", [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(raw)])
    first_capture = read(OUTPUT/"zero10100_retention_training_randomized/execution.json")
    assert first_capture["status"] == "failed"
    summary = dict(status="completed", new_optimizer_updates=0, new_formal_evaluation_episodes=0,
        replay_episodes=2240, single_policy_clips=30, comparison_clips=15,
        all_replays_exact=all(r["reproduction"] == "exact" for r in sources),
        physics_steps_during_rendering=0, formal_results_unchanged=True,
        jobs=sources, scenes=rows, analyzer_sha256=sha(__file__),
        first_capture_metadata_failure=dict(elapsed_s=first_capture["elapsed_s"],
            capture_reused=True, policy_episodes=160, repeated_policy_episodes=0),
        display_repairs=display_repairs,
        total_execution_elapsed_s=first_capture["elapsed_s"]+sum(r["elapsed_s"] for r in sources)+sum(r["elapsed_s"] for r in display_repairs))
    save(DOCS/"summary.json", summary)
    save(DOCS/"artifact_manifest.json", [dict(path=str(p.relative_to(DOCS)), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(DOCS.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"])
    print("VISUAL ANALYSIS COMPLETE", len(rows), "comparisons; exact:", summary["all_replays_exact"])


if __name__ == "__main__":
    main()
