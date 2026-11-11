"""Actual B1 simulator screenshots and first-episode stair videos from model_3497."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
from source_block_check import TASK, CHECKPOINT, CHECKPOINT_SHA


def child(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0",
                      kit_args="--/renderer/multiGpu/enabled=False --/renderer/activeGpu=0").app
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import UsdGeom
    import rl_training.tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from rl_training.source_curriculum.block import axis_vectors
    from rl_training.source_scan.frozen_actor import load_frozen_source_actor
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=120)
    cfg.seed = 42
    cfg.observations.policy.enable_corruption = False
    cfg.viewer.resolution = (1280, 720)
    cfg.sim.render_interval = cfg.decimation * 4
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
    base = env.unwrapped
    terrain, robot = base.scene.terrain, base.scene["robot"]
    obs, _ = env.reset()
    actor = load_frozen_source_actor(CHECKPOINT, device=base.device, expected_sha256=CHECKPOINT_SHA)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 27)
    small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22)

    def render(eye, target):
        base.sim.set_camera_view(eye=eye, target=target)
        for _ in range(8):
            frame = env.render()
        if frame.shape != (720, 1280, 3) or frame.std() < 2:
            raise RuntimeError("blank simulator rendering")
        return Image.fromarray(frame)

    overview = render((160., 120., 230.), (0., 0., 0.))
    overview.save(args.output / "terrain_overview.png")
    if args.overview_only:
        (args.output / "metadata.json").write_text(json.dumps(dict(
            task=TASK, num_envs=120, scene_seed=42, all_screenshots_real_simulation=True,
            eye=[160., 120., 230.], target=[0., 0., 0.], purpose="complete-map framing"), indent=2))
        print("OVERVIEW_COMPLETE", flush=True)
        os._exit(0)
    views = [("flat", 13, "平地"), ("rough", 6, "多尺度粗糙"), ("slope", 0, "坡面"),
             ("stairs_up", 4, "上台阶：中心低，向外走"), ("stairs_down", 5, "下台阶：中心高，向外走"),
             ("slope_rough", 16, "坡面叠加粗糙"), ("friction", 7, "摩擦变化（物理材质不同）")]
    selected = {name: int((terrain.terrain_types == col).nonzero()[0]) for name, col, _ in views}
    for i, path in enumerate(base.scene.env_prim_paths):
        if i not in selected.values():
            UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(path)).MakeInvisible()
    panels = [("完整 10×20 分块地图", overview)]
    for name, col, label in views:
        i = selected[name]
        origin = terrain.env_origins[i].cpu().numpy()
        frame = render((origin + np.array([6., -7., 5.5])).tolist(), (origin + [0., 0., -.05]).tolist())
        frame.save(args.output / f"{name}_L2.png")
        panels.append((label, frame))
    canvas = Image.new("RGB", (2048, 740), (20, 25, 35))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 8), "Lite3 B1 实际仿真 | 六类地形分块 | 所有行固定 L2", font=font, fill="white")
    draw.text((18, 47), "每块8×8米；120个展示环境。起点model_3497；本图验证场景，不代表训练通过。", font=small, fill="#cbd5e1")
    for k, (label, frame) in enumerate(panels):
        x, y = (k % 4)*512, 88 + (k//4)*326
        canvas.paste(frame.resize((512, 288)), (x, y))
        draw.text((x+10, y+291), label, font=small, fill="white")
    canvas.save(args.output / "terrain_montage.png")
    print("SCREENSHOTS_READY", flush=True)
    records = []
    for name, label in (("stairs_up", "上台阶"), ("stairs_down", "下台阶")):
        obs, _ = env.reset()
        i = selected[name]
        origin = terrain.env_origins[i].clone()
        axis = int(base.source_block_state.axes[i])
        direction = axis_vectors(base.source_block_state.axes[i:i+1])[0].cpu().numpy()
        ending = None
        frame_count = 0
        with imageio.get_writer(args.output / f"{name}_L2.mp4", fps=12.5, codec="libx264", quality=7, macro_block_size=16) as writer:
            for step in range(1000):
                root = robot.data.root_link_pos_w[i].cpu().numpy()
                progress = float(base.source_block_state.progress()[0][i])
                base.sim.set_camera_view(eye=(root + [2.5, -3., 1.8]).tolist(), target=(root + [0., 0., -.1]).tolist())
                # Capture before step(): a terminating step automatically resets
                # the robot, and that new episode must not enter this video.
                if step % 4 == 0:
                    frame = Image.fromarray(env.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 1280, 78), fill=(15, 20, 30))
                    draw.text((12, 5), f"固定L2 {label} | 起点 model_3497 | {step*base.step_dt:.2f}s", font=font, fill="white")
                    draw.text((12, 43), f"从中心向{['东','北','西','南'][axis]}；进度 {progress:.2f}米；B1场景验证，非资格评估", font=small, fill="#cbd5e1")
                    writer.append_data(np.asarray(frame))
                    frame_count += 1
                obs, reward, terminated, truncated, _ = env.step(actor(obs).clamp(-100, 100))
                if not bool(torch.isfinite(reward).all()):
                    raise ValueError("non-finite visualization reward")
                if bool(terminated[i] | truncated[i]):
                    ending = dict(step=step+1, seconds=(step+1)*base.step_dt, progress_before_last_step_m=progress,
                                  terminated=bool(terminated[i]), truncated=bool(truncated[i]),
                                  terms=[term for term in base.termination_manager.active_terms if bool(base.termination_manager.get_term(term)[i])])
                    break
            records.append(dict(name=name, env_id=i, axis=axis, initial_origin=origin.cpu().tolist(), frames=frame_count,
                                simulated_seconds=(step+1)*base.step_dt, ending=ending, display_limit_seconds=20))
        print(f"VIDEO {name} {json.dumps(records[-1])}", flush=True)
    report = dict(task=TASK, checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA, num_envs=120,
                  all_screenshots_real_simulation=True, policy_noise=False, selected_env_ids=selected, videos=records,
                  note="B1 scene validation using the unchanged starting actor; no qualification claim. Other robot visuals hidden in closeups only.")
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("VISUALIZATION_COMPLETE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--overview-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.child:
        child(args)
    else:
        from source_block_runtime import run
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--output", str(args.output)]
        if args.overview_only:
            command.append("--overview-only")
        sys.exit(run(command, gpu=0, output=args.output))
