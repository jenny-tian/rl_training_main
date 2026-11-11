"""Real B1.1 simulator views, with an optional unchanged B1 comparison scene."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
from source_block_check import TASK, REFERENCE_TASK, CHECKPOINT, CHECKPOINT_SHA


def child(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0",
                      kit_args="--/renderer/multiGpu/enabled=False --/renderer/activeGpu=0").app
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Usd, UsdGeom, Vt
    import rl_training.tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from rl_training.source_scan.frozen_actor import load_frozen_source_actor

    task = TASK if args.baseline else REFERENCE_TASK
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=120)
    cfg.seed = 42
    cfg.observations.policy.enable_corruption = False
    cfg.viewer.resolution = (1280, 720)
    cfg.sim.render_interval = cfg.decimation * 4
    # Use mesh display colours for review, avoiding the default black
    # material. This changes only rendering, not training configuration.
    cfg.scene.terrain.visual_material = None
    env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    base = env.unwrapped
    terrain = base.scene.terrain
    obs, _ = env.reset()
    actor = load_frozen_source_actor(CHECKPOINT, device=base.device, expected_sha256=CHECKPOINT_SHA)
    for _ in range(25):
        obs, reward, *_ = env.step(actor(obs).clamp(-100, 100))
        if not bool(torch.isfinite(reward).all()):
            raise ValueError("non-finite visualization reward")

    selected = {name: int((terrain.terrain_types == col).nonzero()[0])
                for name, col in (("rough", 6), ("stairs_up", 4), ("stairs_down", 5))}
    camera_records = []

    def render(name, eye, target):
        eye, target = np.asarray(eye).tolist(), np.asarray(target).tolist()
        base.sim.set_camera_view(eye=eye, target=target)
        for _ in range(12):
            frame = env.render()
        if frame.shape != (720, 1280, 3) or frame.std() < 2:
            raise RuntimeError("blank simulator image")
        image = Image.fromarray(frame)
        image.save(args.output / f"{name}.png")
        camera_records.append(dict(name=name, eye=eye, target=target))
        print(f"IMAGE {name}", flush=True)
        return image

    overview = render("overview", [160., 120., 230.], [0., 0., 0.])
    # Only visibility changes: all terrain and collision geometry remains
    # exactly the validated task. Hide other robots to make the tiles legible.
    for i, path in enumerate(base.scene.env_prim_paths):
        if i not in selected.values():
            UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(path)).MakeInvisible()
    origins = {name: terrain.env_origins[i].cpu().numpy() for name, i in selected.items()}
    origin = origins["rough"]
    rough_top = render("rough_top", origin + [0., -.001, 14.], origin)
    rough_oblique = render("rough_oblique", origin + [6., -7., 5.5], origin + [0., 0., -.05])
    rough_close = render("rough_close", origin + [3.5, -2., 1.4], origin + [1.3, 0., 0.])
    palette = np.array([[.19, .35, .55], [.22, .48, .62], [.30, .65, .65],
                        [.61, .76, .52], [.88, .80, .42], [.95, .58, .25]], dtype=np.float32)
    colour_checks = []
    for prim in Usd.PrimRange(base.scene.stage.GetPrimAtPath("/World/ground")):
        path = str(prim.GetPath())
        if prim.GetTypeName() != "Mesh" or not any(f"_c{col:02d}/" in path for col in (4, 5)):
            continue
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        before = hashlib.sha256(vertices.tobytes() + indices.tobytes()).hexdigest()
        top = vertices[indices.reshape(-1, 3), 2].max(-1)
        offset = .175 if "_c04/" in path else 0.
        level = np.rint((top + offset) / .035).astype(int).clip(0, 5)
        colour = mesh.GetDisplayColorPrimvar()
        colour.SetInterpolation(UsdGeom.Tokens.uniform)
        colour.Set(Vt.Vec3fArray.FromNumpy(palette[level]))
        after = hashlib.sha256(np.asarray(mesh.GetPointsAttr().Get()).tobytes()
                               + np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).tobytes()).hexdigest()
        if before != after:
            raise RuntimeError("presentation colours changed geometry")
        colour_checks.append(dict(path=path, geometry_sha256=before, unchanged=True))
    if len(colour_checks) != 20:
        raise RuntimeError(f"expected 20 coloured stair tiles, got {len(colour_checks)}")
    stair_images = {}
    for name in ("stairs_up", "stairs_down"):
        origin = origins[name]
        stair_images[name] = render(name + "_top", origin + [0., -.001, 14.], origin)

    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 25)
    small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)
    canvas = Image.new("RGB", (1920, 854), (20, 25, 35))
    draw = ImageDraw.Draw(canvas)
    title = "原 B1 对照场景" if args.baseline else "B1.1 参考粗糙面 | 方形分块"
    draw.text((16, 5), title + " | 真实仿真截图", font=font, fill="white")
    draw.text((16, 37), "每块 8×8 米；相同起点策略运行 0.5 秒后拍摄；仅场景验证，未训练新模型。", font=small, fill="#cbd5e1")
    panels = [("粗糙地块俯视", rough_top), ("粗糙地面近景", rough_close), ("粗糙地块斜视", rough_oblique),
              ("上台阶俯视：颜色区分高度", stair_images["stairs_up"]),
              ("下台阶俯视：颜色区分高度", stair_images["stairs_down"]), ("完整分块地图", overview)]
    for k, (label, frame) in enumerate(panels):
        x, y = k % 3 * 640, 64 + k // 3 * 395
        canvas.paste(frame.resize((640, 360)), (x, y))
        draw.text((x + 12, y + 363), label, font=small, fill="white")
    canvas.save(args.output / "montage.png")
    metadata = dict(task=task, baseline=args.baseline, num_envs=120, scene_seed=42,
        rough_profile=cfg.source_block_rough_profile, checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
        simulated_seconds_before_images=25 * base.step_dt, all_screenshots_real_simulation=True,
        image_processing="montage resizing and labels only; individual PNGs are raw simulator RGB",
        presentation="mesh display colours; stairs coloured by physical height, geometry unmodified",
        stair_colour_geometry_checks=colour_checks,
        selected_env_ids=selected, selected_origins={k: v.tolist() for k, v in origins.items()},
        cameras=camera_records, policy_training=False)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print("VISUALIZATION_COMPLETE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.child:
        child(args)
    else:
        from source_block_runtime import run
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--output", str(args.output)]
        if args.baseline:
            command.append("--baseline")
        sys.exit(run(command, gpu=args.gpu, output=args.output))
