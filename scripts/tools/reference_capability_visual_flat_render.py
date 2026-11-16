"""Re-render recorded env 0 on a hidden-prototype clone; no policy rollout."""
import argparse
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np

from reference_capability_visual_replay import read, load_npz, compare_arrays, draw_frame, frame_indices
from reference_capability_visual_run import OUTPUT
from reference_stairs_eval import ROOT, save, sha
from reference_eval_checkpoint import register_checkpoint


def collect(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0",
        kit_args="--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1").app
    import isaaclab.app as app_module
    import isaaclab.envs as envs
    import torch
    import imageio.v2 as imageio
    from pxr import UsdGeom
    import reference_stair_allocation_retention as protocol

    source = OUTPUT/f"{args.policy}_retention_nominal"
    original = read(source/"visual_complete.json")
    job = original["job"]
    old_clip = next(c for c in original["clips"] if c["name"] == "flat")
    assert job["policy"] == args.policy and old_clip["env_id"] == 0
    assert original["reproduction"] == "exact"
    register_checkpoint(read(source/"report.json")["model"], Path(job["checkpoint"]), job["checkpoint_sha256"])
    instances = []
    native, launch = envs.ManagerBasedRLEnv, app_module.AppLauncher

    class Prepared(Exception):
        pass

    class PrepareEnvironment(native):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            instances.append(self)
            step, ticks = self.step, 0

            def prepare(action):
                nonlocal ticks
                if ticks == 50:
                    raise Prepared()
                ticks += 1
                return step(action)
            self.step = prepare

    envs.ManagerBasedRLEnv = PrepareEnvironment
    app_module.AppLauncher = lambda *a, **kw: SimpleNamespace(app=app)
    setup = SimpleNamespace(model=read(source/"report.json")["model"], profile="nominal",
        output=args.output, checkpoint_sha256=job["checkpoint_sha256"])
    try:
        protocol.collect(setup)
    except Prepared:
        pass
    finally:
        envs.ManagerBasedRLEnv = native
        app_module.AppLauncher = launch
    assert len(instances) == 1
    base = instances[0]
    initial = load_npz(source/"initial.npz")
    assert all(v["exact"] for v in compare_arrays(load_npz(args.output/"initial.npz"), initial).values())
    states, trace = load_npz(source/"render_states.npz"), load_npz(source/"trace.npz")
    slot = int(np.flatnonzero(states["env_ids"] == 0)[0])
    robot = base.scene["robot"]
    for path in base.scene.env_prim_paths:
        UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(path)).MakeInvisible()
    # Env 0 is the USD clone prototype. Keep it hidden and display an identical
    # robot clone with env 0's saved state, avoiding the static prototype image.
    display_id = 1
    UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(base.scene.env_prim_paths[display_id])).MakeVisible()
    base.render_mode = "rgb_array"
    base.cfg.viewer.resolution = (960, 540)
    base.sim.set_camera_view(eye=old_clip["camera_eye"], target=old_clip["camera_target"])
    env_ids = torch.tensor([display_id], device=base.device)
    n = int(trace["active"][:, 0].sum())
    indices = frame_indices(n)
    destination = args.output/"clips/flat"
    destination.mkdir(parents=True, exist_ok=False)
    frames = []
    error = 0.
    with torch.inference_mode():
        for frame_number, t in enumerate(indices):
            root = torch.tensor(states["root"][t, slot:slot+1], device=base.device)
            joints = torch.tensor(states["joint_pos"][t, slot:slot+1], device=base.device)
            velocity = torch.tensor(states["joint_vel"][t, slot:slot+1], device=base.device)
            robot.write_root_state_to_sim(root, env_ids=env_ids)
            robot.write_joint_state_to_sim(joints, velocity, env_ids=env_ids)
            for _ in range(12 if frame_number == 0 else 2):
                frame = base.render()
            assert frame.shape == (540, 960, 3) and frame[110:510].std() > 2
            error = max(error, float((robot.data.root_state_w[env_ids]-root).abs().max()))
            assert error < 1e-5
            frames.append(draw_frame(frame, label=args.policy, scene="flat", elapsed=(t+1)*.02,
                record=old_clip["record"], command=trace["command"][t, 0], velocity=trace["velocity"][t, 0],
                delta=states["root"][t, slot, :3]-initial["root"][0, :3],
                tilt=float(np.degrees(np.arccos(np.clip(-trace["gravity"][t, 0, 2], -1, 1)))),
                heading_error=None, ended=t == n-1, exact=True, speed=1.))
    writer = imageio.get_writer(destination/"rollout.mp4", fps=10, codec="libx264",
        quality=7, macro_block_size=2, ffmpeg_params=["-threads", "2"])
    try:
        for frame in frames+[frames[-1]]*10:
            writer.append_data(frame)
    finally:
        writer.close()
    for label, frame in (("first", frames[0]), ("middle", frames[len(frames)//2]), ("terminal", frames[-1])):
        imageio.imwrite(destination/f"{label}.png", frame)
    clip = dict(old_clip, path=str(destination), render_display_env_id=display_id,
        maximum_render_pose_error=error, static_prototype_hidden=True)
    save(destination/"video.json", clip)
    assert sha(job["checkpoint"]) == job["checkpoint_sha256"]
    save(args.output/"repair_complete.json", dict(status="completed", policy=args.policy,
        source=str(source), source_root_states_sha256=sha(source/"render_states.npz"),
        original_env_id=0, render_display_env_id=1, physics_steps_during_rendering=0,
        preparation_steps=50, policy_rollout_steps=0, replay_episodes=0,
        new_optimizer_updates=0, initial_exact=True, clip=clip, source_sha256=sha(__file__)))
    print("FLAT DISPLAY REPAIRED", args.policy, "saved env 0 states; no policy rollout", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("zero10100", "B"), required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.output = OUTPUT/f"{args.policy}_flat_display_repair"
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
            gpu=1, output=args.output, timeout=600, minimum_free_mib=10240)
    try:
        collect(args)
        sys.stdout.flush()
        os._exit(0)
    except Exception:
        traceback.print_exc()
        save(args.output/"error.json", dict(error=traceback.format_exc()))
        sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
