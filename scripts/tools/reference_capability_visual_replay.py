"""Capture frozen protocol states, verify reproduction, then render without physics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_eval_checkpoint import register_checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def frame_indices(episode_steps, stride=5):
    """Include the terminal state exactly once; never include a reset state."""
    if episode_steps < 1 or stride < 1:
        raise ValueError("positive episode length and stride required")
    return sorted(set(range(0, episode_steps, stride)) | {episode_steps-1})


def compare_arrays(left, right):
    assert set(left) == set(right), "trace fields changed"
    result = {}
    for key in left:
        a, b = left[key], right[key]
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        exact = bool(a.shape == b.shape and np.array_equal(a, b))
        delta = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape == b.shape and a.size else None
        result[key] = dict(exact=exact, finite=finite, replay_shape=list(a.shape),
                           formal_shape=list(b.shape), maximum_absolute_delta=delta)
    return result


def load_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def same_visual_configuration(left, right):
    """Check every common source; allow only the documented read-only import."""
    left, right = dict(left), dict(right)
    a = {r["path"]: r["sha256"] for r in left.pop("sources")}
    b = {r["path"]: r["sha256"] for r in right.pop("sources")}
    allowed = str(ROOT/"scripts/tools/reference_six_watch.py")
    if set(b)-set(a) or any(a[p] != digest for p, digest in b.items()):
        return False
    if set(a)-set(b) not in (set(), {allowed}):
        return False
    if any(sha(p) != digest for p, digest in a.items()):
        return False
    ignored = {"model", "checkpoint", "model_sha256", "checkpoint_sha256"}
    if left.get("action_mode") == right.get("action_mode") == "mean":
        ignored.add("action_std")
    return {k:v for k,v in left.items() if k not in ignored} == {k:v for k,v in right.items() if k not in ignored}


def verify_capture(output, formal, states, ids):
    initial = compare_arrays(load_npz(output/"initial.npz"), load_npz(formal/"initial.npz"))
    trace = load_npz(output/"trace.npz")
    checks = compare_arrays(trace, load_npz(formal/"trace.npz"))
    report, old = read(output/"report.json"), read(formal/"report.json")
    configuration = same_visual_configuration(read(output/"config.json"), read(formal/"config.json"))
    assert configuration and all(v["exact"] for v in initial.values()), "initial/configuration mismatch"
    assert all(v["finite"] for v in checks.values()), "nonfinite replay"
    assert report["actor_unchanged"] and len(states["root"]) == len(trace["active"])
    origins = load_npz(output/"initial.npz").get("origins")
    for slot, env_id in enumerate(ids):
        n = int(trace["active"][:, env_id].sum())
        assert n > 0 and trace["active"][:n, env_id].all() and not trace["active"][n:, env_id].any()
        expected = trace["root"][:n, env_id]
        observed = states["root"][:n, slot, :3]
        if origins is not None and "feet" in trace:
            observed = observed-origins[env_id]
        elif origins is None:
            # Turn protocol uses one flat tile with origins saved by the capture wrapper.
            observed = observed-states["origins"][slot]
        assert np.array_equal(observed, expected), "recorded body state does not match pre-reset trace"
        assert round(report["records"][env_id]["duration_s"]/.02) == n
    result = dict(status="exact" if report["records"] == old["records"] and all(v["exact"] for v in checks.values()) else "trace_difference",
        configuration_exact=configuration, initial=initial, trace=checks,
        records_exact=report["records"] == old["records"], frozen_actor=True,
        selected_records=[dict(env_id=i, formal=old["records"][i], replay=report["records"][i]) for i in ids],
        capture_is_pre_reset=True, rendering_is_state_replay=True,
        physics_steps_during_rendering=0, optimizer_updates=0, formal_success_counts_unchanged=True)
    save(output/"capture_validation.json", result)
    return result, report, trace


def draw_frame(frame, *, label, scene, elapsed, record, command, velocity, delta, tilt,
               heading_error, ended, exact, speed):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    draw.rectangle((0, 0, 960, 105), fill=(17, 24, 36))
    draw.text((12, 6), f"{label} | {scene} | env {record['env_id']} | {speed:g}x", font=font, fill="white")
    draw.text((12, 30), f"t={elapsed:.2f}s  vx cmd/actual={command[0]:.2f}/{velocity[0]:.2f} m/s"
              f"  yaw cmd/actual={command[2]:.2f}/{velocity[5]:.2f} rad/s", font=font, fill="white")
    extra = f"  heading error={heading_error:.1f} deg" if heading_error is not None else ""
    draw.text((12, 54), f"progress x={delta[0]:.2f} m  lateral y={delta[1]:.2f} m  tilt={tilt:.1f} deg{extra}",
              font=font, fill="white")
    status = f"ENDED: {record['outcome']} (terminal pose held)" if ended else "running"
    draw.text((12, 78), status, font=font, fill=(255, 207, 105) if ended else (151, 230, 174))
    draw.rectangle((0, 515, 960, 540), fill=(17, 24, 36))
    note = "Exact formal-trajectory reproduction" if exact else "Replay differs from formal trajectory"
    draw.text((12, 516), f"Isaac Lab state replay | {note} | frozen policy", font=font, fill="white")
    return np.asarray(image)


def render_states(base, job, states, report, trace, validation, output):
    import imageio.v2 as imageio
    import torch
    from pxr import UsdGeom

    base.render_mode = "rgb_array"
    base.cfg.viewer.resolution = (960, 540)
    robot = base.scene["robot"]
    for path in base.scene.env_prim_paths:
        UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(path)).MakeInvisible()
    assert not base.sim.has_rtx_sensors(), "unexpected physics camera sensor"
    initial = load_npz(output/"initial.npz")
    clips = []
    for slot, selection in enumerate(job["selections"]):
        i, scene = selection["env_id"], selection["name"]
        record = report["records"][i]
        n = int(trace["active"][:, i].sum())
        stride = 1 if scene == "startup_randomized" else 5
        fps = 20 if stride == 1 else 10
        indices = frame_indices(n, stride)
        path = output/"clips"/scene
        path.mkdir(parents=True, exist_ok=False)
        visible = UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(base.scene.env_prim_paths[i]))
        visible.MakeVisible()
        start = initial["root"][i, :3]
        if job["kind"] == "turn":
            target = start+np.array([.0, .0, -.05])
            eye = target+np.array([1.6, -2.7, 2.3])
        elif scene == "startup_randomized":
            target = start+np.array([.0, .0, -.05])
            eye = target+np.array([1.3, -2.0, 1.1])
        else:
            target = start+np.array([1.45, .0, .10])
            eye = target+np.array([-1.4, -4.0, 2.7])
        base.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
        env_ids = torch.tensor([i], device=base.device)
        writer = imageio.get_writer(path/"rollout.mp4", fps=fps, codec="libx264",
            quality=7, macro_block_size=2, ffmpeg_params=["-threads", "2"])
        max_pose_error = 0.
        first_frame = last_frame = middle_frame = None
        try:
            with torch.inference_mode():
                for frame_number, t in enumerate(indices):
                    root = torch.tensor(states["root"][t, slot:slot+1], device=base.device)
                    joints = torch.tensor(states["joint_pos"][t, slot:slot+1], device=base.device)
                    velocities = torch.tensor(states["joint_vel"][t, slot:slot+1], device=base.device)
                    robot.write_root_state_to_sim(root, env_ids=env_ids)
                    robot.write_joint_state_to_sim(joints, velocities, env_ids=env_ids)
                    for _ in range(12 if frame_number == 0 else 2):
                        frame = base.render()
                    assert frame.shape == (540, 960, 3) and frame[110:510].std() > 2, "blank replay frame"
                    max_pose_error = max(max_pose_error, float((robot.data.root_state_w[env_ids]-root).abs().max()))
                    assert max_pose_error < 1e-5, "render advanced the physical state"
                    delta = states["root"][t, slot, :3]-start
                    tilt = float(np.degrees(np.arccos(np.clip(-trace["gravity"][t, i, 2], -1, 1))))
                    heading_error = (float(np.degrees(abs(trace["heading_error"][t, i])))
                        if scene == "heading_right" else None)
                    frame = draw_frame(frame, label=job["policy"], scene=scene, elapsed=(t+1)*.02,
                        record=record, command=trace["command"][t, i], velocity=trace["velocity"][t, i],
                        delta=delta, tilt=tilt, heading_error=heading_error, ended=t == n-1,
                        exact=validation["status"] == "exact", speed=fps*stride*.02)
                    writer.append_data(frame)
                    if first_frame is None:
                        first_frame = frame
                    if frame_number == len(indices)//2:
                        middle_frame = frame
                    last_frame = frame
                for _ in range(fps):
                    writer.append_data(last_frame)
        finally:
            writer.close()
        for name, frame in (("first", first_frame), ("middle", middle_frame), ("terminal", last_frame)):
            imageio.imwrite(path/f"{name}.png", frame)
        visible.MakeInvisible()
        item = dict(name=scene, env_id=i, record=record, fps=fps, simulation_speed=fps*stride*.02,
            frame_steps=[t+1 for t in indices], held_terminal_frames=fps, total_frames=len(indices)+fps,
            replay_exact=validation["status"] == "exact", maximum_render_pose_error=max_pose_error,
            camera_eye=eye.tolist(), camera_target=target.tolist(), path=str(path))
        save(path/"video.json", item)
        clips.append(item)
        print("RENDERED", job["policy"], scene, record["outcome"], len(indices), flush=True)
    return clips


def collect_and_render(job, output):
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=True, device="cuda:0",
        kit_args="--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1")
    app = launcher.app
    import isaaclab.app as app_module
    import isaaclab.envs as envs
    import torch

    formal = Path(job["formal"])
    original = read(formal/"report.json")
    register_checkpoint(original["model"], Path(job["checkpoint"]), job["checkpoint_sha256"])
    ids = [s["env_id"] for s in job["selections"]]
    native = envs.ManagerBasedRLEnv
    base_instances, buffers = [], {k: [] for k in ("root", "joint_pos", "joint_vel")}
    recovery = Path(job["recover_capture"]) if job.get("recover_capture") else None

    class ReadyToRender(Exception):
        pass

    class CaptureEnvironment(native):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            base_instances.append(self)
            if job.get("heading"):
                from reference_low_stairs_heading_probe import attach_feedback
                attach_feedback(self)
            compute = self.reward_manager.compute
            calls = 0

            def capture(dt):
                nonlocal calls
                value = compute(dt)
                if calls >= 50:
                    data = self.scene["robot"].data
                    for key, tensor in (("root", data.root_state_w),
                                        ("joint_pos", data.joint_pos), ("joint_vel", data.joint_vel)):
                        buffers[key].append(tensor[ids].detach().cpu().numpy().copy())
                calls += 1
                return value

            self.reward_manager.compute = capture
            if recovery is not None:
                step = self.step
                steps = 0

                def stop_before_rollout(action):
                    nonlocal steps
                    if steps == 50:
                        raise ReadyToRender()
                    steps += 1
                    return step(action)

                self.step = stop_before_rollout

    envs.ManagerBasedRLEnv = CaptureEnvironment
    original_launcher = app_module.AppLauncher
    args = SimpleNamespace(model=original["model"], checkpoint_sha256=job["checkpoint_sha256"],
        profile=job["profile"], output=output, repeats=20, steps=None, video_env=None,
        action_mode="mean", height_set="low_stairs", direction=job.get("direction"), gpu=1)
    try:
        if job["kind"] == "retention":
            import reference_stair_allocation_retention as protocol
            # Its collector owns AppLauncher; use the single already-running application.
            app_module.AppLauncher = lambda *a, **kw: SimpleNamespace(app=app)
            protocol.collect(args)
        elif job["kind"] == "turn":
            import reference_turn_diagnose as protocol
            protocol.collect(args, app)
        else:
            import reference_isaac_stairs_diagnose as protocol
            protocol.collect(args, app)
    except ReadyToRender:
        assert recovery is not None
    finally:
        envs.ManagerBasedRLEnv = native
        app_module.AppLauncher = original_launcher
    assert len(base_instances) == 1
    base = base_instances[0]
    if recovery is not None:
        setup = compare_arrays(load_npz(output/"initial.npz"), load_npz(recovery/"initial.npz"))
        assert all(r["exact"] for r in setup.values()) and not any(buffers.values())
        for name in ("initial.npz", "config.json"):
            shutil.move(output/name, output/f"render_setup_{name}")
        for name in ("initial.npz", "config.json", "report.json", "trace.npz"):
            shutil.copy2(recovery/name, output/name)
        states = load_npz(recovery/"render_states.npz")
        save(output/"recovery.json", dict(source=str(recovery), policy_rollout_steps=0,
            preparation_steps=50, initial_exact=True, reason="Source metadata comparison included an extra read-only statistics import."))
    else:
        states = {k: np.asarray(v) for k, v in buffers.items()}
        states["env_ids"] = np.asarray(ids)
        states["origins"] = base.scene.env_origins[ids].cpu().numpy().copy()
    assert all(np.isfinite(v).all() for v in states.values())
    np.savez_compressed(output/"render_states.npz", **states)
    validation, report, trace = verify_capture(output, formal, states, ids)
    print("REPRODUCTION", job["policy"], job["name"], validation["status"], flush=True)
    clips = render_states(base, job, states, report, trace, validation, output)
    assert sha(job["checkpoint"]) == job["checkpoint_sha256"]
    save(output/"visual_complete.json", dict(status="completed", job=job, clips=clips,
        reproduction=validation["status"], replay_episodes=report["episodes"],
        newly_simulated_replay_episodes=0 if recovery is not None else report["episodes"],
        new_optimizer_updates=0, physics_steps_during_rendering=0,
        source_sha256=sha(__file__), formal_success_counts_unchanged=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    job = next(j for j in plan["jobs"] if j["id"] == args.job)
    assert sha(job["checkpoint"]) == job["checkpoint_sha256"]
    assert all(sha(s["path"]) == s["sha256"] for s in plan["sources"])
    assert all(sha(s["path"]) == s["sha256"] for s in plan["formal_manifests"])
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
            gpu=1, output=args.output, timeout=2400, minimum_free_mib=10240)
    try:
        collect_and_render(job, args.output.resolve())
        sys.stdout.flush()
        os._exit(0)
    except Exception:
        traceback.print_exc()
        save(args.output/"error.json", dict(error=traceback.format_exc()))
        sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
