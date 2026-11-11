"""Frozen B3 simulation worker. Hooks only observe pre-reset data; no PPO exists."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
from source_block_b3_protocol import CASES, FIELDS, REPEATS, SEED, models, schedule, summarize


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def collect(args, app):
    import gymnasium as gym
    import numpy as np
    import torch
    import rl_training.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import quat_from_euler_xyz, wrap_to_pi
    from rl_training.source_scan.frozen_actor import load_frozen_source_actor

    controlled = args.mode == "controlled"
    num_envs = len(CASES) * REPEATS if controlled else 1024
    task = "Flat-Deeprobotics-Lite3-v0" if args.domain == "flat" else "SourceBlock-B2-Deeprobotics-Lite3-v0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
    cfg.seed = SEED
    cfg.observations.policy.enable_corruption = False
    if controlled:
        material = cfg.events.randomize_rigid_body_material
        material.params.update(static_friction_range=(.9, .9), dynamic_friction_range=(.9, .9),
                               restitution_range=(0., 0.), make_consistent=True)
        if args.domain == "block":
            from rl_training.tasks.manager_based.locomotion.velocity.terrains.source_block_terrain import SourceBlockTerrainImporter

            class DiagnosticFlatImporter(SourceBlockTerrainImporter):
                def _compute_env_origins_curriculum(self, n, origins):
                    super()._compute_env_origins_curriculum(n, origins)
                    ids = torch.arange(n, device=self.device)
                    self.terrain_types = 9 + ids % 4
                    self.terrain_levels = (ids // 4) % 10
                    return origins[self.terrain_levels, self.terrain_types]

            cfg.scene.terrain.class_type = DiagnosticFlatImporter
    if args.video_env is not None:
        cfg.viewer.resolution = (960, 540)
        cfg.sim.render_interval = cfg.decimation * 5
        cfg.scene.terrain.visual_material = None
    env = gym.make(task, cfg=cfg, **({"render_mode": "rgb_array"} if args.video_env is not None else {}))
    base = env.unwrapped
    args.environment = env
    obs, _ = env.reset()
    robot = base.scene["robot"]
    terrain = base.scene.terrain
    term = base.command_manager.get_term("base_velocity")
    sensor = base.scene["contact_forces"]
    spec = next(m for m in models() if m["name"] == args.model)
    actor = load_frozen_source_actor(Path(spec["checkpoint"]), device=base.device, expected_sha256=spec["sha256"])
    actor_before = {k: digest(v.cpu().numpy()) for k, v in actor.state_dict().items()}
    columns = (terrain.terrain_types.cpu().numpy().copy() if args.domain == "block" else np.full(num_envs, -1))

    if controlled:
        generator = torch.Generator().manual_seed(SEED)
        perturbation = ((torch.rand((REPEATS, 9), generator=generator) * 2 - 1)
                        * torch.tensor([.08, .08, .1, .1, .1, .1, .05, .05, .1]))
        perturbation = perturbation.repeat(len(CASES), 1).to(base.device)
        roots = robot.data.default_root_state.clone()
        roots[:, :3] += base.scene.env_origins
        roots[:, :2] += perturbation[:, :2]
        roots[:, 3:7] = quat_from_euler_xyz(perturbation[:, 2] * 0, perturbation[:, 2] * 0, perturbation[:, 2])
        roots[:, 7:13] = perturbation[:, 3:9]
        robot.write_root_state_to_sim(roots)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
        base.sim.forward()
        base.scene.update(base.step_dt)
        initial_heading = robot.data.heading_w.clone()

        def set_commands():
            # Round the integer control tick to avoid 7.999999... boundary shifts.
            t = round(base.common_step_counter * base.step_dt, 8)
            term.is_heading_env[:] = False
            term.is_standing_env[:] = False
            for index, case in enumerate(CASES):
                part = slice(index * REPEATS, (index + 1) * REPEATS)
                vx, vy, wz, heading = schedule(case, t)
                term.vel_command_b[part] = term.vel_command_b.new_tensor((vx, vy, wz))
                term.heading_target[part] = initial_heading[part] + (heading or 0.)
                term.is_heading_env[part] = heading is not None
                term.is_standing_env[part] = heading is None and vx == vy == wz == 0
            # Reuse the actual original heading controller for heading cases.
            term._update_command()

        original_command_compute = base.command_manager.compute

        def command_compute(dt):
            original_command_compute(dt)
            set_commands()

        base.command_manager.compute = command_compute
        set_commands()
        obs = base.observation_manager.compute()
        groups = {case: np.arange(num_envs) // REPEATS == k for k, case in enumerate(CASES)}
    elif args.domain == "flat":
        groups = {"original_flat": np.ones(num_envs, bool)}
    else:
        groups = {"dedicated_turn_c09": columns == 9, "other_omni_c10_12": np.isin(columns, [10, 11, 12]),
                  "straight_flat_c13": columns == 13, "other_terrain": ~np.isin(columns, [9, 10, 11, 12, 13]),
                  "stairs_up_c04": columns == 4, "stairs_down_c05": columns == 5}

    initial = dict(root=robot.data.root_state_w.cpu().numpy().copy(),
        joint_pos=robot.data.joint_pos.cpu().numpy().copy(), joint_vel=robot.data.joint_vel.cpu().numpy().copy(),
        command=term.command.cpu().numpy().copy(), observation=obs["policy"].cpu().numpy().copy(),
        materials=robot.root_physx_view.get_material_properties().cpu().numpy().copy(),
        origins=base.scene.env_origins.cpu().numpy().copy(), columns=columns)
    np.savez_compressed(args.output / "initial.npz", **initial)
    initial_hashes = {k: digest(v) for k, v in initial.items()}
    reward_names = list(base.reward_manager.active_terms)
    term_names = list(base.termination_manager.active_terms)
    body_names = list(sensor.body_names)
    duration = 20 if controlled or args.domain == "flat" else 30
    steps = args.steps or round(duration / base.step_dt)
    trace = np.empty((steps, num_envs, len(FIELDS) + len(reward_names) + len(body_names)), dtype=np.float32)
    ended = np.zeros((steps, num_envs), dtype=bool)
    terminated = np.zeros_like(ended)
    flags = np.zeros((steps, num_envs, len(term_names)), dtype=bool)
    first_alive = np.ones(num_envs, bool)
    records = []
    frames = []
    frame_steps = []
    if args.video_env is not None:
        from pxr import UsdGeom
        for i, path in enumerate(base.scene.env_prim_paths):
            if i != args.video_env:
                UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(path)).MakeInvisible()
        # USD clones inherit env_0 opinions. Author visibility on the selected
        # clone after hiding the source so its robot does not inherit invisibility.
        selected_path = base.scene.env_prim_paths[args.video_env]
        selected = UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(selected_path))
        selected.MakeVisible()
        if selected.ComputeVisibility() == UsdGeom.Tokens.invisible:
            raise RuntimeError("selected robot inherited hidden source visibility")
    original_reward_compute = base.reward_manager.compute
    index = 0
    pre_command = None

    def reward_compute(dt):
        reward = original_reward_compute(dt)
        rates = base.reward_manager._step_reward
        if not torch.allclose(reward, rates.sum(-1) * dt, atol=1e-5, rtol=1e-4):
            raise AssertionError("reward rate units or term ordering mismatch")
        data = robot.data
        forces = sensor.data.net_forces_w_history.norm(dim=-1).amax(1)
        invalid = torch.zeros(num_envs, device=base.device)
        height = data.root_pos_w[:, 2].clone()
        if args.domain == "block":
            hits = base.scene["height_scanner_base"].data.ray_hits_w[..., 2]
            valid_hits = torch.isfinite(hits) & (hits.abs() <= 1e6)
            invalid = (~valid_hits.all(-1)).float()
            height -= torch.where(valid_hits, hits, 0.).mean(-1)
        values = torch.cat((pre_command, data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3],
            wrap_to_pi(term.heading_target - data.heading_w)[:, None], term.is_heading_env[:, None].float(),
            term.is_standing_env[:, None].float(),
            (base.action_manager.action - base.action_manager.prev_action).square().sum(-1, keepdim=True),
            height[:, None], data.projected_gravity_b[:, :2].norm(dim=-1, keepdim=True),
            data.root_pos_w - base.scene.env_origins, invalid[:, None], rates, forces), dim=-1)
        if not torch.isfinite(values).all():
            raise FloatingPointError("nonfinite diagnostic trace")
        trace[index] = values.cpu().numpy()
        flags[index] = torch.stack([base.termination_manager.get_term(n) for n in term_names], -1).cpu().numpy()
        ended[index] = base.termination_manager.dones.cpu().numpy()
        terminated[index] = base.termination_manager.terminated.cpu().numpy()
        for env_id in np.flatnonzero(first_alive & ended[index]):
            contacts = trace[index, env_id, len(FIELDS)+len(reward_names):]
            records.append(dict(env_id=int(env_id), step=index+1, time_s=(index+1)*dt,
                case=CASES[env_id // REPEATS] if controlled else None, column=int(columns[env_id]),
                failed=bool(terminated[index, env_id]), terms=[n for n, v in zip(term_names, flags[index, env_id]) if v],
                command=trace[index, env_id, :3].tolist(), velocity=trace[index, env_id, 3:6].tolist(),
                base_height=float(height[env_id]), gravity_xy=float(trace[index, env_id, 11]),
                contacts_n={n: float(f) for n, f in zip(body_names, contacts) if f > 1.}))
        if args.video_env is not None and first_alive[args.video_env] and (index % 5 == 0 or ended[index, args.video_env]):
            position = data.root_pos_w[args.video_env].cpu().numpy()
            base.sim.set_camera_view(eye=(position + [2., -2.5, 1.3]).tolist(), target=position.tolist())
            # This installed render() skips sim.render() when recompute=True.
            # Use the working public path and warm up the RGB annotator first.
            for _ in range(12 if index == 0 else 2):
                frame = env.render()
            if frame.shape != (540, 960, 3) or frame.std() < 2:
                raise RuntimeError("blank or malformed simulator frame")
            frames.append(frame.copy())
            frame_steps.append(index + 1)
        first_alive[:] &= ~ended[index]
        return reward

    base.reward_manager.compute = reward_compute
    config = dict(task=task, step_dt=base.step_dt, sim_dt=cfg.sim.dt, decimation=cfg.decimation,
        episode_length_s=cfg.episode_length_s, commands=cfg.commands.base_velocity.to_dict(),
        material_event=cfg.events.randomize_rigid_body_material.to_dict(),
        push_event=cfg.events.randomize_push_robot,
        reset_event=cfg.events.randomize_reset_base.to_dict(), ground_material=cfg.scene.terrain.physics_material.to_dict(),
        rewards={n: base.reward_manager.get_term_cfg(n).to_dict() for n in reward_names},
        terminations={n: base.termination_manager.get_term_cfg(n).to_dict() for n in term_names},
        counts_by_column={str(c): int((columns == c).sum()) for c in np.unique(columns)},
        controlled_overrides="friction .9/.9; paired root perturbations; scheduled commands; B2 flat columns 9-12" if controlled else None)
    save(args.output / "config.json", config)
    with torch.inference_mode():
        for index in range(steps):
            if not app.is_running():
                raise RuntimeError("simulator stopped")
            pre_command = term.command.clone()
            action = actor(obs).clamp(-100, 100)
            obs, reward, done, timeout, _ = env.step(action)
            if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                raise FloatingPointError("nonfinite diagnostic observation/reward")
            if not np.array_equal((done | timeout).cpu().numpy(), ended[index]):
                raise AssertionError("pre-reset termination hook differs from step flags")
            if (index + 1) % 250 == 0:
                print(f"B3 {args.model} {args.domain}/{args.mode} {index+1}/{steps}, first_alive={first_alive.sum()}", flush=True)
    actor_after = {k: digest(v.cpu().numpy()) for k, v in actor.state_dict().items()}
    if actor_before != actor_after or any(p.requires_grad for p in actor.parameters()):
        raise AssertionError("actor changed or is trainable")
    np.savez_compressed(args.output / "trace.npz", trace=trace, ended=ended, terminated=terminated,
                        term_flags=flags, columns=columns)
    report = dict(status="passed", model=spec, domain=args.domain, mode=args.mode, seed=SEED,
        num_envs=num_envs, steps=steps, step_dt=base.step_dt, first_episodes_only=True,
        initial_hashes=initial_hashes, actor_unchanged=True, actor_tensor_hashes=actor_after,
        fields=list(FIELDS)+reward_names+body_names, reward_names=reward_names, body_names=body_names,
        termination_names=term_names, records=records,
        metrics=summarize(trace, ended, terminated, groups, reward_names, base.step_dt, args.mode),
        reward_units="weighted reward rate per second; term sum times dt equals step reward",
        trace_time="pre-step command; post-physics state captured before automatic reset",
        probe=bool(args.steps), video_env=args.video_env)
    if frames:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        labelled = []
        for frame, step in zip(frames, frame_steps):
            im = Image.fromarray(frame)
            draw = ImageDraw.Draw(im)
            values = trace[step-1, args.video_env]
            draw.rectangle((0, 0, 960, 48), fill=(10, 10, 10))
            draw.text((8, 5), f"{args.model} {args.domain}/{args.mode} env={args.video_env} t={step*base.step_dt:.2f}s", fill="white")
            draw.text((8, 23), f"command={np.round(values[:3], 3)}  velocity={np.round(values[3:6], 3)}  terminal={ended[step-1,args.video_env]}", fill="white")
            labelled.append(np.asarray(im))
        imageio.mimsave(args.output / "response.mp4", labelled, fps=10, macro_block_size=2)
        Image.fromarray(labelled[-1]).save(args.output / "last_frame.png")
        report["video"] = dict(env_id=args.video_env, frame_steps=frame_steps,
                                image_processing="text overlay only", fps=10, pre_reset_terminal_frame=True)
    save(args.output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", choices=("flat", "block"), required=True)
    parser.add_argument("--mode", choices=("controlled", "native"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--video-env", type=int)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
                   gpu=args.gpu, output=args.output, timeout=1800)
    app = None
    report = dict(status="failed")
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=args.video_env is not None, device="cuda:0",
            fast_shutdown=args.video_env is not None,
            kit_args="--/renderer/multiGpu/enabled=False --/renderer/activeGpu=0").app
        report = collect(args, app)
    except Exception:
        traceback.print_exc()
        report.update(status="failed", error=traceback.format_exc())
    finally:
        report["cleanup"] = {}
        save(args.output / "report.json", report)
        for name, resource in (("environment", getattr(args, "environment", None)), ("application", app)):
            if resource is not None:
                try:
                    if name == "environment":
                        resource.unwrapped.sim._disable_app_control_on_stop_handle = True
                        resource.unwrapped.sim.stop()
                    if name == "application" and args.video_env is not None:
                        report["cleanup"][name] = "requested_fast_shutdown_exit"
                        save(args.output / "report.json", report)
                    resource.close()
                    report["cleanup"][name] = "returned"
                except Exception:
                    report.update(status="failed", cleanup_error=traceback.format_exc())
                save(args.output / "report.json", report)
        print("B3 " + report["status"], flush=True)
    os._exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    sys.exit(main())
