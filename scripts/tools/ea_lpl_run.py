"""EA-LPL runner: frozen base policy episodes on the Source scene.

The supervisor chooses a GPU, records a manifest, enforces deadlines and
cleans the process group. The child owns Isaac Sim and records the first
`episodes_per_env` episodes of every environment.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
from source_adaptation_runtime import OLD, digest, environment_identity, write_json  # noqa: E402
from source_delivery_light import check_scans, close_comparison_environment, lifecycle_deadline, record_phase  # noqa: E402
from source_live_ppo_supervisor import group_exists, terminate_group  # noqa: E402

ACTOR_SHA256 = "87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c"
MIN_FREE_MIB = 12288
STATE_NAMES = ["vx_mps", "vy_mps", "vz_mps", "velocity_error_squared", "height_above_scan_m", "progress_m",
               "joint_deviation_l2", "joint_velocity_rms", "tilt_xy_norm", "lateral_m", "roll_proxy_rad",
               "pitch_proxy_rad", "yaw_rad", "yaw_rate_radps", "yaw_rate_command_radps"]
ACTION_NAMES = ["mean_action_rms", "clipped_joint_fraction", "projected", "paused", "executed_delta_max"]


def make_ea_environment(spec):
    import gymnasium as gym
    import rl_training.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rl_training.source_scan.training_env import SourceTrainingVecEnv
    from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.ea_lpl_env_cfg import (
        DeeproboticsLite3EaLplEnvCfg,
    )
    cfg = DeeproboticsLite3EaLplEnvCfg(
        source_terrain_level=spec["level"], ea_heading_hold=spec["heading_hold"],
        ea_columns=None if spec["columns"] is None else tuple(spec["columns"]),
        ea_reset_table_seed=spec["reset_table_seed"], ea_reset_namespace=spec["namespace"],
        ea_reset_slots=None if spec["slots"] is None else tuple(spec["slots"]))
    cfg.sim.device = "cuda:0"
    raw = gym.make("SourceTerrain-Deeprobotics-Lite3-v0", cfg=cfg)
    return SourceTrainingVecEnv(RslRlVecEnvWrapper(raw, clip_actions=None))


def check_initial_state(base, spec):
    import numpy as np
    from rl_training.ea_lpl.cfg_checks import check_heading_hold, heading_yaw_rate_command
    from rl_training.ea_lpl.reset_table import DEFAULT_RESET_RANGES, initial_state_offsets
    check_heading_hold(base.cfg.commands.base_velocity, enabled=spec["heading_hold"])
    data = base.scene["robot"].data
    n = base.num_envs
    if spec["reset_table_seed"] is None:
        expected = np.zeros((n, 3))
    else:
        slots = spec["slots"] or list(range(n))
        expected = initial_state_offsets(spec["reset_table_seed"], spec["namespace"], slots, [0] * n,
                                         DEFAULT_RESET_RANGES)
        if getattr(base, "_ea_last_reset_keys", None) != {i: (spec["namespace"], slots[i], 0) for i in range(n)}:
            raise ValueError("reset table keys differ from the spec")
    xy = (data.root_link_pos_w[:, :2] - base.scene.env_origins[:, :2] - data.default_root_state[:, :2])
    if np.abs(xy.cpu().double().numpy() - expected[:, :2]).max() > 1e-4:
        raise ValueError("initial x/y offsets differ from the reset table")
    yaw_error = data.heading_w.cpu().double().numpy() - expected[:, 2]
    if np.abs(np.arctan2(np.sin(yaw_error), np.cos(yaw_error))).max() > 1e-4:
        raise ValueError("initial heading differs from the reset table")
    if spec["heading_hold"]:
        # Isaac Lab samples a random yaw rate at resample time; apply the heading loop before the first action.
        base.command_manager.compute(dt=0.0)
        command = base.command_manager.get_command("base_velocity")[:, 2]
        if float((command - heading_yaw_rate_command(data.heading_w)).abs().max()) > 1e-5:
            raise ValueError("heading-hold command does not follow the registered formula")


def collect_fixed(wrapped, actor, spec, folder, app, report):
    import numpy as np
    import torch
    from rl_training.ea_lpl.episodes import episode_records
    base = wrapped.unwrapped
    n = base.num_envs
    if n != 120 or abs(base.step_dt - 0.02) > 1e-9:
        raise ValueError("EA-LPL runs require 120 environments at 50 Hz")
    identity = environment_identity(base)
    if spec["columns"] is not None and [a["column"] for a in identity["assignments"]] != spec["columns"]:
        raise ValueError("initial terrain columns differ from the spec")
    check_initial_state(base, spec)
    request = replay = None
    if spec["codes"] is not None:
        from rl_training.ea_lpl.basis import BASIS_NAMES, residual_request, stacked_bases
        index = torch.tensor([BASIS_NAMES.index(b) for b in spec["codes"]["basis"]], device=base.device)
        codes_u = torch.tensor(spec["codes"]["u"], device=base.device, dtype=torch.float32)
        request = residual_request(index, codes_u, stacked_bases(device=base.device))
        if spec["codes"]["record_replay"]:
            replay = {key: [] for key in ("base_action", "executed_action", "gravity_xy", "base_height",
                                          "joint_vel", "nonfoot_contact", "control_timed_out")}
    observations = wrapped.get_observations()
    manager = base.reward_manager
    original_compute = manager.compute
    captured = {}
    chunks = {key: [] for key in ("rewards", "state", "columns", "actions", "done")}
    events = []
    finished = np.zeros(n, dtype=np.int64)
    target = spec["episodes_per_env"]

    def compute(dt):
        value = original_compute(dt)
        data = base.scene["robot"].data
        velocity = data.root_lin_vel_b
        gravity = data.projected_gravity_b
        command = base.command_manager.get_command("base_velocity")
        ground = base.scene.sensors["height_scanner_base"].data.ray_hits_w[:, :, 2].mean(-1)
        relative = data.root_link_pos_w - base.scene.terrain.env_origins
        w, x, y, z = data.root_link_quat_w.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        state = torch.stack((
            velocity[:, 0], velocity[:, 1], velocity[:, 2], (velocity[:, :2] - command[:, :2]).square().sum(-1),
            data.root_link_pos_w[:, 2] - ground, relative[:, 0], (data.joint_pos - data.default_joint_pos).norm(dim=-1),
            data.joint_vel.square().mean(-1).sqrt(), gravity[:, :2].norm(dim=-1), relative[:, 1],
            torch.asin(gravity[:, 1].clamp(-1, 1)), torch.asin(gravity[:, 0].clamp(-1, 1)), yaw,
            data.root_ang_vel_b[:, 2], command[:, 2]), dim=-1)
        captured.update(rewards=manager._step_reward.clone(), state=state,
                        columns=base.scene.terrain.terrain_types.clone())
        return value

    manager.compute = compute
    try:
        with torch.inference_mode():
            for step in range(1, spec["max_episode_steps"] * target + 1):
                if not app.is_running():
                    raise RuntimeError("simulator stopped")
                action = actor(observations)
                if request is None:
                    observations, reward, done, extras = wrapped.step(action)
                else:
                    observations, reward, done, extras = wrapped.step_residual(action, request)
                report["transitions"] += n
                trace = wrapped.last_trace
                if replay is not None:
                    state_now = trace.safety_state
                    for key, value in (("base_action", trace.base_action), ("executed_action", trace.executed_action),
                                       ("gravity_xy", state_now.projected_gravity_xy),
                                       ("base_height", state_now.base_height), ("joint_vel", state_now.joint_vel),
                                       ("nonfoot_contact", state_now.nonfoot_contact),
                                       ("control_timed_out", state_now.control_timed_out)):
                        replay[key].append(value.cpu().numpy().copy())
                    if "joint_lower" not in replay:
                        replay.update(joint_lower=state_now.joint_lower.cpu().numpy(),
                                      joint_upper=state_now.joint_upper.cpu().numpy(),
                                      default_joint_pos=state_now.default_joint_pos.cpu().numpy(),
                                      action_scale=state_now.action_scale.cpu().numpy())
                metrics = torch.stack((
                    action.square().mean(-1).sqrt(), (trace.raw_policy_action != trace.base_action).float().mean(-1),
                    trace.decision.projected[:, 0].float(), trace.decision.paused[:, 0].float(),
                    (trace.executed_action - (action if request is None else trace.base_action + request)).abs().amax(-1)),
                    dim=-1)
                for key in ("rewards", "state", "columns"):
                    chunks[key].append(captured[key].cpu().numpy().copy())
                chunks["actions"].append(metrics.cpu().numpy().copy())
                chunks["done"].append(done.cpu().numpy().copy())
                terminal = extras["source_terminal"]
                if terminal is not None:
                    ids = terminal["ids"].cpu().tolist()
                    ending = {k: terminal[k].cpu().tolist() for k in ("steps", "hold", "completed")}
                    terms = {k: v.cpu().tolist() for k, v in terminal["terms"].items()}
                    for j, i in enumerate(ids):
                        events.append(dict(env_id=i, control_step=step, episode_index=int(finished[i]),
                                           episode_step=ending["steps"][j], hold=ending["hold"][j],
                                           completed=ending["completed"][j],
                                           terms=[k for k, v in terms.items() if v[j]]))
                        finished[i] += 1
                check_scans(base)
                if step % 200 == 0:
                    done_envs = int((finished >= target).sum())
                    write_json(folder / "progress.json", dict(step=step, transitions=report["transitions"],
                                                              finished_envs=done_envs))
                    print(f"[ea-lpl] {spec['label']} step={step} finished={done_envs}/{n}", flush=True)
                if (finished >= target).all():
                    break
    finally:
        manager.compute = original_compute
        arrays = {key: np.asarray(value) for key, value in chunks.items()}
        np.savez_compressed(folder / f"{spec['label']}.npz", **arrays)
        write_json(folder / f"{spec['label']}_events.json", events)
    if not all(np.isfinite(arrays[key]).all() for key in ("rewards", "state", "actions")):
        raise FloatingPointError("recorded data contain nonfinite values")
    if (finished < target).any():
        raise RuntimeError("some environments did not finish their registered episodes")
    records = episode_records(arrays, events, identity["assignments"], episodes_per_env=target,
                              control_dt=base.step_dt)
    if replay is not None:
        first_end = np.array([next(r["steps"] for r in records if r["env_id"] == i and r["episode_index"] == 0)
                              for i in range(n)])
        arrays_out = {k: np.asarray(v) for k, v in replay.items()
                      if k not in ("joint_lower", "joint_upper", "default_joint_pos", "action_scale")}
        np.savez_compressed(folder / f"{spec['label']}_replay.npz", **arrays_out,
                            joint_lower=replay["joint_lower"], joint_upper=replay["joint_upper"],
                            default_joint_pos=replay["default_joint_pos"], action_scale=replay["action_scale"],
                            request=request.cpu().numpy(), basis_index=index.cpu().numpy(),
                            u=codes_u.cpu().numpy(), episode_end=first_end)
    return dict(label=spec["label"], initial=identity, records=records, controls=len(arrays["done"]),
                state_names=STATE_NAMES, action_names=ACTION_NAMES, codes=spec["codes"])


def child(spec_path, folder):
    from rl_training.ea_lpl.specs import spec_actor, validate_spec
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    path, sha = spec_actor(spec, default=(OLD, ACTOR_SHA256))
    report = dict(status="starting", transitions=0, label=spec["label"])
    app = wrapped = None
    stack_file = (folder / "python_stacks.log").open("w")
    faulthandler.enable(file=stack_file)

    def phase(name, limit):
        faulthandler.cancel_dump_traceback_later()
        record_phase(folder, name, spec["label"], limit)
        if limit:
            faulthandler.dump_traceback_later(max(1, limit - 5), file=stack_file)

    try:
        phase("application_start", 180)
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device="cuda:0").app
        import torch
        from rl_training.source_scan.frozen_actor import load_frozen_source_actor
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        phase("environment_start", 300)
        wrapped = make_ea_environment(spec)
        actor = load_frozen_source_actor(path, device="cuda:0", expected_sha256=sha)
        phase("controls", 600 * spec["episodes_per_env"])
        report["case"] = collect_fixed(wrapped, actor, spec, folder, app, report)
        report.update(level=spec["level"], slots=spec["slots"])
        report["status"] = "passed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(folder / "result.json", report)
        if wrapped is not None:
            phase("environment_close", 30)
            close_comparison_environment(wrapped)
        if app is not None:
            phase("application_close", 30)
            app.close()
        phase("finished", None)
        faulthandler.disable()
        stack_file.close()
    return 0 if report["status"] == "passed" else 1


def gpu_free_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                  text=True, timeout=10)
    return {int(a): int(b) for a, b in (line.split(",") for line in out.strip().splitlines())}, out


def choose_gpu(request):
    free, _ = gpu_free_mib()
    candidates = [0, 1] if request == "auto" else [int(request)]
    usable = [(free[i], i) for i in candidates if free.get(i, 0) >= MIN_FREE_MIB]
    if not usable:
        raise RuntimeError(f"no GPU among {candidates} has {MIN_FREE_MIB} MiB free: {free}")
    return max(usable)[1]


def supervise(spec_path, folder, gpu_request):
    from rl_training.ea_lpl.specs import spec_actor, validate_spec
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    path, sha = spec_actor(spec, default=(OLD, ACTOR_SHA256))
    if digest(path) != sha:
        raise ValueError("base actor checkpoint changed")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "source", "scripts"], cwd=ROOT, text=True).strip():
        raise RuntimeError("commit source and script changes before a GPU run")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    gpu = choose_gpu(gpu_request)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "spec.json", spec)
    write_json(folder / "manifest.json", dict(
        schema="ea_lpl_run_manifest_v1", label=spec["label"], revision=revision,
        spec_sha256=digest(folder / "spec.json"), actor_sha256=sha, actor_path=str(path), gpu=gpu,
        gpu_free_mib_before=gpu_free_mib()[0], python=sys.executable, argv=sys.argv, started_unix=time.time()))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    started, reason = time.monotonic(), None
    with (folder / "runtime.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", __file__, "--child", "--spec", str(folder / "spec.json"),
                                    "--output", str(folder)], cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                now = time.monotonic()
                reason = (f"wall limit {spec['wall_limit_s']}s exceeded" if now - started > spec["wall_limit_s"]
                          else lifecycle_deadline(folder, now))
                if reason:
                    break
                time.sleep(1)
        finally:
            if group_exists(process.pid):
                terminate_group(process, 15)
    result = folder / "result.json"
    status = json.loads(result.read_text())["status"] if result.exists() else "missing"
    receipt = dict(exit_code=process.poll(), deadline_failure=reason, elapsed_s=time.monotonic() - started,
                   process_group_empty=not group_exists(process.pid), status=status, finished_unix=time.time())
    write_json(folder / "execution.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if status == "passed" and reason is None and receipt["exit_code"] == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="auto", choices=("auto", "0", "1", "2"))
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    raise SystemExit(child(args.spec, args.output) if args.child else supervise(args.spec, args.output, args.gpu))
