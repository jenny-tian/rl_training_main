"""Evaluate a deterministic source actor on the original Flat task for 3000 steps.

The parent admits only physical GPU 0/1 with >=12288 MiB free, watches native
simulator errors and bounds startup, evaluation, and cleanup of its own child.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
from rl_training.source_scan.smoke_artifacts import atomic_json  # noqa: E402
from source_curriculum_check import runtime_failure  # noqa: E402

TASK = "Flat-Deeprobotics-Lite3-v0"
NUM_ENVS = 1024
SEED = 20_261_109
STEPS = 3000
MIN_FREE_MIB = 12288
EXPECTED_TERMINATIONS = {"time_out", "illegal_contact", "bad_orientation_2"}


def artifact_paths(output):
    return {"report": output, "log": output.with_suffix(".sim.log"), "phase": output.with_suffix(".phase.json")}


def phase(args, name, limit):
    atomic_json(artifact_paths(args.output)["phase"],
                dict(name=name, started_monotonic=time.monotonic(), limit_seconds=limit))


def choose_gpu(request):
    if request not in ("auto", "0", "1"):
        raise ValueError("retention evaluations may use only physical GPU 0 or 1")
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True, timeout=10)
    free = {int(index): int(memory) for index, memory in (line.split(",") for line in output.strip().splitlines())}
    candidates = [0, 1] if request == "auto" else [int(request)]
    usable = [(free.get(gpu, 0), gpu) for gpu in candidates if free.get(gpu, 0) >= MIN_FREE_MIB]
    if not usable:
        raise RuntimeError(f"no GPU among {candidates} has {MIN_FREE_MIB} MiB free: {free}")
    memory, gpu = max(usable)
    return gpu, memory


def collect(wrapped, actor, app):
    """Use the pre-step command and only post-step states that were not reset."""
    import torch
    from rl_training.ea_lpl.retention import RetentionMetrics

    base = wrapped.unwrapped
    if base.num_envs != NUM_ENVS or set(base.termination_manager.active_terms) != EXPECTED_TERMINATIONS:
        raise ValueError("Flat environment count or termination contract differs")
    dimensions = base.observation_manager.group_obs_dim
    if tuple(dimensions["policy"]) != (45,) or tuple(dimensions["critic"]) != (48,):
        raise ValueError("Flat actor/critic observations must remain 45/48")
    observations = wrapped.get_observations()
    metrics = RetentionMetrics(device=base.device)
    with torch.inference_mode():
        for step in range(1, STEPS + 1):
            if not app.is_running():
                raise RuntimeError("simulator stopped during retention evaluation")
            # command_manager.compute() may resample or change heading commands inside step().
            command = base.command_manager.get_command("base_velocity").clone()
            action = actor(observations)
            if action.shape != (NUM_ENVS, 12) or not bool(torch.isfinite(action).all()):
                raise FloatingPointError("actor action must be finite [1024, 12]")
            observations, reward, done, _ = wrapped.step(action)
            if not bool(torch.isfinite(reward).all()) or any(
                not bool(torch.isfinite(observations[name]).all()) for name in ("policy", "critic")
            ):
                raise FloatingPointError("nonfinite Flat reward or observation")
            data = base.scene["robot"].data
            manager = base.termination_manager
            # This installed TerminationManager keeps current term flags across auto-reset.
            # Mask done envs: their root data now describe the next episode's reset state.
            metrics.update(command, data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2], done,
                           manager.get_term("illegal_contact"), manager.get_term("bad_orientation_2"))
            if step % 250 == 0:
                print(f"[flat-retention] step={step}/{STEPS}", flush=True)
    return metrics.summary()


def child(args):
    report = dict(task=TASK, checkpoint=str(args.checkpoint), checkpoint_sha256=args.sha256, seed=SEED,
                  num_envs=NUM_ENVS, requested_steps=STEPS, completed_steps=0, clip_actions=100,
                  policy_observation_corruption=False, safety_filter=False, status="starting", cleanup={})
    app = wrapped = raw = None
    faulthandler.enable()
    try:
        phase(args, "application_start", 180)
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device="cuda:0").app
        import gymnasium as gym
        import torch
        import rl_training.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rl_training.source_scan.frozen_actor import load_frozen_source_actor

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        phase(args, "environment_start", 300)
        cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=NUM_ENVS)
        cfg.seed = SEED
        cfg.observations.policy.enable_corruption = False
        # Leave the original Flat command sampler, randomization and observation terms intact.
        raw = gym.make(TASK, cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=100)
        actor = load_frozen_source_actor(args.checkpoint, device="cuda:0", expected_sha256=args.sha256)
        report["termination_terms"] = sorted(wrapped.unwrapped.termination_manager.active_terms)
        phase(args, "controls", 1800)
        report["metrics"] = collect(wrapped, actor, app)
        report.update(status="passed", completed_steps=STEPS, actor_observation_dim=45, critic_observation_dim=48)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        # Persist before potentially blocking native close; the parent also requires both closes to return.
        atomic_json(args.output, report)
        environment = wrapped if wrapped is not None else raw
        for name, resource in (("environment", environment), ("application", app)):
            if resource is None:
                report["cleanup"][name] = "not_created"
                continue
            phase(args, name + "_close", 30)
            report["cleanup"][name] = "closing"
            atomic_json(args.output, report)
            try:
                if name == "environment":
                    # Established teardown pattern: suppress Isaac Lab's wait-for-play STOP callback.
                    sim = resource.unwrapped.sim
                    sim._disable_app_control_on_stop_handle = True
                    sim.stop()
                    if sim.is_playing():
                        raise RuntimeError("Flat timeline did not stop before teardown")
                resource.close()
                report["cleanup"][name] = "returned"
            except Exception:
                report.update(status="failed", cleanup_error=traceback.format_exc())
                report["cleanup"][name] = "failed"
                traceback.print_exc()
            atomic_json(args.output, report)
        phase(args, "finished", 30)
    return 0 if report["status"] == "passed" else 1


def stop_child(process):
    """Signal only the new process group created for this evaluation."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGKILL:
            break


def supervise(args):
    paths = artifact_paths(args.output)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("preserve existing retention evidence; choose a new output path")
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.sha256:
        raise ValueError("checkpoint hash differs from --sha256")
    gpu, free = choose_gpu(args.gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failures, deadline = [], None
    lines = queue.Queue()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--checkpoint",
               str(args.checkpoint), "--sha256", args.sha256, "--output", str(args.output), "--gpu", str(gpu)]
    with paths["log"].open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                   start_new_session=True)

        def read_output():
            for line in process.stdout:
                log.write(line)
                log.flush()
                lines.put(line)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            while process.poll() is None or reader.is_alive() or not lines.empty():
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    line = ""
                if runtime_failure(line):
                    failures.append(line.strip())
                if line.startswith("[flat-retention]"):
                    print(line, end="", flush=True)
                now = time.monotonic()
                if now - started > 2400:
                    deadline = "evaluation exceeded 2400s wall limit"
                if paths["phase"].exists():
                    state = json.loads(paths["phase"].read_text())
                    if now - state["started_monotonic"] > state["limit_seconds"]:
                        deadline = f"{state['name']} exceeded {state['limit_seconds']}s"
                if failures or deadline:
                    break
        finally:
            stop_child(process)
            reader.join(timeout=5)
            process.stdout.close()
        # Include errors emitted during cleanup or while stopping the child.
        while not lines.empty():
            line = lines.get_nowait()
            if runtime_failure(line):
                failures.append(line.strip())
    report = json.loads(args.output.read_text()) if args.output.exists() else dict(
        task=TASK, checkpoint=str(args.checkpoint), checkpoint_sha256=args.sha256, num_envs=NUM_ENVS,
        seed=SEED, requested_steps=STEPS, completed_steps=0, status="failed")
    healthy = (process.returncode == 0 and not failures and deadline is None and report.get("status") == "passed"
               and report.get("completed_steps") == STEPS
               and report.get("cleanup") == {"environment": "returned", "application": "returned"})
    report.update(status="passed" if healthy else "failed", runtime=dict(
        physical_gpu=gpu, admission_free_mib=free, simulator_log=str(paths["log"]), child_exit=process.returncode,
        fatal_messages=failures, deadline_failure=deadline, elapsed_seconds=time.monotonic() - started))
    atomic_json(args.output, report)
    print(json.dumps(dict(status=report["status"], output=str(args.output))), flush=True)
    return 0 if healthy else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=("auto", "0", "1"), default="auto")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{64}", args.sha256) is None:
        parser.error("--sha256 must be 64 lowercase hex characters")
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    return child(args) if args.child else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
