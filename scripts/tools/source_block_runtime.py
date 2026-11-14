"""Supervise an explicitly requested B1 simulator job on physical GPU 0 or 1."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

from source_curriculum_check import runtime_failure

ROOT = Path(__file__).resolve().parents[2]


def fatal_line(line):
    return (runtime_failure(line) or "Contacts have been dropped" in line
            or re.search(r"(?:loss|reward|observation)[^\n]*\b(?:nan|inf)\b", line, re.IGNORECASE) is not None)


def run(command, *, gpu, output, timeout=1800, minimum_free_mib=12288):
    if gpu not in (0, 1):
        raise ValueError("B1 uses only GPU0/1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def memory():
        return [int(x.strip()) for x in subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")]
    used, free = memory()
    if minimum_free_mib <= 0:
        raise ValueError("minimum_free_mib must be positive")
    if free < minimum_free_mib:
        raise RuntimeError(f"GPU{gpu}: {free} MiB free, require >={minimum_free_mib}")
    state = dict(command=command, cwd=str(ROOT), physical_gpu=gpu, admission_free_mib=free,
                 peak_total_gpu_used_mib=used, sampling_interval_seconds=2, status="running",
                 minimum_free_mib=minimum_free_mib,
                 source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 source_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "source/rl_training"), CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    # These jobs are headless. An inherited remote X display can fail GLX
    # before Isaac starts rendering; keep desktop settings outside this child.
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    started = time.monotonic()
    reason = None
    with (output / "runtime.log").open("wb") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state["pid"] = child.pid
        (output / "execution.json").write_text(json.dumps(state, indent=2))
        print(f"START pid={child.pid} gpu={gpu} log={output / 'runtime.log'}", flush=True)
        try:
            with (output / "runtime.log").open(errors="replace") as reader:
                tail = ""
                while child.poll() is None:
                    chunk = tail + reader.read()
                    tail = chunk[-1024:]
                    bad = [line for line in chunk.splitlines() if fatal_line(line)]
                    if bad:
                        reason = bad[0]
                        break
                    if time.monotonic() - started > timeout:
                        reason = "wall deadline exceeded"
                        break
                    used, _ = memory()
                    state["peak_total_gpu_used_mib"] = max(used, state["peak_total_gpu_used_mib"])
                    time.sleep(2)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            text = (output / "runtime.log").read_text(errors="replace")
            bad = [line for line in text.splitlines() if fatal_line(line)]
            state.update(exit_code=child.returncode, failure=reason or (bad[0] if bad else None),
                         fatal_messages=bad, elapsed_s=time.monotonic() - started)
            state["status"] = "completed" if child.returncode == 0 and state["failure"] is None else "failed"
            (output / "execution.json").write_text(json.dumps(state, indent=2, allow_nan=False))
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = ROOT / "logs/rsl_rl/deeprobotics_lite3_source_curriculum/2026-11-10_16-00-10_full_seed42/model_3497.pt"
    import hashlib
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != "47889ee59d07416287332164a38566a812d6ebbc0be637e13752829f8891d49c":
        raise ValueError("B1 pilot checkpoint identity mismatch")
    command = [sys.executable, "scripts/reinforcement_learning/rsl_rl/train.py",
               "--task", "SourceBlock-L2-Deeprobotics-Lite3-v0", "--num_envs", "4096", "--seed", "42", "--headless",
               "--resume", "--load_run", "2026-11-10_16-00-10_full_seed42", "--checkpoint", "model_3497.pt",
               "--max_iterations", "30", "--run_name", "block_l2_b1_pilot30"]
    sys.exit(run(command, gpu=args.gpu, output=args.output))
