"""Verify the preflight archive using recorded resources; no live GPU query."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "source/rl_training")]

from scripts.tools import source_parallel_terrain_startup_supervisor as supervisor
from scripts.tools.source_policy_prequalification import verify_hashes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review():
    receipt = json.loads((OUT / "preflight_receipt.json").read_text())
    preparation_path = ROOT / receipt["preparation_path"]
    proposal_path = ROOT / receipt["proposal_path"]
    assert sha(preparation_path) == receipt["preparation_sha256"]
    assert sha(proposal_path) == receipt["proposal_sha256"]
    preparation = json.loads(preparation_path.read_text())
    verify_hashes(preparation["input_sha256"])
    assert sha(Path(preparation["checkpoint"])) == receipt["checkpoint_sha256"]
    assert receipt["runtime_authorized"] is True
    assert receipt["episodes_executed"] == 0
    for key in ("simulation_started", "worker_launched", "supervisor_run_launched", "ppo_initialized",
                "policy_updated", "run_directory_created", "other_gpu_used", "baseline_episode_count_changed"):
        assert receipt[key] is False, key
    assert receipt["processes_signalled"] == [] and receipt["runtime_retries"] == 0
    assert (OUT / "baseline_progress.json").read_bytes() == (preparation_path.parent / "baseline_progress.json").read_bytes()
    continuation = json.loads((OUT / "continuation.json").read_text())
    proposal = json.loads(proposal_path.read_text())
    for key in ("command", "environment", "runtime", "run_directory"):
        assert continuation[key] == proposal[key], key
    assert continuation["authorization_already_granted"] is True
    assert continuation["new_scope_approval_required"] is False
    gpu = (OUT / "gpu_snapshot.csv").read_text().strip()
    processes = (OUT / "gpu_compute_snapshot.csv").read_text().strip()
    gpu_row = gpu.split(", ")
    assert int(gpu_row[4].split()[0]) - int(gpu_row[3].split()[0]) >= 12288
    pids = [int(row.split(", ")[1]) for row in processes.splitlines() if row.startswith(gpu_row[1] + ", ")]
    assert pids == receipt["resource_admission"]["gpu0_pids"] and len(pids) == 4
    owners = (OUT / "process_owners.txt").read_text().splitlines()[1:]
    assert {int(row.split()[0]) for row in owners} == set(pids)
    assert all(row.split()[2] == "mfy" for row in owners)
    original_capture = supervisor._capture
    supervisor._capture = lambda command, **kwargs: processes if any(arg.startswith("--query-compute-apps") for arg in command) else gpu
    try:
        supervisor._gpu_preflight(0, root=ROOT, minimum_available_mib=12288)
    except RuntimeError as error:
        assert str(error) == receipt["initial_preflight_error"] == "GPU 0 already has a compute process"
    else:
        raise AssertionError("recorded state must reject GPU0 admission")
    finally:
        supervisor._capture = original_capture
    return dict(status="preflight_archive_consistent_runtime_not_started", input_hashes_verified=len(preparation["input_sha256"]),
                archived_resource_rejection_reproduced=True, simulation_started=False, episodes_executed=0,
                baseline_unchanged=True, existing_authorization_preserved=True, live_gpu_queried_by_review=False)


if __name__ == "__main__":
    print(json.dumps(review(), indent=2))
