"""Bounded old/B visual review: ten frozen captures and thirty selected clips."""
from pathlib import Path
import argparse
import subprocess
import sys

from reference_capability_visual_replay import read
from reference_stairs_eval import ROOT, save, sha
from reference_stair_allocation_complete import OUTPUT as FORMAL

OUTPUT = ROOT/"logs/reference_six/capability_visual_20261116"
DOCS = ROOT/"docs/lite3-capability-visual-20261116"


def build_plan():
    complete = read(FORMAL/"complete.json")
    assert complete["status"] == "completed"
    specs = [
        ("retention_training_randomized", "retention", "training_randomized",
         [("rough_randomized", 22), ("startup_randomized", 5)], None),
        ("retention_nominal", "retention", "nominal",
         [("flat", 0), ("rough_nominal", 30), ("slope_up", 50), ("slope_down", 70),
          ("friction_lateral", 90), ("friction_longitudinal", 110),
          ("slope_rough_up", 130), ("slope_rough_down", 156)], None),
        ("turn_nominal", "turn", "nominal",
         [("turn_right", 100), ("heading_right", 260), ("arc_left", 280)], None),
        ("heading_up_nominal", "stairs", "nominal", [("stairs_up_5cm", 200)], "up"),
        ("heading_down_nominal", "stairs", "nominal", [("stairs_down_5cm", 200)], "down"),
    ]
    jobs, formal_files = [], set()
    for name, kind, profile, selections, direction in specs:
        for policy in ("zero10100", "B"):
            if kind == "retention":
                formal = FORMAL/"retention"/policy/profile
            else:
                entry = next(r for r in complete["matrix"] if r["variant"] == "B" and r["name"] == name)
                formal = Path(entry["baseline"] if policy == "zero10100" else entry["path"])
            config, report = read(formal/"config.json"), read(formal/"report.json")
            checkpoint = config["checkpoint"]
            digest = config.get("model_sha256", config.get("checkpoint_sha256"))
            assert digest == sha(checkpoint) and report["actor_unchanged"]
            jobs.append(dict(id=f"{policy}_{name}", policy=policy, name=name, kind=kind,
                profile=profile, direction=direction, heading=kind == "stairs",
                formal=str(formal), checkpoint=checkpoint, checkpoint_sha256=digest,
                replay_episodes=report["episodes"],
                selections=[dict(name=n, env_id=i, original=report["records"][i]) for n, i in selections]))
            formal_files.update(formal/f for f in ("initial.npz", "trace.npz", "report.json", "config.json", "execution.json"))
    scripts = ("reference_capability_visual_replay.py", "reference_capability_visual_run.py",
        "reference_stair_allocation_retention.py", "reference_turn_diagnose.py",
        "reference_isaac_stairs_diagnose.py", "reference_low_stairs_heading_probe.py",
        "reference_stair_allocation_heading.py", "source_block_runtime.py")
    sources = [ROOT/"scripts/tools"/name for name in scripts]
    recovery = OUTPUT/"zero10100_retention_training_randomized"
    assert (recovery/"render_states.npz").is_file()
    jobs[0]["recover_capture"] = str(recovery)
    plan = dict(status="prepared", physical_gpu=1, training=False, optimizer_updates=0,
        new_formal_evaluation_episodes=0, maximum_replay_episodes=2240,
        maximum_capture_jobs=10, render_only_recovery_jobs=1, maximum_execution_jobs=11,
        selected_scenes=15, selected_clips=30,
        no_automatic_retry=True, no_deployment=True, jobs=jobs,
        rendering="Re-run original full batches, capture pre-reset root/joints, compare initial/config/all traces, then render saved states without physics.",
        capture_mismatch="Initial/config/nonfinite mismatch stops; finite trace differences are recorded and labelled, never counted as formal successes.",
        video_timing="Ordinary clips sample 0.1 s at 10 fps; startup samples 0.02 s at 20 fps (0.4x). Final state held 1 s.",
        selection_rule="Fixed representative starts, including known failures; selections frozen before rendering, not a new statistical sample.",
        sources=[dict(path=str(p), sha256=sha(p)) for p in sources],
        formal_manifests=[dict(path=str(p), sha256=sha(p)) for p in sorted(formal_files)],
        previous_complete_sha256=sha(FORMAL/"complete.json"),
        repair="Preserve first completed 160-episode capture; repair metadata comparison and resume rendering after 50 preparation steps, with zero repeated policy rollout.",
        original_plan_sha256=sha(OUTPUT/"plan.json"))
    assert sum(j["replay_episodes"] for j in jobs) == 2240
    assert sum(len(j["selections"]) for j in jobs) == 30
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan = build_plan()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/"plan_v2.json"
    if path.exists():
        assert read(path) == plan
    else:
        save(path, plan)
        save(DOCS/"execution_plan_v2.json", plan)
    if args.prepare_only:
        print("PREPARED", plan["maximum_capture_jobs"], "capture jobs;", plan["selected_clips"], "clips; one render-only recovery; no training")
        return
    if (OUTPUT/"complete.json").exists():
        raise RuntimeError("visual stage already ended")
    completed = []
    for job in plan["jobs"]:
        output = OUTPUT/(job["id"]+"_render_only" if job.get("recover_capture") else job["id"])
        if output.exists():
            assert (output/"visual_complete.json").exists() and read(output/"execution.json")["status"] == "completed", "no automatic retry"
        else:
            print("VISUAL JOB", job["id"], flush=True)
            subprocess.run([sys.executable, str(ROOT/"scripts/tools/reference_capability_visual_replay.py"),
                "--plan", str(path), "--job", job["id"], "--output", str(output)], cwd=ROOT, check=True)
        result = read(output/"visual_complete.json")
        assert result["status"] == "completed" and result["new_optimizer_updates"] == 0
        completed.append(dict(job_id=job["id"], path=str(output), replay_episodes=result["replay_episodes"],
            reproduction=result["reproduction"], clips=result["clips"]))
        save(OUTPUT/"progress.json", dict(completed_jobs=len(completed), total_jobs=10, matrix=completed))
    assert sum(r["replay_episodes"] for r in completed) == 2240
    assert all(sha(p["path"]) == p["sha256"] for p in plan["sources"]+plan["formal_manifests"])
    assert sha(FORMAL/"complete.json") == plan["previous_complete_sha256"]
    save(OUTPUT/"complete.json", dict(status="completed", new_optimizer_updates=0,
        new_formal_evaluation_episodes=0, replay_episodes=2240, clips=30, matrix=completed,
        no_deployment=True, previous_results_unchanged=True))
    print("VISUAL CAPTURE COMPLETE: 30 clips, 2240 excluded replay episodes, zero training", flush=True)


if __name__ == "__main__":
    main()
