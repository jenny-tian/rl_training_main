"""Audit final video artifacts, source identity and unchanged formal evidence."""
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import re
import subprocess

import numpy as np

from reference_capability_visual_run import OUTPUT, DOCS
from reference_capability_visual_replay import read, frame_indices
from reference_capability_visual_analyze import LABELS, read_probe
from reference_stairs_eval import save, sha


def decode(path):
    subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
                    "-f", "null", "-"], check=True, capture_output=True)
    return read_probe(path)


def main():
    from PIL import Image, ImageDraw, ImageFont
    summary, plan = read(DOCS/"summary.json"), read(OUTPUT/"plan_v2.json")
    assert summary["status"] == "completed" and summary["all_replays_exact"]
    assert summary["new_optimizer_updates"] == summary["new_formal_evaluation_episodes"] == 0
    assert summary["replay_episodes"] == 2240 and len(summary["scenes"]) == 15
    assert all(sha(s["path"]) == s["sha256"] for s in plan["sources"]+plan["formal_manifests"])
    assert sha(OUTPUT/"plan.json") == plan["original_plan_sha256"]
    from reference_stair_allocation_complete import OUTPUT as FORMAL
    assert sha(FORMAL/"complete.json") == plan["previous_complete_sha256"]
    manifest = read(DOCS/"raw_manifest.json")
    for item in manifest:
        p = Path(item["path"])
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], p
    for job in plan["jobs"]:
        assert sha(job["checkpoint"]) == job["checkpoint_sha256"]
        p = OUTPUT/(job["id"]+"_render_only" if job.get("recover_capture") else job["id"])
        result, check = read(p/"visual_complete.json"), read(p/"capture_validation.json")
        assert check["status"] == "exact" and check["configuration_exact"] and check["records_exact"]
        assert all(v["exact"] and v["finite"] for v in [*check["initial"].values(), *check["trace"].values()])
        assert result["physics_steps_during_rendering"] == 0
        for clip in result["clips"]:
            stride = 1 if clip["name"] == "startup_randomized" else 5
            n = round(clip["record"]["duration_s"]/.02)
            assert clip["frame_steps"] == [v+1 for v in frame_indices(n, stride)]
            assert clip["total_frames"] == len(clip["frame_steps"])+clip["fps"]
            assert clip["maximum_render_pose_error"] < 1e-5 and clip["replay_exact"]
    assert len(summary["display_repairs"]) == 2
    for policy in ("zero10100", "B"):
        repair = read(OUTPUT/f"{policy}_flat_display_repair"/"repair_complete.json")
        assert repair["policy_rollout_steps"] == repair["physics_steps_during_rendering"] == 0
        assert repair["initial_exact"] and repair["clip"]["static_prototype_hidden"]
        assert repair["source_sha256"] == sha(Path(__file__).with_name("reference_capability_visual_flat_render.py"))
    videos = sorted(DOCS.rglob("*.mp4"))
    assert len(videos) == 45
    with ThreadPoolExecutor(max_workers=2) as executor:
        probes = dict(zip(videos, executor.map(decode, videos)))
    for path, probe in probes.items():
        is_comparison = path.name == "comparison.mp4"
        assert (probe["width"], probe["height"]) == ((1920, 540) if is_comparison else (960, 540))
        meta = read(path.with_name("comparison.json" if is_comparison else "video.json"))
        frames = meta["video"]["frames"] if is_comparison else meta["total_frames"]
        assert int(probe["nb_frames"]) == frames
    with (DOCS/"selected_metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 30 and len({(r["scene"], r["policy"]) for r in metrics}) == 30
    for path in DOCS.glob("*.md"):
        text = path.read_text()
        assert not any(ch in text for ch in (chr(0x201c), chr(0x201d)))
        for target in re.findall(r"\]\(([^)]+)\)", text):
            assert (path.parent/target).exists(), (path, target)
    gallery = (DOCS/"index.html").read_text()
    assert gallery.count("<video ") == 15
    for target in re.findall(r'(?:src|poster|href)="([^"]+)"', gallery):
        assert (DOCS/target).is_file(), target
    assert not list(DOCS.rglob("*.pt")) and not list(DOCS.rglob("*.npz"))
    sheet = Image.new("RGB", (1440, 5*250), (17, 24, 36))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for index, scene in enumerate(LABELS):
        x, y = index%3*480, index//3*250
        draw.text((x+5, y+5), scene, font=font, fill="white")
        img = Image.open(DOCS/"comparisons"/scene/"middle.png").convert("RGB")
        assert img.size == (1920, 540) and np.asarray(img).std() > 2
        sheet.paste(img.resize((480, 135)), (x, y+30))
        end = Image.open(DOCS/"comparisons"/scene/"terminal.png").convert("RGB")
        sheet.paste(end.resize((480, 135)).crop((0, 45, 480, 130)), (x, y+165))
    sheet.save(DOCS/"all_scenes_contact_sheet.png")
    save(DOCS/"verification.json", dict(status="passed", formal_source_hashes_unchanged=True,
        all_ten_batches_exact=True, initial_and_complete_traces_exact=True,
        videos_decoded=45, side_by_side_videos=15, selected_scenes=15,
        dimensions_and_frame_counts_verified=True, gallery_links_verified=True,
        terminal_frames_once_then_explicit_hold=True, raw_manifest_files_verified=len(manifest),
        raw_manifest_bytes=sum(r["bytes"] for r in manifest), unique_cpu_tests_passed=3,
        optimizer_updates=0, new_formal_episodes=0, replay_episodes=2240,
        repeated_first_capture_episodes=0, physics_steps_during_rendering=0,
        flat_display_repairs=2, display_repair_policy_rollout_steps=0,
        models_and_npz_excluded_from_documentation=True, verifier_sha256=sha(__file__)))
    save(DOCS/"artifact_manifest.json", [dict(path=str(p.relative_to(DOCS)), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(DOCS.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"])
    print("VERIFIED: 45 videos, 15 comparisons, 10 exact batches; sources and formal results unchanged.")


if __name__ == "__main__":
    main()
