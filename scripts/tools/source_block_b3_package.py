"""Validate selected real replays and prepare a reviewable B3 artifact manifest."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from source_block_b3_protocol import ROOT, FIELDS

OUT = ROOT / "docs/lite3-block-b3-diagnosis-20261111"
RAW = ROOT / "logs/stage1/block_b3"
SELECTIONS = (
    ("original3497_right", "block_original3497_right", "block", "terrain3497", 96),
    ("terrain42_right", "block_terrain42_right", "block", "terrain3497_seed42", 96),
    ("flat43_failure", "flat_seed43_failure_v4", "flat", "flat1998_seed43", 113),
)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def package():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont

    target = OUT / "videos"
    target.mkdir(exist_ok=True)
    validations, sources, images = [], [], []
    for name, directory, domain, model, env_id in SELECTIONS:
        directory = RAW / "videos" / directory
        matrix = RAW / "matrix" / domain / "controlled" / model
        report = json.loads((directory / "report.json").read_text())
        execution = json.loads((directory / "execution.json").read_text())
        reference = json.loads((matrix / "report.json").read_text())
        assert report["status"] == "passed" and execution["status"] == "completed" and execution["exit_code"] == 0
        assert report["initial_hashes"] == reference["initial_hashes"]
        assert report["cleanup"] == {"environment": "returned", "application": "requested_fast_shutdown_exit"}
        replay, baseline = np.load(directory / "trace.npz"), np.load(matrix / "trace.npz")
        n = len(replay["trace"])
        delta = float(abs(replay["trace"][:, :, 3:6]-baseline["trace"][:n, :, 3:6]).max())
        same_ends = np.array_equal(replay["ended"], baseline["ended"][:n])
        assert delta == 0 and same_ends, "replay must exactly reproduce diagnostic velocities and endings"
        records = [r for r in report["records"] if r["env_id"] == env_id]
        note = "First environment in sustained-right-turn condition, same index for both models."
        if name == "flat43_failure":
            candidates = [r for r in reference["records"] if r["failed"] and r["case"] == "turn_right"]
            selected = min(candidates, key=lambda r: (r["step"], r["env_id"]))
            assert env_id == selected["env_id"] and records == [selected]
            note = "Earliest failed sustained-right-turn first episode in Flat seed43, tie broken by env_id."
            end = selected["step"]
            values = replay["trace"][:end, env_id]
            with (target / "failure_trace.csv").open("w", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["time_s", *report["fields"]])
                for i, row in enumerate(values):
                    writer.writerow([(i+1)*.02, *row.tolist()])
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            t = (np.arange(end)+1)*.02
            axes[0].plot(t, values[:, 2], "k--", label="Command yaw")
            axes[0].plot(t, values[:, 5], label="Actual yaw")
            axes[0].set_ylabel("rad/s")
            axes[1].plot(t, values[:, 10], label="Base height (m)")
            axes[1].plot(t, values[:, 11], label="Gravity XY norm")
            for body in ("TORSO", "HL_SHANK", "HR_THIGH"):
                axes[2].plot(t, values[:, report["fields"].index(body)], label=body)
            axes[2].set_ylabel("Contact history peak (N)")
            axes[2].set_xlabel("Time (s)")
            for ax in axes:
                ax.axvline(selected["time_s"], color="red", ls=":")
                ax.legend(loc="upper left")
                ax.grid(alpha=.2)
            fig.suptitle("Flat seed43 / env113: actual pre-reset terminal contact at 5.44 s")
            fig.tight_layout()
            fig.savefig(target / "failure_contact.png", dpi=150)
            plt.close(fig)
        validations.append(dict(name=name, model=model, domain=domain, env_id=env_id,
            selection=note, same_initial_hashes=True, max_velocity_difference_from_matrix=delta,
            same_termination_flags=same_ends, compared_steps=n, selected_endings=records,
            video=report["video"], cleanup=report["cleanup"], execution=execution))
        for source, destination in (("response.mp4", name+".mp4"), ("last_frame.png", name+".png")):
            shutil.copyfile(directory / source, target / destination)
        images.append(Image.open(directory / "last_frame.png").convert("RGB"))
        for filename in ("report.json", "config.json", "execution.json", "runtime.log", "initial.npz", "trace.npz", "response.mp4", "last_frame.png"):
            path = directory / filename
            sources.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(target / "original3497_right.mp4"),
        "-i", str(target / "terrain42_right.mp4"), "-filter_complex", "hstack=inputs=2:shortest=1",
        "-c:v", "libx264", "-crf", "20", str(target / "right_turn_comparison.mp4")], check=True)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 25)
    canvas = Image.new("RGB", (1920, 600), (20, 25, 35))
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), "同一初始状态、同一持续右转指令 -0.5 rad/s；左：原 3497，右：续训 100 次、种子 42", font=font, fill="white")
    canvas.paste(images[0], (0, 60))
    canvas.paste(images[1], (960, 60))
    canvas.save(target / "comparison.png")
    rejected = []
    for name, reason in (
        ("flat_seed43_failure", "Gym wrapper rejected recompute argument; no valid video."),
        ("flat_seed43_failure_v2", "Black frames and app-close hang; own child terminated with SIGKILL after SIGTERM did not exit."),
        ("flat_seed43_failure_v3", "Ground rendered but selected Flat robot inherited source invisibility; visual evidence rejected.")):
        execution = json.loads((RAW / "videos" / name / "execution.json").read_text())
        rejected.append(dict(directory=str(RAW / "videos" / name), reason=reason, execution=execution))
    save(target / "metadata.json", dict(status="passed", selected=validations, rejected_attempts=rejected,
        image_processing="original simulator RGB plus labels; comparison is resizing-free horizontal concatenation"))
    save(OUT / "video_artifact_manifest.json", sources)
    print("Validated 3 visible replay videos; initial states, velocities and terminal flags match the matrix exactly.")


if __name__ == "__main__":
    package()
