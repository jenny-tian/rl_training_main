"""Package B1.1 verification and actual simulator captures without training artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "logs/stage1/block_reference_rough"
OUT = ROOT / "docs/lite3-block-reference-rough-b11-20261111"


def package():
    sources = {"geometry.json": RAW / "geometry_v1.json", "cpu_tests.xml": RAW / "cpu_tests.xml"}
    validation = dict(training_started=False, new_policy_updates=0, simulator={})
    suite = ET.parse(RAW / "cpu_tests.xml").getroot().find("testsuite")
    assert int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0
    assert int(suite.attrib["tests"]) == 166
    validation["cpu"] = dict(tests=166, failures=0, errors=0, seconds=float(suite.attrib["time"]))
    for count in (120, 4096):
        directory = RAW / f"check{count}_v2"
        check = json.loads((directory / "checks.json").read_text())
        runtime = json.loads((directory / "execution.json").read_text())
        assert check["all_passed"] and len(check["checks"]) == 20
        assert runtime["status"] == "completed" and not runtime["fatal_messages"]
        assert check["statistics"]["physical_difficulty"] == "L2_except_reference_uniform_rough"
        sources[f"check{count}.json"] = directory / "checks.json"
        sources[f"check{count}_runtime.json"] = directory / "execution.json"
        validation["simulator"][str(count)] = dict(checks_passed=len(check["checks"]),
            elapsed_seconds=runtime["elapsed_s"], peak_whole_gpu_used_mib=runtime["peak_total_gpu_used_mib"],
            admission_free_mib=runtime["admission_free_mib"], physical_gpu=runtime["physical_gpu"],
            rough_reset_probe=check["rough_reset_probe"], rough_collision_probe=check["rough_collision_probe"],
            endpoint_probe=check["endpoint_probe"])
    geometry = json.loads((RAW / "geometry_v1.json").read_text())
    assert geometry["all_passed"] and geometry["legacy_B1_tiles_identical"] == 200
    validation["geometry"] = {k: v for k, v in geometry.items() if k != "tiles"}
    for folder, prefix in (("visuals_v2", "visuals"), ("baseline_visuals_v1", "visuals/baseline")):
        runtime = json.loads((RAW / folder / "execution.json").read_text())
        metadata = json.loads((RAW / folder / "metadata.json").read_text())
        assert runtime["status"] == "completed" and not runtime["fatal_messages"]
        assert metadata["all_screenshots_real_simulation"] and not metadata["policy_training"]
        for path in (RAW / folder).iterdir():
            if path.suffix == ".png" or path.name in ("metadata.json", "execution.json"):
                sources[f"{prefix}/{path.name}"] = path
    for relative, source in sources.items():
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    canvas = Image.new("RGB", (1920, 1190), (20, 25, 35))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 29)
    small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 23)
    draw.text((20, 8), "粗糙面真实仿真对照 | 相同相对机位、光照与展示材质", font=font, fill="white")
    for i, (folder, title) in enumerate((("baseline_visuals_v1", "原 B1：局部条带起伏"),
                                       ("visuals_v2", "B1.1：参考项目大面积随机起伏"))):
        draw.text((i * 960 + 20, 47), title, font=font, fill="#cbd5e1")
        for j, name in enumerate(("rough_top", "rough_close")):
            frame = Image.open(RAW / folder / f"{name}.png")
            canvas.paste(frame.resize((960, 540)), (i * 960, 90 + j * 550))
    canvas.save(OUT / "visuals/rough_before_after.png")
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    code_paths = [
        "source/rl_training/rl_training/source_curriculum/block.py",
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/source_block.py",
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_block_terrain.py",
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_block_env_cfg.py",
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/__init__.py",
        "scripts/tools/source_block_check.py", "scripts/tools/source_block_geometry_audit.py",
        "scripts/tools/source_block_reference_rough_check.py", "scripts/tools/source_block_reference_rough_audit.py",
        "scripts/tools/source_block_reference_rough_visualize.py", "tests/source_curriculum/test_block_reference_rough.py"]
    for path in code_paths:
        assert subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT) == (ROOT / path).read_bytes()
    native = Path("/home/tjx/IsaacLab/source/isaaclab/isaaclab/terrains")
    references = [native / "height_field/hf_terrains.py", native / "height_field/utils.py", native / "config/rough.py",
        Path("/home/tjx/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py")]
    originals = sorted(set(sources.values()) | {p for p in RAW.rglob("*") if p.is_file()})
    def record(path):
        return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = dict(implementation_commit=commit, b11_complete=True, b2_started=False,
        runs_executed_from_worktree_before_commit=True,
        note="Final checks use the committed code. Visuals_v2 preceded only the statistics-label correction; terrain/reset/render code unchanged. First check runs and dark preview are retained as history.",
        source_files=[record(ROOT / p) for p in code_paths], references=[record(p) for p in references],
        original_artifacts=[record(p) for p in originals])
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n" for p in files))
    print(json.dumps(dict(output=str(OUT), delivered_files=len(files), original_files=len(originals), validation=validation), indent=2))


if __name__ == "__main__":
    package()
