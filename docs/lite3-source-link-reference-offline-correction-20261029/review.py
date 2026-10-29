"""CPU-only archive review; never import IsaacLab or start simulation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/lite3-source-uphill-boundary-diagnostic-recheck-20261029"
sys.path[:0] = [str(ROOT), str(ROOT / "source/rl_training")]

import torch

from rl_training.source_scan.rigid_body_reference import REFERENCE_POINTS, validate_body_reference_state
from rl_training.source_scan.tile_bounds import source_tile_bounds_terminated
from rl_training.source_scan.tile_bounds_diagnostics import validate_tile_bounds_trace
from scripts.tools.source_policy_prequalification import make_run_manifest, select_case, verify_hashes

ARTICULATION = Path("/home/tjx/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py")
MATH = Path("/home/tjx/IsaacLab/source/isaaclab/isaaclab/utils/math.py")
PARAMETERS = dict(size=(12., 4.), flat_length_m=3., collision_radius_m=0.27,
                  numerical_margin_m=0.001, control_period_s=0.02)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((OUT / name).read_text())


def extracted_function(path, name, namespace):
    """Execute only a reviewed arithmetic function, with no imports/decorators."""
    matches = [node for node in ast.walk(ast.parse(path.read_text()))
               if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    node = matches[0]
    node.decorator_list = []
    assert not any(isinstance(child, (ast.Import, ast.ImportFrom)) for child in ast.walk(node))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def compare_installed_arithmetic():
    namespace = {"torch": torch}
    quat_apply = extracted_function(MATH, "quat_apply", namespace)
    namespace["math_utils"] = NS(quat_apply=quat_apply)
    body_velocity = extracted_function(ARTICULATION, "body_link_vel_w", namespace)
    body_linear_slice = extracted_function(ARTICULATION, "body_link_lin_vel_w", namespace)
    root_velocity = extracted_function(ARTICULATION, "root_link_vel_w", namespace)
    root_linear_slice = extracted_function(ARTICULATION, "root_link_lin_vel_w", namespace)

    def evaluate(pos, quat, offset, com_vel, omega):
        dummy = NS(body_com_vel_w=torch.cat((com_vel, omega), -1),
                   body_link_quat_w=quat, body_com_pos_b=offset, _sim_timestamp=1,
                   _body_link_vel_w=NS(timestamp=0, data=None))
        dummy.body_link_vel_w = body_velocity(dummy)
        vel = body_linear_slice(dummy)
        root = NS(root_com_vel_w=dummy.body_com_vel_w[:, 0], root_link_quat_w=quat[:, 0],
                  body_com_pos_b=offset, _sim_timestamp=1, _root_link_vel_w=NS(timestamp=0, data=None))
        root.root_link_vel_w = root_velocity(root)
        assert torch.equal(root_linear_slice(root), vel[:, 0])
        state = dict(body_positions_w=pos, body_linear_velocities_w=vel,
                     body_link_quats_w_wxyz=quat, body_com_offsets_link_m=offset,
                     body_com_positions_w=pos + quat_apply(quat, offset),
                     body_com_linear_velocities_w=com_vel, body_angular_velocities_w=omega,
                     reference_points=REFERENCE_POINTS, state_dtype=str(pos.dtype))
        conversion = validate_body_reference_state(state)
        assert conversion["status"] == "verified"
        count = pos.shape[0]
        def guard(v):
            return source_tile_bounds_terminated(pos, v, torch.zeros((1, 1, 3), dtype=pos.dtype),
                torch.zeros(count, dtype=torch.long), torch.zeros(count, dtype=torch.long), **PARAMETERS)
        return vel, guard(com_vel), guard(vel), conversion

    planar = []
    for name, offset_x, quat_wz, com_vy, omega_z, expected_vy, expected_old, expected_new in (
        ("zero_offset_rotation", 0., (1., 0.), 0.3, 20., 0.3, True, True),
        ("offset_pure_translation", 0.01, (1., 0.), 0.3, 0., 0.3, True, True),
        ("rotation_removes_mixed_point_hit", 0.01, (1., 0.), 0.3, 20., 0.1, True, False),
        ("rotation_adds_previously_missed_hit", 0.01, (1., 0.), 0., -20., 0.2, False, True),
        ("rotated_offset_180deg", 0.01, (0., 1.), 0.3, 20., 0.5, True, True),
    ):
        pos = torch.tensor([[[0., 1.726, 0.]]], dtype=torch.float64)
        quat = torch.tensor([[[quat_wz[0], 0., 0., quat_wz[1]]]], dtype=torch.float64)
        offset = torch.tensor([[[offset_x, 0., 0.]]], dtype=torch.float64)
        com = torch.tensor([[[0., com_vy, 0.]]], dtype=torch.float64)
        omega = torch.tensor([[[0., 0., omega_z]]], dtype=torch.float64)
        vel, old, new, _ = evaluate(pos, quat, offset, com, omega)
        assert abs(vel[0, 0, 1].item() - expected_vy) < 1e-12
        assert old.item() == expected_old and new.item() == expected_new
        planar.append(dict(case=name, link_y_m=1.726, upper_y_m=1.729,
                           com_vy_mps=com_vy, link_vy_mps=vel[0, 0, 1].item(),
                           legacy_prediction_y_m=1.726+com_vy*0.02,
                           corrected_prediction_y_m=1.726+expected_vy*0.02,
                           legacy_hit=old.item(), corrected_hit=new.item()))

    randomized = []
    generator = torch.Generator(device="cpu").manual_seed(101029)
    for dtype in (torch.float32, torch.float64):
        count = 4096
        pos = torch.zeros((count, 1, 3), dtype=dtype)
        pos[..., 1] = 1.72 + torch.rand((count, 1), dtype=dtype, generator=generator) * 0.009
        quat = torch.randn((count, 1, 4), dtype=dtype, generator=generator)
        quat /= torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        offset = (torch.rand((count, 1, 3), dtype=dtype, generator=generator)-0.5) * 0.06
        com = torch.randn((count, 1, 3), dtype=dtype, generator=generator) * 0.3
        omega = torch.randn((count, 1, 3), dtype=dtype, generator=generator) * 20
        _, old, new, conversion = evaluate(pos, quat, offset, com, omega)
        randomized.append(dict(dtype=str(dtype), cases=count, conversion=conversion,
                               changed_flags=int((old != new).sum()),
                               removed_hits=int((old & ~new).sum()), added_hits=int((~old & new).sum())))
    return dict(scope="extracted installed arithmetic on synthetic CPU buffers; no live PhysX validation",
                installed_body_and_root_functions_agree=True, analytic_planar_cases=planar,
                randomized_cases=randomized, random_seed=101029)


def review():
    preparation = read("preparation.json")
    verify_hashes(preparation["input_sha256"])
    assert sha(Path(preparation["checkpoint"])) == preparation["checkpoint_sha256"]
    assert preparation["simulation_run"] is preparation["runtime_authorized"] is preparation["ppo_authorized"] is False
    assert len(preparation["cases"]) == 1
    case = select_case(preparation, "slope_L1_uphill")
    assert (case["geometry_seed"], case["initial_state_seed"], case["geometry_difficulty"]) == (10001, 20060, 0.3384848514104325)
    old_prep = json.loads((ROOT / "docs/lite3-source-uphill-drift-boundary-offline-20261029/preparation.json").read_text())
    for key in ("cases", "command", "termination", "runtime", "checkpoint", "checkpoint_sha256", "safety_height"):
        assert preparation[key] == old_prep[key], key
    assert preparation["input_sha256"]["source/rl_training/rl_training/source_scan/tile_bounds.py"] == old_prep["input_sha256"]["source/rl_training/rl_training/source_scan/tile_bounds.py"]
    preserved = 0
    for label in ("lite3-source-uphill-drift-boundary-offline-20261029", "lite3-source-uphill-boundary-diagnostic-recheck-20261029",
                  "lite3-source-policy-prequalification-flat-20261029", "lite3-source-policy-prequalification-rough-l1-20261029",
                  "lite3-source-policy-prequalification-uphill-l1-20261029"):
        archive = ROOT / "docs" / label
        for line in (archive / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            assert sha(archive / name) == expected, str(archive / name)
            preserved += 1
    assert (OUT / "baseline_progress.json").read_bytes() == (PREVIOUS / "baseline_progress.json").read_bytes()
    old_record = json.loads((PREVIOUS / "result.json").read_text())["record"]
    old_review = validate_tile_bounds_trace(PREVIOUS / "tile_bounds.jsonl", old_record, preparation["expected_body_names"])
    assert old_review == json.loads((PREVIOUS / "tile_bounds_review.json").read_text())
    comparison = compare_installed_arithmetic()
    counts = {}
    for label in ("focused", "regression"):
        suites = ET.parse(OUT / f"{label}.junit.xml").getroot().findall("testsuite")
        counts[label] = {k: sum(int(s.attrib[k]) for s in suites) for k in ("tests", "errors", "failures", "skipped")}
        assert all(counts[label][key] == 0 for key in ("errors", "failures", "skipped"))
    assert counts["regression"]["tests"] > 615
    stage = read("next_stage.json")
    assert stage["runtime_authorized"] is False and stage["automatic_retries"] == 0
    assert stage["case"] == case and stage["preparation_sha256"] == sha(OUT / "preparation.json")
    assert not Path(stage["run_directory"]).exists()
    generated = make_run_manifest(preparation, case, revision="offline-only", gpu="not-queried",
                                  processes="not-queried", preparation_sha256=sha(OUT / "preparation.json"))
    assert generated["link_reference_verification_required"] is True
    assert generated["tile_bounds_schema"] == "lite3_tile_bounds_predicate_v2"
    assert generated["max_control_steps"] == 2000 and generated["num_envs"] == 1
    assert not any(n == "pxr" or n.startswith(("isaaclab", "isaacsim", "omni.")) for n in sys.modules)
    return dict(status="offline_correction_verified_runtime_not_run", simulation_run=False, ppo_initialized=False,
                input_hashes_verified=len(preparation["input_sha256"]), prior_archive_files_unchanged=preserved,
                old_predicate_evaluations_replayed=old_review["evaluations_verified"], cpu_comparison=comparison,
                tests=counts, baseline_progress_unchanged=True, future_run_spec_checked=True,
                historical_corrected_terminal_prediction="unrecoverable_from_existing_logs",
                g3_qualified=False, g4_qualified=False,
                limitations=["Synthetic CPU arithmetic cannot prove live buffer timing, PhysX state, logging overhead or future outcome.",
                             "No new six-family success rate, gait learning or policy qualification follows.",
                             "The prior isolated descendant-cleanup test failure has no established cause and is not claimed fixed.",
                             "The historical torso integration residual remains unisolated; the thigh reference correction is a separate issue."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = review()
    if args.write:
        with (OUT / "post_review.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
