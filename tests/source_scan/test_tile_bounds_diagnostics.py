"""Branch attribution, immutable snapshots and automatic-reset telemetry on CPU."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from rl_training.source_scan.tile_bounds import source_tile_bounds_terminated
from rl_training.source_scan.tile_bounds_diagnostics import TileBoundsRecorder, json_state, validate_tile_bounds_trace
from scripts.tools import source_policy_prequalification as qualification

PARAMETERS = dict(size=(12.0, 4.0), flat_length_m=3.0, collision_radius_m=0.27,
                  numerical_margin_m=0.001, control_period_s=0.02)


def snapshot(*, branch=None):
    positions = torch.zeros((1, 17, 3), dtype=torch.float64)
    velocities = torch.zeros_like(positions)
    origins = torch.zeros((1, 1, 3), dtype=torch.float64)
    levels = torch.tensor([0])
    if branch == "current":
        positions[0, 5, 1] = 1.74
    elif branch == "prediction":
        positions[0, 5, 1] = 1.72
        velocities[0, 5, 1] = 1.0
    elif branch == "invalid_velocity":
        velocities[0, 9, 0] = float("nan")
    elif branch == "invalid_origin":
        origins[0, 0, 1] = float("inf")
    elif branch == "invalid_index":
        levels[0] = -1
    details = {}
    flag = source_tile_bounds_terminated(positions, velocities, origins, levels, torch.tensor([0]),
                                          **PARAMETERS, diagnostics=details)
    details.update(body_ids=list(range(17)), body_names=list(map(str, range(17))),
                   common_step_counter=81, episode_steps=torch.tensor([1]))
    return details, flag


@pytest.mark.parametrize("branch", [None, "current", "prediction", "invalid_velocity", "invalid_origin", "invalid_index"])
def test_recorded_predicate_reconstructs_all_branches_without_invalid_json(tmp_path, branch):
    details, flag = snapshot(branch=branch)
    path = tmp_path / "tile_bounds.jsonl"
    TileBoundsRecorder(path)(details)
    record = dict(steps=1, completed=False,
                  termination_terms=["source_terrain_tile_bounds"] if flag.item() else [])
    report = validate_tile_bounds_trace(path, record, list(map(str, range(17))))
    assert report["terminal_boundary_flag"] is (branch is not None)
    if branch in ("current", "prediction"):
        assert report["terminal_geometric_hits"][0]["body_id"] == 5
        assert report["terminal_geometric_hits"][0]["branch"] == branch
    if branch == "prediction":
        assert not details["outside_now"].item()
    if branch == "invalid_index":
        assert report["terminal_valid_indices"] is False
    if branch in ("invalid_origin", "invalid_velocity"):
        assert report["terminal_invalid_state"] is True
    json.loads(path.read_text(), parse_constant=lambda _: pytest.fail("invalid JSON constant"))


@pytest.mark.parametrize("axis,side", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_equality_is_inside_and_one_ulp_outside_is_attributed(axis, side):
    details, _ = snapshot()
    bound = details["safe_lower_xy_w" if side == 0 else "safe_upper_xy_w"][0, axis]
    positions = details["body_positions_w"]
    positions[0, 7, axis] = bound
    def evaluate():
        return source_tile_bounds_terminated(positions, torch.zeros_like(positions),
            torch.zeros((1, 1, 3), dtype=positions.dtype), torch.tensor([0]), torch.tensor([0]),
            **PARAMETERS, diagnostics={})
    assert not evaluate().item()
    positions[0, 7, axis] = torch.nextafter(bound, bound.new_tensor(float("-inf" if side == 0 else "inf")))
    assert evaluate().item()


def test_snapshot_survives_mutation_of_all_live_inputs_and_returned_flag():
    pos = torch.zeros((2, 17, 3))
    vel = torch.zeros_like(pos)
    origins = torch.zeros((2, 2, 3))
    origins[1, 1, 0] = 12
    pos[1, :, 0] = 12
    pos[1, 8, 1] = 1.72
    vel[1, 8, 1] = 1
    levels, types = torch.tensor([0, 1]), torch.tensor([0, 1])
    details = {}
    flags = source_tile_bounds_terminated(pos, vel, origins, levels, types, **PARAMETERS, diagnostics=details)
    assert flags.tolist() == [False, True]
    before = json_state(details)
    for tensor in (pos, vel, origins, levels, types, flags):
        tensor.zero_()
    assert json_state(details) == before
    assert details["terrain_levels"].tolist() == [0, 1]
    assert details["indexed_origin_xy_w"].tolist() == [[0, 0], [12, 0]]


def test_nonfinite_z_does_not_change_existing_xy_only_predicate():
    details, _ = snapshot()
    pos, vel = details["body_positions_w"], details["body_linear_velocities_w"]
    pos[0, 4, 2] = float("nan")
    vel[0, 8, 2] = float("inf")
    observed = {}
    flag = source_tile_bounds_terminated(pos, vel, details["terrain_origins_w"],
        details["terrain_levels"], details["terrain_types"], **PARAMETERS, diagnostics=observed)
    assert not flag.item() and not observed["invalid_state"].item()
    assert json_state(observed)["body_positions_w"][0][4][2] == "nan"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "stale_step", "wrong_body", "prediction", "wrong_result", "reset_origin", "parameter"])
def test_incomplete_stale_or_corrupted_trace_is_rejected(tmp_path, fault):
    details, _ = snapshot(branch="prediction")
    path = tmp_path / "trace.jsonl"
    TileBoundsRecorder(path)(details)
    row = json.loads(path.read_text())
    record = dict(steps=1, completed=False, termination_terms=["source_terrain_tile_bounds"])
    if fault == "missing":
        path.write_text("")
    elif fault == "duplicate":
        path.write_text(path.read_text() * 2)
    else:
        if fault == "stale_step": row["episode_steps"] = [0]
        elif fault == "wrong_body": row["body_ids"][5] = 7
        elif fault == "prediction": row["outside_prediction_by_axis"][0][5][1] = False
        elif fault == "wrong_result": record["termination_terms"] = []
        elif fault == "reset_origin": row["terrain_origins_w"][0][0][1] = 4
        elif fault == "parameter": row["parameters"]["collision_radius_m"] = 0.1
        path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="tile-bound telemetry"):
        validate_tile_bounds_trace(path, record, list(map(str, range(17))))


def load_wrapper(monkeypatch):
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = NS
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    path = Path(__file__).resolve().parents[2] / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py"
    spec = importlib.util.spec_from_file_location("tile_termination_test_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_wrapper_captures_before_reset_and_preserves_terminal_sample(tmp_path, monkeypatch):
    wrapper = load_wrapper(monkeypatch)
    d, _ = snapshot(branch="prediction")
    data = NS(body_pos_w=d["body_positions_w"], body_lin_vel_w=d["body_linear_velocities_w"],
              root_quat_w=torch.tensor([[1., 0., 0., 0.]]), root_lin_vel_w=torch.tensor([[0.3, 0.1, 0.]]),
              root_lin_vel_b=torch.tensor([[0.3, 0.1, 0.]]), projected_gravity_b=torch.tensor([[0., 0., -1.]]))
    class Scene(dict): pass
    scene = Scene(robot=NS(data=data, body_names=d["body_names"]))
    scene.terrain = NS(terrain_origins=d["terrain_origins_w"], terrain_levels=d["terrain_levels"], terrain_types=d["terrain_types"])
    env = NS(scene=scene, num_envs=1, step_dt=0.02, common_step_counter=81, episode_length_buf=torch.tensor([1]),
             command_manager=NS(get_command=lambda _: torch.tensor([[0.3, 0., 0.]])))
    before, observe = qualification.sample_callbacks(env, tmp_path, capture_tile_bounds=True)
    parameters = {k: v for k, v in PARAMETERS.items() if k != "control_period_s"}
    flags = wrapper.source_terrain_tile_bounds(env, NS(name="robot", body_ids=slice(None)), **parameters)
    assert flags.tolist() == [True]
    before([0], 1)
    for tensor in (data.body_pos_w, data.body_lin_vel_w, data.root_lin_vel_w,
                   scene.terrain.terrain_origins, env.episode_length_buf): tensor.zero_()
    observe(1, {"forward_progress_m": 3.9})
    row = json.loads((tmp_path / "samples.jsonl").read_text())
    assert row["kinematics_phase"] == "pre_reset"
    assert row["kinematics"]["root_lin_vel_w_mps"][1] == pytest.approx(0.1)
    result = validate_tile_bounds_trace(tmp_path / "tile_bounds.jsonl",
        dict(steps=1, completed=False, termination_terms=["source_terrain_tile_bounds"]), d["body_names"])
    assert result["terminal_geometric_hits"] == [dict(branch="prediction", body_id=5, body_name="5", axis="y")]
