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
from rl_training.source_scan.rigid_body_reference import (
    BODY_APIS, capture_body_reference_state, capture_root_reference_state, validate_body_reference_state,
)
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


def test_real_wrapper_captures_before_reset_and_preserves_terminal_sample(tmp_path, monkeypatch, reference_data):
    wrapper = load_wrapper(monkeypatch)
    d, _ = snapshot(branch="prediction")
    data = reference_data(d["body_positions_w"], d["body_linear_velocities_w"])
    data.root_link_lin_vel_w[:] = data.root_com_lin_vel_w[:] = torch.tensor([[0.3, 0.1, 0.]])
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
        dict(steps=1, completed=False, termination_terms=["source_terrain_tile_bounds"]), d["body_names"], require_link_reference=True)
    assert result["terminal_geometric_hits"] == [dict(branch="prediction", body_id=5, body_name="5", axis="y")]
    assert result["reference_conversion"]["status"] == "verified"
    assert row["kinematics"]["root_reference_state"]["body_linear_velocities_w"][0][0][1] == pytest.approx(0.1)


def wrapper_env(data):
    class Scene(dict): pass
    count = data.body_link_pos_w.shape[0]
    scene = Scene(robot=NS(data=data, body_names=list(map(str, range(17)))))
    origins = torch.zeros((count, 2, 3), dtype=data.body_link_pos_w.dtype)
    origins[:, :, 0] = torch.arange(count)[:, None] * 12
    scene.terrain = NS(terrain_origins=origins, terrain_levels=torch.arange(count), terrain_types=torch.zeros(count, dtype=torch.long))
    return NS(scene=scene, step_dt=0.02, common_step_counter=81, episode_length_buf=torch.ones(count, dtype=torch.long))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["zero_offset", "translation", "remove_hit", "add_hit", "rotated_offset"])
def test_wrapper_prediction_uses_link_velocity_with_analytic_rotation_oracle(monkeypatch, reference_data, dtype, mode):
    data = reference_data(dtype=dtype)
    data.body_link_pos_w[0, 5, 1] = 1.726
    offset = 0.0 if mode == "zero_offset" else 0.01
    data.body_com_pos_b[0, 5, 0] = offset
    data.body_com_lin_vel_w[0, 5, 1] = 0.3
    omega = {"zero_offset": 20, "translation": 0, "remove_hit": 20, "add_hit": -20, "rotated_offset": 20}[mode]
    if mode == "add_hit":
        data.body_com_lin_vel_w[0, 5, 1] = 0.0
    world_offset = torch.tensor([offset, 0, 0], dtype=dtype)
    if mode == "rotated_offset":
        # A 180-degree Z rotation maps +X COM offset to -X. The resulting
        # link Y velocity is +0.5 m/s, derived directly from planar motion.
        data.body_link_quat_w[0, 5] = torch.tensor([0, 0, 0, 1], dtype=dtype)
        world_offset[0] *= -1
    data.body_link_ang_vel_w[0, 5, 2] = omega
    data.body_com_pos_w[:] = data.body_link_pos_w
    data.body_com_pos_w[0, 5] += world_offset
    expected_vy = float(data.body_com_lin_vel_w[0, 5, 1]) - omega * float(world_offset[0])
    data.body_link_lin_vel_w[:] = data.body_com_lin_vel_w
    data.body_link_lin_vel_w[0, 5, 1] = expected_vy
    env = wrapper_env(data)
    observed = []
    wrapper = load_wrapper(monkeypatch)
    params = {k: v for k, v in PARAMETERS.items() if k != "control_period_s"}
    plain = wrapper.source_terrain_tile_bounds(env, NS(name="robot", body_ids=slice(None)), **params)
    env._source_tile_bounds_observer = observed.append
    actual = wrapper.source_terrain_tile_bounds(env, NS(name="robot", body_ids=slice(None)), **params)
    legacy = source_tile_bounds_terminated(data.body_pos_w, data.body_lin_vel_w,
        env.scene.terrain.terrain_origins, torch.tensor([0]), torch.tensor([0]), **PARAMETERS)
    assert actual.tolist() == plain.tolist() == [1.726 + expected_vy * 0.02 > 1.729]
    assert validate_body_reference_state(observed[0])["status"] == "verified"
    if mode in ("zero_offset", "translation"):
        assert actual.tolist() == legacy.tolist()
    elif mode == "remove_hit":
        assert legacy.item() and not actual.item()
    elif mode == "add_hit":
        assert not legacy.item() and actual.item()


@pytest.mark.parametrize("api", ["body_link_pos_w", "body_link_lin_vel_w"])
def test_legacy_aliases_cannot_substitute_for_explicit_link_apis(monkeypatch, reference_data, api):
    data = reference_data()
    env = wrapper_env(data)
    delattr(data, api)
    with pytest.raises(RuntimeError, match="explicit"):
        load_wrapper(monkeypatch).source_terrain_tile_bounds(env, NS(name="robot", body_ids=slice(None)),
            **{k: v for k, v in PARAMETERS.items() if k != "control_period_s"})


@pytest.mark.parametrize("api", list(BODY_APIS.values()))
def test_missing_reference_api_is_rejected(reference_data, api):
    data = reference_data()
    delattr(data, api)
    with pytest.raises(RuntimeError, match="explicit API"):
        capture_body_reference_state(data)


@pytest.mark.parametrize("fault", ["legacy", "missing", "metadata", "shape", "quaternion", "offset", "angular_velocity", "stale_velocity", "com_position"])
def test_new_trace_rejects_corrupted_reference_evidence(tmp_path, reference_data, fault):
    details, flag = snapshot(branch="prediction")
    data = reference_data(details["body_positions_w"], details["body_linear_velocities_w"])
    if fault != "legacy":
        details.update(capture_body_reference_state(data))
    path = tmp_path / "trace.jsonl"
    TileBoundsRecorder(path)(details)
    row = json.loads(path.read_text())
    if fault == "missing": del row["body_com_offsets_link_m"]
    elif fault == "metadata": row["reference_points"]["linear_velocity"] = "com"
    elif fault == "shape": row["body_com_offsets_link_m"][0].pop()
    elif fault == "quaternion": row["body_link_quats_w_wxyz"][0][5][0] = 2
    elif fault == "offset": row["body_com_offsets_link_m"][0][5][0] = 0.1
    elif fault == "angular_velocity":
        row["body_com_offsets_link_m"][0][5][0] = 0.1
        row["body_com_positions_w"][0][5][0] += 0.1
        row["body_angular_velocities_w"][0][5][2] = 20
    elif fault == "stale_velocity": row["body_com_linear_velocities_w"][0][5][1] = 0
    elif fault == "com_position": row["body_com_positions_w"][0][5][1] = 0
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="telemetry"):
        validate_tile_bounds_trace(path, dict(steps=1, completed=False, termination_terms=["source_terrain_tile_bounds"]),
                                  details["body_names"], require_link_reference=True)


def test_nonfinite_reference_is_preserved_but_not_claimed_verified(tmp_path, reference_data):
    details, _ = snapshot(branch="invalid_velocity")
    details.update(capture_body_reference_state(reference_data(details["body_positions_w"], details["body_linear_velocities_w"])))
    path = tmp_path / "trace.jsonl"
    TileBoundsRecorder(path)(details)
    result = validate_tile_bounds_trace(path, dict(steps=1, completed=False, termination_terms=["source_terrain_tile_bounds"]),
                                       details["body_names"], require_link_reference=True)
    assert result["terminal_invalid_state"] is True
    assert result["reference_conversion"]["status"] == "nonfinite_reference_unverifiable"
    assert result["reference_conversion"]["bodies_verified"] == 16


def test_all_reference_snapshots_survive_partial_reset_with_body_id_mapping(monkeypatch, reference_data):
    data = reference_data(env_count=2)
    data.body_link_pos_w[1, :, 0] = data.body_com_pos_w[1, :, 0] = 12
    data.body_link_pos_w[1, 5, 1] = data.body_com_pos_w[1, 5, 1] = 1.726
    data.body_link_lin_vel_w[1, 5, 1] = data.body_com_lin_vel_w[1, 5, 1] = 0.3
    env = wrapper_env(data)
    snapshots = []
    env._source_tile_bounds_observer = snapshots.append
    ids = [5, 0, 16]
    result = load_wrapper(monkeypatch).source_terrain_tile_bounds(env, NS(name="robot", body_ids=ids),
        **{k: v for k, v in PARAMETERS.items() if k != "control_period_s"})
    assert result.tolist() == [False, True]
    captured = json_state(snapshots[0])
    assert captured["body_names"] == ["5", "0", "16"] and captured["body_ids"] == ids
    assert validate_body_reference_state(snapshots[0])["bodies_verified"] == 6
    for api in BODY_APIS.values(): getattr(data, api)[1].zero_()
    env.scene.terrain.terrain_levels[1] = 0
    env.episode_length_buf[1] = 0
    assert captured == json_state(snapshots[0])


def test_explicit_root_state_uses_rotated_offset_and_correct_velocity(reference_data):
    data = reference_data()
    # 90 degrees about Z: r_link=(0.02,0,0) -> r_world=(0,0.02,0).
    data.root_link_quat_w[0] = torch.tensor([2**-0.5, 0, 0, 2**-0.5], dtype=torch.float64)
    data.body_com_pos_b[0, 0, 0] = 0.02
    data.root_com_pos_w[0, 1] = 0.02
    data.root_link_ang_vel_w[0, 2] = 10
    data.root_com_lin_vel_w[0, 0] = 0.1
    data.root_link_lin_vel_w[0, 0] = 0.3
    state = capture_root_reference_state(data)
    assert validate_body_reference_state(state)["status"] == "verified"
    env = NS(scene={"robot": NS(data=data)}, command_manager=NS(get_command=lambda _: torch.tensor([[0.3, 0, 0]])))
    kinematics = qualification.capture_kinematics(env)
    assert kinematics["root_lin_vel_w_mps"][0] == 0.1
    assert kinematics["root_reference_state"]["body_linear_velocities_w"][0][0][0] == 0.3
    data.root_link_lin_vel_w.zero_()
    assert state["body_linear_velocities_w"][0, 0, 0] == 0.3
