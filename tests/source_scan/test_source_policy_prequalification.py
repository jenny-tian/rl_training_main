"""CPU faults and acceptance tests; no simulator or GPU is launched."""

from types import SimpleNamespace as NS
import json
from copy import deepcopy

import pytest
import torch

from rl_training.source_scan.frozen_actor import FrozenSourceActor, LAYER_DIMS, load_frozen_source_actor
from scripts.tools import source_policy_prequalification as qualification


def actor_state():
    state = {"distribution.log_std_param": torch.zeros(12)}
    for index, (n_in, n_out) in enumerate(zip(LAYER_DIMS, LAYER_DIMS[1:])):
        state[f"mlp.{2 * index}.weight"] = torch.full((n_out, n_in), 0.001)
        state[f"mlp.{2 * index}.bias"] = torch.full((n_out,), -0.01)
    return state


def test_actor_is_frozen_deterministic_and_preserves_cpu_rng():
    state = actor_state()
    before_rng = torch.random.get_rng_state().clone()
    actor = FrozenSourceActor(state)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert not actor.training and not any(p.requires_grad for p in actor.parameters())
    observations = torch.full((8, 45), 0.5, requires_grad=True)
    expected = observations.detach()
    for index in range(4):
        expected = torch.nn.functional.linear(expected, state[f"mlp.{2*index}.weight"], state[f"mlp.{2*index}.bias"])
        if index < 3:
            expected = torch.nn.functional.elu(expected)
    actual = actor({"policy": observations})
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["extra_normalizer", "shape", "nonfinite"])
def test_actor_rejects_unregistered_layouts(fault):
    state = actor_state()
    if fault == "extra_normalizer":
        state["obs_normalizer.mean"] = torch.zeros(45)
    elif fault == "shape":
        state["mlp.0.weight"] = torch.zeros(512, 44)
    else:
        state["mlp.0.weight"][0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        FrozenSourceActor(state)


def test_checkpoint_hash_is_checked_before_deserialization(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"not the approved checkpoint")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("must not deserialize"))
    with pytest.raises(ValueError, match="hash"):
        load_frozen_source_actor(checkpoint)


def passing_record():
    return {
        "completed": True, "termination_reason": "completion", "termination_terms": [],
        "stable_hold_steps_required": 50, "stable_hold_steps_observed": 50,
        "termination_capture_test_injected": False, "steps": 50,
        "invalid_state_count": 0, "safety_filter_rejection_or_emergency_count": 0,
        "nonfoot_contact_count": 0, "forward_progress_m": 9.3,
        "min_base_height_m": 0.30, "max_projected_gravity_xy": 0.1,
        "max_joint_speed_rad_s": 5.0, "mean_command_velocity_x_mps": 0.3,
        "action_trace": [{"step": i, "safety_paused": False,
                          **{name: [0.0] * 12 for name in (
                              "a_policy_raw", "a_base", "a_requested", "a_executed_command")}}
                         for i in range(1, 51)],
    }


def test_complete_safe_episode_is_accepted_for_one_condition_only():
    assert qualification.episode_passed(passing_record())


@pytest.mark.parametrize("key,value", [
    ("completed", False), ("termination_reason", "timeout"),
    ("stable_hold_steps_observed", 49), ("stable_hold_steps_required", 20),
    ("termination_terms", ["source_terrain_tile_bounds"]),
    ("termination_capture_test_injected", True), ("steps", 2001),
    ("invalid_state_count", 1), ("safety_filter_rejection_or_emergency_count", 1),
    ("nonfoot_contact_count", 1), ("min_base_height_m", 0.19),
    ("min_base_height_m", float("nan")), ("forward_progress_m", 8.9),
    ("max_projected_gravity_xy", 0.66), ("max_joint_speed_rad_s", 31.0),
    ("mean_command_velocity_x_mps", 0.0), ("action_trace", []),
])
def test_partial_or_unsafe_episode_is_rejected(key, value):
    record = passing_record()
    record[key] = value
    assert not qualification.episode_passed(record)


@pytest.mark.parametrize("fault", ["missing", "paused", "nan", "wrong_step", "candidate_not_executed"])
def test_missing_evidence_and_bad_action_trace_are_rejected(fault):
    record = passing_record()
    if fault == "missing":
        del record["invalid_state_count"]
    elif fault == "paused":
        record["action_trace"][-1]["safety_paused"] = True
    elif fault == "nan":
        record["action_trace"][-1]["a_executed_command"][0] = float("nan")
    elif fault == "wrong_step":
        record["action_trace"][-1]["step"] = 49
    else:
        del record["action_trace"][-1]["a_executed_command"]
    assert not qualification.episode_passed(record)


def test_changed_or_missing_frozen_inputs_block_before_launch(tmp_path):
    item = tmp_path / "input.py"
    item.write_text("version 1")
    hashes = {str(item): qualification._sha256(item)}
    qualification.verify_hashes(hashes)
    item.write_text("version 2")
    with pytest.raises(ValueError, match="frozen input changed"):
        qualification.verify_hashes(hashes)
    with pytest.raises(ValueError, match="empty"):
        qualification.verify_hashes({})


def runtime_env(reference_data):
    terms = {
        "source_terrain_tile_bounds": NS(params={
            "asset_cfg": NS(body_ids=slice(None)), "size": (12.0, 4.0),
            "flat_length_m": 3.0, "collision_radius_m": 0.27, "numerical_margin_m": 0.001,
        }),
        "illegal_contact": NS(params={"threshold": 5.0, "consecutive_steps": 3}),
        "bad_orientation_2": NS(params={"max_xy_norm": 0.65}),
    }
    return NS(
        scene={"robot": NS(body_names=[str(i) for i in range(17)], data=reference_data())},
        termination_manager=NS(active_terms=list(terms), get_term_cfg=terms.__getitem__),
        num_envs=1, step_dt=0.02,
        cfg=NS(sim=NS(dt=0.005), scene=NS(height_scanner_base=NS(update_period=0.005))),
    )


def prepared_case():
    return {"case_id": "multi_scale_rough_L1_forward_lane", "bank_row": 0, "bank_column": 1,
            "subterrain": "multi_scale_rough", "geometry_seed": 10001, "initial_state_seed": 20030,
            "geometry_difficulty": 0.25, "mesh_fingerprint": "a" * 64,
            "isolated_mesh_matches_bank": True, "runtime_status": "not_run",
            "cell": {"cell_id": "multi_scale_rough_L1_forward_lane", "level": 1, "episodes": 1,
                     "geometry_seed_range": [10001, 10001], "initial_state_seed_range": [20030, 20030],
                     "qualification_geometry_difficulty": 0.25}}


def preparation(hashes):
    return {"status": "offline_prepared", "simulation_run": False, "cases": [prepared_case()],
            "input_sha256": hashes, "checkpoint": "registered-checkpoint", "expected_body_names": list(map(str, range(17)))}


def test_run_manifest_contains_only_one_execution_spec_and_no_preparation_state():
    prep = preparation({"source": "hash"})
    result = qualification.make_run_manifest(prep, prepared_case(), revision="revision", gpu="gpu",
                                             processes="", preparation_sha256="hash")
    assert "status" not in result and "simulation_run" not in result and "cases" not in result
    assert result["manifest_kind"] == "immutable_run_spec"
    assert result["episode_count"] == result["cell"]["episodes"] == 1
    assert result["cell"]["geometry_seed_range"] == [10001, 10001]
    assert result["link_reference_verification_required"] is True
    assert result["tile_bounds_schema"] == "lite3_tile_bounds_predicate_v2"


@pytest.mark.parametrize("fault", ["missing_id", "duplicate", "l2", "old_count", "seed_range", "wrong_bank_seed", "flat_column", "geometry", "mesh", "already_run"])
def test_case_selection_rejects_unprepared_or_ambiguous_execution(fault):
    prep = preparation({"source": "hash"})
    case = prep["cases"][0]
    case_id = case["case_id"]
    if fault == "missing_id":
        case_id = None
    elif fault == "duplicate":
        prep["cases"].append(deepcopy(case))
    elif fault == "l2":
        case["cell"]["level"] = 2
    elif fault == "old_count":
        case["cell"]["episodes"] = 5
    elif fault == "seed_range":
        case["cell"]["initial_state_seed_range"] = [20030, 20034]
    elif fault == "wrong_bank_seed":
        case["geometry_seed"] = 10030
    elif fault == "flat_column":
        case["bank_column"] = 0
    elif fault == "geometry":
        case["cell"]["qualification_geometry_difficulty"] = 0.0
    elif fault == "mesh":
        case["isolated_mesh_matches_bank"] = False
    else:
        case["runtime_status"] = "passed"
    with pytest.raises(ValueError):
        qualification.select_case(prep, case_id)


def test_sample_kinematics_keep_terminal_state_before_auto_reset(tmp_path, reference_data):
    data = reference_data()
    data.root_link_lin_vel_w[:] = data.root_com_lin_vel_w[:] = torch.tensor([[0.3, 0.1, 0.]])
    command = torch.tensor([[0.3, 0., 0.]])
    env = NS(scene={"robot": NS(data=data)}, command_manager=NS(get_command=lambda _: command))
    before, observe = qualification.sample_callbacks(env, tmp_path)
    before([0], 3)
    data.root_lin_vel_w.zero_()
    command[:, 2] = 99
    observe(3, {"forward_progress_m": 4.0})
    row = json.loads((tmp_path / "samples.jsonl").read_text())
    assert row["kinematics_phase"] == "pre_reset"
    assert row["kinematics"]["root_lin_vel_w_mps"][1] == pytest.approx(0.1)
    assert row["kinematics"]["command_b_vx_vy_wz"][2] == 0.0
    with pytest.raises(ValueError, match="command changed"):
        observe(4, {})


def test_runtime_readback_accepts_only_complete_mapping_and_frozen_thresholds(reference_data):
    env = runtime_env(reference_data)
    names = env.scene["robot"].body_names
    assert qualification.runtime_readback(env, names)["guard_body_ids"] == list(range(17))


@pytest.mark.parametrize("fault", ["mapping", "permuted_mapping", "term", "dt", "scanner", "margin", "contact", "tilt", "reference", "root_reference", "nan"])
def test_runtime_config_drift_stops_before_policy_steps(fault, reference_data):
    env = runtime_env(reference_data)
    names = env.scene["robot"].body_names
    term = env.termination_manager.get_term_cfg
    if fault == "mapping":
        term("source_terrain_tile_bounds").params["asset_cfg"].body_ids = [0]
    elif fault == "permuted_mapping":
        term("source_terrain_tile_bounds").params["asset_cfg"].body_ids = list(reversed(range(17)))
    elif fault == "term":
        env.termination_manager.active_terms.remove("illegal_contact")
    elif fault == "dt":
        env.step_dt = 0.04
    elif fault == "scanner":
        env.cfg.scene.height_scanner_base.update_period = 0.02
    elif fault == "margin":
        term("source_terrain_tile_bounds").params["collision_radius_m"] = 0.1
    elif fault == "contact":
        term("illegal_contact").params["threshold"] = 50
    elif fault == "tilt":
        term("bad_orientation_2").params["max_xy_norm"] = 0.9
    elif fault == "reference":
        env.scene["robot"].data.body_link_lin_vel_w[0, 5, 1] = 1
    elif fault == "root_reference":
        env.scene["robot"].data.root_link_pos_w = torch.ones((1, 3), dtype=torch.float64)
    else:
        env.scene["robot"].data.body_com_pos_b[0, 5, 2] = float("nan")
    with pytest.raises(ValueError):
        qualification.runtime_readback(env, names)


@pytest.mark.parametrize("fault", [None, "timeout", "missing_result", "unsafe_episode", "changed_input", "missing_trace", "corrupt_trace", "legacy_trace", "reference_mismatch", "nonfinite_reference"])
def test_supervisor_acceptance_requires_complete_unchanged_evidence(tmp_path, monkeypatch, fault, reference_data):
    source = tmp_path / "source.py"
    source.write_text("frozen input")
    prep = tmp_path / "preparation.json"
    prep.write_text(json.dumps(preparation({str(source): qualification._sha256(source)})))
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    monkeypatch.setattr(qualification, "PREPARATION", prep)
    monkeypatch.setattr(qualification, "_capture", lambda command, **k: (
        "" if "--porcelain" in command else "revision"))
    monkeypatch.setattr(qualification, "_gpu_preflight", lambda *a, **k: (
        "0, GPU-frozen-device, Test GPU, 0 MiB, 24000 MiB, 0 %", ""))
    run_dir = tmp_path / "run"

    def monitored(command, **kwargs):
        assert kwargs["startup_timeout_s"] == 120 and kwargs["total_timeout_s"] == 240
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-frozen-device"
        assert "--worker" in command
        if fault != "missing_result":
            record = passing_record()
            record.update(cell_id=prepared_case()["case_id"], geometry_seed=10001, initial_state_seed=20030, episode_index=1)
            if fault == "unsafe_episode":
                record["nonfoot_contact_count"] = 1
            qualification.write_json(run_dir / "result.json", {
                "status": "single_condition_safety_passed", "record": record,
                "case_id": prepared_case()["case_id"],
                "bank_geometry": json.loads((run_dir / "manifest.json").read_text())["bank_geometry"],
                "policy_updated": False, "ppo_initialized": False, "g3_qualified": False,
                "actor_sha256_before": "actor", "actor_sha256_after": "actor",
            })
            if fault != "missing_trace":
                from rl_training.source_scan.tile_bounds import source_tile_bounds_terminated
                from rl_training.source_scan.tile_bounds_diagnostics import TileBoundsRecorder, validate_tile_bounds_trace
                from rl_training.source_scan.rigid_body_reference import capture_body_reference_state
                capture = TileBoundsRecorder(run_dir / "tile_bounds.jsonl")
                for step in range(1, 51):
                    details = {}
                    source_tile_bounds_terminated(torch.zeros((1, 17, 3)), torch.zeros((1, 17, 3)),
                        torch.zeros((1, 1, 3)), torch.tensor([0]), torch.tensor([0]),
                        size=(12., 4.), flat_length_m=3., collision_radius_m=0.27,
                        numerical_margin_m=0.001, control_period_s=0.02, diagnostics=details)
                    details.update(body_ids=list(range(17)), body_names=list(map(str, range(17))),
                                   common_step_counter=step, episode_steps=torch.tensor([step]))
                    if fault != "legacy_trace":
                        data = reference_data(dtype=torch.float32)
                        if fault == "nonfinite_reference":
                            data.body_com_pos_b[0, 5, 2] = float("nan")
                        details.update(capture_body_reference_state(data))
                    capture(details)
                review = validate_tile_bounds_trace(run_dir / "tile_bounds.jsonl", record, list(map(str, range(17))))
                qualification.write_json(run_dir / "tile_bounds_review.json", review)
                if fault == "corrupt_trace":
                    (run_dir / "tile_bounds.jsonl").write_text("")
                elif fault == "reference_mismatch":
                    path = run_dir / "tile_bounds.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[-1]["body_com_linear_velocities_w"][0][5][1] = 3
                    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        if fault == "changed_input":
            source.write_text("changed")
        return {"exit_code": 0, "stop_reason": "total_timeout" if fault == "timeout" else None,
                "collision_cooking_failure": None, "fatal_exception": None}

    monkeypatch.setattr(qualification, "run_monitored_process", monitored)
    assert qualification.supervise(run_dir, prepared_case()["case_id"]) == (0 if fault is None else 1)
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["status"] == ("passed" if fault is None else "failed")
    assert outcome["g3_qualified"] is False
