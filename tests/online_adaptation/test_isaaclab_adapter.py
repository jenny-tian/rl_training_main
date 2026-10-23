from types import SimpleNamespace

import torch

import pytest

from rl_training.online_adaptation.isaaclab_adapter import (
    DEFAULT_HEIGHT_SCAN_CONFIG,
    Lite3IsaacLabAdapter,
    compose_proprio_frame,
)
from rl_training.online_adaptation.terrain_observation import (
    HeightScanConfig,
    summarize_height_scan,
)

pytestmark = pytest.mark.isaaclab


def _base_scan_adapter(surface_z=0.0, clearance=0.31):
    env = _height_scan_env(include_scanner=False)
    env.scene["robot"].data.root_pos_w[:, 2] = surface_z + clearance
    xy = torch.cartesian_prod(torch.tensor([-0.05, 0.0, 0.05]), torch.tensor([-0.05, 0.0, 0.05]))
    hits = torch.cat((xy, torch.full((9, 1), surface_z)), dim=1).unsqueeze(0)
    env.scene.sensors["height_scanner_base"] = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))
    return Lite3IsaacLabAdapter(env, safety_height_source="base_scan")


@pytest.mark.parametrize("surface_z", [0.0, -0.115, 0.6])
def test_safety_clearance_is_invariant_to_terrain_translation(surface_z):
    adapter = _base_scan_adapter(surface_z)
    state = adapter.safety_state(torch.zeros(1, 1, dtype=torch.bool))
    assert state.base_height.item() == pytest.approx(0.31)


def test_flat_scan_preserves_legacy_filter_decision():
    from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
    adapter = _base_scan_adapter()
    scan_state = adapter.safety_state(torch.zeros(1, 1, dtype=torch.bool))
    adapter.safety_height_source = "world_z"
    legacy_state = adapter.safety_state(torch.zeros(1, 1, dtype=torch.bool))
    decisions = [SafetyFilter(SafetyLimits(residual_limits=torch.full((1, 12), 0.2))).project(
        torch.zeros(1, 12), torch.zeros(1, 12), state) for state in (scan_state, legacy_state)]
    assert torch.equal(scan_state.base_height, legacy_state.base_height)
    assert torch.equal(decisions[0].action, decisions[1].action)
    assert torch.equal(decisions[0].paused, decisions[1].paused)


@pytest.mark.parametrize("failure", ["missing", "nan", "inf", "shape", "stale_xy", "low_clearance"])
def test_scan_height_failures_and_elevated_low_clearance_pause_filter(failure):
    from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
    adapter = _base_scan_adapter(0.6, 0.1 if failure == "low_clearance" else 0.31)
    sensors = adapter.env.scene.sensors
    if failure == "missing":
        del sensors["height_scanner_base"]
    elif failure in {"nan", "inf"}:
        sensors["height_scanner_base"].data.ray_hits_w[0, 0, 2] = float(failure)
    elif failure == "shape":
        sensors["height_scanner_base"].data.ray_hits_w = torch.zeros(1, 8, 3)
    elif failure == "stale_xy":
        sensors["height_scanner_base"].data.ray_hits_w[..., 0] += 1.0
    state = adapter.safety_state(torch.zeros(1, 1, dtype=torch.bool))
    decision = SafetyFilter(SafetyLimits(residual_limits=torch.full((1, 12), 0.2))).project(
        torch.zeros(1, 12), torch.zeros(1, 12), state)
    assert torch.isfinite(state.base_height).all()
    assert decision.paused.item()
    assert "height" in decision.reasons


def test_step_edge_uses_highest_surface_in_base_patch():
    adapter = _base_scan_adapter()
    adapter.env.scene.sensors["height_scanner_base"].data.ray_hits_w[0, 0, 2] = 0.05
    assert adapter.safety_state(torch.zeros(1, 1, dtype=torch.bool)).base_height.item() == pytest.approx(0.26)


def test_compose_proprio_frame_has_fixed_order_and_49_dimensions():
    terms = [
        torch.full((2, 3), 1.0),
        torch.full((2, 3), 2.0),
        torch.full((2, 12), 3.0),
        torch.full((2, 12), 4.0),
        torch.full((2, 4), 5.0),
        torch.full((2, 12), 6.0),
        torch.full((2, 3), 7.0),
    ]
    frame = compose_proprio_frame(*terms)
    assert frame.shape == (2, 49)
    assert torch.equal(frame[:, :3], terms[0])
    assert torch.equal(frame[:, -3:], terms[-1])


def test_reward_signals_use_articulation_foot_indices_for_foot_speed():
    sensor_body_names = [
        "TORSO",
        "FL_HIP",
        "FL_THIGH",
        "FL_SHANK",
        "FL_FOOT",
        "FR_HIP",
        "FR_THIGH",
        "FR_SHANK",
        "FR_FOOT",
        "HL_HIP",
        "HL_THIGH",
        "HL_SHANK",
        "HL_FOOT",
        "HR_HIP",
        "HR_THIGH",
        "HR_SHANK",
        "HR_FOOT",
    ]
    robot_body_names = [
        "TORSO",
        "FL_HIP",
        "FR_HIP",
        "HL_HIP",
        "HR_HIP",
        "FL_THIGH",
        "FR_THIGH",
        "HL_THIGH",
        "HR_THIGH",
        "FL_SHANK",
        "FR_SHANK",
        "HL_SHANK",
        "HR_SHANK",
        "FL_FOOT",
        "FR_FOOT",
        "HL_FOOT",
        "HR_FOOT",
    ]
    body_lin_vel_w = torch.zeros(1, len(robot_body_names), 3)
    for index, speed in zip((13, 14, 15, 16), (1.0, 2.0, 3.0, 4.0), strict=True):
        body_lin_vel_w[0, index, 0] = speed
    for index, speed in zip((4, 8, 12), (10.0, 20.0, 30.0), strict=True):
        body_lin_vel_w[0, index, 0] = speed
    sensor_forces = torch.zeros(1, len(sensor_body_names), 3)
    sensor_forces[0, [4, 8, 12, 16], 2] = 2.0
    robot = SimpleNamespace(
        body_names=robot_body_names,
        data=SimpleNamespace(
            joint_pos=torch.zeros(1, 12),
            default_joint_pos=torch.zeros(1, 12),
            soft_joint_pos_limits=torch.stack(
                (torch.full((1, 12), -1.0), torch.full((1, 12), 1.0)), dim=-1
            ),
            body_lin_vel_w=body_lin_vel_w,
            body_pos_w=torch.zeros(1, len(robot_body_names), 3),
            root_lin_vel_b=torch.zeros(1, 3),
            root_ang_vel_b=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            root_pos_w=torch.zeros(1, 3),
            joint_vel=torch.zeros(1, 12),
            applied_torque=torch.zeros(1, 12),
        ),
    )
    robot.find_joints = lambda names, preserve_order: (list(range(12)), names)
    robot.find_bodies = lambda pattern: (
        ([13, 14, 15, 16], ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"])
        if pattern == ".*_FOOT"
        else (list(range(13)), robot_body_names[:13])
    )
    sensor = SimpleNamespace(
        body_names=sensor_body_names,
        data=SimpleNamespace(net_forces_w=sensor_forces),
        find_bodies=lambda pattern: (
            ([4, 8, 12, 16], ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"])
            if pattern == ".*_FOOT"
            else (list(range(13)), sensor_body_names[:13])
        ),
    )
    class Scene:
        sensors = {"contact_forces": sensor}

        def __getitem__(self, name):
            assert name == "robot"
            return robot

    env = SimpleNamespace(
        scene=Scene(),
        command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(1, 3)),
        cfg=SimpleNamespace(track_boundary_x=-7.0),
    )

    adapter = Lite3IsaacLabAdapter(env)

    signals = adapter.reward_signals()

    assert signals.foot_slip.item() == pytest.approx(2.5)
    assert signals.foot_height.shape == (1, 4)
    assert signals.foot_contact.tolist() == [[1.0, 1.0, 1.0, 1.0]]


def test_contact_diagnostics_preserve_peak_force_and_body_name():
    env = _height_scan_env(include_scanner=False)
    callbacks = []
    env.add_pre_reset_callback = callbacks.append
    sensor = env.scene.sensors["contact_forces"]
    sensor.data.net_forces_w[0, 2, 0] = 3.0
    sensor.data.net_forces_w[0, 6, 1] = 8.0

    adapter = Lite3IsaacLabAdapter(env)
    diagnostics = adapter.observe_nonfoot_contact()
    callbacks[0]([0])
    sensor.data.net_forces_w.zero_()

    assert diagnostics == {
        "max_nonfoot_contact_force_n": 8.0,
        "max_nonfoot_contact_body": "FR_THIGH",
    }
    assert adapter.contact_diagnostics() == {
        "episode_max_nonfoot_contact_force_n": 8.0,
        "episode_max_nonfoot_contact_body": "FR_THIGH",
        "terminal_nonfoot_contact_force_n": 8.0,
        "terminal_nonfoot_contact_body": "FR_THIGH",
    }


def _height_scan_env(*, include_scanner=True, ray_hits_w=None):
    robot_body_names = [
        "TORSO",
        "FL_HIP",
        "FR_HIP",
        "HL_HIP",
        "HR_HIP",
        "FL_THIGH",
        "FR_THIGH",
        "HL_THIGH",
        "HR_THIGH",
        "FL_SHANK",
        "FR_SHANK",
        "HL_SHANK",
        "HR_SHANK",
        "FL_FOOT",
        "FR_FOOT",
        "HL_FOOT",
        "HR_FOOT",
    ]
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros(1, 12),
            default_joint_pos=torch.zeros(1, 12),
            soft_joint_pos_limits=torch.stack(
                (torch.full((1, 12), -1.0), torch.full((1, 12), 1.0)), dim=-1
            ),
            body_lin_vel_w=torch.zeros(1, len(robot_body_names), 3),
            body_pos_w=torch.zeros(1, len(robot_body_names), 3),
            root_lin_vel_b=torch.zeros(1, 3),
            root_ang_vel_b=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            root_pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
            joint_vel=torch.zeros(1, 12),
            applied_torque=torch.zeros(1, 12),
        ),
    )
    robot.find_joints = lambda names, preserve_order: (list(range(12)), names)
    robot.find_bodies = lambda pattern: (
        ([13, 14, 15, 16], ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"])
        if pattern == ".*_FOOT"
        else (list(range(13)), robot_body_names[:13])
    )
    contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=torch.zeros(1, len(robot_body_names), 3)),
        find_bodies=lambda pattern: (
            ([13, 14, 15, 16], ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"])
            if pattern == ".*_FOOT"
            else (list(range(13)), robot_body_names[:13])
        ),
    )
    sensors = {"contact_forces": contact_sensor}
    if include_scanner:
        if ray_hits_w is None:
            ray_hits_w = torch.ones(1, 4, 3)
            ray_hits_w[..., 2] = torch.tensor([[[1.0, 1.1], [1.2, 1.3]]]).reshape(-1)
        sensors["height_scanner"] = SimpleNamespace(
            data=SimpleNamespace(
                ray_hits_w=ray_hits_w,
                pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
            )
        )

    class Scene:
        def __init__(self):
            self.sensors = sensors

        def __getitem__(self, name):
            assert name == "robot"
            return robot

    return SimpleNamespace(
        scene=Scene(),
        command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(1, 3)),
        cfg=SimpleNamespace(track_boundary_x=-7.0),
    )


def test_height_scan_source_appends_summary_and_reports_validity():
    relative = torch.tensor([[[0.0, 0.1], [0.2, 0.3]]])
    ray_hits = torch.zeros(1, 4, 3)
    ray_hits[..., 2] = (relative + 1.0).reshape(1, 4)
    adapter = Lite3IsaacLabAdapter(
        _height_scan_env(ray_hits_w=ray_hits),
        terrain_feature_source="height_scan",
        height_scan_config=HeightScanConfig(rows=2, cols=2),
    )

    frame = adapter.frame(torch.zeros(1, 12))

    assert frame.shape == (1, 59)
    expected, _ = summarize_height_scan(
        relative,
        HeightScanConfig(rows=2, cols=2),
    )
    torch.testing.assert_close(frame[:, 49:], expected)
    assert adapter.terrain_valid.tolist() == [[True]]
    assert adapter.terrain_metadata()["terrain_feature_source"] == "height_scan"


def test_default_height_scan_grid_matches_isaac_xy_sensor_layout():
    assert (DEFAULT_HEIGHT_SCAN_CONFIG.rows, DEFAULT_HEIGHT_SCAN_CONFIG.cols) == (15, 23)


def test_height_scan_missing_sensor_fails_closed():
    adapter = Lite3IsaacLabAdapter(
        _height_scan_env(include_scanner=False),
        terrain_feature_source="height_scan",
        height_scan_config=HeightScanConfig(rows=2, cols=2),
    )

    frame = adapter.frame(torch.zeros(1, 12))

    assert frame.shape == (1, 59)
    assert torch.equal(frame[:, 49:], torch.zeros(1, 10))
    assert adapter.terrain_valid.tolist() == [[False]]


def test_height_scan_low_valid_fraction_fails_closed():
    relative = torch.tensor([[[float("nan"), 0.1], [0.2, 0.3]]])
    ray_hits = torch.zeros(1, 4, 3)
    ray_hits[..., 2] = (relative + 1.0).reshape(1, 4)
    adapter = Lite3IsaacLabAdapter(
        _height_scan_env(ray_hits_w=ray_hits),
        terrain_feature_source="height_scan",
        height_scan_config=HeightScanConfig(rows=2, cols=2, min_valid_fraction=1.0),
    )

    frame = adapter.frame(torch.zeros(1, 12))

    assert torch.equal(frame[:, 49:], torch.zeros(1, 10))
    assert adapter.terrain_valid.tolist() == [[False]]


def test_geometry_source_preserves_legacy_frame_without_scanner():
    adapter = Lite3IsaacLabAdapter(_height_scan_env(include_scanner=False))

    frame = adapter.frame(torch.zeros(1, 12))

    assert frame.shape == (1, 49)
    assert adapter.terrain_valid.tolist() == [[True]]
    assert adapter.terrain_metadata()["terrain_feature_source"] == "geometry"


def test_adapter_rejects_unknown_terrain_feature_source():
    with pytest.raises(ValueError, match="terrain_feature_source"):
        Lite3IsaacLabAdapter(
            _height_scan_env(),
            terrain_feature_source="unknown",
        )


def test_proprio_feature_source_keeps_49_dimensional_frame_on_step_geometry():
    env = _height_scan_env(include_scanner=False)
    env.cfg.track_boundary_x = -7.0
    adapter = Lite3IsaacLabAdapter(env, step_height=0.02, terrain_feature_source="proprio")
    frame = adapter.frame(torch.zeros(1, 12))
    assert frame.shape == (1, 49)
    assert adapter.terrain_metadata()["terrain_feature_source"] == "proprio"
