from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_capability_mujoco import (observation, command, geometric_clear, termination,
    make_model, DEFAULT, LIMITS, SCENES, JITTER)


def test_observation_frame_clipping_and_previous_action():
    angle = np.pi/2
    qpos = np.r_[0., 0., .375, np.cos(angle/2), 0., 0., np.sin(angle/2), DEFAULT+.1]
    qvel = np.r_[1., 0., 0., 0., 0., 2., np.full(12, 150.)]
    previous = np.arange(12.)
    obs = observation(SimpleNamespace(qpos=qpos, qvel=qvel), np.array([.4, .1, -.5]), previous)
    assert obs.shape == (45,) and obs.dtype == np.float32
    np.testing.assert_allclose(obs[:9], [0, 0, .5, 0, 0, -1, .4, .1, -.5])
    np.testing.assert_allclose(obs[9:21], .1)
    np.testing.assert_allclose(obs[21:33], 5.)  # clip before 0.05 scaling
    np.testing.assert_array_equal(obs[33:], previous)


def test_commands_and_fixed_matrix_budget():
    assert len(SCENES)*len(JITTER)*2 == 36
    assert np.array_equal(command('flat', .5, 0), [0, 0, 0])
    np.testing.assert_allclose(command('flat', .75, 0), [.2, 0, 0])
    np.testing.assert_allclose(command('stairs_down_5cm', 1., -np.pi+.1), [.4, 0, -.05])
    np.testing.assert_allclose(command('turn_right', 2., 0), [0, 0, -.5])
    np.testing.assert_allclose(command('arc_left', 2., 0), [.3, 0, .5])
    assert not command('arc_left', 1.99, 0).any()


def test_success_requires_rear_feet_support_and_safe_posture():
    feet = np.zeros((4, 3)); feet[:, 0] = 2.8
    assert geometric_clear('stairs_up_5cm', np.array([3., 0, .65]), feet, np.zeros(3))
    feet[-1, 0] = 2.5
    assert not geometric_clear('stairs_up_5cm', np.array([3., 0, .65]), feet, np.zeros(3))
    args = dict(rpy=np.zeros(3), gravity=np.array([0, 0, -1.]), lateral=0., route=True,
        clear=True, dwell=.3, support=2, torso=False)
    assert termination(**args) == 'success'
    assert termination(**dict(args, support=1)) is None
    assert termination(**dict(args, torso=True)) is None
    assert termination(**dict(args, lateral=1.01)) == 'lateral_exit'
    assert termination(**dict(args, rpy=np.array([.6, 0, 0]))) == 'orientation_failure'


def test_compiled_terrain_heights_and_motor_limits():
    stairs = make_model('stairs'); data = mujoco.MjData(stairs); mujoco.mj_forward(stairs, data)
    ids = [mujoco.mj_name2id(stairs, mujoco.mjtObj.mjOBJ_GEOM, f'stair{i}') for i in range(1, 7)]
    np.testing.assert_allclose(data.geom_xpos[ids, 2]+stairs.geom_size[ids, 2], np.arange(1, 7)*.05)
    np.testing.assert_allclose(data.geom_xpos[ids, 0]-stairs.geom_size[ids, 0], [1.15, 1.45, 1.75, 2.05, 2.35, 2.65])
    np.testing.assert_array_equal(stairs.actuator_ctrlrange[:, 1], LIMITS)
    rough = make_model('rough'); data = mujoco.MjData(rough); mujoco.mj_forward(rough, data)
    geom = np.zeros(1, dtype=np.int32)
    for x, bounds in ((.25, (-1e-6, 1e-6)), (1.2, (.009, .061)), (3.2, (.009, .061))):
        distance = mujoco.mj_ray(rough, data, np.array([x, 0., 1.]), np.array([0., 0., -1.]),
            None, 1, 1, geom)
        assert distance >= 0 and bounds[0] <= 1-distance <= bounds[1]
