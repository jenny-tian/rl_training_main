"""Adversarial checks for scoring and policy adapter in the paired stair study."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/tools'))
from reference_stairs_eval import DEFAULT, cleared, observation, posture_bad


def test_front_feet_on_landing_is_not_completion():
    feet = np.array([[3.1, 0., .6], [3.1, 0., .6], [2.6, 0., .5], [2.6, 0., .5]])
    assert not cleared('up', feet, np.array([3., 0., .95]), 4)
    feet[:, 0] = 3.
    assert cleared('up', feet, np.array([3., 0., .95]), 4)
    assert not cleared('up', feet, np.array([3., 0., .95]), 0)
    assert not cleared('up', feet, np.array([3., 0., .65]), 4)


def test_downstairs_requires_rear_feet_off_last_step():
    feet = np.array([[.7, 0., 0.], [.7, 0., 0.], [1.2, 0., .1], [1.2, 0., .1]])
    assert not cleared('down', feet, np.array([.8, 0., .35]), 4)
    feet[:, 0] = .8
    assert cleared('down', feet, np.array([.8, 0., .35]), 4)


def test_observation_default_offset_previous_action_and_body_gravity():
    data = SimpleNamespace(qpos=np.r_[np.zeros(3), [1.,0.,0.,0.], DEFAULT], qvel=np.zeros(18))
    obs = observation(data, [.4,0.,0.], np.arange(12))
    assert obs.shape == (45,)
    np.testing.assert_allclose(obs[3:6], [0.,0.,-1.])
    np.testing.assert_allclose(obs[6:9], [.4,0.,0.])
    np.testing.assert_allclose(obs[9:33], 0.)
    np.testing.assert_allclose(obs[33:], np.arange(12))
    data.qpos[3:7] = [math.cos(math.pi/8),0.,math.sin(math.pi/8),0.]
    tilted = observation(data, [0.,0.,0.], np.zeros(12))
    np.testing.assert_allclose(tilted[3:6], [math.sqrt(.5),0.,-math.sqrt(.5)], atol=1e-12)
    assert posture_bad(tilted[3:6])
    assert not posture_bad(np.array([0.,0.,-1.]))
