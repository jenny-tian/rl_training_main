from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_capability_mujoco_deployment import schedule, window_stats, make_model, ContactScorer


def test_partial_intervals_end_at_exact_budget():
    for seconds, expected_intervals, last_steps in ((5., 417, 8), (6., 500, 12), (20., 1667, 8)):
        intervals = schedule(seconds)
        assert len(intervals) == expected_intervals
        assert intervals[0][0] == 0 and intervals[-1][1] == seconds*1000
        assert sum(end-start for start, end in intervals) == seconds*1000
        assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
        assert all(end-start == 12 for start, end in intervals[:-1])
        assert intervals[-1][1]-intervals[-1][0] == last_steps
    with pytest.raises(ValueError):
        schedule(1.0005)


def test_windows_weight_boundary_intervals_and_require_complete_coverage():
    trace = dict(time=np.array([.012, .024, .030]), velocity=np.zeros((3, 6)), command=np.zeros((3, 3)))
    trace['velocity'][:, 0] = [1., 2., 4.]
    trace['velocity'][:, 5] = [1., 2., 4.]
    result = window_stats(trace, .010, .028)
    assert result['complete']
    assert result['vx_mean'] == pytest.approx((.002+2*.012+4*.004)/.018)
    assert result['yaw_mae'] == pytest.approx(result['vx_mean'])
    assert not window_stats(trace, .010, .032)['complete']
    assert window_stats(trace, .010, .032)['vx_mean'] is None
    ticks = np.array([end/1000 for _, end in schedule(20.)])
    full = dict(time=ticks, velocity=np.zeros((len(ticks), 6)), command=np.zeros((len(ticks), 3)))
    assert all(window_stats(full, lo, hi)['complete'] for lo, hi in ((4., 8.), (10., 14.), (16., 20.)))


def test_fixed_stair_bodies_are_static_and_robot_is_not():
    model = make_model('stairs'); scorer = ContactScorer(model)
    np.testing.assert_array_equal(model.actuator_ctrlrange, np.tile([-30., 30.], (12, 1)))
    for i in range(1, 7):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'stair{i}')
        assert model.geom_bodyid[geom] != 0  # regression: worldbody-only filter misses these
        assert scorer.static[geom]
    assert not any(scorer.static[g] for g in scorer.foot_ids.values())
