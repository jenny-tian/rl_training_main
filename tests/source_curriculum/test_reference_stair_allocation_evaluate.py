from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/tools'))
from reference_stair_allocation_train import simulation_audit
from reference_stair_allocation_evaluate import jobs, record_regressions
from reference_stair_allocation_retention import protocol, outcome, command_speed


def test_sensor_audit_rejects_uninitialized_environment_before_reading_sensors():
    with pytest.raises(ValueError, match='completed restored initial reset'):
        simulation_audit(NS(), None, None)


def test_core_and_retention_budgets_and_terrain_coverage():
    matrix = jobs()
    assert len(matrix) == 10
    assert sum(640 if r[1] == 'turn' else 480 for r in matrix) == 5120
    specs = protocol()
    assert len(specs) == len({r['env_id'] for r in specs}) == 160
    assert {r['row'] for r in specs} == {1,8}
    assert {r['column'] for r in specs} == {0,4,8,10,16,17,18,19}
    assert 160*2*3+5120 == 6080
    for terrain in {r['terrain'] for r in specs}:
        assert sum(r['terrain'] == terrain for r in specs) == 20


def test_regression_gate_rejects_lost_success_even_when_total_is_unchanged():
    old = dict(records=[dict(env_id=0,outcome='success',duration_s=10.), dict(env_id=1,outcome='timeout',duration_s=20.)])
    new = dict(records=[dict(env_id=0,outcome='timeout',duration_s=20.), dict(env_id=1,outcome='success',duration_s=10.)])
    bad = record_regressions(old,new,'stairs')
    assert len(bad) == 1 and bad[0]['env_id'] == 0 and bad[0]['reason'] == 'lost_success'


def test_regression_gate_keeps_all_posture_failures_including_exempt_low_speed():
    old = dict(records=[dict(env_id=0,case='low_left',outcome='orientation_failure',duration_s=2.)])
    new = dict(records=[dict(env_id=0,case='low_left',outcome='orientation_failure',duration_s=.8)])
    assert record_regressions(old,new,'turn')[0]['reason'] == 'new_or_earlier_orientation_failure'
    assert not record_regressions(old,old,'turn')


@pytest.mark.parametrize('changes,expected', [
    ({},'success'), ({'terminated':True},'orientation_failure'),
    ({'lateral':1.01},'lateral_exit'), ({'forward':2.99},None),
    ({'support':1},None), ({'dwell':.24},None),
    ({'forward':2.,'expired':True},'timeout')])
def test_retention_success_requires_progress_dwell_support_and_no_failure(changes,expected):
    values = dict(terminated=False,lateral=0.,forward=3.,support=2,dwell=.26,expired=False)
    values.update(changes)
    assert outcome(**values) == expected


def test_retention_command_schedule():
    assert [command_speed(t) for t in (0.,.5,.75,1.,20.)] == [0.,0.,.2,.4,.4]
