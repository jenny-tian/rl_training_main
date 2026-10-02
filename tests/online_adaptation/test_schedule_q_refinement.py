import numpy as np
import pytest

from docs.analysis.refine_sac_schedule_q import classify_events, decompose


def test_fall_reset_and_unknown_reason_are_not_counted_as_performance_rollback():
    events = [{'reason': r} for r in ('fall_reset', 'performance_degradation', 'new_reason')]
    got = classify_events(events)
    assert got == {'performance_rollbacks': [events[1]], 'fall_resets': [events[0]], 'other_events': [events[2]]}


def test_rare_terminal_dominates_error_without_changing_nonterminal_rmse():
    got = decompose([0, 0, 0], [1, 1, 1], [100, 1, 1], [1, 1, 1], [1, 0, 0])
    assert got['nonterminal_rmse'] == 1
    assert got['terminal_squared_error_fraction'] == pytest.approx(10000/10002)
    assert got['all_rmse'] == pytest.approx(np.sqrt(10002/3))
    assert got['terminal_samples'] == 1


def test_empty_and_terminal_only_strata_remain_explicit():
    assert decompose([0], [0], [0], [0], [1])['all_rmse'] is None
    got = decompose([0], [0], [0], [1], [1])
    assert got['nonterminal_rmse'] is None
    assert got['terminal_squared_error_fraction'] is None


@pytest.mark.parametrize('target,mask', [([float('nan')], [1]), ([0, 1], [1]), ([0], [.5])])
def test_invalid_rows_rejected(target, mask):
    with pytest.raises(ValueError):
        decompose([0], [0], target, mask, [0])
