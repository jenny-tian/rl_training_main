import copy

import pytest

from scripts.reinforcement_learning.online_adaptation.run_lora_ppo_context_arm import retention_checks


def summary():
    return dict(fell=False, safety_violation=False, reset_count=0, evaluation_only=True,
                updates=0, steps=2000, flat_retention=dict(
                    comparison_eligible=True, full_sample_budget=True, steady_sample_count=1750,
                    steady_net_forward_m=10.3, steady_velocity_x_mae=0.01,
                    steady_velocity_xy_l2_mean=0.026))


@pytest.mark.parametrize('field,value', [('fell', True), ('safety_violation', True),
                                       ('reset_count', 1), ('evaluation_only', False),
                                       ('updates', 1), ('steps', 1999)])
@pytest.mark.parametrize('side', ['baseline', 'candidate'])
def test_unsafe_or_incomplete_pair_is_never_accepted(field, value, side):
    pair = dict(baseline=summary(), candidate=summary())
    pair[side][field] = value
    assert not all(retention_checks(**pair).values())


@pytest.mark.parametrize('field,value', [('steady_net_forward_m', 8.39),
    ('steady_velocity_x_mae', 0.016), ('steady_velocity_xy_l2_mean', 0.032),
    ('steady_velocity_xy_l2_mean', float('nan')), ('steady_sample_count', 1749),
    ('full_sample_budget', False), ('comparison_eligible', False)])
def test_motion_and_error_gates_cannot_be_replaced_by_no_fall(field, value):
    baseline = summary()
    candidate = copy.deepcopy(baseline)
    candidate['flat_retention'][field] = value
    assert not all(retention_checks(baseline, candidate).values())


def test_existing_v8_measurement_passes_but_gate_is_not_training_success():
    baseline = summary()
    candidate = summary()
    candidate['flat_retention'].update(steady_net_forward_m=10.35067,
                                     steady_velocity_x_mae=0.01120771,
                                     steady_velocity_xy_l2_mean=0.02553232)
    assert all(retention_checks(baseline, candidate).values())
