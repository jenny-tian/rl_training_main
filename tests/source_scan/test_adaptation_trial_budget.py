"""Incremental accounting checks on copied history; never touch the production ledger."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/tools'))
from source_adaptation_trial import PLAN, evaluation_checks
from rl_training.source_scan.training_budget_ledger import BudgetLedger, CEILINGS


@pytest.fixture
def ledger(tmp_path):
    result = BudgetLedger(tmp_path/'ledger.json')
    result.path.write_bytes((ROOT/'docs/lite3-source-training-delivery-revision-20261104/raw/production_ledger_after.json').read_bytes())
    assert result.charged(1,result.read()) == 259584
    return result


def reserve(ledger, **changes):
    args = dict(run=1, attempt_id='cpu-correction-trial', output_dir=ledger.path.parent/'trial',
                revision='c'*40, expected_reference=ledger.reference(), expected_plan=deepcopy(PLAN))
    args.update(changes)
    return ledger.reserve_adaptation_trial(**args)


def test_increment_preserves_history_and_is_charged_once_even_after_failure(ledger):
    before = ledger.read()
    row = reserve(ledger)
    after = ledger.read()
    assert after['entries'][:-1] == before['entries']
    assert after['ceilings'] == before['ceilings'] == CEILINGS
    assert ledger.charged(1,after) == 265344
    with pytest.raises(ValueError):
        reserve(ledger)
    ledger.finish(row['attempt_id'], accepted=False, attempted=120, returned=0, stored=0, updates=0)
    assert ledger.charged(1,ledger.read()) == 265344
    with pytest.raises(ValueError, match='baseline'):
        reserve(ledger, attempt_id='new-name-does-not-allow-retry')


@pytest.mark.parametrize('fault', ['stale', 'ongoing', 'baseline', 'duplicate', 'run', 'profile', 'updates', 'float_updates', 'ceiling'])
def test_refuses_drift_without_writing(ledger, fault):
    args = {}
    if fault == 'stale':
        args['expected_reference'] = dict(ledger.reference(), sha256='f'*64)
    elif fault in ('ongoing','baseline','duplicate'):
        data = ledger.read()
        row = data['entries'][0]
        if fault == 'ongoing': row['state'] = 'reserved'
        if fault == 'baseline': row['reserved_transitions'] += 1
        if fault == 'duplicate': row['attempt_id'] = 'cpu-correction-trial'
        ledger.path.write_text(json.dumps(data))
    elif fault == 'run':
        args['run'] = 2
    else:
        plan = deepcopy(PLAN)
        key, value = dict(profile=('profile','wrong'), updates=('max_updates',3),
            float_updates=('max_updates',2.0), ceiling=('cumulative_ceiling',265345))[fault]
        plan[key] = value
        args['expected_plan'] = plan
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        reserve(ledger, **args)
    assert ledger.path.read_bytes() == before


def test_finalization_rejects_extra_update_or_transition(ledger):
    row = reserve(ledger)
    for returned, updates in ((5880,2),(5760,3)):
        with pytest.raises(ValueError):
            ledger.finish(row['attempt_id'], accepted=True, attempted=returned, returned=returned, stored=returned, updates=updates)
    ledger.finish(row['attempt_id'], accepted=True, attempted=5760, returned=5760, stored=5760, updates=2)
    assert ledger.read()['entries'][-1]['state'] == 'accepted'
    assert ledger.read()['entries'][:-1] == json.loads((ROOT/'docs/lite3-source-training-delivery-revision-20261104/raw/production_ledger_after.json').read_text())['entries']


def test_behavior_thresholds_distinguish_observed_original_from_stationary_candidate():
    baseline = json.loads((ROOT/'docs/lite3-source-reward-live-verification-20261105/deliverables/summary.json').read_text())
    cases = json.loads((ROOT/'docs/lite3-source-reward-live-verification-20261105/raw/runtime/result.json').read_text())['cases']
    thresholds = baseline['two_update_evaluation_thresholds']
    assert all(evaluation_checks(cases[0], thresholds).values())
    checks = evaluation_checks(cases[1], thresholds)
    assert checks == dict(speed=False, velocity_error=False, clipping=False, endings=True, reward=False)
