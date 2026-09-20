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


@pytest.mark.parametrize('selected_is_latest', [False, True])
def test_arm_evaluates_latest_candidate_even_when_selected_model_is_old(tmp_path, monkeypatch, selected_is_latest):
    import json
    from pathlib import Path
    from scripts.reinforcement_learning.online_adaptation import run_lora_ppo_context_arm as arm

    initial, latest, base = [tmp_path / name for name in ('old.pt', 'new.pt', 'base.pt')]
    for path in (initial, latest, base):
        path.write_bytes(path.name.encode())
    selected = latest if selected_is_latest else initial
    output = tmp_path / 'arm'
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if '--output-dir' in command:
            assert command[command.index('--terrain-reference-checkpoint') + 1] == str(initial)
            directory = Path(command[command.index('--output-dir') + 1])
            record = dict(final_checkpoint=str(selected), latest_checkpoint=str(latest),
                          status='budget_exhausted', stable_success_episode=None,
                          final_validation={'count': 4, 'successes': 3 if selected == initial else 0})
        else:
            directory = Path(command[command.index('--output') + 1])
            assert '--evaluation-only' in command
            assert '--terrain-reference-checkpoint' not in command
            candidate = command[command.index('--checkpoint-in') + 1] if '--checkpoint-in' in command else None
            record = summary()
            if '--flat-retention-probe' in command:
                if candidate == str(latest):
                    record['flat_retention']['steady_velocity_x_mae'] = 0.2
            else:
                assert candidate == str(latest)
                record['completion_qualified'] = False
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'summary.json').write_text(json.dumps(record))

    monkeypatch.setattr(arm.subprocess, 'run', fake_run)
    monkeypatch.setattr(arm.fcntl, 'flock', lambda *args: None)
    monkeypatch.setattr(arm.sys, 'argv', ['arm', '--base-checkpoint', str(base),
        '--initial-checkpoint', str(initial), '--flat-anchor-observations', str(tmp_path / 'anchor'),
        '--output', str(output), '--gpu', '1', '--context', '--terrain-anchor-coef', '100',
        '--max-terrain-anchor-mse', '0.0001'])
    arm.main()
    status = json.loads((output / 'arm_status.json').read_text())
    assert status['status'] == 'complete'
    assert status['latest_candidate_validation']['checkpoint'] == str(latest)
    assert status['latest_candidate_validation']['successes'] == 0
    assert status['latest_candidate_flat_retention_passed'] is False
    assert status['flat_retention_passed'] is (not selected_is_latest)
    targets = status['targets']
    assert len(targets) == (1 if selected_is_latest else 2)
    if not selected_is_latest:
        assert targets[0]['is_initial_checkpoint'] is True
        assert targets[0]['roles'] == ['selected']
    flat_commands = [c for c in commands if '--flat-retention-probe' in c]
    assert len(flat_commands) == (6 if selected_is_latest else 9)
    latest_flat = [c for c in flat_commands if '--checkpoint-in' in c and
                   c[c.index('--checkpoint-in') + 1] == str(latest)]
    assert len(latest_flat) == 3
