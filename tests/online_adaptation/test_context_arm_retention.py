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
            assert float(command[command.index('--actor-lr') + 1]) == 5e-5
            assert float(command[command.index('--final-action-std') + 1]) == 0.01
            assert float(command[command.index('--reference-anchor-coef') + 1]) == 10.0
            assert float(command[command.index('--max-reference-mse') + 1]) == 5e-4
            assert float(command[command.index('--safety-projection-penalty') + 1]) == 10.0
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
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(arm.subprocess, 'run', fake_run)
    lock_calls = []
    def fake_lock(file, flags):
        lock_calls.append(flags)
        if len(lock_calls) == 2:
            assert json.loads((output / 'arm_status.json').read_text())['status'] == 'waiting_for_gpu'
            assert (output / 'arm_plan.json').exists()
            assert not commands
            expected = arm.fcntl.LOCK_EX | (0 if selected_is_latest else arm.fcntl.LOCK_NB)
            assert flags == expected
    monkeypatch.setattr(arm.fcntl, 'flock', fake_lock)
    monkeypatch.setattr(arm.sys, 'argv', ['arm', '--base-checkpoint', str(base),
        '--initial-checkpoint', str(initial), '--flat-anchor-observations', str(tmp_path / 'anchor'),
        '--output', str(output), '--gpu', '1', '--context', '--terrain-anchor-coef', '100',
        '--max-terrain-anchor-mse', '0.0001', '--actor-lr', '0.00005', '--final-action-std', '0.01',
        *(['--wait-for-gpu'] if selected_is_latest else [])])
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
    original_calls = len(commands)
    arm.sys.argv.append('--resume-evaluation')
    arm.main()
    assert len(commands) == original_calls
    base.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='resume configuration'):
        arm.main()


def test_zero_exit_without_summary_retries_and_preserves_failed_attempt(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts.reinforcement_learning.online_adaptation import run_lora_ppo_context_arm as arm
    output = tmp_path / 'probe'
    command = ['python', '-u', 'worker.py', '--evaluation-only', '--output', str(output)]
    calls = []
    def fake_run(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            (output / 'failure.txt').write_text('remote asset unavailable')
        else:
            (output / 'summary.json').write_text('{}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(arm.subprocess, 'run', fake_run)
    arm.run_evaluation(command, output / 'runtime.log', {})
    assert len(calls) == 2
    assert (tmp_path / 'probe.incomplete-1/failure.txt').read_text() == 'remote asset unavailable'
    assert (output / 'summary.json').is_file()
    arm.run_evaluation(command, output / 'runtime.log', {}, resume=True)
    assert len(calls) == 2
    with pytest.raises(ValueError, match='arguments changed'):
        arm.run_evaluation(command + ['--seed', '9'], output / 'runtime.log', {}, resume=True)


def test_missing_evaluation_result_has_bounded_retries(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.reinforcement_learning.online_adaptation import run_lora_ppo_context_arm as arm
    output = tmp_path / 'probe'
    command = ['python', '-u', 'worker.py', '--evaluation-only', '--output', str(output)]
    calls = []
    def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(arm.subprocess, 'run', fake_run)
    with pytest.raises(RuntimeError, match='two attempts'):
        arm.run_evaluation(command, output / 'runtime.log', {})
    assert len(calls) == 2
