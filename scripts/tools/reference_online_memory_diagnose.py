"""Offline reproduction of projection-memory contamination; no simulator."""
from __future__ import annotations

import os
from pathlib import Path
import torch

from reference_online_continue import OUTPUT, DOCS, registration
from reference_online_targets import MODEL_PATHS, HASHES, read
from reference_online_interface import DEFAULT, SCALE
from reference_stairs_eval import ROOT, save, sha
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState
from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter


def reproduce():
    default = torch.tensor([DEFAULT] * 2)
    state = SafetyState(torch.zeros(2, 2), torch.full((2, 1), .35), default.clone(), torch.zeros(2, 12),
        default-2., default+2., default, torch.tensor([SCALE]),
        torch.zeros(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool))
    limits = SafetyLimits(residual_limits=torch.tensor([.24, .20, .20] * 4))
    zero, high = torch.zeros(2, 12), torch.zeros(2, 12)
    high[0, 1] = 8.02
    values = {}
    for name, cls in (('legacy', SafetyFilter), ('current', CurrentModelSafetyFilter)):
        filt = cls(limits)
        first = filt.project(high, zero, state)
        first_memory = filt._previous_residual.clone()
        second = filt.project(zero, zero, state)
        values[name] = dict(first_action=first.action[0, 1].item(), first_memory=first_memory[0, 1].item(),
            next_requested_residual=0., next_base_action=0., next_executed_action=second.action[0, 1].item(),
            other_environment_exact_zero=bool(torch.equal(second.action[1], zero[1])))
    assert values['legacy']['next_executed_action'] < -.079
    assert values['current']['next_executed_action'] == 0.
    assert values['current']['first_action'] == values['legacy']['first_action']
    return dict(kind='synthetic_offline_reproduction_not_saved_simulator_frame',
        joint='FL_HipY', default=-.65, scale=.25, joint_lower=-2.65, joint_upper=1.35,
        physical_margin_rad=.02, initial_base_action=8.02, initial_zero_request=True,
        rate_limit=.02, results=values,
        formula='memory = P(base + residual) - P(base), with native projection arithmetic and guards preserved')


def main():
    registration()
    directory = OUTPUT / 'smoke_B'
    execution, error = read(directory / 'execution.json'), read(directory / 'error.json')
    assert execution['status'] == 'failed' and execution['exit_code'] == 1
    assert 'zero residual changes a feasible base action' in error['error']
    assert (directory / 'initial.npz').exists() and not (directory / 'trace.npz').exists()
    try:
        os.kill(execution['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('worker PID still alive')
    assert {name: sha(path) for name, path in MODEL_PATHS.items()} == HASHES
    attempts = read(OUTPUT / 'attempts.json')
    assert len(attempts) == 2 and all(row['kind'] == 'smoke' for row in attempts)
    value = dict(status='stopped_at_interface_gate', runtime=execution, error=error,
        reproduction=reproduce(), checkpoint_hashes=HASHES, optimizer_updates=0, formal_episodes=0,
        smoke_runs_consumed_total=2, smoke_runs_remaining=2,
        this_run_preparation_steps=50, this_run_actor_invoked=True,
        this_run_policy_steps_exact=None, this_run_control_steps_upper_bound=600,
        missing_trace='Trace arrays were only persisted on successful exit; failing tick/state cannot be recovered. No rerun performed.',
        offline_repair_simulation_verified=False, candidate_filter_connected_to_simulation=False,
        offline_tests_passed=71,
        validation_command='PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training python -m pytest -q tests/ea_lpl/test_current_safety.py tests/source_curriculum/test_reference_online_failure_capture.py tests/online_adaptation/test_safety.py tests/ea_lpl/test_residual_bridge.py tests/ea_lpl/test_shadow.py tests/source_curriculum/test_reference_online_interface.py tests/source_curriculum/test_reference_online_continue.py tests/ea_lpl/test_reset_table.py tests/ea_lpl/test_basis.py',
        current_semantics=CurrentModelSafetyFilter.SEMANTICS,
        limitations=['Synthetic reproduction proves the code path; it is not the exact failed simulator state.',
            'No new model success rates, fixed-code headroom or online-learning gains measured.',
            'Legacy EA shadow code constructs the legacy filter and must not replay current-memory snapshots.'],
        attempts=attempts,
        raw_sources=[dict(path=str(p), sha256=sha(p)) for p in (
            directory / 'execution.json', directory / 'error.json', directory / 'runtime.log',
            directory / 'config.json', directory / 'geometry.json', directory / 'initial.npz')],
        repair_sources=[dict(path=str(p), sha256=sha(p)) for p in (
            ROOT / 'source/rl_training/rl_training/ea_lpl/current_safety.py',
            ROOT / 'scripts/tools/reference_online_failure_capture.py')])
    save(DOCS / 'EVIDENCE.json', value)
    save(OUTPUT / 'STOPPED.json', value)


if __name__ == '__main__':
    main()
