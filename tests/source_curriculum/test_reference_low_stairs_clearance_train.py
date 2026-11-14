"""固定预算、父模型与已验收源码绑定，防止启动错误实验。"""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/tools'))


@pytest.mark.parametrize('flags', [[], ['--updates', '0'], ['--updates', '499'], ['--updates', '501'],
    ['--updates', '500', '--gpu', '0'], ['--updates', '500', '--checkpoint', 'other.pt'],
    ['--updates', '500', '--resume'], ['--updates', '500', '--variant', 'riser_contact']])
def test_only_approved_budget_device_and_initialization_can_start(flags, monkeypatch, tmp_path):
    from reference_low_stairs_clearance_train import main
    output = tmp_path / 'unstarted'
    monkeypatch.setattr(sys, 'argv', ['train', *flags, '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not output.exists()


def test_reviewed_sources_are_unchanged():
    from reference_low_stairs_clearance_train import source_preflight
    assert source_preflight()['status'] == 'passed'


@pytest.mark.parametrize('updates,step,valid', [(0, 200000, True), (500, 210000, True),
    (500, 209980, False), (0, 210000, False), (501, 210020, False)])
def test_adam_steps_match_phase_budget(updates, step, valid):
    from reference_low_stairs_clearance_train import check_algorithm
    state = dict(actor_state_dict={'x': torch.zeros(1)}, critic_state_dict={'x': torch.zeros(1)},
                 optimizer_state_dict={'state': {0: {'step': torch.tensor(step)}}, 'param_groups': [{'lr': .001}]})
    if valid:
        check_algorithm(state, updates)
    else:
        with pytest.raises(ValueError):
            check_algorithm(state, updates)
