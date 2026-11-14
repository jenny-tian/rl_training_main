"""Explicit trial budgets and exact native PPO restoration are required."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))


@pytest.mark.parametrize('args',[
    ['--mode','train'],['--mode','train','--updates','0'],
    ['--mode','train','--updates','501'],['--mode','train','--updates','-1'],
    ['--mode','smoke','--updates','1'],['--mode','restore-check','--updates','1'],
    ['--mode','train','--updates','500','--gpu','0'],
])
def test_invalid_budget_or_device_cannot_create_a_run(args,monkeypatch,tmp_path):
    from reference_low_stairs_trial import main
    output=tmp_path/'unstarted'
    monkeypatch.setattr(sys,'argv',['reference_low_stairs_trial.py',*args,'--output',str(output)])
    with pytest.raises(SystemExit) as error:main()
    assert error.value.code==2 and not output.exists()


def test_reviewed_parent_identifies_low_stairs_and_restored_adam_steps():
    from reference_low_stairs_trial import checked_state
    state,rate=checked_state()
    assert state['infos']['reward_variant']=='low_stairs'
    assert state['infos']['terrain_profile']=='low_stairs'
    assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()}=={200000}
    assert rate==pytest.approx(.0005766503906250003)
