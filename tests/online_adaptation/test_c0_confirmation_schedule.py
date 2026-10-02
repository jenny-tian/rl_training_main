import json
from pathlib import Path
from docs.analysis.run_c0_confirmation import commands


def test_fixed_confirmation_uses_new_seed_and_paired_task_and_flat_windows():
    p=json.loads(Path('docs/lite3_c0_confirmation_protocol_20261002.json').read_text())
    cost=0
    for terrain in p['terrains']:
        jobs=list(commands(p,terrain,'cuda:0'))
        assert len(jobs)==4
        assert jobs[0][0]==('candidate' if terrain==101 else 'base')
        for role,phase,cmd in jobs:
            assert cmd[cmd.index('--seed')+1]=='97'
            assert cmd[cmd.index('--evaluation-trials')+1]=='3'
            assert cmd[cmd.index('--retention-trials')+1]=='2'
            assert f'{role}-{terrain}_options.json' in cmd[cmd.index('--worker-options')+1]
            cost+=6000 if phase=='evaluation' else 2000
    assert cost==p['max_actual_control_transitions']==32000
