import json
from pathlib import Path

from docs.analysis.run_fixed_code_physical import commands


def test_registered_physical_schedule_is_frozen_bounded_and_paired():
    p=json.loads(Path('docs/lite3_c0_physical_protocol_20261002.json').read_text())
    total=0
    for terrain in p['terrains']:
        jobs=list(commands(p,terrain,'cuda:0'))
        assert len(jobs)==12
        assert len({cmd[cmd.index('--session-dir')+1] for _,_,cmd in jobs})==6
        for arm,phase,cmd in jobs:
            assert phase in ('evaluation','retention')
            assert cmd[cmd.index('--seed')+1]=='71'
            assert cmd[cmd.index('--target-warmup-steps')+1]=='0'
            assert cmd[cmd.index('--evaluation-trials')+1]=='1'
            total+=64
    assert total==p['max_actual_control_transitions']==1536
