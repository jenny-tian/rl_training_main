import json
from pathlib import Path
from docs.analysis.run_full_policy_evaluation import commands


def test_full_policy_final_checkpoint_and_original_base_use_paired_starts():
    p=json.loads(Path('docs/lite3_full_policy_historyfix_protocol_20261002.json').read_text())
    p['_candidate_checkpoint']='locked-final.pt'
    total=0
    for terrain in p['evaluation_terrains']:
        jobs=list(commands(p,terrain,'cuda:0'));assert len(jobs)==4
        assert jobs[0][0]==('candidate' if terrain==101 else 'base')
        for role,phase,cmd in jobs:
            assert cmd[cmd.index('--base-checkpoint')+1]==('locked-final.pt' if role=='candidate' else p['base_checkpoint'])
            assert cmd[cmd.index('--seed')+1]=='229'
            assert cmd[cmd.index('--target-warmup-steps')+1]=='0'
            total+=10000 if phase=='evaluation' else 2000
    assert total==p['max_evaluation_control_transitions']==48000
