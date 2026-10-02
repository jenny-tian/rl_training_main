import json
from pathlib import Path

from docs.analysis.run_learning_replication import commands


def test_paired_replication_schedule_has_fixed_seeds_and_no_base_learning():
    p = json.loads(Path('docs/lite3_learning_replication_protocol_20261002.json').read_text())
    for terrain in p['terrains']:
        entries = list(commands(p, terrain, 'cuda:0'))
        assert len(entries) == 10
        assert entries[0][0] == ('candidate-61' if terrain == 101 else 'base-61')
        for seed in p['learner_seeds']:
            selected = [(arm, phase, cmd) for arm, phase, cmd in entries if arm.endswith(str(seed))]
            for arm, phase, cmd in selected:
                assert cmd[cmd.index('--seed')+1] == str(seed)
                assert cmd[cmd.index('--evaluation-trials')+1] == '5'
                if arm.startswith('base'):
                    assert phase != 'adaptation'
                    assert cmd[cmd.index('--target-warmup-steps')+1] == '0'
    # Four independent learners plus four matched base evaluations/retentions.
    assert 4*(p['adaptation_steps']+2*(p['evaluation_steps']+p['retention_trials']*p['retention_steps'])) == p['max_actual_control_transitions']
