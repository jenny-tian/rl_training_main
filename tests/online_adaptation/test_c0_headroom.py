from types import SimpleNamespace

from docs.analysis import run_c0_headroom as runner


def test_grid_ranking_includes_zero_and_excludes_incomplete():
    rows=[dict(index=i,eligible=True,completion=False,normalized_progress=.8,safety_interventions=0) for i in range(9)]
    rows[8].update(completion=True,normalized_progress=.7)
    rows[4].update(completion=True,normalized_progress=.9)
    rows[0].update(eligible=False,completion=True,normalized_progress=1.)
    assert runner.select_grid(rows)==[4,8]
    assert runner.select_grid([rows[4]])==[]
    assert runner.select_grid(rows[1:4])==[1,2]


def test_completed_and_failed_stages_cannot_be_retried(tmp_path,monkeypatch):
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=int(cmd[0]=='fail'))
    monkeypatch.setattr(runner.subprocess,'run',run)
    state={'stages':{}};path=tmp_path/'execution.json'
    for stage in ('pass','fail'):
        runner.run_stage(path,state,stage,[stage],tmp_path)
        runner.run_stage(path,state,stage,[stage],tmp_path)
    assert calls==[['pass'],['fail']]


def test_headroom_budget_counts_grid_and_all_verification_starts():
    from pathlib import Path
    import json
    p=json.loads(Path('docs/lite3_c0_headroom_protocol_20261002.json').read_text())
    per_terrain=(len(p['codes'])*p['grid_trials']+3*p['verification_trials'])*p['trial_steps']
    assert per_terrain==p['per_terrain_max_steps']==36000
    assert len(p['terrains'])*per_terrain==p['max_actual_control_transitions']==72000
    cmd=runner.command(p,101,'cuda:0','verify-rank1',0,p['verification_seed'],3)
    assert cmd[cmd.index('--evaluation-steps')+1]=='6000'
    assert cmd[cmd.index('--phase')+1]=='evaluation'
