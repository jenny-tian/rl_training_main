import json
import sys
from pathlib import Path

import pytest

from docs.analysis import run_episode_learning as runner
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def setup(tmp_path):
    p=json.loads(Path('docs/lite3_c0_episode_learning_protocol_20261002.json').read_text())
    base=tmp_path/'base.pt';base.write_bytes(b'base')
    prior=tmp_path/'prior.json';write_json_atomic(prior,{'terrains':{'101':{'confirmation_pass':True}}})
    root=tmp_path/'runs';root.mkdir()
    p.update(base_checkpoint=str(base),base_sha256=sha256_file(base),prior_confirmation_report=str(prior),
             prior_confirmation_sha256=sha256_file(prior),output=str(root),generations=2)
    path=tmp_path/'protocol.json';write_json_atomic(path,p)
    return path,root/'seed-137-terrain-101'


def fake_executor(calls,interrupt_at=None,fail=False):
    def execute(path,state,stage,cmd,root):
        if state['stages'].get(stage,{}).get('status')=='complete':return True
        if interrupt_at is not None and len(calls)==interrupt_at:raise RuntimeError('interrupted supervisor')
        calls.append(stage)
        directory=Path(cmd[cmd.index('--session-dir')+1]);directory.mkdir(exist_ok=True)
        phase=cmd[cmd.index('--phase')+1];seed=int(cmd[cmd.index('--seed')+1])
        options=json.loads(Path(cmd[cmd.index('--worker-options')+1]).read_text())
        summary=directory/'session_summary.json'
        data=json.loads(summary.read_text()) if summary.exists() else {'spent':{'evaluation':0,'retention':0},'uncertain_steps':0,'trials':{}}
        n=int(cmd[cmd.index('--evaluation-trials' if phase=='evaluation' else '--retention-trials')+1])
        data['spent'][phase]+=10*n
        for i in range(n):
            trial=directory/f'trial-{len(data["trials"]):05d}';trial.mkdir()
            code=options.get('fixed-code',[0.,0.])
            write_json_atomic(trial/'episode_summary.json',{'max_x':4+code[0], 'initial_x':0.,'task_stop_reason':'physical_failure'})
            (trial/'metrics.jsonl').write_text('{"safety_reasons":[],"paused":false}\n')
            data['trials'][trial.name]={'status':'complete','reset_seed':seed+i,'output':str(trial)}
        write_json_atomic(summary,data)
        state['stages'][stage]={'command':cmd,'status':'failed' if fail else 'complete'}
        write_json_atomic(path,state)
        return not fail
    return execute


def invoke(monkeypatch,protocol,resume=False):
    args=['run','--protocol',str(protocol),'--terrain','101','--seed','137','--device','cuda:0']
    if resume:args+=['--resume']
    monkeypatch.setattr(sys,'argv',args)
    runner.main()


def test_interrupted_generation_resume_never_repeats_paid_rollouts(tmp_path,monkeypatch):
    protocol,group=setup(tmp_path);calls=[]
    monkeypatch.setattr(runner,'run_stage',fake_executor(calls,interrupt_at=5))
    with pytest.raises(RuntimeError,match='interrupted'):invoke(monkeypatch,protocol)
    state=json.loads((group/'execution.json').read_text())
    assert state['search_state']['generation']==1 and state['search_state']['pending'] is not None
    monkeypatch.setattr(runner,'run_stage',fake_executor(calls))
    invoke(monkeypatch,protocol,True)
    state=json.loads((group/'execution.json').read_text())
    assert len(calls)==12 and len(set(calls))==12
    assert state['search_state']['generation']==2 and len(state['search_state']['history'])==2
    assert state['costs']=={'training':80,'evaluation':100,'retention':40,'uncertain_steps':0}
    locked=json.loads((group/'final_code.json').read_text())
    assert locked['code']==state['search_state']['mean']
    invoke(monkeypatch,protocol,True)
    assert len(calls)==12
    assert json.loads((group/'final_code.json').read_text())==locked


def test_failed_physical_stage_does_not_promote_final_code(tmp_path,monkeypatch):
    protocol,group=setup(tmp_path);calls=[]
    monkeypatch.setattr(runner,'run_stage',fake_executor(calls,fail=True))
    with pytest.raises(SystemExit):invoke(monkeypatch,protocol)
    state=json.loads((group/'execution.json').read_text())
    assert state['status']=='failed' and state['costs']['training']==10
    assert not (group/'final_code.json').exists()


def test_outer_source_hash_matches_managed_worker_contract():
    from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser
    args=build_play_parser().parse_args(['--base-checkpoint','base.pt','--mode','base','--terrain-split','development','--terrain-seed','101'])
    assert runner.runtime_hash()==runner.managed_trial.worker_configuration(args)['source_sha256']
