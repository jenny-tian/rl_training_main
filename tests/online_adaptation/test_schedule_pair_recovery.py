import json
from types import SimpleNamespace

import pytest

from docs.analysis import run_sac_schedule_pair as runner
from rl_training.online_adaptation.provenance import sha256_file


@pytest.mark.parametrize('fail_reference', [False, True])
def test_resume_never_reexecutes_finished_or_failed_arm(tmp_path, monkeypatch, fail_reference):
    base = tmp_path/'base.pt'
    base.write_bytes(b'fixture')
    p = {'schema':'lite3_sac_actor_schedule_pair_v1','terrains':[101], 'output':str(tmp_path),
         'base_checkpoint':str(base),'base_sha256':sha256_file(base),
         'arms':{'reference':.005,'actor025':.25},'common_worker_options':{}}
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(p))
    for arm, rate in p['arms'].items():
        (tmp_path/f'{arm}_options.json').write_text(json.dumps({'actor-updates-per-transition':rate}))
    jobs = [(arm, phase, [arm, phase]) for arm in p['arms'] for phase in ('adaptation','evaluation','retention')]
    monkeypatch.setattr(runner, 'commands', lambda *args: iter(jobs))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if fail_reference and command==['reference','adaptation'] else 0)

    monkeypatch.setattr(runner.subprocess, 'run', run)
    argv = ['run', '--protocol', str(protocol), '--terrain','101','--device','cuda:0']
    monkeypatch.setattr(runner.sys, 'argv', argv)
    if fail_reference:
        with pytest.raises(SystemExit) as error:
            runner.main()
        assert error.value.code==1
        assert calls==[['reference','adaptation']]+[job[2] for job in jobs if job[0]=='actor025']
    else:
        runner.main()
        assert calls==[job[2] for job in jobs]
    completed = list(calls)
    monkeypatch.setattr(runner.sys, 'argv', argv+['--resume'])
    if fail_reference:
        with pytest.raises(SystemExit):
            runner.main()
    else:
        runner.main()
    assert calls==completed


def test_resume_returns_to_interrupted_phase_and_preserves_completed_phase(tmp_path, monkeypatch):
    base = tmp_path/'base.pt'
    base.write_bytes(b'fixture')
    p = {'schema':'lite3_sac_actor_schedule_pair_v1','terrains':[101], 'output':str(tmp_path),
         'base_checkpoint':str(base),'base_sha256':sha256_file(base),
         'arms':{'reference':.005},'common_worker_options':{}}
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(p))
    (tmp_path/'reference_options.json').write_text(json.dumps({'actor-updates-per-transition':.005}))
    jobs = [('reference', phase, [phase]) for phase in ('adaptation','evaluation','retention')]
    monkeypatch.setattr(runner, 'commands', lambda *args: iter(jobs))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls)==2:
            raise RuntimeError('simulated runner interruption')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', run)
    argv = ['run', '--protocol', str(protocol), '--terrain','101','--device','cuda:0']
    monkeypatch.setattr(runner.sys, 'argv', argv)
    with pytest.raises(RuntimeError, match='interruption'):
        runner.main()
    saved = json.loads((tmp_path/'execution-101.json').read_text())
    assert saved['phases']['reference:adaptation']['status']=='complete'
    assert saved['phases']['reference:evaluation']['status']=='running'
    monkeypatch.setattr(runner.sys, 'argv', argv+['--resume'])
    runner.main()
    assert calls==[['adaptation'],['evaluation'],['evaluation'],['retention']]
