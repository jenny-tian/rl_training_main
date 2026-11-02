import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/tools'))
from source_live_ppo_supervisor import CONTRACT, validate_plan, preflight, terminate_group
from source_live_ppo_validation import validate
from rl_training.source_scan.training_budget_ledger import BudgetLedger
from rl_training.source_scan.training_runner import validate_training_authorization


def test_budget_path_is_shared_git_metadata_and_does_not_dirty_other_worktrees():
    from rl_training.source_scan.training_budget_ledger import canonical_budget_path
    common=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],
                                       cwd=ROOT,text=True).strip())
    expected=common/'source-ppo-ledgers/lite3-l1-v1.json'
    assert canonical_budget_path(ROOT)==expected
    assert canonical_budget_path(common.parent)==expected


def reserve(ledger, **overrides):
    args = dict(run=1, attempt_id='test', transitions=384, purpose='bounded_live_ppo_integration',
                output_dir=ledger.path.parent/'output', revision='a'*40, expected_reference=ledger.reference())
    args.update(overrides)
    return ledger.reserve(**args)


def test_budget_reserves_once_and_keeps_full_charge_after_partial_failure(tmp_path):
    ledger=BudgetLedger(tmp_path/'ledger.json')
    assert not ledger.path.exists()
    reserve(ledger)
    ledger.finish('test', accepted=False, attempted=16, returned=8, stored=8, updates=0)
    assert ledger.read()['entries'][0]['reserved_transitions']==384
    assert ledger.remaining(1)['numerics']['remaining_120x24_updates']==90
    assert ledger.remaining(1)['midpoint']['remaining_120x24_updates']==363
    assert ledger.remaining(1)['pilot']['remaining_120x24_updates']==728
    assert ledger.remaining(2)['numerics']['remaining_120x24_updates']==91
    with pytest.raises(ValueError,match='already has reserved'): reserve(ledger,attempt_id='other-directory')


def test_stale_snapshot_or_corrupt_ledger_cannot_silently_reset_budget(tmp_path):
    ledger=BudgetLedger(tmp_path/'ledger.json')
    initial=ledger.reference(); reserve(ledger)
    with pytest.raises(ValueError,match='differs'): reserve(ledger,run=2,expected_reference=initial)
    ledger.path.write_text('{}')
    with pytest.raises(ValueError,match='invalid'): ledger.reference()


def test_concurrent_budget_writer_fails_without_overwrite(tmp_path):
    ledger=BudgetLedger(tmp_path/'ledger.json')
    with ledger._locked():
        with pytest.raises(RuntimeError,match='writer is active'): reserve(ledger)
    assert not ledger.path.exists()


def test_refuses_budget_overdraw_and_impossible_observed_counts(tmp_path):
    ledger=BudgetLedger(tmp_path/'ledger.json')
    with pytest.raises(ValueError): reserve(ledger,transitions=262145)
    reserve(ledger)
    with pytest.raises(ValueError): ledger.finish('test',accepted=False,attempted=385,returned=8,stored=8,updates=0)
    assert ledger.read()['entries'][0]['state']=='reserved'


def test_formal_zero_start_entrypoint_rejects_spent_seed(tmp_path):
    from test_training_runner import authorization
    plan,kwargs=authorization(tmp_path)
    ledger=BudgetLedger(kwargs['ledger_path']); reserve(ledger)
    plan['budget_ledger']=ledger.reference()
    with pytest.raises(ValueError,match='already has reserved'):
        validate_training_authorization(plan,**kwargs)


@pytest.mark.parametrize('fault',[None,'unapproved','updates','transitions','seed','device','timeout','ledger'])
def test_manifest_fixes_scale_budget_seed_and_launch_permission(tmp_path,fault,monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    ledger=BudgetLedger(tmp_path/'ledger.json')
    kwargs=dict(revision='a'*40,inputs={'source':'b'*64},dependency_hashes={'dependency':'c'*64},
                output_dir=tmp_path/'out',ledger=ledger)
    plan=CONTRACT|{'attempt_id':'test','user_approved':True,'status':'approved',
        'base_revision':'a'*40,'remote_revision_verified':'a'*40,
        'input_sha256':kwargs['inputs'],'dependency_sha256':kwargs['dependency_hashes'],
        'output_dir':str(kwargs['output_dir']),'budget_ledger':ledger.reference()}
    fields={'unapproved':('user_approved',False),'updates':('max_updates',3),'transitions':('max_transitions',385),
        'seed':('ppo_seed',130002),'device':('device','cuda:1'),'timeout':('wall_timeout_s',241)}
    if fault in fields:
        key,value=fields[fault];plan[key]=value
    if fault=='ledger': reserve(ledger)
    if fault is None: assert validate_plan(plan,**kwargs)==plan
    else:
        with pytest.raises((ValueError,PermissionError)): validate_plan(plan,**kwargs)
    assert not kwargs['output_dir'].exists()


def test_unapproved_proposal_refused_before_git_gpu_or_ledger_calls(tmp_path,monkeypatch):
    import source_live_ppo_supervisor as supervisor
    path=tmp_path/'proposal.json';path.write_text(json.dumps({'user_approved':False}))
    def forbidden(*args,**kwargs): raise AssertionError('backend called before approval')
    monkeypatch.setattr(supervisor,'_git_state',forbidden)
    monkeypatch.setattr(subprocess,'check_output',forbidden)
    with pytest.raises(PermissionError): preflight(NS(manifest=path),approved=True)


def test_own_process_group_is_terminated_without_other_processes():
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    terminate_group(process,grace=.1)
    assert process.poll() is not None
    with pytest.raises(ProcessLookupError): os.killpg(process.pid,0)


def test_startup_and_wall_deadlines_remain_distinct():
    from source_live_ppo_supervisor import deadline_expired
    assert not deadline_expired(119,'',CONTRACT)
    assert deadline_expired(121,'',CONTRACT)
    assert not deadline_expired(121,'[live-ppo] initial-ready',CONTRACT)
    assert deadline_expired(241,'[live-ppo] initial-ready',CONTRACT)


@pytest.mark.parametrize('ending',['deadline','zero_exit'])
def test_supervisor_rejects_timeout_or_empty_zero_exit_and_keeps_reservation(tmp_path,monkeypatch,ending):
    import source_live_ppo_supervisor as supervisor
    ledger=BudgetLedger(tmp_path/'ledger.json')
    output=tmp_path/'attempt'
    plan=CONTRACT|{'attempt_id':'supervisor-cpu-fixture','base_revision':'a'*40,
                  'budget_ledger':ledger.reference(),'input_sha256':{},'dependency_sha256':{}}
    monkeypatch.setattr(supervisor,'preflight',lambda args,approved:(plan,ledger))
    monkeypatch.setattr(supervisor,'_input_hashes',lambda:{})
    monkeypatch.setattr(supervisor,'dependencies',lambda:{})
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'0, uuid, CPU-fixture, 18 MiB, 24000 MiB, 24576 MiB, 0 %\n')
    original=subprocess.Popen
    def fake_launch(command,**kwargs):
        program='import time;time.sleep(60)' if ending=='deadline' else 'pass'
        return original([sys.executable,'-c',program],**kwargs)
    monkeypatch.setattr(subprocess,'Popen',fake_launch)
    if ending=='deadline':monkeypatch.setattr(supervisor,'deadline_expired',lambda *args:True)
    assert supervisor.run(NS(output_dir=output,checkpoint=tmp_path/'unused'))==1
    execution=json.loads((output/'execution.json').read_text())
    assert execution['process_group_empty'] is True
    assert execution['timed_out']==(ending=='deadline')
    row=ledger.read()['entries'][0]
    assert row['state']=='failed' and row['reserved_transitions']==384


def test_child_rejects_unapproved_manifest_without_isaac_or_cuda():
    from source_live_ppo_probe import check_child_authorization
    before=set(sys.modules)
    initialized=torch.cuda.is_initialized()
    with pytest.raises(PermissionError):check_child_authorization({'user_approved':False},None)
    assert torch.cuda.is_initialized()==initialized
    assert not any(name.startswith(('isaaclab','isaacsim','omni.')) for name in set(sys.modules)-before)


@pytest.fixture(scope='module')
def live_fixture(tmp_path_factory):
    from moving_cpu_fixture import build_artifacts
    folder=tmp_path_factory.mktemp('live_ppo_synthetic_cpu')
    output,_=build_artifacts(folder,live_ppo=True)
    report=json.loads(output.read_text())
    plan={'attempt_id':report['attempt_id'],'base_revision':report['base_revision'],
          'checkpoint_sha256':report['checkpoint_sha256']}
    execution={'returncode':0,'process_group_empty':True,'timed_out':False,'monitor_errors':[]}
    return folder,plan,execution


def test_complete_cpu_fixture_checks_actual_updates_bootstrap_gae_and_motion(live_fixture):
    folder,plan,execution=live_fixture
    result=validate(folder,plan,execution,allow_synthetic=True)
    assert result['accepted'],result
    assert result['ppo_updates']==2 and result['event_review']['motion_samples']==192
    assert result['ordinary_failure_episodes']==1
    (folder/'acceptance.json').write_text(json.dumps(result,indent=2))
    assert not validate(folder,plan,execution)['accepted']


@pytest.mark.parametrize('fault',['sampled_actions','gae','optimizer_steps','bootstrap','model_nan','incomplete','monitor','fake_live'])
def test_corrupt_math_or_partial_evidence_refused_even_with_recomputed_file_hash(tmp_path,live_fixture,fault):
    source,plan,execution=live_fixture
    folder=tmp_path/'case'
    def copy_file(a,b):
        if str(a).endswith('.gz'): return os.link(a,b)
        return shutil.copyfile(a,b)
    shutil.copytree(source,folder,ignore=shutil.ignore_patterns('*.source.jsonl'),copy_function=copy_file)
    report=json.loads((folder/'result.json').read_text())
    execution=dict(execution)
    name='rollout_1.pt'
    if fault in ('optimizer_steps','model_nan'): name='model_1.pt'
    if fault in ('sampled_actions','gae','bootstrap','optimizer_steps','model_nan'):
        payload=torch.load(folder/name,weights_only=True)
        if fault=='sampled_actions': payload['actions'][0,0,0]+=1
        if fault=='gae': payload['advantages'][0,0,0]+=.5
        if fault=='bootstrap': payload['rewards'][11,0,0]+=1
        if fault=='optimizer_steps': next(iter(payload['optimizer']['state'].values()))['step'].fill_(19)
        if fault=='model_nan': payload['actor']['mlp.0.weight'][0,0]=float('nan')
        torch.save(payload,folder/name)
        report['evidence_sha256'][name]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
    if fault=='incomplete': (folder/'rollout_2.pt').unlink()
    if fault=='monitor': execution['monitor_errors']=['injected GPU monitor failure']
    if fault=='fake_live': report['synthetic_cpu_fixture']=False
    (folder/'result.json').write_text(json.dumps(report))
    result=validate(folder,plan,execution,allow_synthetic=True)
    assert not result['accepted'],result
