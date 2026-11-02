"""CPU fixtures exercise contracts; none constitutes real runtime qualification."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/tools'))
import source_prepare_cumulative_training as prepare
import source_training_supervisor as supervisor
from rl_training.source_scan.cumulative_training import validate_authorization, validate_supervised_child
from rl_training.source_scan.runtime_readiness import selected_scale_proposal
from rl_training.source_scan.training_budget_ledger import BudgetLedger
from rl_training.source_scan.training_runner import validate_training_authorization


def accepted_ledger(tmp_path):
    ledger = BudgetLedger(tmp_path/'ledger.json')
    archived = ROOT/'docs/lite3-source-live-ppo-integration-runtime-20261102/raw/budget_ledger_after.json'
    ledger.path.write_bytes(archived.read_bytes())
    return ledger


def reserve(ledger, **overrides):
    args = dict(run=1, attempt_id='cpu-numeric-fixture', output_dir=ledger.path.parent/'trial',
        revision='c'*40, expected_reference=ledger.reference(), expected_plan=ledger.numerics_plan(1))
    args.update(overrides)
    return ledger.reserve_numerics(**args)


def fixture_authorization(tmp_path):
    ledger = accepted_ledger(tmp_path)
    deps = {}
    for index in range(44):
        path = tmp_path/f'dependency_{index}'
        path.write_text(str(index))
        deps[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle = tmp_path/'bundle'
    prepare.write_bundle(bundle, ledger=ledger, revision='c'*40, remote='c'*40,
        inputs={'cpu_fixture':'b'*64}, dependency_hashes=deps, frozen=True)
    plan = json.loads((bundle/'run1.authorization.proposal.json').read_text())
    plan.update(status='approved', user_approved=True, output_dir=str(tmp_path/'trial'))
    runtime = json.loads(Path(plan['runtime_qualification']['path']).read_text())
    runtime.update(status='qualified_for_bounded_numerics', measured_collection_transitions_per_second=700.)
    for key in ('g0_passed','g1_passed','g2_passed','g3_passed','g4_collection_passed'): runtime[key] = True
    rewrite_receipt(plan, 'runtime_qualification', runtime)
    protocol = json.loads(Path(plan['source_selection_lock']['path']).read_text())
    protocol.update(status='locked', user_approved=True)
    rewrite_receipt(plan, 'source_selection_lock', protocol)
    metrics = json.loads(Path(plan['metrics_qualification']['path']).read_text())
    metrics.update(status='qualified', live_normalized_metric_collector_qualified=True)
    rewrite_receipt(plan, 'metrics_qualification', metrics)
    kwargs = dict(run=1, budget_node='numerics', output_dir=tmp_path/'trial',
        checkpoint_sha256=plan['checkpoint_sha256'], revision='c'*40,
        input_sha256=plan['input_sha256'], ledger_path=ledger.path)
    return ledger, plan, kwargs


def rewrite_receipt(plan, key, value):
    path = Path(plan[key]['path'])
    path.write_text(json.dumps(value))
    plan[key]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_cumulative_rounding_retains_previous_charge_and_no_production_write(tmp_path):
    ledger = accepted_ledger(tmp_path)
    original = ledger.path.read_bytes()
    budget = ledger.numerics_plan(1)
    nodes = list(budget['nodes'].values())
    assert [node['remaining_updates'] for node in nodes] == [90,363,728]
    assert [node['planned_transitions'] for node in nodes] == [259200,1045440,2096640]
    assert [node['cumulative_transitions'] for node in nodes] == [259584,1045824,2097024]
    assert [node['unused_transitions'] for node in nodes] == [2560,2752,128]
    assert ledger.numerics_plan(2)['nodes']['numerics']['remaining_updates'] == 91
    assert ledger.path.read_bytes() == original
    row = reserve(ledger)
    assert row['reserved_transitions'] == 259200
    assert ledger.charged(1,ledger.read()) == 259584
    with pytest.raises(ValueError): ledger.numerics_plan(1)
    ledger.finish(row['attempt_id'], accepted=False, attempted=120, returned=0, stored=0, updates=0)
    assert ledger.charged(1,ledger.read()) == 259584
    with pytest.raises(ValueError,match='finalized'):
        ledger.finish(row['attempt_id'], accepted=True, attempted=120, returned=0, stored=0, updates=0)


@pytest.mark.parametrize('fault',['stale','changed_plan','concurrent','failed_diagnostic','corrupt_diagnostic'])
def test_reservation_refuses_changed_or_ineligible_history(tmp_path,fault):
    ledger = accepted_ledger(tmp_path)
    if fault in ('failed_diagnostic','corrupt_diagnostic'):
        data = ledger.read()
        data['entries'][0]['state' if fault=='failed_diagnostic' else 'returned_transitions'] = 'failed' if fault=='failed_diagnostic' else 383
        ledger.path.write_text(json.dumps(data))
        with pytest.raises(ValueError): ledger.numerics_plan(1)
    elif fault=='concurrent':
        with ledger._locked():
            with pytest.raises(RuntimeError): reserve(ledger)
    elif fault=='stale':
        with pytest.raises(ValueError): reserve(ledger,expected_reference={'path':str(ledger.path),'sha256':None})
    else:
        changed = ledger.numerics_plan(1); changed['nodes']['numerics']['remaining_updates'] = 91
        with pytest.raises(ValueError): reserve(ledger,expected_plan=changed)


@pytest.mark.parametrize('fault',[None,'approval','updates','budget','seed','initialization','midpoint',
    'missing_runtime','runtime_not_run','metrics_not_run','metric_collector','protocol_unapproved',
    'changed_receipt','dependency','stale_ledger'])
def test_authorization_requires_all_separate_qualifications(tmp_path,fault):
    ledger,plan,kwargs = fixture_authorization(tmp_path)
    fields = {'approval':('user_approved',False),'updates':('max_updates',91),
        'budget':('max_transitions',262080),'seed':('ppo_seed',130002),
        'initialization':('initialization','reuse_diagnostic_weights')}
    if fault in fields:
        key,value = fields[fault]; plan[key] = value
    if fault=='midpoint': kwargs['budget_node'] = 'midpoint'
    if fault=='missing_runtime': del plan['runtime_qualification']
    if fault in ('runtime_not_run','metrics_not_run','metric_collector','protocol_unapproved'):
        key = {'runtime_not_run':'runtime_qualification','metrics_not_run':'metrics_qualification',
            'metric_collector':'metrics_qualification','protocol_unapproved':'source_selection_lock'}[fault]
        payload = json.loads(Path(plan[key]['path']).read_text())
        if fault=='metric_collector': payload['live_normalized_metric_collector_qualified'] = False
        elif fault=='protocol_unapproved': payload.update(status='proposed',user_approved=False)
        else: payload['status'] = 'not_run'
        rewrite_receipt(plan,key,payload)
    if fault=='changed_receipt': Path(plan['runtime_qualification']['path']).write_text('{}')
    if fault=='dependency': Path(next(iter(plan['dependency_sha256']))).write_text('changed')
    if fault=='stale_ledger': reserve(ledger)
    if fault is None:
        assert validate_training_authorization(plan,**kwargs)['max_updates'] == 90
    else:
        with pytest.raises((ValueError,PermissionError,NotImplementedError)):
            validate_training_authorization(plan,**kwargs)


@pytest.mark.parametrize('fault',[None,'ticket_missing','reservation_changed','inputs_changed'])
def test_child_requires_existing_exact_parent_reservation(tmp_path,fault):
    ledger,plan,kwargs = fixture_authorization(tmp_path)
    lock = validate_authorization(plan,**kwargs)
    row = reserve(ledger)
    ticket = {'schema':'lite3_source_numerics_supervision_v1','authorization':lock,'reservation':row}
    args = {key:kwargs[key] for key in ('output_dir','revision','input_sha256','ledger_path')}
    if fault=='ticket_missing': del ticket['reservation']
    if fault=='reservation_changed': ticket['reservation']['reserved_transitions'] += 1
    if fault=='inputs_changed': args['input_sha256'] = {}
    if fault is None: assert validate_supervised_child(plan,ticket,**args) == lock
    else:
        with pytest.raises(ValueError): validate_supervised_child(plan,ticket,**args)


def test_preparation_templates_are_unapproved_and_read_only(tmp_path):
    ledger = accepted_ledger(tmp_path); before = ledger.path.read_bytes()
    result = prepare.write_bundle(tmp_path/'bundle',ledger=ledger,revision='a'*40,remote=None,
        inputs={},dependency_hashes={})
    assert result['formal_launch_ready'] is False and result['cuda_initialized'] is False
    assert result['production_budget_before'] == result['production_budget_after']
    for row in result['authorizations'].values():
        plan = json.loads(Path(row['path']).read_text())
        assert plan['user_approved'] is False and plan['remote_revision_verified'] is None
    assert ledger.path.read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare.write_bundle(tmp_path/'bundle',ledger=ledger,revision='a'*40,remote=None,inputs={},dependency_hashes={})


def test_selected_scale_plan_reports_missing_execution_and_dense_cost():
    plan = selected_scale_proposal()
    assert plan['user_approved'] is False and plan['startup_ladder'] == [16,32,60,120]
    assert plan['evidence_strategy']['dense_frame_bytes'] == 67393440
    assert plan['evidence_strategy']['dense_G3_bytes_before_JSON'] == 539147520000
    assert plan['evidence_strategy']['sparse_force_only_pairs_must_be_included']
    assert plan['evidence_strategy']['implementation_status'] == 'not_implemented_for_120_envs'
    assert plan['stages']['G4_end_to_end']['status'] == 'pending_bounded_numerics'
    assert plan['stages']['G3']['all_old_policy_cells_must_succeed'] is False


def test_unapproved_supervisor_and_unsupervised_entrypoint_do_not_start_backend(tmp_path,monkeypatch):
    import source_training_entrypoint as entry
    path = tmp_path/'authorization.json'; path.write_text(json.dumps({'user_approved':False}))
    def forbidden(*a,**kw): raise AssertionError('runtime called')
    monkeypatch.setattr(supervisor,'validate_before_simulation',forbidden)
    with pytest.raises(PermissionError): supervisor.run(NS(authorization=path))
    path.write_text(json.dumps({'schema':prepare.SCHEMA}))
    monkeypatch.setattr(entry,'validate_before_simulation',lambda *a:({'schema':prepare.SCHEMA},'a'*40,{}))
    monkeypatch.setattr(sys,'argv',['entry','--authorization',str(path),'--checkpoint',str(tmp_path/'none'),
        '--output-dir',str(tmp_path/'output'),'--run','1','--budget-node','numerics'])
    before = set(sys.modules)
    with pytest.raises(ValueError,match='requires source_training_supervisor'): entry.main()
    assert not (tmp_path/'output').exists()
    assert not torch.cuda.is_initialized()
    assert not any(name.startswith(('isaaclab','isaacsim','omni.')) for name in set(sys.modules)-before)


def test_startup_and_wall_timeouts_are_distinct():
    assert not supervisor.deadline_expired(119,'')
    assert supervisor.deadline_expired(121,'')
    assert not supervisor.deadline_expired(121,'[source-numerics] initial-ready')
    assert supervisor.deadline_expired(901,'[source-numerics] initial-ready')


@pytest.mark.parametrize('fault',['run','node','version'])
def test_supervised_cli_contract_checked_before_runtime_import(tmp_path,monkeypatch,fault):
    import source_training_entrypoint as entry
    import rl_training.source_scan.cumulative_training as cumulative
    import rl_training.source_scan.training_budget_ledger as budget_module
    output = tmp_path/'trial'; output.mkdir()
    authorization = tmp_path/'plan.json'; authorization.write_text('{}')
    ticket = output/'supervision.json'; ticket.write_text('{}')
    checkpoint = tmp_path/'checkpoint'; checkpoint.write_bytes(b'CPU fixture')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(entry,'CHECKPOINT_SHA256',hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(entry,'_git_state',lambda:('a'*40,'a'*40,''))
    monkeypatch.setattr(entry,'_input_hashes',lambda:{})
    monkeypatch.setattr(cumulative,'validate_supervised_child',lambda *a,**kw:{'run':1,'budget_node':'numerics'})
    monkeypatch.setattr(budget_module,'canonical_budget_path',lambda *a:tmp_path/'unused')
    monkeypatch.setattr(entry.importlib.metadata,'version',lambda *a:'wrong' if fault=='version' else '5.0.1')
    monkeypatch.setattr(sys,'argv',['entry','--authorization',str(authorization),'--checkpoint',str(checkpoint),
        '--output-dir',str(output),'--supervised-ticket',str(ticket),
        '--run','2' if fault=='run' else '1','--budget-node','midpoint' if fault=='node' else 'numerics'])
    before = set(sys.modules)
    with pytest.raises((ValueError,RuntimeError)): entry.main()
    assert not (output/'attempt.json').exists() and not torch.cuda.is_initialized()
    assert not any(name.startswith(('isaaclab','isaacsim','omni.')) for name in set(sys.modules)-before)


@pytest.mark.parametrize('ending',['deadline','zero_exit'])
def test_cpu_dummy_process_failure_preserves_full_reservation(tmp_path,monkeypatch,ending):
    ledger,plan,kwargs = fixture_authorization(tmp_path)
    authorization = tmp_path/'approved_cpu_fixture.json'; authorization.write_text(json.dumps(plan))
    monkeypatch.setattr(supervisor,'validate_before_simulation',lambda *a:(plan,'c'*40,plan['input_sha256']))
    monkeypatch.setattr(supervisor,'dependencies',lambda:plan['dependency_sha256'])
    monkeypatch.setattr(supervisor,'_input_hashes',lambda:plan['input_sha256'])
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'0, uuid, CPU fixture, 18 MiB, 24000 MiB, 24576 MiB, 0 %\n')
    original = subprocess.Popen
    def launch(command,**kw):
        program = 'import time;time.sleep(60)' if ending=='deadline' else 'pass'
        return original([sys.executable,'-c',program],**kw)
    monkeypatch.setattr(subprocess,'Popen',launch)
    if ending=='deadline': monkeypatch.setattr(supervisor,'deadline_expired',lambda *a:True)
    assert supervisor.run(NS(authorization=authorization,output_dir=kwargs['output_dir'],
        checkpoint=tmp_path/'unused',run=1)) == 1
    execution = json.loads((kwargs['output_dir']/'execution.json').read_text())
    assert execution['process_group_empty'] is True
    assert execution['timed_out'] == (ending=='deadline')
    row = ledger.read()['entries'][-1]
    assert row['state']=='failed' and row['reserved_transitions']==259200
    assert ledger.charged(1,ledger.read())==259584


def result_fixture(tmp_path):
    from test_source_training import make_actor
    from source_live_ppo_validation import state_digest
    ledger,plan,_ = fixture_authorization(tmp_path)
    folder = tmp_path/'result'; (folder/'training').mkdir(parents=True)
    state = make_actor().state_dict(); digest = state_digest(state)
    torch.save(state,folder/'training/actor_update_90.pt')
    report = dict(status='completed',run=1,data_kind='source_simulation',num_envs=120,rollout_length=24,
        completed_updates=90,attempted_transition_upper_bound=259200,returned_transitions=259200,
        stored_transitions=259200,real_source_training_transitions=259200,real_source_ppo_updates=90,
        source_backward_checks=1800,source_optimizer_steps_started=1800,source_optimizer_steps_completed=1800,
        checkpoint_sha256=plan['checkpoint_sha256'],base_revision=plan['base_revision'],input_sha256=plan['input_sha256'],
        cumulative_budget=plan['cumulative_budget'],initial_actor_sha256='a'*64,initial_critic_sha256='a'*64,
        exposure={'fixture':{'transitions':259200}},elapsed_seconds=400.,
        update_receipts=[dict(update=i,returned_transitions=2880,stored_transitions=2880,
            losses={'value':.1,'surrogate':-.02},actor_sha256=digest,critic_sha256='b'*64) for i in range(1,91)])
    (folder/'attempt.json').write_text(json.dumps(dict(status='completed',environment_close_returned=True,application_close_state='closing')))
    execution = dict(returncode=0,process_group_empty=True,timed_out=False,monitor_errors=[])
    return folder,plan,report,execution


@pytest.mark.parametrize('fault',[None,'nonfinite','missing_loss','partial','no_actor_change','no_critic_change',
    'checkpoint','slow','timeout','close'])
def test_result_validation_refuses_incomplete_or_inconsistent_evidence(tmp_path,fault):
    folder,plan,report,execution = result_fixture(tmp_path)
    if fault=='nonfinite': report['update_receipts'][0]['losses']['value'] = float('nan')
    if fault=='missing_loss': report['update_receipts'][0]['losses'] = {}
    if fault=='partial': report['update_receipts'].pop()
    if fault=='no_actor_change': report['initial_actor_sha256'] = report['update_receipts'][-1]['actor_sha256']
    if fault=='no_critic_change': report['initial_critic_sha256'] = 'b'*64
    if fault=='checkpoint': report['update_receipts'][-1]['actor_sha256'] = 'b'*64
    if fault=='slow': report['elapsed_seconds'] = 500.
    if fault=='timeout': execution['timed_out'] = True
    if fault=='close': (folder/'attempt.json').write_text('{}')
    (folder/'training/run_report.json').write_text(json.dumps(report))
    result = supervisor.validate_result(folder,plan,execution)
    assert result['accepted'] == (fault is None)
    assert result['formal_pilot_authorized'] is False and result['training_ready'] is False
    if fault: assert result['reasons']


@pytest.mark.parametrize('model',['actor','critic'])
def test_frozen_training_parameters_refused_before_collection(tmp_path,model):
    from scripts.tools.source_training_numerics_check import make_runner
    from rl_training.source_scan.source_training import TransitionBudget
    from rl_training.source_scan.training_runner import SourceTrainingRunner
    runner = make_runner(compact=True,rollout_length=4)
    getattr(runner.alg,model).requires_grad_(False)
    with pytest.raises(ValueError,match='trainable'):
        SourceTrainingRunner(runner,run=1,budget=TransitionBudget(8,4,32),
            output_dir=tmp_path/'output',checkpoint_sha256='a'*64)
