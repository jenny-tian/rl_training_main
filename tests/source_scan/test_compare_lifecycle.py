"""CPU regressions for the stalled actor switch; no simulator launch."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import source_delivery_light as light
from rl_training.source_scan.delivery_stage import reference,read_reference,slot_ledger,job_contract
from rl_training.source_scan.smoke_artifacts import SmokeArtifactWriter
from source_training_entrypoint import _input_hashes


def test_actor_metrics_are_durable_before_teardown_failure(tmp_path):
    calls=[]
    class Sim:
        _disable_app_control_on_stop_handle=False
        def stop(self):
            assert self._disable_app_control_on_stop_handle
            calls.append('stop')
        def is_playing(self):return False
    def close():
        calls.append('close')
        saved=json.loads((tmp_path/'result.json').read_text())
        assert saved['actors']==[dict(label='old',first_episodes=[dict(completed=False)])]
        assert saved['steps_written']==1 and calls==['phase','stop','close']
        raise RuntimeError('simulated native close failure')
    report=dict(attempt_id='CPU_fixture',synthetic_cpu_fixture=True,actors=[])
    writer=SmokeArtifactWriter(tmp_path/'result.json',report)
    writer.append_step(dict(step=1,finite=True))
    report['actors']=[dict(label='old',first_episodes=[dict(completed=False)])]
    wrapped=NS(unwrapped=NS(sim=Sim()),close=close)
    def phase(name,label,limit):
        assert (name,label,limit)==('environment_close','old',15)
        calls.append('phase')
    try:
        with pytest.raises(RuntimeError,match='simulated native close failure'):
            light.persist_actor_and_close(writer,report,wrapped,phase,'compare','old')
    finally:writer.close()


def test_new_environment_and_close_have_independent_deadlines(tmp_path):
    light.record_phase(tmp_path,'environment_close','old',15)
    first=json.loads((tmp_path/'phase.json').read_text())
    assert light.lifecycle_deadline(tmp_path,first['started_mono']+14) is None
    assert 'environment_close' in light.lifecycle_deadline(tmp_path,first['started_mono']+16)
    light.record_phase(tmp_path,'environment_start','new',120)
    second=json.loads((tmp_path/'phase.json').read_text())
    assert light.lifecycle_deadline(tmp_path,second['started_mono']+119) is None
    assert 'actor=new' in light.lifecycle_deadline(tmp_path,second['started_mono']+121)
    light.record_phase(tmp_path,'actor_controls','new')
    assert light.lifecycle_deadline(tmp_path,second['started_mono']+1000) is None


@pytest.mark.parametrize('fault',[None,'extra_launch','changed_original_slots'])
def test_repair_contract_preserves_original_failure_and_limits(tmp_path,fault):
    root=Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/.git/source-delivery-evidence/lite3-training-delivery-20261104-v1')
    if not (root/'run/compare/receipt.json').exists():pytest.skip('requires archived failed compare')
    plan=json.loads((root/'run/compare/run_manifest.json').read_text())
    plan.update(input_sha256=_input_hashes(),output_dir=str(root/'run/compare_repair_20261105'))
    supplement=dict(schema='lite3_compare_repair_supplement_v1',status='approved',user_approved=True,
        user_message='继续',stage_id='lite3-training-delivery-20261104-v1',job='compare',num_envs=120,
        control_steps=4000,diagnostic_transitions=480000,training_transitions=0,ppo_updates=0,
        max_launches=1,startup_timeout_s=120,wall_timeout_s=1800,cleanup_grace_s=15,automatic_retry=False,
        source_revision=plan['base_revision'],input_sha256=plan['input_sha256'],output_dir=plan['output_dir'],
        slot_ledger=str(root/'compare_repair_slots.json'),previous_failed_compare=reference(root/'run/compare/receipt.json','compare'),
        original_slots=reference(root/'slots.json','slot_ledger'))
    if fault=='extra_launch':supplement['max_launches']=2
    if fault=='changed_original_slots':supplement['original_slots']['sha256']='0'*64
    p=tmp_path/'repair.json';p.write_text(json.dumps(supplement));plan['compare_supplement']=reference(p,'compare_supplement')
    original=(root/'slots.json').read_bytes()
    if fault:
        with pytest.raises((PermissionError,ValueError)):slot_ledger(plan)
    else:
        assert slot_ledger(plan)==root/'compare_repair_slots.json'
        contract=job_contract('compare',plan)
        assert contract['control_steps']==4000 and contract['ppo_updates']==0
        assert contract['attempt_id'].endswith('compare-repair-20261105')
    assert (root/'slots.json').read_bytes()==original
