"""Verify archive integrity and CPU fixture relationships; never launches a simulator."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]


def check(condition,message):
    if not condition:raise ValueError(message)


def main():
    checked=0
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(HERE/name).resolve()
        check(path.is_relative_to(HERE),'archive path escaped root')
        check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'archive hash differs: '+name)
        checked+=1
    raw=HERE/'raw'
    def read(name):return json.loads((raw/name).read_text())
    provenance=read('provenance.json')
    for name,value in provenance['raw_files'].items():
        payload=(raw/name).read_bytes()
        check(len(payload)==value['bytes'] and hashlib.sha256(payload).hexdigest()==value['sha256'],
            'source evidence changed: '+name)
    preparation=read('preparation.json')
    for name in ('real_source_transitions','real_source_ppo_updates','isaac_sim_launches','gpu_jobs'):
        check(preparation[name]==0,'unexpected runtime work during CPU preparation')
    check(preparation['real_G1_passed'] is False and preparation['formal_training_ready'] is False
        and preparation['cuda_initialized'] is False,'preparation falsely claims live qualification')
    check((raw/'production_ledger_before.json').read_bytes()==(raw/'production_ledger_after.json').read_bytes(),
        'CPU preparation changed production budget')
    ledger=read('production_ledger_after.json')
    check(len(ledger['entries'])==1 and ledger['entries'][0]['state']=='accepted'
        and ledger['entries'][0]['reserved_transitions']==384 and ledger['entries'][0]['completed_updates']==2,
        'accepted prior diagnostic charge changed')
    check(read('input_hashes_before.json')==read('input_hashes_after.json') and len(read('input_hashes_before.json'))==238,
        'production input snapshot differs')
    check(read('dependencies_before.json')==read('dependencies_after.json') and len(read('dependencies_before.json'))==51,
        'installed dependency snapshot differs')
    proposal=read('g1.proposal.json')
    check(proposal['user_approved'] is False and proposal['status']=='proposed'
        and proposal['remote_revision_verified'] is None,'draft proposal falsely approved or frozen')
    check(proposal['num_envs']==16 and proposal['control_steps']==1 and proposal['ppo_updates']==0
        and proposal['policy_loaded'] is False and proposal['diagnostic_transitions']==16,'G1 scope expanded')
    check(proposal['input_sha256']==read('input_hashes_after.json') and proposal['dependency_sha256']==read('dependencies_after.json'),
        'G1 draft differs from checked source/dependencies')
    execution=read('regression_execution.json')
    check(execution['returncode']==0 and execution['environment_overrides']['CUDA_VISIBLE_DEVICES']=='',
        'regression failed or CPU mask changed')
    for name,count in (('regression.junit.xml',647),('focused_final.junit.xml',102)):
        suite=ET.parse(raw/name).getroot().find('testsuite')
        check(suite is not None and int(suite.attrib['tests'])==count,'test count differs: '+name)
        check(all(int(suite.attrib[key])==0 for key in ('errors','failures','skipped')),'test failures: '+name)
    from source_scale_startup_validation import validate
    fixture=read('cpu_example_g1/provenance.json')
    check(fixture['synthetic_cpu_fixture'] is True and fixture['live_runtime_qualification'] is False,
        'CPU fixture mislabeled as runtime qualification')
    result=validate(raw/'cpu_example_g1',read('cpu_example_g1/fixture_manifest.json'),read('cpu_example_g1/fixture_execution_inputs.json'))
    check(result['accepted'] is True and result['training_ready'] is False,'CPU fixture relationship review failed: '+str(result))
    check(result==read('cpu_example_g1/fixture_acceptance.json'),'CPU fixture review differs from archived result')
    import torch
    check(not torch.cuda.is_initialized(),'archive reviewer initialized CUDA')
    return dict(status='passed_cpu_adapter_archive_review',hashed_files=checked,registered_tests=647,
        focused_tests=102,production_inputs=238,installed_dependencies=51,production_budget_unchanged=True,
        cpu_fixture_relations_rechecked=True,real_G1_passed=False,real_source_transitions=0,
        real_source_ppo_updates=0,isaac_sim_launches=0,gpu_jobs=0,formal_training_ready=False)


if __name__=='__main__':
    result=main();(HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
