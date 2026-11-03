"""CPU-only review of G2 preparation, original failures and final evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent


def check(condition,message):
    if not condition:raise ValueError(message)


def main(source_root):
    raw=HERE/'raw'
    def read(name):return json.loads((raw/name).read_text())
    lines=(HERE/'SHA256SUMS').read_text().splitlines();listed=set()
    for line in lines:
        digest,name=line.split('  ',1);path=(HERE/name).resolve();listed.add(name)
        check(path.is_relative_to(HERE) and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'archive bytes differ: '+name)
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS','review.json') and '__pycache__' not in p.parts}
    check(listed==actual and len(lines)==len(listed),'archive checksum coverage differs')
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,value in provenance['files'].items():
        payload=(HERE/name).read_bytes()
        check(len(payload)==value['bytes'] and hashlib.sha256(payload).hexdigest()==value['sha256'],'original bytes differ: '+name)
    inputs=read('input_hashes_before.json');deps=read('dependencies_before.json')
    check(inputs==read('input_hashes_after.json') and len(inputs)==243,'production inputs changed during regression')
    check(deps==read('dependencies_after.json') and len(deps)==51,'installed dependencies changed')
    for name,digest in inputs.items():check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'source differs: '+name)
    for name,digest in deps.items():check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'dependency differs: '+name)
    before=(raw/'production_ledger_before.json').read_bytes();after=(raw/'production_ledger_after.json').read_bytes();reference=read('production_ledger_reference.json')
    check(before==after and hashlib.sha256(before).hexdigest()==reference['sha256'],'production budget changed')
    ledger=json.loads(after)
    check(len(ledger['entries'])==1 and ledger['entries'][0]['state']=='accepted'
        and ledger['entries'][0]['reserved_transitions']==384 and ledger['entries'][0]['completed_updates']==2,'prior PPO charge changed')
    prep=read('preparation.json')
    check(prep['real_120_env_G2_passed'] is False and prep['formal_training_ready'] is False
        and prep['cuda_initialized'] is False,'CPU preparation falsely claims real qualification')
    for name in ('gpu_queries','gpu_jobs','isaac_sim_launches','real_source_transitions','real_source_ppo_updates'):
        check(prep[name]==0,'unauthorized runtime work: '+name)
    sys.path[:0]=[str(source_root/'source/rl_training'),str(source_root/'scripts/tools')]
    from rl_training.source_scan.scale_reset import contract_for,same_value,validate_predecessor,PREDECESSOR_120
    from rl_training.source_scan.scale_startup import PREDECESSOR_16,PREDECESSOR_32,PREDECESSOR_60
    from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
    from source_scale_reset_validation import validate
    from source_scale_startup_validation import validate as validate_g1
    import torch
    checkpoint=read('checkpoint_reference.json')
    check(checkpoint['sha256']==CHECKPOINT_SHA256 and hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest()==CHECKPOINT_SHA256,'original checkpoint differs')
    plan=read('g2.proposal.json')
    check(plan['status']=='proposed' and plan['user_approved'] is False and plan['remote_revision_verified'] is None,'working draft falsely frozen/approved')
    check(plan['input_sha256']==inputs and plan['dependency_sha256']==deps and plan['budget_ledger']==reference,'draft input snapshot differs')
    for key,value in contract_for().items():check(same_value(plan.get(key),value),'G2 draft contract differs: '+key)
    check(read('predecessor_reference.json')==PREDECESSOR_120,'predecessor reference differs')
    validate_predecessor(plan['predecessor_qualification'],source_root=source_root)
    for prior in (PREDECESSOR_16,PREDECESSOR_32,PREDECESSOR_60,PREDECESSOR_120):
        folder=source_root/prior['archive']/'raw';load=lambda name:json.loads((folder/name).read_text())
        check(validate_g1(folder,load('run_manifest.json'),load('execution.json'))==load('acceptance.json'),'historical G1 acceptance changed')
    expected=[('focused.initial.junit.xml',190,1),('transaction.initial.junit.xml',1,1),
        ('focused_fix.junit.xml',2,0),('regression.junit.xml',929,0)]
    for name,count,failures in expected:
        suite=ET.parse(raw/name).getroot().find('testsuite')
        check(suite is not None and int(suite.attrib['tests'])==count and int(suite.attrib['failures'])==failures
            and int(suite.attrib['errors'])==0 and int(suite.attrib['skipped'])==0,'test record differs: '+name)
    full=ET.parse(raw/'regression.junit.xml').getroot().find('testsuite')
    check(sum(v.attrib.get('classname','').endswith(('.test_scale_reset','.test_scale_reset_transaction')) for v in full.findall('testcase'))==105,'new G2 test count differs')
    for name,code in [('focused.initial_execution.json',1),('transaction.initial_execution.json',1),('focused_fix_execution.json',0),('regression_execution.json',0)]:
        record=read(name)
        check(record['returncode']==code and record['environment_overrides']['CUDA_VISIBLE_DEVICES']==''
            and record['gpu_jobs']==record['isaac_sim_launches']==0,'test execution/mask differs: '+name)
    findings=read('development_findings.json');validation=read('validation_findings.json')
    check('inference tensor outside InferenceMode' in findings['error'] and validation['acceptance_rules_relaxed'] is False,'failure/fix history missing')
    check('Inplace update to inference tensor' in (raw/'development_failure/events.jsonl').read_text(),'original inference failure missing')
    partial=validate(raw/'development_failure',read('development_failure/fixture_manifest.json'),read('development_failure/fixture_execution_inputs.json'))
    check(partial['accepted'] is False,'unfinished original fixture falsely accepted')
    fixture=read('cpu_example/provenance.json')
    check(fixture['synthetic_cpu_fixture'] is True and fixture['real_runtime_qualification'] is False,'CPU fixture mislabeled')
    accepted=validate(raw/'cpu_example',read('cpu_example/fixture_manifest.json'),read('cpu_example/fixture_execution_inputs.json'))
    check(accepted['accepted'] is True and accepted==read('cpu_example/fixture_acceptance.json'),'synthetic G2 acceptance differs')
    check(accepted['reset_batches']==4 and accepted['selected_environment_resets']==184 and accepted['assignment_count']==304
        and accepted['complete_assignment_blocks']==2 and accepted['tail_assignment_count']==64
        and accepted['diagnostic_transitions']==480 and accepted['ground_positive_qualification'] is False,'CPU scope expanded')
    launch=read('launch_command.proposal.json')
    check(launch['status']=='proposed_not_executed' and launch['separate_user_approval_required'] is True,'GPU launch falsely approved')
    check(not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled(),'CPU reviewer initialized CUDA or leaked inference mode')
    return dict(status='passed_120_env_G2_cpu_preparation_archive_review',hashed_archive_files=len(listed),registered_tests=929,
        new_G2_tests=105,initial_focused_passed=189,initial_focused_failed=1,post_fix_targeted_tests=2,
        production_inputs=243,installed_dependencies=51,predecessor_120_60_32_16_verified=True,
        historical_G1_acceptance_unchanged=True,production_budget_unchanged=True,real_120_env_G2_passed=False,
        real_source_transitions=0,real_source_ppo_updates=0,gpu_queries=0,gpu_jobs=0,isaac_sim_launches=0,
        cuda_initialized=False,formal_training_ready=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,default=HERE.parents[1])
    args=parser.parse_args();result=main(args.source_root.resolve())
    (HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
