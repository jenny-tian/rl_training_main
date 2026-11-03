"""CPU archive review of the 120-env preparation; no simulator or GPU work."""
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
    checked=0
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(HERE/name).resolve()
        check(path.is_relative_to(HERE),'archive path escaped root')
        check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'archive SHA mismatch: '+name);checked+=1
    listed={line.split('  ',1)[1] for line in (HERE/'SHA256SUMS').read_text().splitlines()}
    actual={str(path.relative_to(HERE)) for path in HERE.rglob('*') if path.is_file()
        and path.name not in ('SHA256SUMS','review.json') and '__pycache__' not in path.parts}
    check(listed==actual and len(listed)==checked,'archive checksum coverage differs')
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,value in provenance['files'].items():
        data=(HERE/name).read_bytes()
        check(len(data)==value['bytes'] and hashlib.sha256(data).hexdigest()==value['sha256'],'original bytes differ: '+name)
    before=read('input_hashes_before.json');after=read('input_hashes_after.json')
    check(before==after and len(before)==238,'production inputs changed during regression')
    deps=read('dependencies_before.json')
    check(deps==read('dependencies_after.json') and len(deps)==51,'installed dependencies changed')
    for name,digest in before.items():
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'source snapshot differs: '+name)
    for name,digest in deps.items():
        check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'dependency differs: '+name)
    ledger_bytes=(raw/'production_ledger_before.json').read_bytes()
    check(ledger_bytes==(raw/'production_ledger_after.json').read_bytes(),'CPU preparation changed production budget')
    reference=read('production_ledger_reference.json')
    check(hashlib.sha256(ledger_bytes).hexdigest()==reference['sha256'],'ledger reference differs')
    ledger=read('production_ledger_after.json')
    check(len(ledger['entries'])==1 and ledger['entries'][0]['state']=='accepted'
        and ledger['entries'][0]['reserved_transitions']==384 and ledger['entries'][0]['completed_updates']==2,
        'prior PPO diagnostic charge changed')
    prep=read('preparation.json')
    check(prep['real_120_env_G1_passed'] is False and prep['formal_training_ready'] is False
        and prep['cuda_initialized'] is False,'preparation falsely claims live qualification')
    for name in ('gpu_queries','gpu_jobs','isaac_sim_launches','real_source_transitions','real_source_ppo_updates'):
        check(prep[name]==0,'CPU preparation performed unauthorized runtime work')
    sys.path[:0]=[str(source_root/'source/rl_training'),str(source_root/'scripts/tools')]
    from rl_training.source_scan.scale_startup import contract_for,validate_predecessor,PREDECESSOR_16,PREDECESSOR_32,PREDECESSOR_60
    from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
    from collections import Counter
    from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
    from source_scale_startup_validation import validate
    import torch
    checkpoint=read('checkpoint_reference.json')
    check(checkpoint['sha256']==CHECKPOINT_SHA256
        and hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest()==CHECKPOINT_SHA256,'original checkpoint differs')
    plan=read('g1_120.proposal.json')
    check(plan['status']=='proposed' and plan['user_approved'] is False
        and plan['remote_revision_verified'] is None,'working-tree draft falsely approved/frozen')
    check(plan['input_sha256']==after and plan['dependency_sha256']==deps
        and plan['budget_ledger']==reference,'proposal differs from checked source/budget')
    for name,value in contract_for(120).items():check(plan.get(name)==value,'draft contract differs: '+name)
    check(read('predecessor_reference.json')==PREDECESSOR_60,'predecessor reference differs')
    validate_predecessor(120,plan['predecessor_qualification'],source_root=source_root)
    for prior in (PREDECESSOR_16,PREDECESSOR_32,PREDECESSOR_60):
        folder=source_root/prior['archive']/'raw'
        prior_read=lambda name:json.loads((folder/name).read_text())
        check(validate(folder,prior_read('run_manifest.json'),prior_read('execution.json'))==prior_read('acceptance.json'),
            'historical real runtime acceptance differs: '+prior['archive'])
    for name,count in (('focused.junit.xml',247),('regression.junit.xml',824)):
        suite=ET.parse(raw/name).getroot().find('testsuite')
        check(suite is not None and int(suite.attrib['tests'])==count,'test count differs: '+name)
        check(all(int(suite.attrib[key])==0 for key in ('failures','errors','skipped')),'test failures: '+name)
    for name in ('focused_execution.json','regression_execution.json'):
        execution=read(name)
        check(execution['returncode']==0 and execution['environment_overrides']['CUDA_VISIBLE_DEVICES']==''
            and execution['gpu_jobs']==0 and execution['isaac_sim_launches']==0,'CPU test execution differs: '+name)
    fixture=read('cpu_example_120/provenance.json')
    check(fixture['synthetic_cpu_fixture'] is True and fixture['real_runtime_qualification'] is False,
        'CPU fixture mislabeled as runtime qualification')
    accepted=validate(raw/'cpu_example_120',read('cpu_example_120/fixture_manifest.json'),read('cpu_example_120/fixture_execution_inputs.json'))
    check(accepted['accepted'] is True and accepted==read('cpu_example_120/fixture_acceptance.json'),
        'CPU fixture relationship review differs')
    check(accepted['diagnostic_transitions']==120 and accepted['ground_positive_qualification'] is False,
        'CPU fixture scope expanded')
    report=read('cpu_example_120/result.json')
    check(Counter(ticket['subterrain'] for ticket in report['initial_assignments'])==SOURCE_RESET_BLOCK_COUNTS,
        'CPU initial 120-ticket quota differs')
    launch=read('launch_command.proposal.json')
    check(launch['status']=='proposed_not_executed' and launch['separate_user_approval_required'] is True,
        'runtime command falsely approved')
    check(not torch.cuda.is_initialized(),'CPU reviewer initialized CUDA')
    return dict(status='passed_120_env_G1_cpu_preparation_archive_review',hashed_archive_files=checked,
        registered_tests=824,new_120_env_tests=70,focused_tests=247,production_inputs=238,installed_dependencies=51,
        predecessor_60_and_32_and_16_env_verified=True,historical_16_and_32_and_60_env_acceptance_unchanged=True,
        production_budget_unchanged=True,real_120_env_G1_passed=False,real_source_transitions=0,
        real_source_ppo_updates=0,gpu_queries=0,gpu_jobs=0,isaac_sim_launches=0,
        cuda_initialized=False,formal_training_ready=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=HERE.parents[1])
    args=parser.parse_args();result=main(args.source_root.resolve())
    (HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
