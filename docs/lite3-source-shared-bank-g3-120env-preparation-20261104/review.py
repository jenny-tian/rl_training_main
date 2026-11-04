"""CPU-only review of G3 implementation, synthetic traces and unapproved scope."""
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
    raw=HERE/'raw';read=lambda n:json.loads((raw/n).read_text());listed=set()
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(HERE/name).resolve()
        check(name not in listed and path.is_relative_to(HERE),'invalid checksum path')
        listed.add(name);check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'archive changed: '+name)
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
        and p.name not in ('SHA256SUMS','review.json') and '__pycache__' not in p.parts}
    check(actual==listed,'archive checksum coverage differs')
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,value in provenance['files'].items():
        payload=(HERE/name).read_bytes();check(len(payload)==value['bytes']
            and hashlib.sha256(payload).hexdigest()==value['sha256'],'original evidence changed: '+name)
    before=read('input_hashes_before.json');after=read('input_hashes_after.json')
    added={'source/rl_training/rl_training/source_scan/scale_motion.py','source/rl_training/rl_training/source_scan/scale_motion_evidence.py',
        'scripts/tools/source_scale_motion_collection.py','scripts/tools/source_scale_motion_probe.py',
        'scripts/tools/source_scale_motion_validation.py','scripts/tools/source_scale_motion_supervisor.py'}
    check(len(before)==243 and len(after)==249 and set(after)-set(before)==added
        and all(after.get(k)==v for k,v in before.items()),'qualified G2 input changed')
    check(after==read('input_hashes_for_regression.json'),'final source changed during regression')
    for name,digest in after.items():check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'source differs: '+name)
    for name,digest in read('test_source_sha256.json').items():
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'test source differs: '+name)
    deps=read('dependencies_after.json');check(deps==read('dependencies_before.json') and len(deps)==51,'dependencies changed')
    for name,digest in deps.items():check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'dependency differs: '+name)
    ledger=(raw/'production_ledger_before.json').read_bytes();reference=read('production_ledger_reference.json')
    check(ledger==(raw/'production_ledger_after.json').read_bytes()
        and hashlib.sha256(ledger).hexdigest()==reference['sha256'],'production budget changed')
    entries=json.loads(ledger)['entries']
    check(len(entries)==1 and entries[0]['state']=='accepted' and entries[0]['reserved_transitions']==384
        and entries[0]['completed_updates']==2,'prior PPO charge changed')
    for name,digest in read('prior_archive_sha256.json').items():
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'prior archive changed: '+name)
    prep=read('preparation.json')
    check(prep['real_G3_qualified'] is False and prep['real_G4_qualified'] is False
        and prep['formal_training_ready'] is False and prep['new_output_created'] is False,'CPU preparation claims runtime qualification')
    for key in ('gpu_queries','gpu_jobs','isaac_sim_launches','real_source_transitions','real_source_ppo_updates'):
        check(prep[key]==0,'runtime work outside CPU approval: '+key)
    suites=[('focused.initial',61,0),('protocol',23,0),('streaming_hash',3,0),
        ('interrupted.regression',297,2),('interrupted2.regression',789,2),('regression',1043,0)]
    for name,count,returncode in suites:
        suite=ET.parse(raw/(name+'.junit.xml')).getroot().find('testsuite')
        check(int(suite.attrib['tests'])==count and all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped')),
            'test results differ: '+name)
        if name.startswith('interrupted'):
            check(read(name.replace('.regression','')+'.regression_execution.json')['returncode']==returncode,
                'interrupted regression status differs')
    execution=read('regression_execution.json')
    check(execution['returncode']==0 and execution['registered_test_files']==29
        and execution['environment_overrides']['CUDA_VISIBLE_DEVICES']=='' and execution['gpu_jobs']==execution['isaac_sim_launches']==0,
        'final regression execution differs')
    full=ET.parse(raw/'regression.junit.xml').getroot().find('testsuite')
    check(sum(v.attrib.get('classname','').endswith(('.test_scale_motion','.test_scale_motion_protocol'))
        for v in full.findall('testcase'))==92,'new G3 case count differs')
    sys.path[:0]=[str(source_root/'source/rl_training'),str(source_root/'scripts/tools')]
    from rl_training.source_scan.scale_motion import contract_for,same_value,validate_predecessor,G2,FLAT
    from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
    from source_scale_motion_validation import validate
    import torch
    validate_predecessor(G2,source_root=source_root)
    check(read('predecessor_reference.json')==dict(G2=G2,Flat=FLAT),'predecessor references differ')
    draft=read('g3.proposal.draft.json')
    check(draft['status']=='proposed' and draft['user_approved'] is False and draft['remote_revision_verified'] is None,
        'development draft falsely frozen/approved')
    for key,value in contract_for().items():check(same_value(draft.get(key),value),'G3 contract differs: '+key)
    check(draft['input_sha256']==after and draft['dependency_sha256']==deps and draft['budget_ledger']==reference,'draft inputs differ')
    checkpoint=read('checkpoint_reference.json')
    check(checkpoint['sha256']==CHECKPOINT_SHA256 and hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest()==CHECKPOINT_SHA256,
        'original checkpoint changed')
    for name,valid in [('cpu_example',True),('cpu_corrupt_actor',False)]:
        folder=raw/name;result=validate(folder,read(name+'/fixture_manifest.json'),
            read(name+'/fixture_execution_inputs.json'),Path(checkpoint['path']),cpu_profile=(5,4))
        check(result==read(name+'/fixture_acceptance.json') and result['accepted'] is False
            and result.get('cpu_evidence_validated',False) is valid,'CPU evidence acceptance differs: '+name)
        check(read(name+'/provenance.json')['synthetic_cpu_fixture'] is True,'synthetic trace mislabeled')
        if valid:
            check(result['deferred_final_recovery_ids']==[3] and result['physics_substeps']==20
                and result['diagnostic_transitions']==600,'CPU tail recovery accounting differs')
            real=validate(folder,read(name+'/fixture_manifest.json'),read(name+'/fixture_execution_inputs.json'),Path(checkpoint['path']))
            check(real['accepted'] is False and real['reasons'],'CPU profile falsely qualifies full G3')
        else:check('frozen actor output differs' in result['reasons'][0],'corrupt actor evidence accepted')
    estimate=read('evidence_budget_estimate.json')
    check(estimate['projected_uncompressed_bytes_under_G2_activity']<draft['events_uncompressed_byte_limit']
        and estimate['compression_ratio_not_guaranteed'] is True and estimate['throughput_qualification'] is False,'estimate misrepresented')
    launch=read('launch_command.proposal.json')
    check(launch['status']=='proposed_not_executed' and launch['separate_user_approval_required'] is True,'GPU launch falsely approved')
    check(not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled(),'CPU reviewer initialized CUDA or leaked inference mode')
    return dict(status='passed_120_env_G3_CPU_preparation_review',archive_review_passed=True,hashed_archive_files=len(listed),
        registered_tests=1043,registered_test_files=29,new_G3_tests=92,production_inputs=249,new_production_inputs=6,
        qualified_G2_inputs_unchanged=True,installed_dependencies=51,independent_Flat_prerequisite_bound=True,
        synthetic_tail_recovery_verified=True,corrupt_actor_evidence_rejected=True,production_budget_unchanged=True,
        real_G3_qualified=False,real_G4_qualified=False,formal_training_ready=False,gpu_queries=0,gpu_jobs=0,
        isaac_sim_launches=0,real_source_transitions=0,real_source_ppo_updates=0,cuda_initialized=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,default=HERE.parents[1]);args=parser.parse_args()
    result=main(args.source_root.resolve());(HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
