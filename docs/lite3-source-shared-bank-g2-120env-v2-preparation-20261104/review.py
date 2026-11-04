"""CPU review of the G2 cache-baseline repair and unapproved v2 preparation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent


def check(condition,message):
    if not condition:raise ValueError(message)


def main(source_root):
    raw=HERE/'raw';read=lambda name:json.loads((raw/name).read_text())
    listed=set()
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(HERE/name).resolve()
        check(name not in listed and path.is_relative_to(HERE),'invalid checksum path: '+name)
        listed.add(name);check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'archive changed: '+name)
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
        and p.name not in ('SHA256SUMS','review.json') and '__pycache__' not in p.parts}
    check(actual==listed,'archive checksum coverage differs')
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,record in provenance['files'].items():
        payload=(HERE/name).read_bytes()
        if record.get('encoding')=='gzip':payload=gzip.decompress(payload)
        check(len(payload)==record['original_bytes'] and hashlib.sha256(payload).hexdigest()==record['original_sha256'],
            'original evidence differs: '+name)
    inputs=read('input_hashes_after.json');deps=read('dependencies_after.json')
    check(len(inputs)==243 and len(deps)==51,'source/dependency inventory incomplete')
    changed={name for name,value in inputs.items() if read('input_hashes_before.json').get(name)!=value}
    expected_changed={'source/rl_training/rl_training/source_scan/scale_diagnostic.py',
        'source/rl_training/rl_training/source_scan/scale_reset.py','scripts/tools/source_scale_reset_collection.py',
        'scripts/tools/source_scale_reset_probe.py','scripts/tools/source_scale_reset_validation.py'}
    check(changed==expected_changed,'unexpected production source changes')
    check(deps==read('dependencies_before.json'),'installed dependencies changed')
    for name,digest in inputs.items():check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'source differs: '+name)
    for name,digest in read('test_source_sha256.json').items():
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'test source differs: '+name)
    for name,digest in deps.items():check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'dependency differs: '+name)
    before=(raw/'production_ledger_before.json').read_bytes();reference=read('production_ledger_reference.json')
    check(before==(raw/'production_ledger_after.json').read_bytes()
        and hashlib.sha256(before).hexdigest()==reference['sha256'],'production budget changed')
    check(hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest()==reference['sha256'],'live production budget changed')
    for name,digest in read('prior_archive_sha256.json').items():
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'prior archive changed: '+name)
    for name,value in read('historical_G2_reviews.json').items():
        check(value==json.loads((source_root/'docs'/name/'review.json').read_text()),'historical G2 review differs')
    prep=read('preparation.json')
    for name in ('gpu_queries','gpu_jobs','isaac_sim_launches','real_source_transitions','real_source_ppo_updates'):
        check(prep[name]==0,'runtime work outside CPU approval: '+name)
    check(prep['real_120_env_G2_passed'] is False and prep['formal_training_ready'] is False,'CPU work claims GPU qualification')
    expected_tests=[('initial_red',2,1),('focused',120,0),('regression',prep['registered_tests'],0)]
    for name,count,failures in expected_tests:
        suite=ET.parse(raw/(name+'.junit.xml')).getroot().find('testsuite')
        check(suite is not None and int(suite.attrib['tests'])==count and int(suite.attrib['failures'])==failures
            and int(suite.attrib['errors'])==int(suite.attrib['skipped'])==0,'test result differs: '+name)
        execution=read(name+'_execution.json')
        check(execution['returncode']==int(failures>0) and execution['environment_overrides']['CUDA_VISIBLE_DEVICES']==''
            and execution['gpu_jobs']==execution['isaac_sim_launches']==0,'CPU test execution differs: '+name)
    full=ET.parse(raw/'regression.junit.xml').getroot().find('testsuite')
    check(read('regression_execution.json')['registered_test_files']==27 and sum(
        value.attrib.get('classname','').endswith(('.test_scale_reset','.test_scale_reset_transaction','.test_scale_reset_contact_cache'))
        for value in full.findall('testcase'))==127,'registered file/G2 test count differs')
    check('reset changed untouched buffer: sensor/contact_forces/_is_outdated' in (raw/'initial_red.log').read_text(),
        'pre-fix regression did not reproduce the original error')
    sys.path[:0]=[str(source_root/'source/rl_training'),str(source_root/'scripts/tools')]
    from rl_training.source_scan.scale_reset import contract_for,same_value,validate_predecessor
    from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
    from source_scale_reset_validation import validate
    from source_parallel_reset_lifecycle import check_transition
    from rl_training.source_scan.moving_diagnostic import restore_tensor
    import torch
    draft=read('g2.v2.proposal.draft.json')
    check(draft['status']=='proposed' and draft['user_approved'] is False
        and draft['remote_revision_verified'] is None,'working draft falsely frozen or approved')
    for name,value in contract_for().items():check(same_value(draft.get(name),value),'v2 draft contract differs: '+name)
    check(draft['input_sha256']==inputs and draft['dependency_sha256']==deps and draft['budget_ledger']==reference,'draft inputs differ')
    check(not Path(draft['output_dir']).exists(),'new GPU attempt directory already exists')
    validate_predecessor(draft['predecessor_qualification'],source_root=source_root)
    checkpoint=read('checkpoint_reference.json')
    check(checkpoint['sha256']==CHECKPOINT_SHA256 and hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest()==CHECKPOINT_SHA256,
        'original checkpoint changed')
    failure=source_root/'docs/lite3-source-shared-bank-g2-120env-runtime-20261104/raw/events.jsonl'
    events=[json.loads(line) for line in failure.read_text().splitlines()]
    a=next(v['before'] for v in events if v['phase']=='scale_reset_attempt')
    b=next(v['after'] for v in events if v['phase']=='scale_reset_after')
    try:check_transition({k:restore_tensor(v) for k,v in a.items()},{k:restore_tensor(v) for k,v in b.items()},[0])
    except AssertionError as error:check(str(error)=='reset changed untouched buffer: sensor/contact_forces/_is_outdated','original rejection differs')
    else:raise ValueError('original GPU failure was incorrectly accepted')
    for name,accepted in [('cpu_example',True),('cpu_partial_failure',False)]:
        with tempfile.TemporaryDirectory(prefix='lite3-g2-v2-cpu-review-') as directory:
            folder=Path(directory)
            for path in (raw/name).iterdir():
                if path.name.endswith('.gz'):(folder/path.name.removesuffix('.gz')).write_bytes(gzip.decompress(path.read_bytes()))
                else:shutil.copyfile(path,folder/path.name)
            result=validate(folder,read(name+'/fixture_manifest.json'),read(name+'/fixture_execution_inputs.json'))
            check(result==read(name+'/fixture_acceptance.json') and result['accepted'] is accepted,'CPU artifact acceptance differs: '+name)
            report=read(name+'/result.json')
            if not accepted:
                check(report['reset_attempts']==2 and report['reset_batches']==1 and report['simulation_control_steps']==1
                    and report['motion_samples']==4 and report['completed_diagnostic_transitions']==120,'partial progress lost')
                check('reset changed untouched buffer: root' in result['reasons'][0]
                    and result['probe_error']==report['error'],'original failure was obscured')
            check(read(name+'/provenance.json')['synthetic_cpu_fixture'] is True,'CPU artifact mislabeled')
    launch=read('launch_command.proposal.json')
    check(launch['status']=='proposed_not_executed' and launch['separate_user_approval_required'] is True,'GPU launch falsely approved')
    check(not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled(),'CPU review initialized CUDA or leaked inference mode')
    return dict(status='passed_120_env_G2_v2_cpu_repair_preparation_review',archive_review_passed=True,
        hashed_archive_files=len(listed),registered_tests=prep['registered_tests'],registered_test_files=27,
        production_inputs=243,changed_production_inputs=5,installed_dependencies=51,
        original_GPU_failure_still_rejected=True,actual_adapter_callback_covered=True,
        partial_failure_progress_verified=True,historical_G2_archives_unchanged=True,production_budget_unchanged=True,
        real_120_env_G2_passed=False,gpu_queries=0,gpu_jobs=0,isaac_sim_launches=0,real_source_transitions=0,
        real_source_ppo_updates=0,cuda_initialized=False,formal_training_ready=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,default=HERE.parents[1]);args=parser.parse_args()
    result=main(args.source_root.resolve());(HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
