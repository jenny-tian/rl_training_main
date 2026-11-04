"""Recheck the one real 120-env G2 v2 run on CPU; never launch a simulator."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

HERE=Path(__file__).resolve().parent
REVISION='717f21fbfe7d08b957208d487b8e97f0c3af0bea'
PROPOSAL_SHA256='797a8defa878684b24e19134d2a065e6010411227ef81608dc7d88edc63ed6ec'


def check(condition,message):
    if not condition:raise ValueError(message)


def main(source_root):
    raw=HERE/'raw';read=lambda name:json.loads((raw/name).read_text())
    listed=set()
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(HERE/name).resolve()
        check(name not in listed and path.is_relative_to(HERE),'invalid checksum path')
        listed.add(name);check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'archive changed: '+name)
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
        and p.name not in ('SHA256SUMS','review.json') and '__pycache__' not in p.parts}
    check(actual==listed,'checksum coverage differs')
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,record in provenance['files'].items():
        payload=(HERE/name).read_bytes()
        if record.get('encoding')=='gzip':payload=gzip.decompress(payload)
        check(len(payload)==record['original_bytes'] and hashlib.sha256(payload).hexdigest()==record['original_sha256'],
            'original evidence differs: '+name)
    frozen=read('frozen_proposal.json');plan=read('run_manifest.json')
    check(hashlib.sha256((raw/'frozen_proposal.json').read_bytes()).hexdigest()==PROPOSAL_SHA256,'frozen scope changed')
    check(frozen['status']=='proposed' and frozen['user_approved'] is False
        and plan==read('approved_manifest.json')==frozen|dict(status='approved',user_approved=True),'approval changed scope')
    check(plan['base_revision']==plan['remote_revision_verified']==REVISION,'executed revision differs')
    check(read('approval.json')['proposal_sha256']==PROPOSAL_SHA256,'approval binding differs')
    preflight=read('preflight.json')
    check(preflight['status']=='passed' and preflight['revision']==preflight['remote_revision']==REVISION
        and preflight['working_tree_clean'] is True and preflight['repository_private'] is True
        and preflight['proposal_sha256']==PROPOSAL_SHA256 and preflight['gpu0_free_mib']>=12288
        and preflight['disk_free_bytes']>=4294967296 and preflight['cuda_initialized'] is False,'preflight evidence differs')
    for before,after,key,count in [('input_hashes_before.json','input_hashes_after.json','input_sha256',243),
            ('dependencies_before.json','dependencies_after.json','dependency_sha256',51)]:
        check(read(before)==read(after)==plan[key] and len(plan[key])==count,'input/dependency snapshots changed')
    for name,digest in plan['input_sha256'].items():
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'use original 717f21f source: '+name)
    for name,digest in plan['dependency_sha256'].items():
        check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'installed dependency changed: '+name)
    before=(raw/'production_ledger_before.json').read_bytes()
    check(before==(raw/'production_ledger_after.json').read_bytes()
        and hashlib.sha256(before).hexdigest()==plan['budget_ledger']['sha256'],'production budget changed')
    ledger=json.loads(before)
    check(len(ledger['entries'])==1 and ledger['entries'][0]['state']=='accepted'
        and ledger['entries'][0]['reserved_transitions']==384 and ledger['entries'][0]['completed_updates']==2,'prior PPO charge differs')
    prep=source_root/'docs/lite3-source-shared-bank-g2-120env-v2-preparation-20261104'
    for name,digest in [('SHA256SUMS','7e8cae12ebd14b47b1fa67262d06087f6353cca3b3b536838a18c5b688a0fb44'),
            ('review.json','3965787b8f312c0b843413525bccaf393d7fe24b44d996f7563a4089ad45840a')]:
        check(hashlib.sha256((prep/name).read_bytes()).hexdigest()==digest,'CPU preparation anchor changed')
    for line in (prep/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);check(hashlib.sha256((prep/name).read_bytes()).hexdigest()==digest,'CPU preparation evidence changed')
    check(read('preparation_review.json')==json.loads((prep/'review.json').read_text()),'preflight preparation review differs')
    for name,digest in json.loads((prep/'raw/prior_archive_sha256.json').read_text()).items():
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'prior G2 archive changed: '+name)
    check(read('input_hashes_before.json')==json.loads((prep/'raw/input_hashes_after.json').read_text()),'runtime source differs from tested repair')
    audit=read('attempt_audit.json');invocation=read('supervisor_execution.json');execution=read('execution.json');post=read('postflight.json')
    check(audit['output_existed_before'] is False and audit['approved_file_existed_before'] is False
        and audit['prior_matching_supervisor_or_child_found'] is False and audit['old_v1_output_preserved'] is True,'attempt audit differs')
    check(invocation['single_supervisor_invocation'] is True and invocation['automatic_retry'] is False
        and invocation['returncode']==0 and invocation['watchdog_exceeded'] is False,'supervisor invocation differs')
    check(invocation['command']==json.loads((prep/'raw/launch_command.proposal.json').read_text())['argv'],'launch command changed')
    check(execution['returncode']==0 and execution['process_group_empty'] is True and execution['timed_out'] is False
        and execution['monitor_errors']==[] and '480s' in execution['command'] and '--kill-after=15s' in execution['command'],
        'process or timeout evidence differs')
    ticket=read('supervision.json')
    check(ticket==dict(manifest_sha256=hashlib.sha256((raw/'run_manifest.json').read_bytes()).hexdigest(),revision=REVISION),'child ticket differs')
    sys.path[:0]=[str(source_root/'source/rl_training'),str(source_root/'scripts/tools')]
    from source_scale_reset_validation import validate
    from rl_training.source_scan.scale_reset import validate_predecessor
    from rl_training.source_scan.moving_diagnostic import restore_tensor
    import torch
    validate_predecessor(plan['predecessor_qualification'],source_root=source_root)
    with tempfile.TemporaryDirectory(prefix='lite3-G2-v2-CPU-review-') as directory:
        folder=Path(directory)
        for path in raw.iterdir():
            if path.name.endswith('.gz'):(folder/path.name.removesuffix('.gz')).write_bytes(gzip.decompress(path.read_bytes()))
            else:shutil.copyfile(path,folder/path.name)
        accepted=validate(folder,plan,execution)
        check(accepted==read('acceptance.json')==read('independent_acceptance.json') and accepted['accepted'] is True,'independent G2 acceptance differs')
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    metrics=read('derived_metrics.json');report=read('result.json')
    check(metrics['event_counts']==dict(Counter(event['phase'] for event in events)),'event counts differ')
    attempts=[v for v in events if v['phase']=='scale_reset_attempt']
    check(len(attempts)==4 and all(len(v['before'])==42
        and not bool(restore_tensor(v['before']['sensor/contact_forces/_is_outdated']).any()) for v in attempts),
        'actual cache-baseline repair evidence differs')
    check(report['observation_history_terms']==[] and report['completed_diagnostic_transitions']==480
        and report['reset_attempts']==report['reset_batches']==report['simulation_control_steps']==4,'actual progress differs')
    check(accepted['assignment_count']==304 and accepted['complete_assignment_blocks']==2
        and accepted['tail_assignment_count']==64 and accepted['selected_environment_resets']==184,'ticket count differs')
    frames=[v for v in events if v['phase']=='sparse_contact_sample']
    check([v['used_contact_count'] for v in frames]==metrics['contact_used_points_by_substep']
        and sum(len(v['normal_forces_N']) for v in frames)==3194,'contact point count differs')
    positive_envs=sorted(int(k) for k,v in report['contact_capture']['ground_contact_points'].items() if v)
    check(sum(report['contact_capture']['ground_contact_points'].values())==metrics['positive_ground_points_total']==92
        and positive_envs==metrics['positive_ground_environment_ids'] and len(positive_envs)==23
        and max(abs(v) for frame in frames for v in frame['normal_forces_N'])==metrics['max_absolute_normal_force_N'],
        'limited positive-force evidence differs')
    rows=lambda text:[[v.strip() for v in row] for row in csv.reader(text.splitlines()) if row]
    samples=[json.loads(line) for line in (raw/'gpu_samples.jsonl').read_text().splitlines()]
    gpu0=[next(v for v in rows(sample['gpu']) if v[0]=='0') for sample in samples]
    worker=post['worker_pid'];kit=(raw/'kit.log').read_text(errors='replace')
    check(set(map(int,re.findall(r'launching the process .* from process (\d+)\.',kit)))=={worker},'worker identity differs')
    own=[int(v[3].split()[0]) for sample in samples for v in rows(sample['processes']) if v[1]==str(worker)]
    check(len(samples)==19 and len(own)==15 and max(own)==metrics['worker_sampled_peak_mib']==2954,'worker memory samples differ')
    check(max(int(v[3].split()[0]) for v in gpu0)==metrics['gpu0_sampled_peak_mib']==9094
        and min(int(v[4].split()[0]) for v in gpu0)==metrics['gpu0_sampled_min_free_mib']==15031,'card memory samples differ')
    check(post['process_group_empty'] is True and post['worker_absent_from_proc'] is True
        and post['worker_absent_from_gpu_table'] is True and post['worker_pid_from_kit_log'] is True,'cleanup evidence differs')
    check(str(worker) not in {v[1] for v in rows((raw/'compute_postflight.csv').read_text())},'worker remained in GPU table')
    check(hashlib.sha256(Path(post['checkpoint_path']).read_bytes()).hexdigest()==plan['checkpoint_sha256'],'original checkpoint changed')
    for name,record in read('log_paths.json').items():
        payload=(raw/name).read_bytes();text=payload.decode(errors='replace')
        check(len(payload)==record['bytes'] and hashlib.sha256(payload).hexdigest()==record['sha256'],'original log differs: '+name)
        check(text.count('[Error]')==text.count('Traceback (most recent call last):')==0,'runtime error or traceback present')
    check(not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled(),'CPU review initialized CUDA or leaked inference mode')
    return dict(status='passed_bounded_120_env_G2_v2_runtime_archive_review',accepted=True,executed_revision=REVISION,
        hashed_archive_files=len(listed),production_inputs=243,installed_dependencies=51,runtime_GPU_launches=1,review_GPU_launches=0,
        real_read_only_diagnostic_transitions=480,reset_batches=4,selected_environment_resets=184,physics_substeps=16,
        assignment_count=304,complete_assignment_blocks=2,tail_assignment_count=64,actual_snapshot_fields=42,
        reset_boundary_qualified=True,reset_first_recovery_step_qualified=True,production_budget_unchanged=True,
        previous_failed_G2_preserved=True,real_source_training_transitions=0,real_source_ppo_updates=0,
        ground_positive_qualified=False,g3_g4_qualified=False,training_ready=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,default=HERE.parents[1]);args=parser.parse_args()
    result=main(args.source_root.resolve());(HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
