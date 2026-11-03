"""CPU review of the single real 120-env G1 attempt, without replaying physics."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def check(condition,message):
    if not condition:raise ValueError(message)


def main(source_root):
    raw=HERE/'raw'
    def read(name):return json.loads((raw/name).read_text())
    provenance=json.loads((HERE/'provenance.json').read_text());checked=0
    lines=(HERE/'SHA256SUMS').read_text().splitlines()
    for line in lines:
        digest,name=line.split('  ',1);path=(HERE/name).resolve()
        check(path.is_relative_to(HERE),'archive path escaped root')
        check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'archive SHA mismatch: '+name);checked+=1
    listed={line.split('  ',1)[1] for line in lines}
    actual={str(path.relative_to(HERE)) for path in HERE.rglob('*') if path.is_file()
        and path.name not in ('SHA256SUMS','review.json') and '__pycache__' not in path.parts}
    check(listed==actual and len(listed)==checked,'archive checksum coverage differs')
    for name,value in provenance['files'].items():
        payload=(HERE/name).read_bytes()
        check(len(payload)==value['bytes'] and hashlib.sha256(payload).hexdigest()==value['sha256'],'original bytes differ: '+name)
    frozen=read('frozen_proposal.json');approved=read('approved_manifest.json');plan=read('run_manifest.json')
    digest=hashlib.sha256((raw/'frozen_proposal.json').read_bytes()).hexdigest()
    check(digest=='2a653d981dc31a5f3157062d82933da210ec5ff5e702af8fed6b74c4f295e67b','frozen proposal changed')
    check(frozen['user_approved'] is False and frozen['status']=='proposed','proposal approval state differs')
    check(approved==frozen|{'status':'approved','user_approved':True} and plan==approved,'approved scope changed')
    check(plan['base_revision']==plan['remote_revision_verified']=='a455b59aa9e429b3c210e731432bf4cc96d23be0','executed revision differs')
    approval=read('approval.json')
    check(approval['proposal_sha256']==digest and approval['approved_manifest_sha256']==
        hashlib.sha256((raw/'approved_manifest.json').read_bytes()).hexdigest(),'approval does not bind frozen scope')
    check(read('input_hashes_before.json')==read('input_hashes_after.json')==plan['input_sha256']
        and len(plan['input_sha256'])==238,'production inputs changed')
    check(read('dependencies_before.json')==read('dependencies_after.json')==plan['dependency_sha256']
        and len(plan['dependency_sha256'])==51,'installed dependencies changed')
    for name,digest in plan['input_sha256'].items():
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,
            'source differs; use --source-root for revision a455b59: '+name)
    for name,digest in plan['dependency_sha256'].items():
        check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'audited dependency differs: '+name)
    before=(raw/'production_ledger_before.json').read_bytes();after=(raw/'production_ledger_after.json').read_bytes()
    check(before==after and hashlib.sha256(before).hexdigest()==plan['budget_ledger']['sha256'],'production budget changed')
    ledger=json.loads(after)
    check(len(ledger['entries'])==1 and ledger['entries'][0]['state']=='accepted'
        and ledger['entries'][0]['reserved_transitions']==384 and ledger['entries'][0]['completed_updates']==2,'prior PPO charge changed')
    invocation=read('supervisor_execution.json')
    check(invocation['single_supervisor_invocation'] is True and invocation['automatic_retry'] is False
        and invocation['returncode']==0 and invocation['command'].count('run')==1,'supervisor invocation differs')
    command=invocation['command']
    check(command[command.index('--num-envs')+1]=='120'
        and invocation['environment_overrides']['CUDA_VISIBLE_DEVICES']=='0','selected launch scale/device differs')
    check(invocation['environment_overrides']['GIT_CONFIG_VALUE_0']=='https://github.com/','Git transport mapping differs')
    audit=read('attempt_audit.json')
    check(audit['output_existed_before'] is False and audit['approved_file_existed_before'] is False
        and audit['prior_matching_supervisor_or_child_found'] is False,'initial attempt audit differs')
    preflight=read('preflight.json')
    check(preflight['runtime_GPU_launches']==0 and preflight['isaac_sim_launches']==0
        and preflight['cuda_initialized'] is False,'CPU preflight mislabeled as runtime attempt')
    sys.path[:0]=[str(source_root/'source/rl_training'),str(source_root/'scripts/tools')]
    from source_scale_startup_validation import validate
    from rl_training.source_scan.sparse_contact import reconstruct_frame
    from rl_training.source_scan.scale_startup import validate_predecessor
    from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
    from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
    import torch
    check(not torch.cuda.is_initialized(),'CPU reviewer initialized CUDA')
    validate_predecessor(120,plan['predecessor_qualification'],source_root=source_root)
    execution=read('execution.json');acceptance=validate(raw,plan,execution)
    check(acceptance['accepted'] and acceptance==read('acceptance.json')==read('independent_acceptance.json'),
        'CPU acceptance differs from runtime parent')
    report=read('result.json')
    check(Counter(t['subterrain'] for t in report['initial_assignments'])==SOURCE_RESET_BLOCK_COUNTS
        and {t['column'] for t in report['initial_assignments']}==set(range(15)),
        'initial 120-ticket block differs from registered quotas')
    events=[json.loads(line) for line in (raw/'events.jsonl').read_text().splitlines()]
    topology={k:v for k,v in next(v for v in events if v['phase']=='sparse_contact_topology').items() if k!='phase'}
    check((topology['sensor_count'],topology['filter_count'],topology['capacity'])==(2040,2064,1048576),'runtime topology differs')
    points=[]
    for frame in [v for v in events if v['phase']=='sparse_contact_sample']:
        decoded=reconstruct_frame(frame,topology);points.append(decoded['used_contact_count'])
        check(not decoded['matrix'].any() and not decoded['net'].any() and all(v==0 for v in frame['normal_forces_N']),
            'positive-force scope differs; re-review actual data')
        for (i,j),count in zip(frame['pair_indices'],frame['pair_counts']):
            target=topology['target_paths'][topology['filter_orders'][topology['sensor_filter_order'][i]][j]]
            check(target in topology['ground_colliders'] and count>0,'contact class differs')
    check(points==[48,376,48,376],'detailed point counts differ')
    check(acceptance['ground_positive_qualification'] is False and not any(acceptance['ground_contact_points'].values()),
        'G1 falsely claims supporting-force qualification')
    metrics=read('derived_metrics.json')
    check([v['detailed_points'] for v in metrics['frames']]==points
        and metrics['events_bytes']==(raw/'events.jsonl').stat().st_size,'derived contact summary differs')
    samples=[json.loads(line) for line in (raw/'gpu_samples.jsonl').read_text().splitlines()]
    gpu0=[next(csv.reader([line])) for sample in samples for line in sample['gpu'].splitlines() if line.startswith('0,')]
    check(len(samples)==11 and max(int(r[3].strip().split()[0]) for r in gpu0)==8956
        and min(int(r[4].strip().split()[0]) for r in gpu0)==15169,'GPU resource samples differ')
    post=read('postflight.json');pids=set(post['own_sampled_worker_pids'])
    own=[row for sample in samples for row in csv.reader(sample['processes'].splitlines()) if row and int(row[1]) in pids]
    check(pids=={131533} and max(int(row[3].strip().split()[0]) for row in own)==2952,'own GPU worker samples differ')
    after_pids={int(row[1]) for row in csv.reader((raw/'compute_after.csv').read_text().splitlines()) if row}
    check(not (pids&after_pids) and post['process_group']==execution['process_group']==131532
        and post['process_group_empty'] is True and post['own_worker_pids_absent'] is True,'postflight cleanup incomplete')
    check(post['checkpoint_unchanged'] is True and post['checkpoint_sha256']==CHECKPOINT_SHA256
        and hashlib.sha256(Path(post['checkpoint_path']).read_bytes()).hexdigest()==CHECKPOINT_SHA256
        and post['predecessor_60_and_32_and_16_env_verified'] is True,'postflight inputs incomplete')
    log=(raw/'runtime.log').read_text(errors='replace')
    check(log.count('Unresolved reference prim path')==480 and log.count('Skipping NVIDIA GPU due CUDA being in bad state')==6,
        'warning inventory differs')
    for name in ('runtime.log','kit.log','isaaclab.log'):
        text=(raw/name).read_text(errors='replace')
        check('Traceback (most recent call last):' not in text and '[Error]' not in text,'runtime log has a failure: '+name)
    check(not torch.cuda.is_initialized(),'CPU reviewer started a GPU context')
    return dict(status='passed_bounded_120_env_G1_archive_review',hashed_archive_files=checked,
        executed_revision=plan['base_revision'],production_inputs=238,installed_dependencies=51,accepted=True,
        real_read_only_diagnostic_transitions=120,physics_substeps=4,real_source_training_transitions=0,
        real_source_ppo_updates=0,runtime_GPU_launches=1,review_GPU_launches=0,
        predecessor_60_and_32_and_16_env_verified=True,initial_120_ticket_block_verified=True,contact_point_samples=sum(points),all_recorded_contact_forces_zero=True,
        ground_positive_qualified=False,production_budget_unchanged=True,training_ready=False,g3_g4_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=HERE.parents[1])
    args=parser.parse_args();result=main(args.source_root.resolve())
    (HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
