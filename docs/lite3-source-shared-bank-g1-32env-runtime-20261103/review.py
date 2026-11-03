"""CPU review of the single real 32-env G1 attempt, without replaying physics."""
import argparse
import csv
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
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(HERE/name).resolve()
        check(path.is_relative_to(HERE),'archive path escaped root')
        check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'archive SHA mismatch: '+name);checked+=1
    for name,value in provenance['files'].items():
        payload=(HERE/name).read_bytes()
        check(len(payload)==value['bytes'] and hashlib.sha256(payload).hexdigest()==value['sha256'],'original bytes differ: '+name)
    frozen=read('frozen_proposal.json');approved=read('approved_manifest.json');plan=read('run_manifest.json')
    check(hashlib.sha256((raw/'frozen_proposal.json').read_bytes()).hexdigest()==
        'a29e86b23d6e037564a21c7e061ce931e806727d1def593cc3cfc38de3a124e3','frozen proposal changed')
    check(frozen['user_approved'] is False and frozen['status']=='proposed','proposal approval state differs')
    check(approved==frozen|{'status':'approved','user_approved':True} and plan==approved,'approved scope changed')
    check(plan['base_revision']==plan['remote_revision_verified']=='7cb2c14b7a07067240c7812eb26bbf5808b0e6f0','executed revision differs')
    check(read('approval.json')['proposal_sha256']==hashlib.sha256((raw/'frozen_proposal.json').read_bytes()).hexdigest(),
        'approval does not bind the frozen proposal')
    check(read('input_hashes_before.json')==read('input_hashes_after.json')==plan['input_sha256']
        and len(plan['input_sha256'])==238,'production inputs changed')
    check(read('dependencies_before.json')==read('dependencies_after.json')==plan['dependency_sha256']
        and len(plan['dependency_sha256'])==51,'installed dependencies changed')
    for name,digest in plan['input_sha256'].items():
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,
            'source differs; use --source-root for revision 7cb2c14: '+name)
    for name,digest in plan['dependency_sha256'].items():
        check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'audited dependency differs: '+name)
    before=(raw/'production_ledger_before.json').read_bytes();after=(raw/'production_ledger_after.json').read_bytes()
    check(before==after and hashlib.sha256(before).hexdigest()==plan['budget_ledger']['sha256'],'production budget changed')
    ledger=json.loads(after)
    check(len(ledger['entries'])==1 and ledger['entries'][0]['state']=='accepted'
        and ledger['entries'][0]['reserved_transitions']==384 and ledger['entries'][0]['completed_updates']==2,'prior PPO charge changed')
    transport=read('preflight_transport_failure.json')
    check(transport['real_GPU_attempts_at_failure']==0 and transport['isaac_sim_launches_at_failure']==0
        and transport['worktree_or_repository_configuration_changed'] is False,'HTTPS failure mislabeled as simulation attempt')
    invocation=read('supervisor_execution.json')
    check(invocation['single_supervisor_invocation'] is True and invocation['automatic_retry'] is False
        and invocation['command'].count('run')==1,'supervisor invocation count differs')
    check(invocation['environment_overrides']['GIT_CONFIG_VALUE_0']=='https://github.com/','Git transport fallback differs')
    audit=read('interruption_audit.json')
    check(audit['output_existed_before'] is False and audit['prior_matching_supervisor_or_child_found'] is False,
        'interrupted-turn attempt audit differs')
    sys.path[:0]=[str(source_root/'source/rl_training'),str(source_root/'scripts/tools')]
    from source_scale_startup_validation import validate
    from rl_training.source_scan.sparse_contact import reconstruct_frame
    from rl_training.source_scan.scale_startup import validate_predecessor
    import torch
    check(not torch.cuda.is_initialized(),'CPU reviewer initialized CUDA')
    validate_predecessor(32,plan['predecessor_qualification'],source_root=source_root)
    execution=read('execution.json');acceptance=validate(raw,plan,execution)
    check(acceptance['accepted'] and acceptance==read('acceptance.json')==read('independent_acceptance.json'),
        'CPU acceptance differs from runtime parent')
    events=[json.loads(line) for line in (raw/'events.jsonl').read_text().splitlines()]
    topology={k:v for k,v in next(v for v in events if v['phase']=='sparse_contact_topology').items() if k!='phase'}
    check((topology['sensor_count'],topology['filter_count'],topology['capacity'])==(544,568,262144),'runtime topology differs')
    points=[]
    for frame in [v for v in events if v['phase']=='sparse_contact_sample']:
        decoded=reconstruct_frame(frame,topology);points.append(decoded['used_contact_count'])
        check(not decoded['matrix'].any() and not decoded['net'].any() and all(v==0 for v in frame['normal_forces_N']),
            'positive-force scope differs; re-review actual data')
    check(points==[20,96,20,96],'detailed point counts differ')
    check(acceptance['ground_positive_qualification'] is False and not any(acceptance['ground_contact_points'].values()),
        'G1 falsely claims supporting-force qualification')
    samples=[json.loads(line) for line in (raw/'gpu_samples.jsonl').read_text().splitlines()]
    gpu0=[next(csv.reader([line])) for sample in samples for line in sample['gpu'].splitlines() if line.startswith('0,')]
    check(len(samples)==5 and max(int(r[3].strip().split()[0]) for r in gpu0)==8762
        and min(int(r[4].strip().split()[0]) for r in gpu0)==15363,'GPU resource samples differ')
    check('4055214' not in (raw/'compute_after.csv').read_text(),'runtime worker remains in final GPU table')
    post=read('postflight.json')
    check(post['process_group_empty'] is True and post['own_worker_pids_absent'] is True
        and post['checkpoint_unchanged'] is True and post['predecessor_16_env_verified'] is True,'postflight incomplete')
    log=(raw/'runtime.log').read_text(errors='replace')
    check(log.count('Unresolved reference prim path')==128 and log.count('Skipping NVIDIA GPU due CUDA being in bad state')==6,
        'warning inventory differs')
    check('Traceback (most recent call last):' not in log and '[Error]' not in log,'runtime console has a failure')
    check(not torch.cuda.is_initialized(),'CPU reviewer started a GPU context')
    return dict(status='passed_bounded_32_env_G1_archive_review',hashed_archive_files=checked,
        executed_revision=plan['base_revision'],production_inputs=238,installed_dependencies=51,accepted=True,
        real_read_only_diagnostic_transitions=32,physics_substeps=4,real_source_training_transitions=0,
        real_source_ppo_updates=0,runtime_GPU_launches=1,review_GPU_launches=0,prelaunch_HTTPS_failure_GPU_launches=0,
        predecessor_16_env_verified=True,contact_point_samples=sum(points),all_recorded_contact_forces_zero=True,
        ground_positive_qualified=False,production_budget_unchanged=True,training_ready=False,g3_g4_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=HERE.parents[1])
    args=parser.parse_args();result=main(args.source_root.resolve())
    (HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
