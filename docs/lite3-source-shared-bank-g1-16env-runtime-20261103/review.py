"""Verify the real G1 archive on CPU; source hashes bind the executed verifier."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent


def check(condition,message):
    if not condition:raise ValueError(message)


def main(source_root):
    raw=HERE/'raw'
    def read(name):return json.loads((raw/name).read_text())
    provenance=json.loads((HERE/'provenance.json').read_text())
    checked=0
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(HERE/name).resolve()
        check(path.is_relative_to(HERE),'archive path escaped root')
        check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'archive SHA mismatch: '+name)
        checked+=1
    for name,value in provenance['files'].items():
        data=(HERE/name).read_bytes()
        check(len(data)==value['bytes'] and hashlib.sha256(data).hexdigest()==value['sha256'],'original bytes differ: '+name)
    plan=read('run_manifest.json');frozen=read('frozen_proposal.json');approved=read('approved_manifest.json')
    check(plan['base_revision']==plan['remote_revision_verified']=='8b2aec0230c4dd2e85dc085a0cdf8cb418f0f5e9','executed revision differs')
    check(hashlib.sha256((raw/'frozen_proposal.json').read_bytes()).hexdigest()==
        '5e83df3a37c0bb1c542622703ccd54f71630cb82f8ec89afd41bc39ff4e4c7de','frozen proposal changed')
    check(frozen['user_approved'] is False and frozen['status']=='proposed','proposal approval state differs')
    check(approved==frozen|{'status':'approved','user_approved':True} and plan==approved,'approved scope changed')
    check(read('approval.json')['proposal_sha256']==hashlib.sha256((raw/'frozen_proposal.json').read_bytes()).hexdigest(),
        'user approval no longer binds frozen proposal')
    check(read('input_hashes_before.json')==read('input_hashes_after.json')==plan['input_sha256']
        and len(plan['input_sha256'])==238,'production source inputs changed')
    check(read('dependencies_before.json')==read('dependencies_after.json')==plan['dependency_sha256']
        and len(plan['dependency_sha256'])==51,'installed dependencies changed')
    for name,digest in plan['input_sha256'].items():
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,
            'executed source differs; use --source-root for revision 8b2aec0: '+name)
    for name,digest in plan['dependency_sha256'].items():
        check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'audited runtime dependency differs: '+name)
    before=(raw/'production_ledger_before.json').read_bytes();after=(raw/'production_ledger_after.json').read_bytes()
    check(before==after and hashlib.sha256(before).hexdigest()==plan['budget_ledger']['sha256'],'production budget changed')
    ledger=json.loads(after)
    check(len(ledger['entries'])==1 and ledger['entries'][0]['state']=='accepted'
        and ledger['entries'][0]['reserved_transitions']==384 and ledger['entries'][0]['completed_updates']==2,
        'prior real PPO diagnostic charge changed')
    sys.path[:0]=[str(source_root/'source/rl_training'),str(source_root/'scripts/tools')]
    from source_scale_startup_validation import validate
    from rl_training.source_scan.sparse_contact import reconstruct_frame
    import torch
    check(not torch.cuda.is_initialized(),'CPU reviewer initialized CUDA')
    execution=read('execution.json');acceptance=validate(raw,plan,execution)
    check(acceptance['accepted'] and acceptance==read('acceptance.json')==read('independent_acceptance.json'),
        'independent acceptance differs from the parent')
    check(execution['returncode']==0 and execution['process_group_empty'] is True and execution['monitor_errors']==[],
        'runtime process evidence incomplete')
    events=[json.loads(line) for line in (raw/'events.jsonl').read_text().splitlines()]
    topology={k:v for k,v in next(value for value in events if value['phase']=='sparse_contact_topology').items() if k!='phase'}
    frames=[value for value in events if value['phase']=='sparse_contact_sample']
    points=[]
    for frame in frames:
        decoded=reconstruct_frame(frame,topology);points.append(decoded['used_contact_count'])
        check(not decoded['matrix'].any() and not decoded['net'].any() and all(v==0 for v in frame['normal_forces_N']),
            'unexpected positive-force scope; re-review actual evidence')
    check(points==[8,44,8,44],'measured detailed contact counts differ')
    check(acceptance['ground_positive_qualification'] is False and not any(acceptance['ground_contact_points'].values()),
        'G1 falsely reports supporting-force qualification')
    report=read('result.json');cleanup=read('result.cleanup.json')
    check(cleanup['environment']=='returned' and cleanup['application'] in ('closing','returned')
        and cleanup['persistence_errors']==[],'cleanup failed')
    samples=[json.loads(line) for line in (raw/'gpu_samples.jsonl').read_text().splitlines()]
    gpu0=[next(csv.reader([line])) for sample in samples for line in sample['gpu'].splitlines() if line.startswith('0,')]
    used=[int(row[3].strip().split()[0]) for row in gpu0];free=[int(row[4].strip().split()[0]) for row in gpu0]
    check(len(samples)==3 and max(used)==7637 and min(free)==16488,'resource samples differ')
    check('3978576' not in (raw/'compute_after.csv').read_text(),'runtime worker remains in final GPU sample')
    log=(raw/'runtime.log').read_text(errors='replace')
    check(log.count('Unresolved reference prim path')==64 and log.count('Skipping NVIDIA GPU due CUDA being in bad state')==6,
        'warning inventory differs')
    check('Traceback (most recent call last):' not in log and '[Error]' not in log,'runtime console contains a failure')
    check(read('postflight.json')['process_group_empty'] is True and read('postflight.json')['checkpoint_unchanged'] is True,
        'postflight integrity check missing')
    check(not torch.cuda.is_initialized(),'CPU reviewer started a GPU context')
    return dict(status='passed_bounded_16_env_G1_archive_review',hashed_archive_files=checked,
        executed_revision=plan['base_revision'],production_inputs=238,installed_dependencies=51,
        accepted=True,real_read_only_diagnostic_transitions=16,physics_substeps=4,
        real_source_training_transitions=0,real_source_ppo_updates=0,runtime_gpu_launches=1,review_gpu_launches=0,
        contact_point_samples=sum(points),all_recorded_contact_forces_zero=True,ground_positive_qualified=False,
        production_budget_unchanged=True,training_ready=False,g3_g4_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=HERE.parents[1])
    args=parser.parse_args();result=main(args.source_root.resolve())
    (HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
