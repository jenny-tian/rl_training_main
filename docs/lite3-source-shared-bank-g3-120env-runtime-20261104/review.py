"""Review a failed G3 archive; this never grants runtime qualification."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main(folder,artifact_root=None):
    count=0
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        assert digest(folder/name)==expected,name
        count+=1
    load=lambda name:json.loads((folder/name).read_text())
    report=load('raw/run/result.json');execution=load('raw/run/execution.json')
    acceptance=load('raw/run/acceptance.json');post=load('raw/orchestration/postflight.json')
    before=load('raw/orchestration/input_hashes_before.json')
    assert before==load('raw/orchestration/input_hashes_after.json') and len(before)==249
    assert load('raw/orchestration/dependencies_before.json')==load('raw/orchestration/dependencies_after.json')
    assert len(load('raw/orchestration/dependencies_before.json'))==51
    assert (folder/'raw/orchestration/production_ledger_before.json').read_bytes()==(folder/'raw/orchestration/production_ledger_after.json').read_bytes()
    assert digest(folder/'raw/orchestration/production_ledger_after.json')=='b1d224ba20a93c93c99189d700aeb145d506259f56462eb506def64f8f888189'
    proposal=load('raw/orchestration/frozen_proposal.json');approved=load('raw/orchestration/approved_manifest.json')
    assert digest(folder/'raw/orchestration/frozen_proposal.json')=='424fee88b75b2830d991b1cc7724b0763ccf8f9c77bb6c90d7826855f3eafc4a'
    assert [k for k in proposal if proposal[k]!=approved[k]]==['status','user_approved']
    assert approved==load('raw/run/run_manifest.json') and approved['user_approved'] is True
    assert report['status']=='failed' and 'FloatingPointError: nonfinite scanner evidence' in report['error']
    assert report['simulation_control_steps']==507 and report['completed_diagnostic_transitions']==60840
    assert report['motion_samples']==2028 and report['steps_written']==506
    assert report['actor_sha256_before']==report['actor_sha256_after']
    assert acceptance['accepted'] is False and acceptance['training_ready'] is False
    assert execution['process_group_empty'] and execution['validation_process_group_empty']
    assert execution['timed_out'] is False and execution['validation_returncode']==1
    assert post['status']=='passed' and post['cuda_initialized'] is False
    assert load('raw/orchestration/process_cleanup_audit.json')['owned_pids_absent']
    assert load('raw/orchestration/stop_authorization.json')['run_already_exited']
    manifest=load('bulk_manifest.json');store=artifact_root or Path(manifest['artifact_root'])
    assert len(manifest['files'])==2
    for entry in manifest['files']:
        path=store/entry['path']
        assert path.stat().st_size==entry['size_bytes'],str(path)
        assert digest(path)==entry['sha256'],str(path)
    assert digest(store/'run/events.jsonl.gz')==report['events_sha256']
    assert digest(store/'run/result.steps.jsonl')==report['steps_sha256']
    count_steps=0;endings=0
    with (store/'run/result.steps.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);count_steps+=1
            assert row['step']==count_steps
            endings+=sum(row['done'])
    assert count_steps==506 and endings==2
    return dict(schema='lite3_failed_g3_archive_review_v1',archive_integrity_verified=True,
        runtime_accepted=False,runtime_evidence_fully_replayed=False,
        failure='nonfinite scanner evidence',source_revision=approved['base_revision'],
        hashed_archive_files=count,bulk_files_verified=2,complete_control_records=count_steps,
        simulated_controls=507,ppo_updates_this_attempt=0,production_ledger_unchanged=True,
        source_inputs_unchanged=249,dependencies_unchanged=51,process_cleanup_recorded=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path)
    args=parser.parse_args()
    print(json.dumps(main(Path(__file__).resolve().parent,args.artifact_root),indent=2))
