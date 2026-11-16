"""交付前只读核对来源、预算、历史文件和检查结果，归档完成记录。"""
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from reference_low_stairs_reward_state_probe import OUTPUT,DOCS,BASELINES,CHECKPOINT,DIGEST,read,sources
from reference_stairs_eval import sha,save


def main():
    summary=read(DOCS/'summary.json');replay=read(DOCS/'landing_candidate_summary.json')
    recovery=read(DOCS/'recovery_gate_audit.json');complete=read(OUTPUT/'complete.json')
    assert all(x['status']=='passed' for x in (summary,replay,recovery,complete))
    assert summary['new_episodes']==complete['new_episodes']==480
    assert all(x['optimizer_updates']==0 for x in (summary,replay,recovery,complete))
    assert replay['new_simulation_episodes']==recovery['new_simulation_episodes']==0
    assert complete['sources']==sources()
    assert sha(CHECKPOINT)==DIGEST
    checked={}
    manifests=[read(DOCS/'raw_manifest.json'),summary['sources'],replay['sources'],replay['implementation_sources'],
        recovery['sources'],recovery['implementation_sources']]
    for rows in manifests:
        for row in rows:
            path=row['path']
            if path not in checked:checked[path]=sha(path)
            assert checked[path]==row['sha256'],path
    tests={}
    for key in ('probe','recovery','candidate'):
        suites=ET.parse(DOCS/f'{key}_tests.xml').getroot().findall('testsuite')
        assert suites and all(int(s.get(k))==0 for s in suites for k in ('errors','failures','skipped'))
        tests[key]=sum(int(s.get('tests')) for s in suites)
        assert tests[key]==summary[key+'_tests_passed']
    assert sum(tests.values())==53
    jobs=[]
    for profile in BASELINES:
        execution=read(OUTPUT/profile/'execution.json');capture=read(OUTPUT/profile/'capture_complete.json')
        assert execution['status']=='completed' and execution['exit_code']==0 and not execution['fatal_messages']
        assert capture['episodes']==240 and capture['optimizer_updates']==0 and capture['sources']==sources()
        process=Path('/proc')/str(execution['pid'])/'cmdline'
        still_running=process.exists() and execution['command'][1].encode() in process.read_bytes().split(b'\0')
        assert not still_running
        jobs.append(dict(profile=profile,pid=execution['pid'],process_exited=True,elapsed_s=execution['elapsed_s'],
            peak_total_gpu_used_mib=execution['peak_total_gpu_used_mib'],episodes=capture['episodes']))
    assert sum(j['episodes'] for j in jobs)==480
    gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,process_name,used_gpu_memory',
                                 '--format=csv,noheader'],text=True).strip().splitlines()
    names=('reference_low_stairs_reward_state_analyze.py','reference_low_stairs_landing_candidate.py',
        'reference_low_stairs_landing_replay.py','reference_low_stairs_reward_state_plot.py',
        'reference_low_stairs_recovery_gate.py','reference_low_stairs_recovery_gate_audit.py',Path(__file__).name)
    implementations=[dict(path=str(Path(__file__).with_name(n)),sha256=sha(Path(__file__).with_name(n))) for n in names]
    artifacts=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(DOCS.rglob('*'))
        if p.is_file() and p.name!='completion_audit.json']
    save(DOCS/'completion_audit.json',dict(status='passed',new_simulation_episodes=480,optimizer_updates=0,
        next_stage_authorized=False,checkpoint_sha256=DIGEST,checked_source_and_raw_files=len(checked),
        jobs=jobs,tests=tests,total_tests=sum(tests.values()),gpu1_compute_processes_at_audit=gpu,
        implementation_sources=implementations,artifacts=artifacts))
    print(dict(status='passed',episodes=480,updates=0,tests=sum(tests.values()),verified_files=len(checked),
        jobs=jobs,gpu1_compute_processes_at_audit=gpu),flush=True)


if __name__=='__main__':main()
