"""Archive completed repair preparation, without running an optimizer."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_evaluate import verify_training
from reference_low_stairs_analyze import adapter_check, check_log

RAW=ROOT/'logs/reference_six/low_stairs_repair_preparation_20261114'
OUT=ROOT/'docs/lite3-low-stairs-repair-preparation-20261114'


def read(path):
    return json.loads(path.read_text())


def main():
    reports=[];traces=[];initials=[];checks=[]
    for variant in ('riser_contact','yaw_airtime'):
        path=RAW/f'{variant}_v1'
        report,execution=read(path/'report.json'),read(path/'execution.json')
        assert report['status']=='passed' and report['variant']==variant
        assert report['optimizer_updates']==0 and report['actor_unchanged']
        assert report['environments']==120 and report['steps']==600
        assert report['yaw_outside_pure_turn_changed_frames']==0
        assert execution['status']=='completed' and execution['exit_code']==0 and execution['physical_gpu']==1
        checks.append(check_log(path));reports.append(report)
        with np.load(path/'trace.npz') as loaded:traces.append(dict(loaded))
        with np.load(path/'initial.npz') as loaded:initials.append(dict(loaded))
        for file in path.glob('*.json'):shutil.copy2(file,OUT/f'{variant}_{file.name}')
    assert initials[0].keys()==initials[1].keys() and traces[0].keys()==traces[1].keys()
    assert all(np.array_equal(initials[0][k],initials[1][k]) for k in initials[0])
    assert all(np.array_equal(traces[0][k],traces[1][k]) for k in traces[0])
    d=traces[0];valid=d['active'];r=reports[0]
    assert all(np.isfinite(v).all() for v in d.values())
    scope=[]
    for name in sorted(set(r['terrain_names'])):
        ids=np.array([i for i,n in enumerate(r['terrain_names']) if n==name])
        values=d['riser_cost'][:,ids][valid[:,ids]]
        scope.append(dict(terrain=name,scored_frames=int((values>0).sum()),max_cost=float(values.max())))
        if name!='stairs_up':assert (values==0.).all()
    yaw=(np.linalg.norm(d['command'][...,:2],axis=-1)<.1)&(np.abs(d['command'][...,2])>.1)&valid
    changes=(d['yaw_candidate']>d['yaw_original']+1e-6)&valid
    assert not (changes&~yaw).any()
    turn=[]
    for case in sorted(set(r['cases'])):
        ids=np.array([i for i,n in enumerate(r['cases']) if n==case and r['terrain_names'][i]=='flat'])
        if not len(ids):continue
        mask=valid[100:,ids]
        turn.append(dict(case=case,old_mean_rate=float(d['yaw_original'][100:,ids][mask].mean()),
            candidate_mean_rate=float(d['yaw_candidate'][100:,ids][mask].mean()),
            changed_frames=int(changes[100:,ids].sum())))
    checkpoint=adapter_check(verify_training())
    suite=ET.parse(RAW/'tests.xml').getroot().find('testsuite')
    assert int(suite.get('tests'))==49 and all(int(suite.get(k))==0 for k in ('errors','failures','skipped'))
    shutil.copy2(RAW/'tests.xml',OUT/'tests.xml')
    save(OUT/'integration_analysis.json',dict(status='passed',initial_arrays_exact=True,all_trace_arrays_exact=True,
        initial_fields=list(initials[0]),trace_fields=list(traces[0]),scoring_scope=scope,turn=turn,
        first_episode_failures=r['first_episode_failures'],
        note='Both variants ran identical frozen physical trajectories. Their reward arithmetic differs as intended. '
             'Failures remain; this is not a trained-policy comparison.'))
    names=('reference_low_stairs_repair.py','reference_low_stairs_repair_validate.py',
           'reference_low_stairs_failure_audit.py','reference_low_stairs_repair_verify.py')
    files=[ROOT/'scripts/tools'/name for name in names]
    files.append(ROOT/'tests/source_curriculum/test_reference_low_stairs_repair.py')
    save(OUT/'verification.json',dict(status='passed',status_scope='diagnosis and frozen integration, not policy qualification',
        optimizer_updates=0,formal_training_started=False,online_learning_started=False,
        tests_passed=49,checkpoint=checkpoint,runtime_checks=checks,
        frozen_control_steps=120*600*2,all_initial_and_trace_arrays_exact=True,
        sources=[dict(path=str(p),sha256=sha(p)) for p in files]))
    save(OUT/'raw_manifest.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(RAW.rglob('*')) if p.is_file()])
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in paths))
    print(json.dumps(dict(status='passed',tests=49,optimizer_updates=0,scope=scope,turn=turn),ensure_ascii=False))


if __name__=='__main__':main()
