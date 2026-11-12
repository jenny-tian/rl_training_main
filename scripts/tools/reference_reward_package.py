"""Package completed, optimizer-free reward validation evidence."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from reference_lite3_loader import ROOT, REFERENCE, REFERENCE_COMMIT
from reference_stairs_eval import save, sha, MODELS, TRAIN
from reference_reward_variants import VARIANTS, BASE_HEIGHT_M

RAW = ROOT/'logs/reference_six/reward_fixes_20261112'
OUT = ROOT/'docs/lite3-reward-fixes-20261112'
RUNS = ('stand_gate','height_target','core','core_no_phase_v2','core_airtime_nonnegative','core_no_gait')


def read(path):
    return json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summaries=[]; variants={}; manifest=[]; executions=[]
    for name in ('calibration_v2', *RUNS):
        run=RAW/name
        report=read(run/'report.json'); execution=read(run/'execution.json')
        config=read(run/'config.json'); audit=read(run/'reward_override_audit.json')
        assert execution['status']=='completed' and not execution['fatal_messages']
        assert report['status']=='passed' and report['optimizer_updates']==0 and report['actor_unchanged']
        assert report['all_finite'] and config['reward_source_sha256']==sha(Path(__file__).with_name('reference_reward_variants.py'))
        assert report['checkpoint_sha256']==sha(MODELS['update10000'])
        assert set(audit['changes'])==set(VARIANTS[report['variant']])
        with np.load(run/'trace.npz') as a:
            assert all(np.isfinite(a[k]).all() for k in a.files)
        if name=='calibration_v2':
            assert report['steps']==300 and report['environments']==64
            assert report['calibration_target_verified']
            assert abs(report['mean_last_second_height']-BASE_HEIGHT_M)<.005
            shutil.copy2(run/'trace.npz',OUT/'calibration_trace.npz')
        else:
            assert report['steps']==240 and report['environments']==120 and len(report['families'])==6
            variants[report['variant']]=audit
        if report['stand_gate_max_error'] is not None:
            assert report['stand_gate_max_error']<1e-6
        summaries.append(dict(run=name,**{k:v for k,v in report.items() if k not in ('results','sources')}))
        executions.append(dict(run=name,**execution))
    protocols=[]
    for direction in ('up','down'):
        run=RAW/f'{direction}_protocol_smoke'
        report=read(run/'report.json');execution=read(run/'execution.json');geometry=read(run/'geometry.json')
        assert report['status']=='passed' and execution['status']=='completed'
        assert report['probe'] and report['episodes']==24 and report['steps']==250
        assert report['direction']==direction and report['actor_unchanged']
        assert report['checkpoint_sha256']==sha(MODELS['update10000'])
        for geom in geometry[1:]:
            sign=1 if direction=='up' else -1
            assert len(geom['fronts'])>=5 and np.allclose(np.diff([0.]+geom['tops']),sign*geom['height'],atol=.002)
        with np.load(run/'trace.npz') as a:
            assert all(np.isfinite(a[k]).all() for k in a.files)
        protocols.append(dict(direction=direction,episodes=24,steps=250,geometry=geometry,
                              protocol_smoke_only=True,actor_unchanged=True))
        executions.append(dict(run=run.name,**execution))
    old=read(TRAIN/'report.json')
    for row in old['sources']: assert sha(Path(row['path']))==row['sha256'],row['path']
    assert len(old['sources'])==20
    prior=read(ROOT/'docs/lite3-code-turn-audit-20261112/verification.json')
    for row in prior['models']: assert sha(Path(row['path']))==row['sha256']
    for row in read(ROOT/'docs/lite3-code-turn-audit-20261112/reward_code_audit.json')['sources'].values():
        assert sha(Path(row['path']))==row['sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REFERENCE,text=True).strip()==REFERENCE_COMMIT
    subprocess.run(['git','diff','--exit-code','HEAD','--','source/rl_training'],cwd=REFERENCE,check=True)
    resume=RAW/'resume_smoke'
    restored=read(resume/'checkpoint_restore_audit.json')
    resume_report=read(resume/'report.json'); resume_execution=read(resume/'execution.json')
    assert resume_report['status']=='passed' and resume_execution['status']=='completed'
    assert resume_report['completed_updates']==0 and resume_report['cumulative_updates']==10000
    assert resume_report['environment_steps']==100 and not resume_report['from_scratch']
    assert restored['actor_exact'] and restored['critic_exact'] and restored['optimizer_exact']
    assert restored['optimizer_steps']==200000 and restored['next_iteration']==10000
    assert restored['sha256']==sha(MODELS['update10000'])
    executions.append(dict(run='resume_smoke',**resume_execution))
    save(OUT/'checkpoint_restore_audit.json',restored)
    tests=ET.parse(RAW/'tests.xml').getroot().find('testsuite')
    assert int(tests.get('tests'))==33
    assert all(int(tests.get(k))==0 for k in ('failures','errors','skipped'))
    shutil.copy2(RAW/'tests.xml',OUT/'tests.xml')
    trial=read(RAW/'core_no_phase/error.json')
    save(OUT/'validation_summary.json',dict(status='passed',reward_jobs=summaries,protocol_jobs=protocols,
        resume_smoke={k:v for k,v in resume_report.items() if k!='sources'},
        attempts=[dict(run='calibration_v1',status='not_started',reason='GPU0 free memory below 12288 MiB admission threshold'),
                  dict(run='core_no_phase',status='failed_check_corrected',error=trial['error'],
                       correction='Zero-weight names remain registered; validate zero contribution and absent phase cache instead.')]))
    save(OUT/'variants.json',variants)
    save(OUT/'executions.json',executions)
    save(OUT/'verification.json',dict(status='passed',optimizer_updates=0,training_started=False,
        calibrated_height_m=BASE_HEIGHT_M,tests_passed=33,reward_variants_passed=6,final_restore_without_updates_passed=True,
        reference_commit=REFERENCE_COMMIT,common_training_source_files_unchanged=20,
        checkpoint_sha256=sha(MODELS['update10000']),all_four_prior_checkpoints_unchanged=True,
        reference_and_isaac_reward_sources_unchanged=True,gpu=1,gpu1_user_authorized_for_validation_only=True,
        protocol_smoke_directions=['up','down'],behavior_improvement_claimed=False,
        source_files=[dict(path=str(p),sha256=sha(p)) for p in sorted([
            *[ROOT/'scripts/tools'/name for name in ('reference_reward_variants.py','reference_reward_validate.py',
              'reference_reward_package.py','reference_reward_resume.py','reference_six_train.py','reference_isaac_stairs_diagnose.py',
              'reference_isaac_stairs_analyze.py')],
            ROOT/'tests/source_curriculum/test_reference_reward_variants.py',
            ROOT/'tests/source_curriculum/test_reference_isaac_stairs_analysis.py'])]))
    for p in sorted(RAW.rglob('*')):
        if p.is_file(): manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    save(OUT/'raw_manifest.json',manifest)
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in paths))
    print(json.dumps(dict(status='passed',artifacts=len(paths),variants=6,tests=33,optimizer_updates=0)))


if __name__=='__main__': main()
