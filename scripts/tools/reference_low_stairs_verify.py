"""Archive bounded low-stair preparation checks; never launch a training job."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from reference_stairs_eval import ROOT, save, sha

RAW=ROOT/'logs/reference_six/low_stairs_preparation_20261112'
OUT=ROOT/'docs/lite3-low-stairs-preparation-20261112'


def read(path):return json.loads(path.read_text())


def main():
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    checks=[]
    for name in ('integration_v2','ppo_validate3','eval_up_probe','eval_down_probe'):
        run=RAW/name;report=read(run/'report.json');execution=read(run/'execution.json')
        assert report['status']=='passed' and execution['status']=='completed'
        lines=(run/'runtime.log').read_text(errors='replace').splitlines()
        errors=[line for line in lines if '[Error]' in line or 'switching to software' in line]
        assert not errors,(name,errors)
        for path in run.glob('*.json'):shutil.copy2(path,OUT/f'{name}_{path.name}')
        checks.append(dict(run=name,status='passed',elapsed_s=execution['elapsed_s'],gpu=execution['physical_gpu'],
                           errors=errors,optimizer_updates=report.get('completed_updates',0)))
    integration=read(RAW/'integration_v2/report.json')
    assert integration['optimizer_updates']==0 and integration['actor_unchanged']
    geometry=read(RAW/'integration_v2/geometry.json')
    assert geometry['all_40_stair_tiles_in_2_to_5_cm']
    for name in ('eval_up_probe','eval_down_probe'):
        report=read(RAW/name/'report.json')
        assert report['probe'] and report['steps']==100 and report['episodes']==12
        assert report['height_set']=='low_stairs' and report['actor_unchanged']
        assert {row['height'] for row in report['groups']}=={0.,.02,.03,.05}
    train=read(RAW/'ppo_validate3/report.json')
    assert train['from_scratch'] and train['checkpoint_loaded'] is None
    assert train['completed_updates']==3 and train['environment_steps']==72 and train['interactions']==294912
    assert train['terrain_profile']==train['reward_variant']=='low_stairs'
    state=torch.load(train['output_checkpoint'],map_location='cpu',weights_only=True)
    assert sha(train['output_checkpoint'])==train['output_sha256']
    assert state['infos']['validation_only'] and state['infos']['from_scratch']
    assert {int(x['step']) for x in state['optimizer_state_dict']['state'].values()}=={60}
    for key in ('actor_state_dict','critic_state_dict'):
        assert all(torch.isfinite(value).all() for value in state[key].values())
    # Original checkpoints and the reference project's source are immutable.
    old=read(ROOT/'docs/lite3-code-turn-audit-20261112/verification.json')
    for row in old['models']:assert sha(row['path'])==row['sha256']
    reference_rows=[r for r in old['common_training_sources'] if r['path'].startswith('/home/tjx/rl_training/')]
    assert reference_rows
    for row in reference_rows:assert sha(row['path'])==row['sha256']
    for name in ('reference_low_stairs_cfg.py','reference_low_stairs_rewards.py'):
        recorded=read(RAW/'ppo_validate3/reward_override_audit.json')['low_stairs_sources'][name]
        assert sha(ROOT/'scripts/tools'/name)==recorded
    suite=ET.parse(RAW/'tests.xml').getroot().find('testsuite')
    assert int(suite.get('tests'))==59 and all(int(suite.get(k))==0 for k in ('errors','failures','skipped'))
    shutil.copy2(RAW/'tests.xml',OUT/'tests.xml')
    shutil.copy2(RAW/'integration_v2/terrain_geometry.png',OUT/'terrain_geometry.png')
    files=sorted(p for p in RAW.rglob('*') if p.is_file())
    save(OUT/'raw_manifest.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    source_names=('reference_low_stairs_cfg.py','reference_low_stairs_rewards.py','reference_low_stairs_validate.py',
                  'reference_low_stairs_verify.py','reference_six_train.py','reference_isaac_stairs_diagnose.py')
    source_files=[ROOT/'scripts/tools'/name for name in source_names]
    source_files.append(ROOT/'tests/source_curriculum/test_reference_low_stairs.py')
    save(OUT/'verification.json',dict(status='passed',status_scope='configuration and bounded integration, not locomotion qualification',
        checks=checks,tests_passed=59,geometry=geometry,validation_adam_steps=60,formal_training_started=False,
        old_checkpoints_unchanged=True,reference_source_files_unchanged=len(reference_rows),
        online_heights_evaluated=False,validation_model_not_a_training_initializer=True,
        sources=[dict(path=str(p),sha256=sha(p)) for p in source_files]))
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in paths))
    print(json.dumps(dict(status='passed',tests=59,validation_updates=3,artifacts=len(paths),formal_training=False)))


if __name__=='__main__':main()
