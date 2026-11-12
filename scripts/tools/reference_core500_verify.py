"""Verify the completed core500 budget, checkpoint and frozen evaluation artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from reference_stairs_eval import ROOT, MODELS, load_actor, sha, save
from reference_core500_evaluate import TRAINING, OUTPUT

OUT=ROOT/'docs/lite3-core500-results-20261112'


def read(path):return json.loads(path.read_text())


def verify_checkpoint():
    import torch
    import yaml
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    torch.set_num_threads(1);torch.manual_seed(20261112)
    report=read(TRAINING/'report.json')
    path=Path(report['output_checkpoint'])
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert sha(path)==report['output_sha256']
    assert state['iter']==10499 and state['infos']['completed_updates']==10500
    assert state['infos']['phase_updates']==500 and state['infos']['reward_variant']=='core'
    assert not state['infos']['from_scratch']
    assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()}=={210000}
    for key in ('actor_state_dict','critic_state_dict'):
        assert all(torch.isfinite(v).all() for v in state[key].values())
    with np.load(OUTPUT/'core10500_turn_nominal/initial.npz') as initial:
        actual=torch.from_numpy(initial['observation'].copy())
    inputs=torch.cat((torch.randn(2048,45),actual))
    obs=TensorDict({'policy':inputs},batch_size=[len(inputs)])
    cfg=yaml.safe_load((TRAINING/'params/agent.yaml').read_text())['actor'];cfg.pop('class_name')
    assert cfg['obs_normalization'] is False
    native=MLPModel(obs,{'actor':['policy']},'actor',12,**cfg).eval()
    native.load_state_dict(state['actor_state_dict'],strict=True)
    MODELS['core10500']=path
    adapted=load_actor('core10500')
    with torch.no_grad():difference=(native(obs)-adapted(inputs)).abs().max().item()
    assert difference==0.
    old=read(ROOT/'docs/lite3-code-turn-audit-20261112/verification.json')
    for row in old['models']:assert sha(Path(row['path']))==row['sha256']
    for row in old['common_training_sources']:assert sha(Path(row['path']))==row['sha256']
    reward_source=Path(__file__).with_name('reference_reward_variants.py')
    assert sha(reward_source)==read(TRAINING/'reward_override_audit.json')['source_sha256']
    return dict(path=str(path),sha256=sha(path),cumulative_updates=10500,phase_updates=500,adam_steps=210000,
        actor_adapter_max_difference=difference,actor_adapter_samples=len(inputs),parameters_finite=True,
        original_four_checkpoints_unchanged=True,original_twenty_training_sources_unchanged=True,
        core_reward_source_unchanged=True,action_std=state['actor_state_dict']['distribution.log_std_param'].exp().tolist())


def training_metrics():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator=EventAccumulator(str(TRAINING/'training'),size_guidance={'scalars':0});accumulator.Reload()
    tags=('Train/mean_reward','Train/mean_episode_length','Episode_Termination/bad_orientation_2',
          'Episode_Termination/time_out','Metrics/base_velocity/error_vel_xy','Metrics/base_velocity/error_vel_yaw',
          'Episode_Reward/base_height_l2','Episode_Reward/stand_still','Policy/mean_std')
    metrics={}
    for tag in tags:
        values=accumulator.Scalars(tag)
        assert len(values)==500 and values[0].step==10000 and values[-1].step==10499
        array=np.array([v.value for v in values]);assert np.isfinite(array).all()
        metrics[tag]=dict(first50_mean=float(array[:50].mean()),last50_mean=float(array[-50:].mean()),final=float(array[-1]),
                          values=array.tolist())
    report=read(TRAINING/'report.json')
    return dict(scalars=metrics,terminated=report['terminated'],truncated=report['truncated'],
        aggregate_termination_fraction=report['terminated']/(report['terminated']+report['truncated']),
        note='TensorBoard episode values are logger averages, not paired evaluation success rates. Rewards differ from the old training objective.')


def verify_runtime_logs(summary):
    runs={Path(row[key]) for row in summary['matrix'] for key in ('baseline','candidate')}
    runs.update(OUTPUT/name for name in ('video_turn','video_stairs'))
    rows=[]
    for run in sorted(runs):
        lines=(run/'runtime.log').read_text(errors='replace').splitlines()
        fallback=[line for line in lines if re.search(
            r'switching to software|GPU solver failed|GPU Bp pipeline failed|no suitable CUDA GPU',line,re.I)]
        assert not fallback,(str(run),fallback)
        errors=[line for line in lines if '[Error]' in line]
        allowed=('No device could be created.', 'Invalid getDeviceInfo parameters.')
        unexpected=[line for line in errors if not ('[gpu.foundation.plugin]' in line and any(x in line for x in allowed))]
        assert not unexpected,(str(run),unexpected)
        if run.name.startswith('video_'):assert not errors
        rows.append(dict(run=str(run),physics_fallback_messages=0,renderer_startup_errors=errors))
    return rows


def main():
    summary=read(OUT/'summary.json');assert summary['status']=='passed'
    checkpoint=verify_checkpoint()
    metrics=training_metrics();save(OUT/'training_metrics.json',metrics)
    test_file=OUTPUT/'tests.xml';suite=ET.parse(test_file).getroot().find('testsuite')
    assert int(suite.get('tests'))==17 and all(int(suite.get(k))==0 for k in ('errors','failures','skipped'))
    shutil.copy2(test_file,OUT/'tests.xml')
    save(OUT/'verification.json',dict(status='passed',checkpoint=checkpoint,tests_passed=17,
        approved_budget_updates=500,actual_updates=500,actual_interactions=49152000,
        candidate_episodes=1600,paired_baseline_episodes=1600,video_replays_excluded_from_formal_count=True,
        runtime_log_checks=verify_runtime_logs(summary),
        initial_pairing=[{k:v for k,v in row.items() if k!='kind'} for row in summary['matrix']],
        video_replays={name:read(OUT/'clips'/name/'replay_validation.json') for name in ('video_turn','video_stairs')},
        source_files=[dict(path=str(p),sha256=sha(p)) for p in sorted([
            *[ROOT/'scripts/tools'/name for name in ('reference_core500_evaluate.py','reference_core500_analyze.py',
              'reference_core500_verify.py','reference_eval_checkpoint.py','reference_turn_diagnose.py','reference_isaac_stairs_diagnose.py')],
            ROOT/'tests/source_curriculum/test_reference_eval_checkpoint.py'])]))
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in paths))
    print(json.dumps(dict(status='passed',updates=500,candidate_episodes=1600,adapter_max_difference=0.,tests=17,artifacts=len(paths))))


if __name__=='__main__':main()
