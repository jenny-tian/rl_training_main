"""Paired behavioral screening of A500; execution success is not acceptance."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from reference_stairs_eval import ROOT, MODELS, load_actor, sha
from reference_six_watch import read, save
from reference_low_stairs_a_evaluate import TRAINING, OUTPUT, MODEL, verify_training, verify_run, same_configuration
from reference_low_stairs_evaluate import verify_run as verify_baseline
from reference_core500_evaluate import same_initial
from reference_core500_analyze import write_csv, compare_video, paired_turn_rows
from reference_turn_analyze import analyze as analyze_turn, WINDOWS
from reference_turn_diagnose import CASES
from reference_isaac_stairs_analyze import analyze as analyze_stairs
from reference_low_stairs_analyze import stairs_heights, check_log, failure_diagnostics, flat_heading_diagnostics, read_agent
from reference_low_stairs_failure_audit import stair_features, LEGS, DT

OUT = ROOT / 'docs/lite3-low-stairs-a-trial-20261114'
OUTCOMES = ('success', 'stuck', 'orientation_failure', 'lateral_exit', 'timeout')


def adapter_check(checkpoint):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    torch.set_num_threads(1)
    torch.manual_seed(20261114)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    with np.load(OUTPUT / 'turn_nominal/initial.npz') as initial:
        actual = torch.from_numpy(initial['observation'].copy())
    inputs = torch.cat((torch.randn(2048, 45), actual))
    obs = TensorDict({'policy': inputs}, batch_size=[len(inputs)])
    native = MLPModel(obs, {'actor': ['policy']}, 'actor', 12, **read_agent()).eval()
    native.load_state_dict(state['actor_state_dict'], strict=True)
    MODELS[MODEL] = checkpoint
    adapted = load_actor(MODEL)
    with torch.inference_mode():
        difference = (native(obs) - adapted(inputs)).abs().max().item()
    assert difference == 0.
    return dict(status='passed',samples=len(inputs),max_output_difference=difference,
                checkpoint=str(checkpoint),sha256=sha(checkpoint),adam_steps=210000)


def training_summary():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    report = read(TRAINING / 'report.json')
    events = EventAccumulator(str(TRAINING / 'training'), size_guidance={'scalars': 0})
    events.Reload()
    metrics = {}
    for tag in ('Train/mean_reward', 'Train/mean_episode_length', 'Episode_Termination/bad_orientation_2',
                'Episode_Termination/time_out', 'Metrics/base_velocity/error_vel_xy',
                'Metrics/base_velocity/error_vel_yaw', 'Episode_Reward/stair_riser_contact',
                'Episode_Reward/stair_clearance', 'Episode_Reward/stair_progress', 'Episode_Reward/stair_landing'):
        values = events.Scalars(tag)
        assert len(values) == 500 and [v.step for v in values] == list(range(10000, 10500))
        array = np.array([v.value for v in values])
        assert np.isfinite(array).all()
        metrics[tag] = dict(first20=float(array[:20].mean()),first100=float(array[:100].mean()),
                            last100=float(array[-100:].mean()),last=float(array[-1]))
    coverage = np.array(report['coverage_by_column_level'])
    moving = np.array(report['translation_frames_by_column_level'])
    contact = np.array(report['collision_frames_by_column_level'])
    assert coverage.shape == moving.shape == contact.shape == (20,10)
    assert coverage.sum() == report['interactions'] and np.all(contact <= moving) and np.all(moving <= coverage)
    rows = []
    for family in dict.fromkeys(report['terrain_column_names']):
        columns = [i for i,name in enumerate(report['terrain_column_names']) if name == family]
        for level in range(10):
            total, translated, collisions = (int(a[columns,level].sum()) for a in (coverage,moving,contact))
            rows.append(dict(family=family,level=level,environment_frames=total,translation_frames=translated,
                riser_penalty_frames=collisions,penalty_fraction_of_translation=collisions/translated if translated else None))
    return dict(status='passed',updates=500,interactions=report['interactions'],metrics=metrics,
                initial_level_histogram=report['initial_level_histogram'],
                final_level_histogram=report['final_terrain_level_histogram'],
                initial_learning_rate=report['initial_learning_rate'],final_learning_rate=report['final_learning_rate'],
                coverage=rows,execution=read(TRAINING/'execution.json'),first20=read(TRAINING/'first20_check.json'))


def contact_diagnostics(report,cfg,data):
    rows, approaches = [], []
    for record in report['records']:
        if not record['height']:
            continue
        i = record['env_id']
        n = int(data['active'][:,i].sum())
        f,force = data['feet'][:n,i],data['forces'][:n,i]
        geometry = cfg['geometry'][record['column']]
        x = stair_features(f,force,geometry)
        tilt = np.degrees(np.arccos(np.clip(-data['gravity'][:n,i,2],-1.,1.)))
        ids = np.flatnonzero(tilt > 20.)
        first_tilt = int(ids[0]) if len(ids) else n
        before = (np.arange(n)<first_tilt) & (data['root'][:n,i,0]>.85)
        rows.append(dict(profile=report['profile'],env_id=i,height=record['height'],speed=record['speed'],
            outcome=record['outcome'],before_tilt_s=float(before.sum()*DT),
            collision_foot_seconds=float(x['collision'][before].sum()*DT),
            wall_foot_seconds=float(x['wall'][before].sum()*DT),
            collision_fraction=float(x['collision'][before].mean()) if before.any() else None,
            first_tilt20_s=(first_tilt+1)*DT if first_tilt<n else None,
            final_support_count=float(x['support'][-25:].sum(-1).mean())))
        edge,top = geometry['fronts'][0],geometry['tops'][0]
        for j,leg in enumerate(LEGS[:2]):
            reaches = np.flatnonzero((f[:,j,0]>=edge-.032)&(f[:,j,0]>np.abs(f[:,j,1])+.04))
            if not len(reaches) or reaches[0]>=first_tilt:
                continue
            t = int(reaches[0])
            landed = np.flatnonzero((np.arange(n)>=t)&(f[:,j,0]>=edge+.022)&x['support'][:,j]
                                   &(np.abs(x['ground'][:,j]-top)<.005))
            approaches.append(dict(profile=report['profile'],env_id=i,height=record['height'],speed=record['speed'],
                outcome=record['outcome'],leg=leg,approach_s=(t+1)*DT,bottom_above_top=float(x['bottom'][t,j]-top),
                landing_after_approach_s=float((landed[0]-t)*DT) if len(landed) else None))
    return rows,approaches


def screen(heights,groups,pairs,turn_failures):
    checks = []
    def require(name,actual,target,minimum=True):
        checks.append(dict(name=name,actual=actual,target=target,comparison='>=' if minimum else '<=',
                           passed=bool(actual>=target if minimum else actual<=target)))
    def height(direction,h):
        return next(r for r in heights if (r['policy'],r['profile'],r['direction'],r['height']) ==
                    ('candidate','nominal',direction,h))
    for speed,target in ((.32,8),(.4,10),(.6,14)):
        group = next(r for r in groups if (r['policy'],r['profile'],r['direction'],r['height'],r['speed']) ==
                     ('candidate','nominal','up',.05,speed))
        require(f'nominal_up_5cm_{speed}_success',group['success'],target)
    require('nominal_up_5cm_orientation_failures',height('up',.05)['orientation_failure'],11,False)
    for h,target in ((.02,60),(.03,54)):
        require(f'nominal_up_{h}_success',height('up',h)['success'],target)
    for h in (.02,.03,.05):
        require(f'nominal_down_{h}_success',height('down',h)['success'],60)
    for direction in ('up','down'):
        require(f'nominal_flat_{direction}_success',height(direction,0.)['success'],60)
    worsening = [r for r in pairs if r['candidate_mae']>r['baseline_mae']*1.1+1e-12]
    expected = {(profile,case,window) for profile in ('nominal','training_randomized') for case in CASES for window in WINDOWS}
    missing = sorted(expected-{(r['profile'],r['case'],r['window']) for r in pairs})
    require('turn_windows_with_mae_worse_than_10_percent',len(worsening),0,False)
    require('turn_new_paired_orientation_failures',sum(r['new_failures'] for r in turn_failures),0,False)
    require('turn_missing_paired_windows',len(missing),0,False)
    random_changes = []
    for r in groups:
        if r['policy']!='candidate' or r['profile']!='training_randomized':
            continue
        old = next(o for o in groups if o['policy']=='baseline' and
                   (o['profile'],o['direction'],o['height'],o['speed']) == (r['profile'],r['direction'],r['height'],r['speed']))
        random_changes.append(dict(direction=r['direction'],height=r['height'],speed=r['speed'],
            baseline_success=old['success'],candidate_success=r['success'],success_delta=r['success']-old['success'],
            baseline_orientation=old['orientation_failure'],candidate_orientation=r['orientation_failure']))
    return dict(status='passed' if all(c['passed'] for c in checks) else 'failed',
        scope='Prespecified engineering screen; no automatic deployment or training extension',
        checks=checks,turn_error_regressions=worsening,turn_missing_windows=missing,
        random_groups=random_changes,random_decreases_require_review=[r for r in random_changes if r['success_delta']<0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    complete=read(OUTPUT/'complete.json')
    assert complete['status']=='passed' and complete['candidate_episodes']==1600
    checkpoint,digest=verify_training()
    assert digest==complete['candidate_sha256']
    heights,groups,episodes,rewards,turns,windows,pairs,failures,contacts,approaches,flat=[],[],[],[],[],[],[],[],[],[],[]
    turn_failures,stair_pairs,checks,logs=[],[],[],[]
    raw=set()
    for entry in complete['matrix']:
        old,new=Path(entry['baseline']),Path(entry['candidate'])
        assert same_initial(old,new)
        old_cfg,new_cfg=read(old/'config.json'),read(new/'config.json')
        assert same_configuration(old_cfg,new_cfg)
        reports,pair_windows={},{}
        for policy,directory in (('baseline',old),('candidate',new)):
            if policy=='baseline':
                report=verify_baseline(directory,entry['kind'],entry['profile'],entry['direction'])
            else:
                report=verify_run(directory,entry['kind'],entry['profile'],digest,entry['direction'])
            reports[policy]=report
            logs.append(check_log(directory))
            if entry['kind']=='turn':
                _,data,rows,grouped=analyze_turn(directory)
                turns.extend(dict(policy=policy,**r) for r in grouped)
                windows.extend(dict(policy=policy,**r) for r in rows)
                pair_windows[policy]=rows
            else:
                _,cfg,data,ep,rw,_=analyze_stairs(directory)
                heights.extend(dict(policy=policy,**r) for r in stairs_heights(report))
                episodes.extend(dict(policy=policy,**r) for r in ep)
                rewards.extend(dict(policy=policy,**r) for r in rw)
                failures.extend(dict(policy=policy,**r) for r in failure_diagnostics(report,data))
                flat.extend(dict(policy=policy,**r) for r in flat_heading_diagnostics(report,data))
                groups.extend(dict(policy=policy,profile=entry['profile'],direction=entry['direction'],
                    **{k:v for k,v in g.items() if k!='outcomes'},
                    **{k:g['outcomes'].get(k,0) for k in OUTCOMES}) for g in report['groups'])
                if entry['direction']=='up':
                    cr,ap=contact_diagnostics(report,cfg,data)
                    contacts.extend(dict(policy=policy,**r) for r in cr)
                    approaches.extend(dict(policy=policy,**r) for r in ap)
            assert all(np.isfinite(v).all() for v in data.values())
            checks.append(dict(policy=policy,run=str(directory),episodes=report['episodes'],
                               arrays_finite=True,actor_unchanged=True,initial_equal=True,common_config_equal=True))
            raw.update(p for p in directory.iterdir() if p.is_file())
            for name in ('report.json','config.json','execution.json'):
                shutil.copy2(directory/name,out/f'{policy}_{directory.name}_{name}')
        a,b=({r['env_id']:r for r in reports[k]['records']} for k in ('baseline','candidate'))
        assert a.keys()==b.keys()
        if entry['kind']=='turn':
            pairs.extend(paired_turn_rows(pair_windows['baseline'],pair_windows['candidate'],entry['profile']))
            for case in CASES:
                ids=[i for i,r in a.items() if r['case']==case]
                assert all(a[i]['case']==b[i]['case'] and a[i]['repeat']==b[i]['repeat'] for i in ids)
                turn_failures.append(dict(profile=entry['profile'],case=case,starts=len(ids),
                    baseline_failures=sum(a[i]['outcome']=='orientation_failure' for i in ids),
                    candidate_failures=sum(b[i]['outcome']=='orientation_failure' for i in ids),
                    new_failures=sum(a[i]['outcome']!='orientation_failure' and b[i]['outcome']=='orientation_failure' for i in ids),
                    recovered=sum(a[i]['outcome']=='orientation_failure' and b[i]['outcome']!='orientation_failure' for i in ids)))
        else:
            for i in a:
                assert all(a[i][k]==b[i][k] for k in ('height','speed','repeat','column'))
                stair_pairs.append(dict(profile=entry['profile'],direction=entry['direction'],env_id=i,
                    height=a[i]['height'],speed=a[i]['speed'],repeat=a[i]['repeat'],
                    baseline_outcome=a[i]['outcome'],candidate_outcome=b[i]['outcome'],
                    gained_success=a[i]['outcome']!='success' and b[i]['outcome']=='success',
                    lost_success=a[i]['outcome']=='success' and b[i]['outcome']!='success'))
    assert sum(c['episodes'] for c in checks if c['policy']=='candidate')==1600
    for name,rows in [('stairs_heights',heights),('stairs_groups',groups),('stairs_episodes',episodes),
                      ('stairs_paired',stair_pairs),('stairs_reward_windows',rewards),('turn_groups',turns),
                      ('turn_windows',windows),('turn_paired',pairs),('turn_failures',turn_failures),
                      ('failure_diagnostics',failures),('contacts_before_tilt',contacts),
                      ('first_front_approaches',approaches),('flat_heading',flat)]:
        write_csv(out/f'{name}.csv',rows)
    video_checks={}
    for video in complete['videos']:
        directory=Path(video['candidate'])
        verify_run(directory,video['kind'],video['profile'],digest,video['direction'],video['video_env'])
        logs.append(check_log(directory))
        dest=out/'clips'/video['name']
        compare_video(directory,Path(video['formal']),dest)
        video_checks[video['name']]=read(dest/'replay_validation.json')
        raw.update(p for p in directory.iterdir() if p.is_file())
    training=training_summary();adapter=adapter_check(checkpoint)
    screening=screen(heights,groups,pairs,turn_failures)
    save(out/'screening.json',screening);save(out/'training_summary.json',training)
    write_csv(out/'training_terrain_coverage.csv',training['coverage'])
    save(out/'summary.json',dict(status='passed',status_scope='artifact verification; screening is separate',
        candidate_sha256=digest,candidate_episodes=1600,reused_baseline_episodes=1600,training_updates=500,
        stairs_heights=heights,stairs_groups=groups,turn_paired=pairs,turn_failures=turn_failures,
        screening=screening,matrix=complete['matrix'],videos=complete['videos'],
        caveats=['One training seed and fixed evaluation starts; no deployment qualification.',
                 'Common reference reward scores are not replayed A training rewards.',
                 'Turn errors use paired complete windows; failures retain all starts.',
                 'Contact diagnostics are geometry/force proxies sampled every 20 ms, not exact reward replay.',
                 'Native terrain curriculum restarted; use measured coverage when interpreting the 500-update budget.']))
    save(out/'verification.json',dict(status='passed',formal_candidate_episodes=1600,adapter=adapter,
        rollout_checks=checks,runtime_logs=logs,video_checks=video_checks,video_replays_excluded=True,
        additional_training_performed_by_analyzer=False))
    for name in ('report.json','execution.json','restore.json','first20_check.json','terrain_audit.json',
                 'repair_audit.json','runtime_overrides.json','implementation_sources.json','geometry.json'):
        shutil.copy2(TRAINING/name,out/f'training_{name}')
    raw.update(p for p in TRAINING.rglob('*') if p.is_file())
    raw.update(p for p in OUTPUT.iterdir() if p.is_file())
    save(out/'raw_manifest.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(raw)])
    plot(out,heights,pairs,training)
    print(f'A500 analysis complete; behavioral screen: {screening["status"]}',flush=True)


def plot(out,heights,pairs,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,profile in enumerate(('nominal','training_randomized')):
        for col,direction in enumerate(('up','down')):
            ax=axes[row,col]
            for policy,label,offset in (('baseline','Low stairs 10000',-.18),('candidate','A 10500',.18)):
                rows=[next(r for r in heights if (r['policy'],r['profile'],r['direction'],r['height'])==
                           (policy,profile,direction,h)) for h in (0.,.02,.03,.05)]
                ax.bar(np.arange(4)+offset,[r['success'] for r in rows],.36,label=label)
            ax.set_xticks(range(4),['Flat','2 cm','3 cm','5 cm']);ax.set_ylim(0,65)
            ax.set_ylabel('Full course completed / 60');ax.set_title(f'{profile} / {direction}')
    axes[0,0].legend();fig.tight_layout();fig.savefig(out/'stairs_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(14,7))
    for ax,profile in zip(axes,('nominal','training_randomized')):
        rows=[next((r for r in pairs if (r['profile'],r['case'],r['window'])==(profile,case,'steady')),None) for case in CASES]
        for policy,label in (('baseline','Low stairs 10000'),('candidate','A 10500')):
            ax.plot(range(len(CASES)),[r[f'{policy}_mae'] if r else np.nan for r in rows],marker='o',label=label)
        ax.set_xticks(range(len(CASES)),[f'{case}\nn={r["n"] if r else 0}' for case,r in zip(CASES,rows)],rotation=35,ha='right')
        ax.set_ylabel('Paired yaw MAE (rad/s)');ax.set_title(profile);ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'turn_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,family in zip(axes,('stairs_up','stairs_down')):
        rows=[r for r in training['coverage'] if r['family']==family]
        ax.bar([r['level'] for r in rows],[r['environment_frames'] for r in rows])
        ax.set_xlabel('Native terrain level');ax.set_ylabel('Environment control frames');ax.set_title(family)
    fig.tight_layout();fig.savefig(out/'stair_level_coverage.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
