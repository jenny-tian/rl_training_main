"""Reproducible offline diagnosis of saved first episodes, with explicit limits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_evaluate import OUTPUT, TRAINING
from reference_core500_analyze import write_csv

LEGS = ('FL', 'FR', 'HL', 'HR')
DT = .02


def longest(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return float((stops - starts).max(initial=0) * DT)


def stair_features(feet, forces, geometry):
    """Geometry-based proxies, not per-contact manifold classifications.

    Native stairs are square rings. Surface estimates use max(|x|,|y|).
    Collision probes are restricted to the +x face, away from corners.
    Measured edge coordinates bracket the actual face within about 2 mm.
    """
    edges, tops = np.asarray(geometry['fronts']), np.asarray(geometry['tops'])
    radius = np.abs(feet[..., :2]).max(-1)
    levels = (radius[..., None] >= edges).sum(-1)
    ground = np.r_[0., tops][levels]
    bottom = feet[..., 2] - .022
    support = (forces[..., 2] > 1.) & (np.abs(bottom - ground) <= .025)
    dx = feet[..., 0, None] - edges
    central = feet[..., 0] > np.abs(feet[..., 1]) + .04
    collision = ((np.abs(dx) < .032) & (bottom[..., None] < tops - .005)).any(-1)
    collision &= central & (forces[..., 0] < -5.)
    # Stricter subset: foot centre is below the upper surface, in front of a
    # vertical face, as required by the new horizontal-ray contact term.
    wall = ((dx <= .002) & (dx >= -.032) & (feet[..., 2, None] < tops)
            & (feet[..., 2, None] > np.r_[0., tops[:-1]] + .002)).any(-1)
    wall &= central & (forces[..., 0] < -5.)
    return dict(ground=ground, bottom=bottom, support=support, collision=collision, wall=wall)


def analyze_stairs(directory, joint_limits=None):
    report = json.loads((directory/'report.json').read_text())
    cfg = json.loads((directory/'config.json').read_text())
    geom = json.loads((directory/'geometry.json').read_text())
    with np.load(directory/'trace.npz') as loaded:
        data = {k: loaded[k] for k in ('active','root','feet','forces','gravity','torque','joint_pos','joint_vel')}
    limits = np.array([36. if 'Knee' in name else 24. for name in cfg['joint_names']])
    speed_limits = np.array([17.3 if 'Knee' in name else 26.2 for name in cfg['joint_names']])
    if joint_limits is not None:
        assert joint_limits['joint_names']==cfg['joint_names']
        position_limits=np.asarray(joint_limits['soft_joint_limits'])
    rows, crossings, snapshots = [], [], []
    for record in report['records']:
        if not record['height']:
            continue
        i = record['env_id']
        n = np.count_nonzero(data['active'][:, i])
        assert data['active'][:n, i].all() and not data['active'][n:, i].any()
        f, force = data['feet'][:n, i], data['forces'][:n, i]
        x = stair_features(f, force, geom[record['column']])
        gravity = data['gravity'][:n, i]
        tilt = np.degrees(np.arccos(np.clip(-gravity[:, 2], -1., 1.)))
        # Stop at the first 20-degree tilt. Do not mix recovery/post-fall
        # collisions with evidence preceding the onset of loss of posture.
        tilt_ids = np.flatnonzero(tilt > 20.)
        first_tilt = int(tilt_ids[0]) if len(tilt_ids) else n
        before = (np.arange(n) < first_tilt) & (data['root'][:n,i,0] > .85)
        sat = np.abs(data['torque'][:n,i]) >= .95 * limits
        row = dict(profile=report['profile'], **record, before_tilt_s=float(before.sum()*DT),
                   first_tilt20_s=float((first_tilt+1)*DT) if first_tilt<n else None,
                   max_tilt_deg=float(tilt.max()),
                   collision_before_tilt=bool(x['collision'][before].any()),
                   collision_foot_seconds=float(x['collision'][before].sum()*DT),
                   wall_foot_seconds=float(x['wall'][before].sum()*DT),
                   collision_foot_fraction=float(x['collision'][before].mean()) if before.any() else None,
                   torque_saturated_fraction=float(sat[before].mean()) if before.any() else None,
                   velocity_near_limit_fraction=float((np.abs(data['joint_vel'][:n,i][before])>=.95*speed_limits).mean()) if before.any() else None,
                   max_joint_speed=float(np.abs(data['joint_vel'][:n,i]).max()),
                   # Last 0.5 s is descriptive of failure, not its cause.
                   final_support_count=float(x['support'][-25:].sum(-1).mean()),
                   final_support_left=float(x['support'][-25:,[0,2]].mean()),
                   final_support_right=float(x['support'][-25:,[1,3]].mean()))
        if joint_limits is not None:
            q=data['joint_pos'][:n,i][before]
            row['soft_position_limit_fraction']=float(((q<position_limits[:,0])|(q>position_limits[:,1])).mean()) if before.any() else None
        for j,leg in enumerate(LEGS):
            row[f'{leg}_collision_s'] = float(x['collision'][before,j].sum()*DT)
            row[f'{leg}_longest_collision_s'] = longest(x['collision'][:,j] & before)
            row[f'{leg}_final_support'] = float(x['support'][-25:,j].mean())
        rows.append(row)
        # First approach to each edge by each foot; captures insufficient
        # leading-sphere clearance before contact, even if crossing fails.
        for j,leg in enumerate(LEGS):
            for edge,top in zip(geom[record['column']]['fronts'],geom[record['column']]['tops']):
                reaches = np.flatnonzero((f[:,j,0] >= edge-.032)
                                         & (f[:,j,0] > np.abs(f[:,j,1])+.04))
                if not len(reaches):
                    continue
                t = int(reaches[0])
                if t >= first_tilt:
                    continue
                land = np.flatnonzero((np.arange(n)>=t) & (f[:,j,0]>=edge+.022)
                    & x['support'][:,j] & (np.abs(x['ground'][:,j]-top)<.005))
                crossings.append(dict(profile=report['profile'], env_id=i, height=record['height'],
                    speed=record['speed'], outcome=record['outcome'], leg=leg, edge=edge,
                    approach_s=float((t+1)*DT), bottom_above_top=float(x['bottom'][t,j]-top),
                    landing_after_approach_s=float((land[0]-t)*DT) if len(land) else None,
                    landing_x_past_edge=float(f[land[0],j,0]-edge) if len(land) else None,
                    landing_y=float(f[land[0],j,1]) if len(land) else None))
        if report['profile']=='nominal' and i in (140,180,200):
            ids=np.flatnonzero(x['collision'].any(-1))
            for label,t in [('first_collision',int(ids[0])),('first_tilt20',min(first_tilt,n-1)),('last',n-1)]:
                snapshots.append(dict(env_id=i,label=label,time_s=float((t+1)*DT),
                    tilt_deg=float(tilt[t]),root=data['root'][t,i].tolist(),feet=f[t].tolist(),
                    forces=force[t].tolist(),support=x['support'][t].tolist(),
                    torque=data['torque'][t,i].tolist(),joint_names=cfg['joint_names']))
    return rows, crossings, snapshots


def grouped(rows):
    result=[]
    for profile,height,speed in sorted({(r['profile'],r['height'],r['speed']) for r in rows}):
        subset=[r for r in rows if (r['profile'],r['height'],r['speed'])==(profile,height,speed)]
        result.append(dict(profile=profile,height=height,speed=speed,starts=len(subset),
            successes=sum(r['outcome']=='success' for r in subset),
            collision_before_tilt=sum(r['collision_before_tilt'] for r in subset),
            **{k:float(np.mean([r[k] for r in subset if r[k] is not None]))
               for k in ('collision_foot_seconds','wall_foot_seconds','collision_foot_fraction',
                         'torque_saturated_fraction','final_support_count','final_support_left','final_support_right')}))
    return result


def plot_failure(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=OUTPUT/'up_nominal'
    with np.load(path/'trace.npz') as loaded:
        data={k:loaded[k] for k in ('feet','forces','gravity')}
    geometry=json.loads((path/'geometry.json').read_text())[3]
    i=180;keep=np.arange(125,350);t=(keep+1)*DT
    feet,force=data['feet'][keep,i],data['forces'][keep,i]
    x=stair_features(feet,force,geometry)
    tilt=np.degrees(np.arccos(np.clip(-data['gravity'][keep,i,2],-1.,1.)))
    fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True,layout='constrained')
    for j,leg in enumerate(LEGS[:2]):
        axes[0].plot(t,feet[:,j,0],label=leg)
        axes[1].plot(t,x['bottom'][:,j]*100,label=leg)
        axes[2].plot(t,np.maximum(-force[:,j,0],0.),label=leg)
    axes[0].axhline(geometry['fronts'][0]-.022,color='k',ls='--',label='Sphere reaches riser')
    axes[0].set_ylabel('Front foot x (m)');axes[0].legend(ncol=3,fontsize=9)
    axes[1].axhline(5.,color='k',ls='--',label='First tread: 5 cm')
    axes[1].set_ylabel('Foot bottom (cm)');axes[1].legend(ncol=3,fontsize=9)
    axes[2].set_ylabel('Opposing foot force (N)');axes[2].legend(ncol=2,fontsize=9)
    axes[3].plot(t,tilt,color='tab:red',label='Body tilt')
    axes[3].axhline(20.,color='k',ls='--',label='20 deg')
    axes[3].set_ylabel('Body tilt (deg)');axes[3].set_xlabel('Time since episode start (s)')
    axes[3].legend(ncol=2,fontsize=9)
    for ax in axes:
        ax.axvline(3.56,color='gray',ls=':',lw=1)
        ax.axvline(5.24,color='gray',ls=':',lw=1)
        ax.grid(alpha=.2)
    fig.suptitle('Frozen low-stair policy: 5 cm, 0.32 m/s, nominal, fixed repeat 0\n'
                 'First riser contact 3.56 s; first tilt above 20 deg at 5.24 s')
    fig.savefig(output/'first_riser_failure.png',dpi=160)
    fig.savefig(output/'first_riser_failure.pdf')
    plt.close(fig)


def turn_audit():
    from reference_reward_code_audit import probe_commands
    # This file is the before-reward-override audit. Commands were not changed.
    audit=json.loads((TRAINING/'configuration_audit.json').read_text())
    command, sources=probe_commands(audit['candidate_environment']['commands']['base_velocity'])
    overrides=json.loads((TRAINING/'reward_override_audit.json').read_text())
    rows=[]
    for profile in ('nominal','training_randomized'):
        path=OUTPUT/f'turn_{profile}'
        cfg=json.loads((path/'config.json').read_text())
        report=json.loads((path/'report.json').read_text())
        with np.load(path/'trace.npz') as loaded:
            d={k:loaded[k] for k in ('active','velocity','reward_rates')}
        names=cfg['reward_names']
        for case in ('low_left','low_right','turn_left','turn_right','fast_left','fast_right'):
            specs=[r for r in report['records'] if r['case']==case]
            ids=[r['env_id'] for r in specs if d['active'][200:400,r['env_id']].all()]
            rates=d['reward_rates'][200:400,ids]
            # These terms are unchanged in low_stairs; the evaluation gait
            # multiplier is recorded and may differ from training history.
            yaw=rates[:,:,names.index('feet_air_time_ang_z_lite3')]
            rows.append(dict(profile=profile,case=case,complete_windows=len(ids),
                actual_yaw=float(d['velocity'][200:400,ids,5].mean()),
                angular_airtime_rate=float(yaw.mean()),
                negative_airtime_rate=float(np.minimum(yaw,0.).mean()),
                positive_airtime_rate=float(np.maximum(yaw,0.).mean()),
                phase_rate=float(rates[:,:,names.index('phase_foot_trajectory_exp')].mean()),
                tracking_rate=float(rates[:,:,names.index('track_ang_vel_z_exp')].mean())))
    return dict(command_sampling=command,sources=sources,turn_windows=rows,
        current_override_terms=sorted(overrides['changes']),
        maximum_tracking_gain_from_static={str(w):float(1.5*(1.-np.exp(-w*w/.5))) for w in (.2,.5,.8)},
        caveat='Unchanged angular airtime/phase function scores in frozen evaluation, not historical training returns. '
               'Standing gate and height have already been repaired and are NOT inferred from common-reference scores.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--integration',type=Path,help='Successful frozen run providing actual joint limits')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    limits=json.loads((args.integration/'report.json').read_text()) if args.integration else None
    if limits is not None:
        assert limits['status']=='passed' and limits['actor_unchanged'] and limits['optimizer_updates']==0
    rows,crossings,snapshots=[],[],[]
    for name in ('up_nominal','up_training_randomized'):
        r,c,s=analyze_stairs(OUTPUT/name,limits);rows+=r;crossings+=c;snapshots+=s
    write_csv(args.output/'stairs_episodes.csv',rows)
    write_csv(args.output/'stairs_groups.csv',grouped(rows))
    write_csv(args.output/'foot_approaches.csv',crossings)
    save(args.output/'snapshots.json',snapshots)
    turn=turn_audit();save(args.output/'turn_audit.json',turn)
    plot_failure(args.output)
    save(args.output/'diagnosis.json',dict(status='passed',episodes=len(rows),groups=grouped(rows),
        analytic_geometry_tolerance_m=.002,foot_radius_m=.022,
        collision_definition='Near +x riser, foot bottom below top, opposing force >5 N; a proxy, not a contact manifold.',
        torque_definition='Applied torque >=95% of configured actuator limit at recorded 20 ms control frames; not every 5 ms physics tick.',
        limitations=['Association does not identify the unique training cause.',
                     'Ground estimates use square-ring geometry; very near edges have millimetre uncertainty.',
                     'No measurement of non-foot contacts in the saved stair traces.'],
        sources=[dict(path=str(OUTPUT/name/'trace.npz'),sha256=sha(OUTPUT/name/'trace.npz'))
                 for name in ('up_nominal','up_training_randomized','turn_nominal','turn_training_randomized')]))
    print(json.dumps(dict(stair_episodes=len(rows),groups=grouped(rows),turn=turn),ensure_ascii=False))


if __name__=='__main__':main()
