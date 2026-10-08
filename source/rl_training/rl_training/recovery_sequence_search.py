"""Bounded transient action search and post-handoff recovery diagnostics."""
import numpy as np


def sample_population(rng, mean, std, incumbent, *, population, bound):
    if population < 3 or bound <= 0 or mean.shape != std.shape or incumbent.shape != mean.shape:
        raise ValueError('invalid search population')
    samples=np.clip(rng.normal(mean,std,size=(population,*mean.shape)),-bound,bound).astype(np.float32)
    samples[0]=0.;samples[1]=incumbent
    return samples


def update_distribution(samples, scores, *, elite_count, std_floor):
    if not 1 < elite_count <= len(samples) or not np.isfinite(scores).all() or std_floor<=0:
        raise ValueError('invalid elite update')
    # Stable ties use candidate index; no outcome-dependent replacement trials.
    elite=np.argsort(-np.asarray(scores),kind='stable')[:elite_count]
    mean=samples[elite].mean(axis=0)
    std=np.maximum(samples[elite].std(axis=0),std_floor)
    return mean,std,elite


def prefix_at_step(prefix, step, hold_steps):
    if step<0 or hold_steps<1 or prefix.ndim!=3 or prefix.shape[-1]!=12:
        raise ValueError('invalid action prefix')
    index=step//hold_steps
    return prefix[:,index] if index<prefix.shape[1] else np.zeros_like(prefix[:,0])


def short_recovery_metrics(root, forces, active, done, *, handoff_step, horizon, upright_deg=10., hold=25, loaded_steps=5, success=None):
    """Only score first episodes; require recovery after the search prefix ends."""
    count=int(active.sum())
    if count<1 or not np.array_equal(active,np.arange(len(active))<count):
        raise ValueError('first-episode active mask must be a nonempty prefix')
    state=root[:count];force=forces[:count]
    tilt=np.rad2deg(np.arccos(np.clip(1-2*np.sum(state[:,4:6]**2,axis=-1),-1,1)))
    loaded=(force[...,2]>5).sum(axis=-1)>=2
    streak=0;best=0;grounded=False
    for j in range(count):
        if j>=handoff_step and tilt[j]<=upright_deg:streak+=1
        else:streak=0
        best=max(best,streak)
        if streak>=hold and loaded[j-hold+1:j+1].sum()>=loaded_steps:grounded=True
    survived=count==horizon and not bool(done[:count].any())
    completed=bool(success is not None and np.asarray(success)[:count].any())
    positive=bool((survived or completed) and grounded)
    score=1000*positive+2*count-float(tilt.mean())+.5*best
    return dict(active_controls=count,survived_horizon=bool(survived),completed_course=completed,short_recovery=positive,
                longest_post_handoff_upright=best,mean_tilt_deg=float(tilt.mean()),score=float(score))
