"""Frozen 12 ms / 30 Nm evaluation; preserves the historical 20 ms entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

import reference_capability_mujoco as baseline
from reference_capability_mujoco import (ROOT, MODELS, SCENES, DEFAULT, SCALE,
    terrain_kind, angles, command, observation, geometric_clear, termination,
    actor, policy_action, rotation, contacts, state_hash, read, save, sha, verify_sources)
from reference_capability_mujoco_contact_audit import support_from_contacts

OUTPUT = ROOT/'logs/reference_six/capability_mujoco_deployment_20261116'
DOCS = ROOT/'docs/lite3-capability-mujoco-deployment-20261116'
PERIOD_MS = 12
RADII = (.025, .03, .04)
LIMITS = np.full(12, 30.)
LEGS = ('FL', 'FR', 'HL', 'HR')


def schedule(seconds):
    """Integer milliseconds avoid drift and cap the final partial interval exactly."""
    total = round(seconds*1000)
    if total <= 0 or not np.isclose(total/1000, seconds, rtol=0, atol=1e-10):
        raise ValueError('Duration must be a positive integer number of milliseconds')
    return [(start, min(start+PERIOD_MS, total)) for start in range(0, total, PERIOD_MS)]


def make_model(kind):
    model = baseline.make_model(kind)
    model.actuator_ctrlrange[:] = np.column_stack((-LIMITS, LIMITS))
    assert model.opt.timestep == .001 and np.all(model.actuator_ctrllimited)
    return model


def load_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def episode_path(record, root=baseline.OUTPUT):
    if record.get('smoke', False):
        return root/'smoke'/f'{record["policy"]}_{record["scene"]}'
    return root/'episodes'/f'{record["policy"]}_{record["scene"]}_{record["repeat"]}'


class ContactScorer:
    """Reconstruct contacts on a separate state; never perturb rollout warmstart."""
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.foot_ids = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
            f'{leg}_FOOT_collision') for leg in LEGS}
        self.static = model.body_weldid[model.geom_bodyid] == 0
        self.names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
            for i in range(model.ngeom)]

    def score(self, qpos, qvel, torque):
        model, data = self.model, self.data
        data.qpos[:] = qpos; data.qvel[:] = qvel; data.ctrl[:] = torque
        data.qacc_warmstart[:] = 0.
        mujoco.mj_forward(model, data)
        entries = []
        force = np.zeros(6)
        for i in range(data.ncon):
            con = data.contact[i]
            if self.static[con.geom1] == self.static[con.geom2]:
                continue
            geom = con.geom2 if self.static[con.geom1] else con.geom1
            parts = self.names[geom].split('_')
            if len(parts) < 2 or parts[0] not in LEGS or parts[1] not in ('FOOT', 'SHANK'):
                continue
            mujoco.mj_contactForce(model, data, i, force)
            distance = np.linalg.norm(con.pos-data.geom_xpos[self.foot_ids[parts[0]]])
            entries.append((parts[0], parts[1], float(distance), abs(float(force[0]))))
        return np.array([support_from_contacts(entries, r) for r in RADII]), entries


def window_stats(trace, lo, hi):
    end = trace['time']
    begin = np.r_[0., end[:-1]]
    weights = np.maximum(0., np.minimum(end, hi)-np.maximum(begin, lo))
    covered = float(weights.sum())
    complete = bool(np.isclose(covered, hi-lo, rtol=0, atol=1e-9))
    def avg(values):
        return float(np.dot(weights, values)/covered) if complete else None
    return dict(start_s=lo, end_s=hi, covered_s=covered, samples=int((weights > 0).sum()),
        complete=complete, vx_mean=avg(trace['velocity'][:, 0]),
        yaw_mean=avg(trace['velocity'][:, 5]),
        yaw_mae=avg(np.abs(trace['velocity'][:, 5]-trace['command'][:, 2])))


def validate_baseline():
    """Zero new episodes: require exact agreement with every prior sensitivity event."""
    baseline.verify_sources(read(baseline.OUTPUT/'plan.json'))
    records = read(baseline.OUTPUT/'complete.json')['records']
    audit = read(baseline.OUTPUT/'contact_audit.json')['records']
    results = []
    for record, prior in zip(records, audit, strict=True):
        assert all(record[k] == prior[k] for k in ('policy', 'scene', 'repeat'))
        path = episode_path(record)/'trace.npz'
        assert sha(path) == record['trace_sha256']
        trace = load_npz(path)
        scorer = ContactScorer(baseline.make_model(terrain_kind(record['scene'])))
        scores = [None]*3
        dwell = 0.
        for i in range(len(trace['time'])):
            supports, _ = scorer.score(trace['qpos'][i], trace['qvel'][i], trace['torque'][i])
            # Match the historical audit's 20 ms accumulation exactly.
            route = record['scene'] not in ('turn_right', 'arc_left')
            clear = geometric_clear(record['scene'], trace['qpos'][i, :3], trace['feet'][i], record['initial_xyz']) if route else False
            dwell = dwell+baseline.PERIOD if clear else 0.
            for j, support in enumerate(supports):
                result = termination(trace['rpy'][i], trace['gravity'][i],
                    trace['qpos'][i, 1]-record['initial_xyz'][1], route, clear, dwell,
                    support, bool(trace['torso'][i]))
                if result and scores[j] is None:
                    scores[j] = dict(outcome=result, duration_s=float(trace['time'][i]),
                        lateral_m=float(trace['qpos'][i, 1]-record['initial_xyz'][1]))
        for j, radius in enumerate(RADII):
            if scores[j] is None:
                scores[j] = {k: record[k] for k in ('outcome', 'duration_s', 'lateral_m')}
            assert scores[j] == prior['distal_contact_sensitivity'][str(radius)], (record, radius, scores[j])
        results.append(dict(policy=record['policy'], scene=record['scene'], repeat=record['repeat'],
            states=len(trace['time']), sensitivity=dict(zip(map(str, RADII), scores))))
        print('BASELINE CONTACT MATCH', record['policy'], record['scene'], record['repeat'], flush=True)
    save(OUTPUT/'baseline_contact_validation.json', dict(status='passed', physics_steps=0,
        policy_steps=0, records=results, source_sha256=sha(__file__)))


def audit():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    old_plan = read(baseline.OUTPUT/'plan.json')
    baseline.verify_sources(old_plan)
    validate_baseline()
    sources = {Path(item['path']) for item in old_plan['sources']}
    sources.update((Path(__file__), Path(__file__).with_name('reference_capability_mujoco_contact_audit.py'),
        DOCS/'PLAN.md', OUTPUT/'baseline_contact_validation.json', baseline.OUTPUT/'plan.json',
        baseline.OUTPUT/'complete.json', baseline.OUTPUT/'smoke.json', baseline.OUTPUT/'contact_audit.json'))
    old_records = read(baseline.OUTPUT/'complete.json')['records']+read(baseline.OUTPUT/'smoke.json')['records']
    for r in old_records:
        sources.update((episode_path(r)/'initial.npz', episode_path(r)/'trace.npz'))
    for kind in ('flat', 'stairs', 'rough'):
        old, new = baseline.make_model(kind), make_model(kind)
        for field in ('body_mass', 'geom_pos', 'geom_size', 'geom_type', 'hfield_data', 'jnt_pos'):
            np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
        assert np.array_equal(new.actuator_ctrlrange[:, 1], LIMITS)
    plan = dict(status='prepared', formal_episode_limit=36, smoke_episode_limit=4, optimizer_updates=0,
        period_ms=12, physics_step_ms=1, torque_limits=LIMITS.tolist(), nominal_pd=[30., 1.],
        primary_support_radius_m=.03, sensitivity_radii_m=list(RADII),
        startup='reuse historical settled qpos/qvel/qacc_warmstart, no new settling',
        episodes=old_plan['episodes'], actor_parity=old_plan['actor_parity'],
        sources=[dict(path=str(p), sha256=sha(p)) for p in sorted(sources)])
    save(OUTPUT/'plan.json', plan)
    save(DOCS/'execution_plan.json', plan)
    print('AUDIT PASSED; NO NEW POLICY EPISODES', flush=True)


def run_episode(name, scene, repeat, seconds=20., smoke=False):
    spec = dict(policy=name, scene=scene, repeat=repeat, smoke=smoke)
    destination = episode_path(spec, OUTPUT)
    destination.mkdir(parents=True, exist_ok=False)
    model = make_model(terrain_kind(scene)); data = mujoco.MjData(model)
    initial = load_npz(episode_path(spec)/'initial.npz')
    data.qpos[:] = initial['qpos']; data.qvel[:] = initial['qvel']; data.time = 1.
    mujoco.mj_forward(model, data)
    data.qacc_warmstart[:] = initial['qacc_warmstart']
    initial_hash = state_hash(data)
    np.savez_compressed(destination/'initial.npz', **initial)
    start = data.qpos[:3].copy()
    policy = actor(name); scorer = ContactScorer(model)
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{leg}_FOOT') for leg in LEGS]
    buffers = {key: [] for key in ('time', 'qpos', 'qvel', 'feet', 'rpy', 'gravity', 'velocity',
        'command', 'observation', 'action', 'raw_action', 'support', 'support_radii', 'torso', 'torque')}
    last = np.zeros(12); dwell = 0.
    saturation = torque_samples = 0
    scores = [None]*3
    begun = time.monotonic()
    outcome = None
    for begin_ms, end_ms in schedule(seconds):
        cmd = command(scene, begin_ms/1000, angles(data.qpos[3:7])[2])
        obs = observation(data, cmd, last)
        raw_action = policy_action(policy, obs)
        last = np.clip(raw_action, -100., 100.)
        target = DEFAULT+SCALE*last
        for _ in range(end_ms-begin_ms):
            torque = 30*(target-data.qpos[7:])-data.qvel[6:]
            saturation += int((np.abs(torque) > LIMITS).sum()); torque_samples += 12
            data.ctrl[:] = np.clip(torque, -LIMITS, LIMITS)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(np.r_[data.qpos, data.qvel]).all() or np.any(data.warning.number):
            raise FloatingPointError('Stop; invalid simulation. No automatic rerun.')
        rot = rotation(data.qpos[3:7]); gravity = rot.T @ [0., 0., -1.]
        rpy = angles(data.qpos[3:7]); feet = data.xpos[foot_ids].copy()
        supports, _ = scorer.score(data.qpos, data.qvel, data.ctrl)
        _, torso = contacts(model, data, set(scorer.foot_ids.values()))
        route = scene not in ('stand', 'turn_right', 'arc_left')
        clear = geometric_clear(scene, data.qpos[:3], feet, start) if route else False
        dwell = dwell+(end_ms-begin_ms)/1000 if clear else 0.
        values = dict(time=end_ms/1000, qpos=data.qpos.copy(), qvel=data.qvel.copy(), feet=feet,
            rpy=rpy, gravity=gravity, velocity=np.r_[rot.T@data.qvel[:3], data.qvel[3:6]],
            command=cmd, observation=obs, action=last, raw_action=raw_action, support=int(supports[1]),
            support_radii=supports, torso=torso, torque=data.ctrl.copy())
        for key, value in values.items():
            buffers[key].append(value)
        for j, support in enumerate(supports):
            result = termination(rpy, gravity, data.qpos[1]-start[1], route, clear, dwell, support, torso)
            if result and scores[j] is None:
                scores[j] = dict(outcome=result, duration_s=end_ms/1000,
                    lateral_m=float(data.qpos[1]-start[1]))
        if scores[1]:
            outcome = scores[1]['outcome']
            break
    arrays = {key: np.asarray(value) for key, value in buffers.items()}
    duration = float(arrays['time'][-1])
    if outcome is None:
        outcome = 'survived' if not route or smoke else 'timeout'
    for j in range(3):
        if scores[j] is None:
            scores[j] = dict(outcome=outcome if duration == seconds else 'censored',
                duration_s=duration, lateral_m=float(data.qpos[1]-start[1]))
    tilt = np.degrees(np.arccos(np.clip(-arrays['gravity'][:, 2], -1., 1.)))
    record = dict(**spec, outcome=outcome, duration_s=duration,
        progress_m=float((-1 if scene == 'stairs_down_5cm' else 1)*(data.qpos[0]-start[0])),
        lateral_m=float(data.qpos[1]-start[1]), drift_xy_m=float(np.linalg.norm(data.qpos[:2]-start[:2])),
        max_tilt_deg=float(tilt.max()), initial_xyz=start.tolist(), final_xyz=data.qpos[:3].tolist(),
        initial_state_sha256=initial_hash, model_sha256=MODELS[name][1], source_sha256=sha(__file__),
        mean_vx_after_2s=window_stats(arrays, 2., duration)['vx_mean'] if duration > 2 else None,
        torque_saturation_fraction=saturation/torque_samples, torso_contact_samples=int(arrays['torso'].sum()),
        max_action_abs=float(np.abs(arrays['raw_action']).max()), optimizer_updates=0,
        windows=[window_stats(arrays, lo, hi) for lo, hi in ((4., 8.), (10., 14.), (16., 20.))],
        sensitivity=dict(zip(map(str, RADII), scores)), physics_steps=end_ms, elapsed_s=time.monotonic()-begun)
    if smoke:
        record['smoke_passed'] = bool(outcome == 'survived' and
            (record['max_tilt_deg'] < 15 and record['drift_xy_m'] < .25 if scene == 'stand' else record['mean_vx_after_2s'] > .15))
    np.savez_compressed(destination/'trace.npz', **arrays)
    record['trace_sha256'] = sha(destination/'trace.npz')
    save(destination/'report.json', record)
    assert sha(MODELS[name][0]) == MODELS[name][1]
    print(name, scene, repeat, outcome, f'{duration:.3f}s', flush=True)
    return record


def run():
    plan = read(OUTPUT/'plan.json'); verify_sources(plan)
    if (OUTPUT/'run_started.json').exists():
        raise FileExistsError('Run already started; no automatic rerun')
    save(OUTPUT/'run_started.json', dict(phase='smoke', optimizer_updates=0))
    smoke = []
    for scene, seconds in (('stand', 5.), ('flat', 6.)):
        for name in MODELS:
            smoke.append(run_episode(name, scene, 0, seconds, True))
            passed = smoke[-1]['smoke_passed']
            save(OUTPUT/'smoke.json', dict(status=('passed' if len(smoke) == 4 else 'running') if passed else 'failed', records=smoke))
            if not passed:
                print('STOP: SMOKE FAILED; NO FURTHER EPISODES', flush=True)
                return
    records = []
    for spec in plan['episodes']:
        records.append(run_episode(spec['policy'], spec['scene'], spec['repeat']))
        save(OUTPUT/'progress.json', dict(formal_completed=len(records), formal_limit=36, last=records[-1]))
    old = read(baseline.OUTPUT/'complete.json')['records']+read(baseline.OUTPUT/'smoke.json')['records']
    for r, prior in zip(records+smoke, old, strict=True):
        assert r['initial_state_sha256'] == prior['initial_state_sha256']
    for scene in SCENES:
        for repeat in range(3):
            assert len({r['initial_state_sha256'] for r in records if r['scene'] == scene and r['repeat'] == repeat}) == 1
    verify_sources(plan)
    save(OUTPUT/'complete.json', dict(status='completed', formal_episodes=36, smoke_episodes=4,
        optimizer_updates=0, identical_cross_profile_initial_states=True, records=records))
    print('COMPLETE: 36 formal, 4 smoke, 0 optimizer updates', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('audit', 'run'))
    args = parser.parse_args()
    audit() if args.phase == 'audit' else run()
