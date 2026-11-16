"""Bounded, frozen zero10100/B comparison on isolated MuJoCo deployment assets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from reference_stairs_eval import (ROOT, DEPLOY, ROBOT_XML, STAIR_XML, JOINTS,
    DEFAULT, SCALE, save, sha, load_actor, policy_action, rotation, contacts)
from reference_eval_checkpoint import register_checkpoint

OUTPUT = ROOT/'logs/reference_six/capability_mujoco_20261116'
DOCS = ROOT/'docs/lite3-capability-mujoco-20261116'
MODELS = {
    'zero10100': (ROOT/'logs/reference_six/low_stairs_zero_preservation_20261115/train_001_100/model_updates_10100.pt',
        '290d94746c467b5cd21ef74c47a908011521c1f145da35ba7b94f1b92d9b7497'),
    'B': (ROOT/'logs/reference_six/stair_allocation_20261116/B/train_001_100/model_updates_10200.pt',
        'aba3a52e362fbb6f6c860bb8db384c12a84d13f6f39b53cae16465eaf2a84b79')}
SCENES = ('flat', 'stairs_up_5cm', 'stairs_down_5cm', 'rough', 'turn_right', 'arc_left')
DT, PERIOD = .001, .02
LIMITS = np.tile([24., 24., 36.], 4)
JITTER = np.random.default_rng(20261116).uniform([-.05, -.1, -.03], [.05, .1, .03], (3, 3))


def read(path):
    return json.loads(Path(path).read_text())


def terrain_kind(scene):
    return 'stairs' if scene.startswith('stairs') else 'rough' if scene == 'rough' else 'flat'


def rough_heights():
    heights = np.random.default_rng(20261116).integers(1, 7, (100, 100))*.01
    # Columns follow world x; x=-2..7.9, y=-4.95..4.95, both at exactly 0.1 m.
    heights[:, :29] = 0.
    return heights


def build_scene(kind):
    if kind not in ('flat', 'stairs', 'rough'):
        raise ValueError(kind)
    root = ET.parse(ROBOT_XML).getroot()
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', str((ROBOT_XML.parent/mesh.get('file')).resolve()))
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', str(DT)); option.set('gravity', '0 0 -9.81')
    for motor, limit in zip(root.findall('./actuator/motor'), LIMITS, strict=True):
        motor.set('ctrlrange', f'{-limit} {limit}'); motor.set('ctrllimited', 'true')
    if kind == 'stairs':
        stairs = copy.deepcopy(ET.parse(STAIR_XML).getroot().find('./worldbody/body'))
        for index, geom in enumerate(stairs.findall('geom'), 1):
            xyz = [float(x) for x in geom.get('pos').split()]
            size = [float(x) for x in geom.get('size').split()]
            xyz[2] = size[2] = index*.05/2
            geom.set('pos', ' '.join(map(str, xyz))); geom.set('size', ' '.join(map(str, size)))
        root.find('worldbody').append(stairs)
    if kind == 'rough':
        ET.SubElement(root.find('asset'), 'hfield', name='rough_field', nrow='100', ncol='100', size='4.95 4.95 0.06 0.1')
        ET.SubElement(root.find('worldbody'), 'geom', name='rough_surface', type='hfield',
            hfield='rough_field', pos='2.95 0 0', rgba='0.45 0.50 0.43 1', friction='1 0.005 0.0001')
        # Avoid duplicate coplanar floor contacts on the flat approach.
        root.find('worldbody').remove(root.find('./worldbody/geom[@name="floor"]'))
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='640', offheight='360')
    return ET.tostring(root, encoding='unicode')


def make_model(kind):
    model = mujoco.MjModel.from_xml_string(build_scene(kind))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(v[0])) for v in model.actuator_trnid]
    assert names == JOINTS and (model.nq, model.nv, model.nu) == (19, 18, 12)
    assert np.array_equal(model.jnt_qposadr[1:], np.arange(7, 19))
    assert np.array_equal(model.jnt_dofadr[1:], np.arange(6, 18))
    assert np.array_equal(model.actuator_ctrlrange[:, 1], LIMITS)
    if kind == 'rough':
        model.hfield_data[:] = (rough_heights()/.06).ravel()
    return model


def angles(q):
    rot = rotation(q)
    return np.array([math.atan2(rot[2, 1], rot[2, 2]),
        math.asin(np.clip(-rot[2, 0], -1., 1.)), math.atan2(rot[1, 0], rot[0, 0])])


def observation(data, command, last_action):
    gravity = rotation(data.qpos[3:7]).T @ [0., 0., -1.]
    # Isaac clips each raw term before scaling; actions were clipped before use.
    pieces = (data.qvel[3:6], gravity, command, data.qpos[7:]-DEFAULT, data.qvel[6:], last_action)
    scales = (.25, 1., 1., 1., .05, 1.)
    return np.concatenate([np.clip(v, -100., 100.)*s for v, s in zip(pieces, scales)]).astype(np.float32)


def command(scene, t, yaw):
    if scene == 'stand':
        return np.zeros(3)
    if scene in ('turn_right', 'arc_left'):
        return np.zeros(3) if t < 2. else np.array([0., 0., -.5] if scene == 'turn_right' else [.3, 0., .5])
    result = np.array([.4*np.clip((t-.5)/.5, 0., 1.), 0., 0.])
    if scene.startswith('stairs') and result[0] > 0:
        target = math.pi if scene == 'stairs_down_5cm' else 0.
        error = (target-yaw+math.pi)%(2*math.pi)-math.pi
        result[2] = np.clip(.5*error, -.8, .8)
    return result


def prepare(model, scene, repeat):
    data = mujoco.MjData(model)
    dx, dy, dyaw = JITTER[repeat]
    down = scene == 'stairs_down_5cm'
    yaw = (math.pi if down else 0.)+dyaw
    data.qpos[:3] = [3.55+dx if down else .25+dx, dy, .675 if down else .375]
    data.qpos[3:7] = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
    data.qpos[7:] = DEFAULT
    mujoco.mj_forward(model, data)
    for _ in range(1000):
        data.ctrl[:] = np.clip(30*(DEFAULT-data.qpos[7:])-data.qvel[6:], -LIMITS, LIMITS)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert np.max(np.abs(angles(data.qpos[3:7])[:2])) < math.radians(15)
    return data


def geometric_clear(scene, root, feet, start):
    if scene == 'stairs_up_5cm':
        return bool(feet[:, 0].min() > 2.70 and root[0] > 2.95 and .18 < root[2]-.3 < .55)
    if scene == 'stairs_down_5cm':
        return bool(feet[:, 0].max() < 1.10 and root[0] < .85 and .18 < root[2] < .55)
    return bool(root[0]-start[0] >= 3. and .18 < root[2] < .61)


def termination(rpy, gravity, lateral, route, clear, dwell, support, torso):
    if abs(rpy[0]) > math.pi/6 or abs(rpy[1]) > math.pi/4 or gravity[2] > 0 or np.max(np.abs(gravity[:2])) > .7:
        return 'orientation_failure'
    if route and abs(lateral) > 1.:
        return 'lateral_exit'
    if route and clear and dwell >= .25 and support >= 2 and not torso:
        return 'success'
    return None


def actor(name):
    path, digest = MODELS[name]
    register_checkpoint(name, path, digest)
    return load_actor(name)


def state_hash(data):
    return hashlib.sha256(np.concatenate((data.qpos, data.qvel, data.qacc_warmstart)).tobytes()).hexdigest()


def run_episode(name, scene, repeat, destination, seconds=20., smoke=False):
    if destination.exists():
        raise FileExistsError(f'Preserving existing episode: {destination}')
    destination.mkdir(parents=True)
    model = make_model(terrain_kind(scene))
    policy = actor(name)
    data = prepare(model, scene, repeat)
    start = data.qpos[:3].copy()
    initial_hash = state_hash(data)
    np.savez_compressed(destination/'initial.npz', qpos=data.qpos, qvel=data.qvel, qacc_warmstart=data.qacc_warmstart)
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{s}_FOOT') for s in ('FL', 'FR', 'HL', 'HR')]
    foot_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{s}_FOOT_collision') for s in ('FL', 'FR', 'HL', 'HR')}
    assert all(i >= 0 for i in [*foot_ids, *foot_geoms])
    route = scene not in ('stand', 'turn_right', 'arc_left')
    buffers = {key: [] for key in ('time', 'qpos', 'qvel', 'feet', 'rpy', 'gravity', 'velocity',
        'command', 'observation', 'action', 'raw_action', 'support', 'torso', 'torque')}
    last = np.zeros(12)
    dwell = 0.
    saturation = samples = 0
    outcome = None
    begun = time.monotonic()
    for tick in range(round(seconds/PERIOD)):
        cmd = command(scene, tick*PERIOD, angles(data.qpos[3:7])[2])
        obs = observation(data, cmd, last)
        raw_action = policy_action(policy, obs)
        last = np.clip(raw_action, -100., 100.)
        target = DEFAULT+SCALE*last
        for _ in range(20):
            torque = 30*(target-data.qpos[7:])-data.qvel[6:]
            saturation += int((np.abs(torque) > LIMITS).sum()); samples += 12
            data.ctrl[:] = np.clip(torque, -LIMITS, LIMITS)
            mujoco.mj_step(model, data)
        # Sync derived position/contact arrays to the saved post-step state.
        mujoco.mj_forward(model, data)
        if not np.isfinite(np.r_[data.qpos, data.qvel]).all() or np.any(data.warning.number):
            raise FloatingPointError('Invalid simulation; preserve directory and stop without automatic rerun')
        rot = rotation(data.qpos[3:7]); gravity = rot.T @ [0., 0., -1.]
        rpy = angles(data.qpos[3:7]); feet = data.xpos[foot_ids].copy()
        support, torso = contacts(model, data, foot_geoms)
        clear = geometric_clear(scene, data.qpos[:3], feet, start) if route else False
        dwell = dwell+PERIOD if clear else 0.
        values = dict(time=(tick+1)*PERIOD, qpos=data.qpos.copy(), qvel=data.qvel.copy(), feet=feet,
            rpy=rpy, gravity=gravity, velocity=np.r_[rot.T@data.qvel[:3], data.qvel[3:6]],
            command=cmd, observation=obs, action=last, raw_action=raw_action,
            support=support, torso=torso, torque=data.ctrl.copy())
        for key, value in values.items():
            buffers[key].append(value)
        outcome = termination(rpy, gravity, data.qpos[1]-start[1], route, clear, dwell, support, torso)
        if outcome:
            break
    arrays = {key: np.asarray(value) for key, value in buffers.items()}
    if outcome is None:
        outcome = 'survived' if not route or smoke else 'timeout'
    duration = len(arrays['time'])*PERIOD
    tilt = np.degrees(np.arccos(np.clip(-arrays['gravity'][:, 2], -1., 1.)))
    progress = (-1 if scene == 'stairs_down_5cm' else 1)*(data.qpos[0]-start[0])
    windows = []
    for lo, hi in ((4., 8.), (10., 14.), (16., 20.)):
        mask = (arrays['time'] > lo+1e-8) & (arrays['time'] <= hi+1e-8)
        complete = int(mask.sum()) == 200
        windows.append(dict(start_s=lo, end_s=hi, samples=int(mask.sum()), complete=complete,
            vx_mean=float(arrays['velocity'][mask, 0].mean()) if complete else None,
            yaw_mean=float(arrays['velocity'][mask, 5].mean()) if complete else None,
            yaw_mae=float(np.abs(arrays['velocity'][mask, 5]-arrays['command'][mask, 2]).mean()) if complete else None))
    record = dict(policy=name, scene=scene, repeat=repeat, smoke=smoke, outcome=outcome,
        duration_s=duration, progress_m=float(progress), lateral_m=float(data.qpos[1]-start[1]),
        drift_xy_m=float(np.linalg.norm(data.qpos[:2]-start[:2])), max_tilt_deg=float(tilt.max()),
        final_xyz=data.qpos[:3].tolist(), initial_xyz=start.tolist(), initial_state_sha256=initial_hash,
        mean_vx_after_2s=float(arrays['velocity'][arrays['time'] > 2., 0].mean()) if duration > 2. else None,
        max_action_abs=float(np.abs(arrays['raw_action']).max()),
        torque_saturation_fraction=saturation/samples, torso_contact_samples=int(arrays['torso'].sum()),
        windows=windows, model_sha256=MODELS[name][1], optimizer_updates=0,
        source_sha256=sha(__file__), elapsed_s=time.monotonic()-begun)
    if smoke:
        record['smoke_passed'] = bool(outcome == 'survived' and
            (record['max_tilt_deg'] < 15 and record['drift_xy_m'] < .25 if scene == 'stand' else record['mean_vx_after_2s'] > .15))
    np.savez_compressed(destination/'trace.npz', **arrays)
    record['trace_sha256'] = sha(destination/'trace.npz')
    save(destination/'report.json', record)
    assert sha(MODELS[name][0]) == MODELS[name][1]
    print(name, scene, repeat, outcome, f'{duration:.2f}s', flush=True)
    return record


def source_files():
    files = {Path(__file__), Path(__file__).with_name('reference_stairs_eval.py'),
        Path(__file__).with_name('reference_eval_checkpoint.py'), ROBOT_XML, STAIR_XML,
        DOCS/'PLAN.md', DEPLOY/'interface/robot/simulation/mujoco_simulation.py',
        DEPLOY/'run_policy/lite3_test_policy_runner_onnx.cpp', DEPLOY/'run_policy/lite3_test_policy_runner_onnx.h',
        DEPLOY/'state_machine/rl_control_state_onnx.hpp', DEPLOY/'state_machine/standup_state.hpp',
        DEPLOY/'state_machine/parameters/lite3_control_parameters.cpp',
        DEPLOY/'Lite3_description/lite3_mjcf/mjcf/Lite3_stair.xml',
        Path('/home/tjx/IsaacLab/source/isaaclab/isaaclab/terrains/height_field/hf_terrains.py')}
    for mesh in ET.parse(ROBOT_XML).getroot().findall('./asset/mesh'):
        files.add((ROBOT_XML.parent/mesh.get('file')).resolve())
    for path, _ in MODELS.values():
        files.add(path); files.add(path.parent/'configuration.json')
    return [dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(files)]


def verify_sources(plan):
    assert all(sha(p['path']) == p['sha256'] for p in plan['sources'])


def audit():
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    OUTPUT.mkdir(parents=True, exist_ok=False)
    parity = {}
    configs = [read(p.parent/'configuration.json') for p, _ in MODELS.values()]
    for key in ('actions', 'observations'):
        assert configs[0]['environment'][key] == configs[1]['environment'][key]
    for cfg in configs:
        env = cfg['environment']
        assert env['sim']['dt']*env['decimation'] == PERIOD
        assert not cfg['agent']['actor']['obs_normalization']
        assert env['actions']['joint_pos']['joint_names'] == JOINTS
        assert env['actions']['joint_pos']['preserve_order']
        assert cfg['agent']['clip_actions'] == 100
        assert cfg['agent']['actor']['hidden_dims'] == [512, 256, 128]
        assert cfg['agent']['actor']['activation'] == 'elu'
        assert env['scene']['robot']['init_state']['joint_pos'] == {
            '.*HipX_joint': 0., '.*HipY_joint': -.65, '.*Knee_joint': 1.3}
        assert env['actions']['joint_pos']['scale'] == {'.*_HipX_joint': .125, '^(?!.*_HipX_joint).*': .25}
        obs_cfg = env['observations']['policy']
        terms = ('base_ang_vel', 'projected_gravity', 'velocity_commands', 'joint_pos', 'joint_vel', 'actions')
        assert [k for k, v in obs_cfg.items() if isinstance(v, dict)] == list(terms)
        assert [obs_cfg[k]['scale'] for k in terms] == [.25, 1., 1., 1., .05, 1.]
        assert all(obs_cfg[k]['clip'] == [-100, 100] for k in terms)
        assert all(obs_cfg[k]['params']['asset_cfg']['joint_names'] == JOINTS for k in ('joint_pos', 'joint_vel'))
        for key, expected in (('Hip', 24.), ('Knee', 36.)):
            params = env['scene']['robot']['actuators'][key]
            assert (params['stiffness'], params['damping'], params['effort_limit']) == (30., 1., expected)
    observations = torch.from_numpy(np.random.default_rng(42).normal(size=(256, 45)).astype(np.float32))
    for name, (path, digest) in MODELS.items():
        policy = actor(name)
        native = MLPModel(TensorDict({'policy': observations}, batch_size=[256]), {'actor': ['policy']},
            'actor', 12, hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False).eval()
        state = torch.load(path, map_location='cpu', weights_only=True)['actor_state_dict']
        assert set(k for k in state if not k.startswith('mlp.')) == {'distribution.log_std_param'}
        native.load_state_dict({k: v for k, v in state.items() if k.startswith('mlp.')}, strict=True)
        with torch.inference_mode():
            a, b = policy(observations), native(TensorDict({'policy': observations}, batch_size=[256]))
        assert torch.equal(a, b)
        parity[name] = dict(checkpoint_sha256=digest, samples=256, maximum_action_difference=float((a-b).abs().max()))
    geometries = {}
    for kind in ('flat', 'stairs', 'rough'):
        model = make_model(kind)
        (OUTPUT/f'{kind}.xml').write_text(build_scene(kind))
        info = dict(xml_sha256=sha(OUTPUT/f'{kind}.xml'), joints=JOINTS, timestep=model.opt.timestep,
            actuator_limits=model.actuator_ctrlrange.tolist(), body_mass_total=float(model.body_mass.sum()))
        if kind == 'stairs':
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'stair{i}') for i in range(1, 7)]
            data = mujoco.MjData(model); mujoco.mj_forward(model, data)
            tops = data.geom_xpos[ids, 2]+model.geom_size[ids, 2]
            assert np.allclose(tops, np.arange(1, 7)*.05)
            info.update(top_heights=tops.tolist(), x_edges=(data.geom_xpos[ids, 0]-model.geom_size[ids, 0]).tolist(),
                widths=(model.geom_size[ids, 0]*2).tolist())
        if kind == 'rough':
            np.save(OUTPUT/'rough_heights.npy', rough_heights())
            info.update(heights_sha256=sha(OUTPUT/'rough_heights.npy'), heights_m=np.unique(rough_heights()).tolist(),
                spacing_m=.1, flat_approach_ends_x=.8, world_bounds=[-2., 7.9, -4.95, 4.95])
        geometries[kind] = info
    records = [dict(policy=n, scene=s, repeat=r) for s in SCENES for r in range(3) for n in MODELS]
    cpp = (DEPLOY/'run_policy/lite3_test_policy_runner_onnx.cpp').read_text()
    header = (DEPLOY/'run_policy/lite3_test_policy_runner_onnx.h').read_text()
    assert all(text in cpp for text in ('decimation_ = 12;', 'kp_ = 30.', 'kd_ =  1.', 'max_cmd_vel_ << 0.8, 0.8, 0.8;'))
    assert all(text in header for text in ('obs_dim_ = 45', 'act_dim_ = 12', 'omega_scale_ = 0.25', 'dof_vel_scale_ = 0.05'))
    plan = dict(status='prepared', formal_episode_limit=36, smoke_episode_limit=4, optimizer_updates=0,
        policy_period_s=PERIOD, physics_step_s=DT, nominal_pd=[30., 1.], torque_limits=LIMITS.tolist(),
        profile='training_period_and_limits_on_deployment_assets', command_inputs='physical units',
        deployment_original=dict(period_s=.012, torque_limits=[30.]*12, posture_deg=[30, 45]),
        training_differences=['MuJoCo physics 1ms instead of Isaac 5ms', 'no randomized actuator delay',
            'no observation noise or physical parameter randomization', 'different collision/solver implementation'],
        startup='1s policy-independent PD settle, then zero commands and protocol ramp; not C++ standup state machine',
        fixed_jitter=JITTER.tolist(), geometry=geometries, actor_parity=parity, episodes=records,
        render_policy_rollouts=0, sources=source_files(), versions=dict(mujoco=mujoco.__version__, torch=torch.__version__),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    save(OUTPUT/'plan.json', plan); save(DOCS/'execution_plan.json', plan)
    print('AUDIT PASSED: interface, native actor parity, geometry; 36 formal + 4 smoke maximum', flush=True)


def run():
    plan = read(OUTPUT/'plan.json'); verify_sources(plan)
    if (OUTPUT/'run_started.json').exists():
        raise FileExistsError('Run already started; inspect evidence rather than silently rerunning')
    save(OUTPUT/'run_started.json', dict(phase='smoke', optimizer_updates=0))
    smoke = []
    for scene, seconds in (('stand', 5.), ('flat', 6.)):
        for name in MODELS:
            smoke.append(run_episode(name, scene, 0, OUTPUT/'smoke'/f'{name}_{scene}', seconds, True))
    for scene in ('stand', 'flat'):
        assert len({r['initial_state_sha256'] for r in smoke if r['scene'] == scene}) == 1
    save(OUTPUT/'smoke.json', dict(status='passed' if all(r['smoke_passed'] for r in smoke) else 'failed', records=smoke))
    assert all(r['smoke_passed'] for r in smoke), 'Stop after smoke failure; do not expand budget'
    records = []
    for spec in plan['episodes']:
        name, scene, repeat = spec['policy'], spec['scene'], spec['repeat']
        records.append(run_episode(name, scene, repeat, OUTPUT/'episodes'/f'{name}_{scene}_{repeat}'))
        save(OUTPUT/'progress.json', dict(formal_completed=len(records), formal_limit=36, last=records[-1]))
    for scene in SCENES:
        for repeat in range(3):
            pair = [r for r in records if r['scene'] == scene and r['repeat'] == repeat]
            assert len(pair) == 2 and len({r['initial_state_sha256'] for r in pair}) == 1
    verify_sources(plan)
    save(OUTPUT/'complete.json', dict(status='completed', formal_episodes=36, smoke_episodes=4,
        optimizer_updates=0, identical_paired_initial_states=True, records=records))
    print('COMPLETE: 36 formal, 4 smoke, 0 optimizer updates', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('audit', 'run'))
    args = parser.parse_args()
    audit() if args.phase == 'audit' else run()
