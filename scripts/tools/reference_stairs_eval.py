"""Paired, frozen Lite3 stair evaluation using the deployment MuJoCo assets.

No network, robot connection, optimization, or writes to either reference repo.
The two control profiles share the same simulator and robot. The training_nominal
profile matches period/torque limits only; it is not an Isaac Lab reproduction.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = Path('/home/tjx/Lite3_rl_deploy')
TRAIN = ROOT / 'logs/reference_six/training/scratch_seed42_10000'
ROBOT_XML = DEPLOY / 'third_party/deep_robotics_model/Lite3/Lite3_mjcf/mjcf/Lite3.xml'
STAIR_XML = DEPLOY / 'Lite3_description/lite3_mjcf/mjcf/stair.xml'
MODELS = {
    'reference': DEPLOY / 'policy/ppo/policy.onnx',
    # File iterations are zero based: this checkpoint contains 8001 updates.
    'update8001': TRAIN / 'training/model_8000.pt',
    'update10000': TRAIN / 'model_updates_10000.pt',
}
JOINTS = [f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'HL', 'HR')
          for joint in ('HipX', 'HipY', 'Knee')]
DEFAULT = np.tile([0., -.65, 1.3], 4)
SCALE = np.tile([.125, .25, .25], 4)
PROFILES = {'deployment': {'period': .012, 'limits': [30.] * 12},
            'training_nominal': {'period': .020, 'limits': [24., 24., 36.] * 4}}
SPEEDS = (.2, .4, .6, .8)
DT = .001


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_xml(profile, flat=False):
    """Copy only the robot/floor and exact deployment stairs into an isolated scene."""
    root = ET.parse(ROBOT_XML).getroot()
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', str((ROBOT_XML.parent / mesh.get('file')).resolve()))
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', str(DT))
    option.set('gravity', '0 0 -9.81')
    if not flat:
        stair_body = ET.parse(STAIR_XML).getroot().find('./worldbody/body')
        root.find('worldbody').append(copy.deepcopy(stair_body))
    for motor, limit in zip(root.findall('./actuator/motor'), PROFILES[profile]['limits'], strict=True):
        motor.set('ctrlrange', f'{-limit} {limit}')
        motor.set('ctrllimited', 'true')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='640')
    return ET.tostring(root, encoding='unicode')


def load_actor(name, original_onnx=True):
    import torch
    torch.set_num_threads(1)
    if name == 'reference':
        if original_onnx:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(str(MODELS[name]), sess_options=options, providers=['CPUExecutionProvider'])
        import onnx
        from onnx import numpy_helper
        graph = onnx.load(str(MODELS[name]))
        assert [n.op_type for n in graph.graph.node] == ['Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm']
        weights = {v.name: torch.from_numpy(numpy_helper.to_array(v).copy()) for v in graph.graph.initializer}
    else:
        weights = torch.load(MODELS[name], map_location='cpu', weights_only=True)['actor_state_dict']
    layers = []
    for i, (a, b) in enumerate(zip((45, 512, 256, 128), (512, 256, 128, 12))):
        layer = torch.nn.Linear(a, b)
        layer.load_state_dict({k: weights[f'mlp.{i * 2}.{k}'] for k in ('weight', 'bias')})
        layers.append(layer)
        if i < 3:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
        assert torch.isfinite(parameter).all()
    return actor


def policy_action(actor, obs):
    import torch
    with torch.inference_mode():
        inputs = obs.astype(np.float32)
        result = (actor.run(['actions'], {'obs': inputs[None]})[0][0]
                  if hasattr(actor, 'run') else actor(torch.from_numpy(inputs)).numpy())
    if result.shape != (12,) or not np.isfinite(result).all():
        raise FloatingPointError('invalid policy action')
    return result


def rotation(quaternion):
    import mujoco
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def observation(data, command, last_action):
    gravity = rotation(data.qpos[3:7]).T @ np.array([0., 0., -1.])
    return np.concatenate((data.qvel[3:6] * .25, gravity, command,
                           data.qpos[7:] - DEFAULT, data.qvel[6:] * .05, last_action))


def posture_bad(gravity):
    return bool(gravity[2] > 0 or np.any(np.abs(gravity[:2]) > .7))


def cleared(direction, feet, root, support_count):
    """All four feet, including rear feet, must clear the terminal riser."""
    if direction == 'up':
        crossed = np.min(feet[:, 0]) > 2.70 and root[0] > 2.95
        ground = .6
    elif direction == 'down':
        crossed = np.max(feet[:, 0]) < 1.10 and root[0] < .85
        ground = 0.
    else:
        crossed = root[0] > 2.95
        ground = 0.
    return bool(crossed and .18 < root[2] - ground < .55 and support_count >= 2)


def contacts(model, data, foot_geoms):
    import mujoco
    supported = set()
    body_contact = False
    force = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)
        if abs(force[0]) < 1.:
            continue
        pair = (con.geom1, con.geom2)
        for g in pair:
            if g in foot_geoms:
                supported.add(g)
            elif model.geom_bodyid[g] > 0 and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) == 'TORSO_collision':
                body_contact = True
    return len(supported), body_contact


def initial_data(model, direction, repeat):
    import mujoco
    data = mujoco.MjData(model)
    rng = np.random.default_rng(20261112 + repeat)
    jitter = rng.uniform([-0.05, -0.10, -0.03], [0.05, 0.10, 0.03])
    yaw = (math.pi if direction == 'down' else 0.) + jitter[2]
    data.qpos[:3] = [3.55 if direction == 'down' else .25, jitter[1], .975 if direction == 'down' else .375]
    data.qpos[0] += jitter[0]
    data.qpos[3:7] = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
    data.qpos[7:] = DEFAULT
    mujoco.mj_forward(model, data)
    # Identical, policy-independent stand settling. No privileged policy inputs.
    for _ in range(1000):
        data.ctrl[:] = np.clip(30. * (DEFAULT - data.qpos[7:]) - data.qvel[6:],
                               model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
    if posture_bad(observation(data, np.zeros(3), np.zeros(12))[3:6]):
        raise RuntimeError('initial stand settling failed')
    return data


def episode(model, actor, spec, video_path=None):
    import mujoco
    data = initial_data(model, spec['direction'], spec['repeat'])
    initial_hash = hashlib.sha256(np.concatenate((data.qpos, data.qvel)).tobytes()).hexdigest()
    initial_x = float(data.qpos[0])
    sign = -1. if spec['direction'] == 'down' else 1.
    period = PROFILES[spec['profile']]['period']
    decimation = round(period / DT)
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{s}_FOOT') for s in ('FL', 'FR', 'HL', 'HR')]
    foot_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{s}_FOOT_collision') for s in ('FL', 'FR', 'HL', 'HR')}
    last_action, target = np.zeros(12), DEFAULT.copy()
    outcome = 'timeout'
    errors, positions, traces = [], [], []
    max_tilt = 0.
    saturation = 0
    torque_samples = 0
    torso_contact_samples = 0
    max_action_abs = 0.
    numerical_failure = None
    safe_limit_crossed = False
    clear_time = 0.
    renderer = writer = camera = None
    if video_path:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=640, width=960)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 4., 110., -22.
        writer = imageio.get_writer(video_path, fps=25, codec='libx264', quality=7, macro_block_size=16)
    started = time.monotonic()
    try:
        for step in range(25000):
            t = step * DT
            if step % decimation == 0:
                # A 0.5 s zero-command start followed by a fixed, identical 0.5 s ramp.
                vx = spec['speed'] * np.clip((t - .5) / .5, 0., 1.)
                obs = observation(data, np.array([vx, 0., 0.]), last_action)
                try:
                    last_action = policy_action(actor, obs)
                except FloatingPointError:
                    outcome = 'policy_numerical_failure'
                    numerical_failure = dict(t=t, observation_max_abs=float(np.max(np.abs(obs))))
                    break
                max_action_abs = max(max_action_abs, float(np.max(np.abs(last_action))))
                target = DEFAULT + SCALE * last_action
            torque = 30. * (target - data.qpos[7:]) - data.qvel[6:]
            saturation += int(np.count_nonzero(np.abs(torque) > model.actuator_ctrlrange[:, 1]))
            torque_samples += 12
            data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            if step % 10 == 0:
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError('nonfinite rollout')
                rot = rotation(data.qpos[3:7])
                gravity = -rot[2, :]
                max_tilt = max(max_tilt, math.degrees(math.acos(np.clip(-gravity[2], -1., 1.))))
                roll = math.atan2(rot[2, 1], rot[2, 2])
                pitch = math.asin(np.clip(-rot[2, 0], -1., 1.))
                safe_limit_crossed |= abs(roll) > math.pi / 6 or abs(pitch) > math.pi / 4
                support, torso_contact = contacts(model, data, foot_geoms)
                torso_contact_samples += int(torso_contact)
                feet = data.xpos[foot_ids]
                if posture_bad(gravity):
                    outcome = 'orientation_failure'
                elif abs(data.qpos[1]) > 2.5:
                    outcome = 'lateral_exit'
                elif cleared(spec['direction'], feet, data.qpos[:3], 2):
                    # Keep the geometric dwell through normal swing/flight phases.
                    # Require actual landing support at the final success sample.
                    clear_time += .01
                    if clear_time >= .25 and support >= 2 and not torso_contact:
                        outcome = 'success'
                else:
                    clear_time = 0.
                if t >= 1.:
                    body_vel = rot.T @ data.qvel[:3]
                    errors.append(float(np.linalg.norm(body_vel[:2] - [spec['speed'], 0.])))
                positions.append((t, float(data.qpos[0])))
                if step % 50 == 0 or outcome != 'timeout':
                    traces.append(dict(t=round(t, 3), xyz=data.qpos[:3].tolist(),
                                       tilt_deg=math.degrees(math.acos(np.clip(-gravity[2], -1., 1.))),
                                       feet_x=feet[:, 0].tolist(), feet_z=feet[:, 2].tolist(),
                                       body_velocity=(rot.T @ data.qvel[:3]).tolist(),
                                       roll_pitch=[roll,pitch], support=support))
            if writer and (step % 40 == 0 or outcome != 'timeout'):
                from PIL import Image, ImageDraw
                camera.lookat[:] = data.qpos[:3] + [0., 0., -.10]
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 63), fill=(20, 25, 32))
                draw.text((12, 9), f"{spec['model']} | {spec['profile']} | {spec['direction']} | {spec['speed']:.1f} m/s | start {spec['repeat']}", fill='white')
                display_outcome = 'running' if outcome == 'timeout' else outcome
                draw.text((12, 34), f"10 cm risers / 30 cm tread | t={t:.2f}s | {display_outcome} | deterministic simulation", fill='white')
                writer.append_data(np.asarray(frame))
            if outcome != 'timeout':
                break
        if outcome == 'timeout':
            recent = [x for t, x in positions if t >= positions[-1][0] - 5.]
            if sign * (recent[-1] - recent[0]) < .15:
                outcome = 'stuck'
        if writer:
            draw.rectangle((0, 585, 960, 640), fill=(20, 25, 32))
            draw.text((12, 605), f'FINAL: {outcome} | frozen final frame held for 1 second', fill='white')
            for _ in range(25):
                writer.append_data(np.asarray(frame))
        if any(data.warning.number):
            raise RuntimeError(f'MuJoCo physics warning: {data.warning.number.tolist()}')
        return {**spec, 'outcome': outcome, 'duration_s': round((step + 1) * DT, 3),
                'progress_m': sign * (float(data.qpos[0]) - initial_x),
                'final_xyz': data.qpos[:3].tolist(), 'max_tilt_deg': max_tilt,
                'mean_velocity_xy_error': float(np.mean(errors)) if errors else None,
                'torque_saturation_fraction': saturation / torque_samples,
                'torso_contact_samples': torso_contact_samples,
                'max_action_abs': max_action_abs, 'numerical_failure': numerical_failure,
                'deployment_posture_limit_crossed': bool(safe_limit_crossed),
                'initial_state_sha256': initial_hash, 'trace': traces,
                'wall_seconds': time.monotonic() - started}
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()


def make_model(profile, flat=False):
    import mujoco
    model = mujoco.MjModel.from_xml_string(build_xml(profile, flat))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(v[0])) for v in model.actuator_trnid]
    if names != JOINTS or model.nq != 19 or model.nv != 18:
        raise ValueError('deployment model joint mapping mismatch')
    return model


def batch(job):
    name, profile, repeats, speeds, output = job
    actor = load_actor(name)
    model, flat = make_model(profile), make_model(profile, flat=True)
    records = []
    for direction in ('flat', 'up', 'down'):
        for speed in ((.4,) if direction == 'flat' else speeds):
            for repeat in range(repeats):
                spec = dict(model=name, profile=profile, direction=direction, speed=speed, repeat=repeat)
                record = episode(flat if direction == 'flat' else model, actor, spec)
                records.append(record)
                save(Path(output) / 'episodes' / f'{name}_{profile}_{direction}_{speed:.1f}_{repeat:02d}.json', record)
            print(f"DONE {name} {profile} {direction} speed={speed}: "
                  + str({outcome: sum(r['outcome'] == outcome for r in records if r['direction'] == direction and r['speed'] == speed)
                         for outcome in ('success', 'orientation_failure', 'stuck', 'timeout', 'lateral_exit', 'policy_numerical_failure')}), flush=True)
    return records


def audit(output):
    import onnxruntime as ort
    import torch
    import mujoco
    actor = load_actor('reference', original_onnx=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(MODELS['reference']), sess_options=options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(42)
    errors = []
    for _ in range(50):
        obs = rng.uniform(-2., 2., 45).astype(np.float32)
        expected = session.run(['actions'], {'obs': obs[None]})[0][0]
        errors.append(float(np.max(np.abs(policy_action(actor, obs) - expected))))
    if max(errors) > 2e-5:
        raise ValueError(f'ONNX adapter mismatch {max(errors)}')
    for name in MODELS:
        load_actor(name)
    model = make_model('deployment')
    geom = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'stair{i}') for i in range(1, 7)]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tops = data.geom_xpos[geom, 2] + model.geom_size[geom, 2]
    fronts = data.geom_xpos[geom, 0] - model.geom_size[geom, 0]
    if not np.allclose(tops, np.arange(1, 7) * .1) or not np.allclose(fronts, 1.15 + np.arange(6) * .3):
        raise ValueError('deployment stair geometry changed')
    report = dict(status='passed', onnx_actor_max_error=max(errors), models={k:dict(path=str(p), sha256=sha(p)) for k,p in MODELS.items()},
                  reference_backend='original ONNX Runtime CPU, ORT_ENABLE_ALL; adapter parity check is diagnostic only',
                  sources={str(p):sha(p) for p in (ROBOT_XML, STAIR_XML, Path(__file__))},
                  versions=dict(mujoco=mujoco.__version__, torch=torch.__version__, onnxruntime=ort.__version__),
                  stair_tops=tops.tolist(), stair_fronts=fronts.tolist(), profiles=PROFILES,
                  observation_order=['angular_velocity*.25','projected_gravity','command','joint_position-default','joint_velocity*.05','previous_action'],
                  policy_noise=False, observation_noise=False, random_physics=False,
                  initial_jitter=dict(x_m=.05, y_m=.10, yaw_rad=.03),
                  limit_note='Both profiles use MuJoCo deployment assets. training_nominal changes only period and torque limits, not solver, actuator delay, or randomization.',
                  termination_note='Reference gravity threshold, checked every 10 ms. Deployment roll 30/pitch 45 degree threshold is recorded separately, not used to stop.',
                  success_note='All four feet beyond final riser, body beyond margin, base height 0.18-0.55 m sustained 0.25 s; at completion >=2 feet in contact and no torso contact. Normal swing does not reset geometric dwell.',
                  max_episode_s=25., optimization=False)
    save(output / 'audit.json', report)
    for profile in PROFILES:
        (output / f'{profile}.xml').write_text(build_xml(profile))
    return report


def summarize(records):
    groups = []
    for key in sorted({(r['model'],r['profile'],r['direction'],r['speed']) for r in records}):
        rows = [r for r in records if (r['model'],r['profile'],r['direction'],r['speed']) == key]
        errors = [r['mean_velocity_xy_error'] for r in rows if r['mean_velocity_xy_error'] is not None]
        groups.append(dict(zip(('model','profile','direction','speed'),key), episodes=len(rows),
                           **{k:sum(r['outcome']==k for r in rows) for k in ('success','orientation_failure','stuck','timeout','lateral_exit','policy_numerical_failure')},
                           success_rate=sum(r['outcome']=='success' for r in rows)/len(rows),
                           mean_velocity_xy_error=float(np.mean(errors)) if errors else None,
                           deployment_posture_limit_crossed=sum(r['deployment_posture_limit_crossed'] for r in rows)))
    pairing = {}
    for r in records:
        key = (r['profile'],r['direction'],r['speed'],r['repeat'])
        pairing.setdefault(key,set()).add(r['initial_state_sha256'])
    if any(len(v)!=1 for v in pairing.values()):
        raise ValueError('models did not receive identical initial states')
    return dict(status='completed', episodes=len(records), paired_starts=True, groups=groups)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=list(PROFILES))
    parser.add_argument('--models',nargs='+',choices=list(MODELS),default=list(MODELS))
    parser.add_argument('--speeds',nargs='+',type=float,default=SPEEDS)
    parser.add_argument('--video-record',type=Path)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    audit_report=audit(args.output)
    if args.video_record:
        source=json.loads(args.video_record.read_text())
        spec={k:source[k] for k in ('model','profile','direction','speed','repeat')}
        result=episode(make_model(spec['profile'],spec['direction']=='flat'),load_actor(spec['model']),spec,args.output/'rollout.mp4')
        if result['outcome']!=source['outcome'] or result['initial_state_sha256']!=source['initial_state_sha256'] or abs(result['duration_s']-source['duration_s'])>.011:
            raise ValueError('video does not reproduce measured episode')
        save(args.output/'replay.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
        return
    jobs=[(name,profile,args.repeats,args.speeds,str(args.output)) for name in args.models for profile in args.profiles]
    records=[]
    execution=dict(status='running',started_at=datetime.now().astimezone().isoformat(),
                   expected_episodes=len(jobs)*args.repeats*(1+2*len(args.speeds)),workers=args.workers,
                   profiles=args.profiles,speeds=list(args.speeds),repeats=args.repeats,pid=os.getpid())
    save(args.output/'execution.json',execution)
    started=time.monotonic()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for rows in pool.map(batch,jobs):
                records.extend(rows)
        summary=summarize(records)
        assert len(records)==execution['expected_episodes']
        assert all(sha(p)==audit_report['models'][name]['sha256'] for name,p in MODELS.items())
        summary['models_unchanged']=True
        save(args.output/'summary.json',summary)
        execution['status']='completed'
    except Exception as error:
        execution.update(status='failed',error=repr(error))
        raise
    finally:
        execution.update(ended_at=datetime.now().astimezone().isoformat(),elapsed_s=time.monotonic()-started)
        save(args.output/'execution.json',execution)
    print(f'COMPLETED {len(records)} paired episodes',flush=True)


if __name__=='__main__':
    main()
