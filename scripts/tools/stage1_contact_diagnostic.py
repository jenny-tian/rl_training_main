"""Record pre-reset stair contacts while preserving the registered EA-LPL rollout."""
from __future__ import annotations

import argparse
from pathlib import Path

import ea_lpl_run as runner

collect_registered = runner.collect_fixed


def collect_contacts(wrapped, actor, spec, folder, app, report):
    import numpy as np
    import torch

    base = wrapped.unwrapped
    robot = base.scene['robot']
    sensor = base.scene.sensors['contact_forces']
    assignments = runner.environment_identity(base)['assignments']
    selected = [a['env_id'] for a in assignments if a['subterrain'] in ('stairs_ascending', 'stairs_descending')]
    if len(selected) != 16:
        raise ValueError('Expected eight ascending and eight descending environments')
    ids = torch.tensor(selected, device=base.device)
    active = torch.ones(len(selected), dtype=torch.bool, device=base.device)
    buffers = {name: [] for name in ('active', 'steps', 'root_relative', 'root_quaternion', 'gravity',
                                    'base_velocity', 'joint_pos', 'joint_vel', 'joint_target', 'body_relative',
                                    'forces', 'force_history', 'scan_height', 'contact_count', 'terminated',
                                    'truncated', 'rewards')}
    manager = base.reward_manager
    compute_rewards = manager.compute

    def capture(dt):
        value = compute_rewards(dt)
        data = robot.data
        terminated = base.reset_terminated[ids]
        truncated = base.reset_time_outs[ids]
        snapshot = {
            'active': active.clone(),
            'steps': base.episode_length_buf[ids],
            'root_relative': data.root_pos_w[ids] - base.scene.terrain.env_origins[ids],
            'root_quaternion': data.root_quat_w[ids],
            'gravity': data.projected_gravity_b[ids],
            'base_velocity': data.root_lin_vel_b[ids],
            'joint_pos': data.joint_pos[ids],
            'joint_vel': data.joint_vel[ids],
            'joint_target': data.joint_pos_target[ids],
            'body_relative': data.body_pos_w[ids] - data.root_pos_w[ids, None, :],
            'forces': sensor.data.net_forces_w[ids],
            'force_history': sensor.data.net_forces_w_history[ids],
            'scan_height': base.scene.sensors['height_scanner_base'].data.ray_hits_w[ids, :, 2]
                           - base.scene.terrain.env_origins[ids, None, 2],
            'contact_count': base._lite3_illegal_contact_consecutive_counts[ids],
            'terminated': terminated,
            'truncated': truncated,
            'rewards': manager._step_reward[ids],
        }
        for name, tensor in snapshot.items():
            buffers[name].append(tensor.detach().cpu().numpy().copy())
        active.logical_and_(~(terminated | truncated))
        return value

    manager.compute = capture
    try:
        result = collect_registered(wrapped, actor, spec, folder, app, report)
    finally:
        manager.compute = compute_rewards
        arrays = {name: np.asarray(values) for name, values in buffers.items()}
        np.savez_compressed(folder / 'stair_contact_trace.npz', **arrays)
        runner.write_json(folder / 'stair_contact_metadata.json', dict(
            schema='stage1_stair_contacts_v1', capture='after termination/reward, before reset',
            env_ids=selected, terrain=[assignments[i]['subterrain'] for i in selected],
            sensor_body_names=sensor.body_names, robot_body_names=robot.body_names,
            joint_names=robot.joint_names, reward_names=manager.active_terms,
            step_dt=base.step_dt, actor=spec['actor'], level=spec['level'],
            illegal_contact_threshold_n=5.0, illegal_contact_consecutive_steps=3))
    if active.any():
        raise RuntimeError('Some stair environments did not finish their first episode')
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError('Non-finite stair diagnostic data')
    result['diagnostics'] = {'trace': 'stair_contact_trace.npz', 'metadata': 'stair_contact_metadata.json',
                             'trace_sha256': runner.digest(folder / 'stair_contact_trace.npz')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', choices=('auto', '0', '1'), default='auto')
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    runner.collect_fixed = collect_contacts
    # Reuse the existing supervisor, but launch this instrumented child.
    runner.__file__ = str(Path(__file__).resolve())
    return runner.child(args.spec, args.output) if args.child else runner.supervise(args.spec, args.output, args.gpu)


if __name__ == '__main__':
    raise SystemExit(main())
