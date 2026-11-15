"""用冻结父模型的真实状态验证能力保持损失和命令覆盖候选；零次优化。"""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace as NS

import numpy as np
import torch

from reference_stairs_eval import save, sha
from reference_low_stairs_regression_probe import OUTPUT, PARENT, CANDIDATE, HASHES
from reference_low_stairs_regression_analyze import OUT, npz, read, verify_replay
from reference_low_stairs_evaluate import OUTPUT as BASELINE
from reference_low_stairs_preservation import (GROUPS, ACTION_SCALES, capability_groups,
    joint_target_preservation_loss, split_heading_turn_samples)
from reference_reward_code_audit import definitions, COMMAND_BASE, COMMAND


def command_audit(cfg, n=100000):
    ns = dict(torch=torch, CommandTerm=object,
        math_utils=NS(wrap_to_pi=lambda x: (x+math.pi) % (2*math.pi)-math.pi))
    sources = definitions(COMMAND_BASE, ['UniformVelocityCommand'], ns)
    ns['mdp'] = NS(UniformVelocityCommand=ns['UniformVelocityCommand'])
    sources.update(definitions(COMMAND, ['UniformThresholdVelocityCommand'], ns))
    term = object.__new__(ns['UniformThresholdVelocityCommand'])
    term.device = 'cpu'; term.num_envs = n
    term.cfg = NS(**{k: v for k, v in cfg.items() if k != 'ranges'})
    term.cfg.ranges = NS(**cfg['ranges'])
    term.vel_command_b = torch.zeros(n, 3); term.heading_target = torch.zeros(n)
    term.is_heading_env = torch.zeros(n, dtype=torch.bool); term.is_standing_env = torch.zeros(n, dtype=torch.bool)
    term.robot = NS(data=NS(heading_w=torch.zeros(n)))
    term._full_ranges_lin_vel_x = cfg['ranges']['lin_vel_x']; term._full_ranges_lin_vel_y = cfg['ranges']['lin_vel_y']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261114)
        ids = torch.arange(n); term._resample_command(ids)
        original = copy.deepcopy(term)
        generator = torch.Generator().manual_seed(20261115)
        changed = split_heading_turn_samples(term, ids, torch.rand(n, generator=generator),
                                            torch.rand(n, generator=generator))
        original._update_command(); term._update_command()
    keep = torch.ones(n, dtype=torch.bool); keep[changed] = False
    assert torch.equal(original.command[keep], term.command[keep])
    assert torch.equal(original.is_standing_env, term.is_standing_env)
    assert torch.equal(original.heading_target, term.heading_target)
    def counts(value):
        command = value.command
        pure = command[:, :2].norm(dim=1) < 1e-8
        return dict(standing=int((command.norm(dim=1) == 0).sum()),
            heading_turns=int((pure & value.is_heading_env & ~value.is_standing_env).sum()),
            constant_left=int((pure & ~value.is_heading_env & (command[:, 2] > .1)).sum()),
            constant_right=int((pure & ~value.is_heading_env & (command[:, 2] < -.1)).sum()))
    before, after = counts(original), counts(term)
    assert before['standing'] == after['standing'] and before['constant_left'] == before['constant_right'] == 0
    assert after['constant_left'] > 1000 and after['constant_right'] > 1000
    # 更新朝向后，恒定角速度仍保持；避免再次被原生航向处理覆盖。
    expected = term.command[changed].clone(); term.robot.data.heading_w[:] = 1.
    term._update_command(); assert torch.equal(expected, term.command[changed])
    return dict(status='passed', samples=n, changed=len(changed), before=before, after=after,
        untouched_samples_exact=True, standing_flags_exact=True, constant_rate_survives_heading_change=True,
        rate_range_rad_s=[.2, .8], sources=sources)


def main():
    torch.set_num_threads(1)
    bank = {k: [] for k in ('observation', 'parent_action', 'candidate_action', 'group', 'env_id', 'tick', 'profile')}
    checks, sources = [], []
    for profile in ('nominal', 'training_randomized'):
        name = 'turn_parent_nominal_8s' if profile == 'nominal' else 'turn_parent_native_8s'
        path = OUTPUT / name
        checks.append(verify_replay(path, BASELINE / f'turn_{profile}'))
        data, report = npz(path / 'diagnostic.npz'), read(path / 'report.json')
        initial = npz(path / 'initial.npz')
        trace = npz(path / 'trace.npz')
        successful = np.array([r['outcome'] == 'survived' for r in report['records']])
        # 观测来自物理步之前，倾角也对应这一步之前；避免用事后状态分组。
        q0 = initial['root'][:, 3:7]
        tilt0 = np.arccos(np.clip(1-2*(q0[:, 1]**2+q0[:, 2]**2), -1, 1))
        tilt = np.concatenate((tilt0[None], np.arccos(np.clip(-trace['gravity'][:-1, :, 2], -1, 1))))
        command = data['observation'][..., 6:9]
        groups = capability_groups(torch.from_numpy(command), torch.from_numpy(tilt)).numpy()
        valid = trace['active'] & successful[None] & (tilt < np.deg2rad(30)) & (groups >= 0)
        # 只保留每 5 帧一个状态，再按类别定额抽取；训练与验收时不得混用此诊断集。
        valid &= (np.arange(len(groups))[:, None] % 5 == 0)
        for group in range(len(GROUPS)):
            pairs = np.argwhere(valid & (groups == group))
            if not len(pairs):
                continue
            chosen = np.linspace(0, len(pairs)-1, min(1024, len(pairs)), dtype=int)
            t, e = pairs[chosen].T
            for key in ('observation', 'parent_action', 'candidate_action'):
                bank[key].append(data[key][t, e])
            bank['group'].append(np.full(len(t), group)); bank['env_id'].append(e); bank['tick'].append(t)
            bank['profile'].append(np.full(len(t), int(profile == 'training_randomized')))
        sources.extend(dict(path=str(path / name), sha256=sha(path / name))
                       for name in ('diagnostic.npz', 'report.json', 'initial.npz', 'trace.npz'))
    bank = {k: np.concatenate(v) for k, v in bank.items()}
    assert set(bank['group']) == set(range(len(GROUPS)))
    teacher = torch.from_numpy(bank['parent_action']); candidate = torch.from_numpy(bank['candidate_action'])
    groups = torch.from_numpy(bank['group'])
    assert joint_target_preservation_loss(teacher, teacher, groups).item() == 0.
    prediction = candidate.clone().requires_grad_()
    loss = joint_target_preservation_loss(prediction, teacher, groups)
    loss.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0 and teacher.grad is None
    assert torch.equal(candidate, prediction.detach())
    rows = []
    delta = (candidate-teacher) * candidate.new_tensor(ACTION_SCALES)
    for group, name in enumerate(GROUPS):
        value = delta[groups == group]
        rows.append(dict(group=name, samples=len(value), target_rmse_rad=float(value.square().mean().sqrt()),
            target_absolute_p95_rad=float(value.abs().flatten().quantile(.95)),
            mean_target_delta_rad=value.mean(0).tolist()))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'diagnostic_anchor_bank.npz', **bank)
    cfg = read(OUTPUT / 'turn_parent_native_8s/config.json')['cfg']['commands']['base_velocity']
    sampler = command_audit(cfg)
    assert sha(PARENT) == HASHES['parent'] and sha(CANDIDATE) == HASHES['candidate']
    save(OUT / 'preservation_validation.json', dict(status='passed', samples=len(groups),
        group_metrics=rows, candidate_loss=float(loss), parent_loss=0., finite_gradient=True,
        parameters_updated=False, optimizer_updates=0, integrated_into_ppo=False, sampler=sampler,
        frozen_replay=checks, source_files=sources, anchor_sha256=sha(OUT / 'diagnostic_anchor_bank.npz'),
        caveat='只验证候选组件；此库来自诊断协议，若用于未来训练，必须另用独立种子作最终留出验收。'))
    print(json.dumps(dict(status='passed', group_metrics=rows, sampler=sampler), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
