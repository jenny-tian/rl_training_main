"""加快学习 D 的离线检查第一步：把 C0 评估（库里每组参数在两个新地形上各 240 个回合）重放成每次尝试的窗口得分。不占 GPU。

用法：python reference_online_select_attempts.py
窗口得分用学习器自己的代码（c1b_w.WindowSession，单环境）从轨迹重算；先在 c3 训练轨迹的第一个回合上与训练时记录的窗口反馈逐项核对。
输出 logs/reference_six/online_select_offline_20261123/attempts_<目标>.json：
每个评估起点（240 个）、每组参数（库里 3 组加全 0）一条记录，含各窗口得分、结局、整回合回报。
"""
from __future__ import annotations

import json

import numpy as np

from rl_training.ea_lpl import c1b_w
from reference_stairs_eval import ROOT, save

LOGS = ROOT / 'logs/reference_six'
OUT = LOGS / 'online_select_offline_20261123'
SOURCES = (('s2n', 'stairs_up_06', 60000), ('s2n', 'stairs_up_065', 60000), ('bs', 'boxes_06', 40000))
KEYS = ('phase', 'stance', 'feet', 'features', 'root', 'before_root')


def load(directory):
    index = json.loads((directory / 'trace_index.json').read_text())
    chunks = [np.load(directory / c['file']) for c in index['chunks']]
    return {k: np.concatenate([c[k] for c in chunks]) for k in KEYS}


def replay(trace, env, posture, first=0, last=None):
    """一个环境从 first 起的第一个回合；返回每个可用窗口的得分（前后腿反馈的平均）和策略步数。"""
    steps = np.flatnonzero(trace['phase'][first:last, env] == 1) + first
    assert len(steps) and steps[0] > 0 and (np.diff(steps) == 1).all()
    group = c1b_w.WindowSession(False, np.zeros((c1b_w.MAX_WINDOWS, 1, 4)), replicas=1)
    group.begin(0)
    rows = []
    for n, t in enumerate(steps):
        group.observe(0, trace['stance'][t, env], (trace['feet'][t, env, :, 0] - trace['feet'][t - 1, env, :, 0]) / .02,
                      trace['features'][t, env, 0], trace['root'][t, env, 0] - trace['before_root'][t, env, 0])
        if n == len(steps) - 1 and posture is not None: group.finish(0, 0., posture=posture)
        entry = group.tick(False)
        if entry is not None: rows.append([float(np.mean(entry['feedback'][0]))] if entry['used'] else [])
    return rows, len(steps)


def validate(target):
    """c3 训练的第一个回合（选择阶段，不加扰动）：重放的窗口反馈与训练时记录的逐项一致。"""
    base = LOGS / 'online_c1b_library_20261122' / target / 'train'
    index = json.loads((base / 'trace_index.json').read_text())
    z = np.load(base / index['chunks'][0]['file'])
    trace = {k: z[k] for k in KEYS}
    specs = json.loads((base / 'config.json').read_text())['specs']
    windows = json.loads((base / 'windows.json').read_text())
    compared, worst = 0, 0.
    for spec in specs:
        if spec['arm'] != 'Updated': continue
        steps = np.flatnonzero(trace['phase'][:, spec['env_id']] == 1)
        stop = steps[0] + (np.flatnonzero(np.diff(steps) != 1)[0] + 1 if (np.diff(steps) != 1).any() else len(steps))
        whole = (stop - steps[0]) // c1b_w.WINDOW  # 只比较这段轨迹里完整的窗口
        rows, _ = replay(trace, spec['env_id'], None, last=stop)
        for w, entry in enumerate(windows[f"{spec['session']}:Updated"][:whole]):
            assert entry['episode'] == 0
            recorded = [float(np.mean(entry['feedback'][entry['used'].index(spec['repeat'])]))] if spec['repeat'] in entry['used'] else []
            assert len(recorded) == len(rows[w]), (target, spec['session'], spec['repeat'], w)
            if recorded: compared += 1; worst = max(worst, abs(recorded[0] - rows[w][0]))
    assert compared > 1000 and worst < 1e-6, (compared, worst)
    return dict(windows_compared=compared, max_abs_difference=worst)


def main():
    OUT.mkdir(exist_ok=True)
    for target in ('stairs_up_07', 'boxes_07'):
        check = validate(target)
        print(target, 'replay matches the training log:', check, flush=True)
        attempts = {}
        for entry, (variant, source, mark) in enumerate(SOURCES):
            for part in range(4):
                directory = LOGS / 'online_library_transfer_20261122' / target / f'from_{variant}_{source}_{mark}_pooled' / f'target_{mark}_{part:02d}'
                trace = load(directory)
                for r in json.loads((directory / 'episodes.json').read_text())['records']:
                    if r['arm'] == 'Frozen' and entry: continue  # Frozen 组是全 0，三次评估逐回合相同，只取一次
                    rows, steps = replay(trace, r['env_id'], r['outcome'] == 'posture_failure')
                    assert steps == r['policy_steps']
                    key = (r['session'], r['repeat'], entry if r['arm'] == 'Updated' else len(SOURCES))
                    assert key not in attempts
                    attempts[key] = dict(session=r['session'], start=r['repeat'], entry=key[2], outcome=r['outcome'], episode_return=r['episode_return'],
                                         policy_steps=steps, windows=[x[0] for x in rows if x])
            print(target, 'entry', entry, 'done', flush=True)
        assert len(attempts) == 240 * 4
        save(OUT / f'attempts_{target}.json', dict(target=target, entries=[':'.join(map(str, s)) for s in SOURCES] + ['zero'], validation=check,
                                                   attempts=[attempts[k] for k in sorted(attempts)]))


if __name__ == '__main__': main()
