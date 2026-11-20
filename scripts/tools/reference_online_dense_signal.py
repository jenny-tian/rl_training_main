"""用法：python reference_online_dense_signal.py <目标>。不占 GPU 的分析：做法二训练轨迹里，回合内的逐步信号对摆动幅度的分辨能力，与整回合回报比较。
两组都用（Frozen 组的参数是 0 附近的随机扰动，Updated 组在学习中的均值附近扰动）。"""
import json, sys
import numpy as np
from reference_stairs_eval import ROOT
OUT = ROOT / 'logs/reference_six/online_dense_signal_20261120'; OUT.mkdir(exist_ok=True)
target = sys.argv[1]
d = str(ROOT / 'logs/reference_six/online_c1b_s2n_20261119' / target / 'train') + '/'
index = json.load(open(d + 'trace_index.json'))
geometry = json.load(open(d + 'geometry.json')); fronts = list(geometry.values())[0]['fronts']
ARRIVE = fronts[0] - .25  # 机身到这里时前脚在第一级台阶前
N, E = 480, 100
WINDOWS = (100, 200, 400)  # 到达后的 2、4、8 秒
arrival = np.full((N, E), -1, int); x_arrival = np.zeros((N, E)); code = np.full((N, E, 2), np.nan)
progress = np.zeros((N, E, len(WINDOWS))); alive = np.zeros((N, E, len(WINDOWS)), int)
stall = np.zeros((N, E, len(WINDOWS), 2)); swing = np.zeros((N, E, len(WINDOWS), 2)); speed = np.zeros((N, E, len(WINDOWS)))
last_feet = None; last_ep = None
env = np.arange(N)
for chunk in index['chunks']:
    z = np.load(d + chunk['file'])
    phase, episode, age, root, feet, stance, cd, bv = (z[k] for k in ('phase', 'episode', 'age', 'root', 'feet', 'stance', 'code', 'body_velocity'))
    for t in range(len(phase)):
        step = chunk['start'] + t
        ep = episode[t]; pol = phase[t] == 1
        ok = pol & (ep < E)
        new = ok & (arrival[env, np.minimum(ep, E - 1)] < 0) & (root[t, :, 0] >= ARRIVE)
        arrival[env[new], ep[new]] = step; x_arrival[env[new], ep[new]] = root[t, new, 0]
        first = ok & (age[t] <= 1)
        code[env[first], ep[first]] = np.tanh(cd[t, first, 2:])
        if last_feet is not None:
            vx = (feet[t, :, :, 0] - last_feet[:, :, 0]) / .02
            same = ok & (last_ep == ep)
            since = step - arrival[env, np.minimum(ep, E - 1)]
            for w, width in enumerate(WINDOWS):
                m = same & (arrival[env, np.minimum(ep, E - 1)] >= 0) & (since >= 0) & (since < width)
                if not m.any(): continue
                e = ep[m]; i = env[m]
                alive[i, e, w] += 1
                progress[i, e, w] = root[t, m, 0] - x_arrival[i, e]
                speed[i, e, w] += bv[t, m, 0]
                sw = ~stance[t, m]  # (k, 4)
                st = sw & (vx[m] < .10)
                for g, legs in enumerate(((0, 1), (2, 3))):
                    swing[i, e, w, g] += sw[:, legs].sum(1); stall[i, e, w, g] += st[:, legs].sum(1)
        last_feet = feet[t].copy(); last_ep = ep.copy()
    print(chunk['file'], flush=True)
records = json.load(open(d + 'episodes.json'))['records']
ret = np.full((N, E), np.nan); success = np.full((N, E), np.nan); steps = np.zeros((N, E)); outcome = np.full((N, E), '', dtype=object); cstep = np.zeros((N, E))
for r in records:
    if r['episode'] < E:
        ret[r['env_id'], r['episode']] = r['episode_return']; success[r['env_id'], r['episode']] = r['outcome'] == 'success'
        steps[r['env_id'], r['episode']] = r['policy_steps']; outcome[r['env_id'], r['episode']] = r['outcome']; cstep[r['env_id'], r['episode']] = r['control_step']
np.savez_compressed(str(OUT / f'{target}.npz'), arrival=arrival, code=code, progress=progress, alive=alive, stall=stall, swing=swing,
                    speed=speed, ret=ret, success=success, steps=steps, outcome=outcome.astype(str), cstep=cstep)
print('saved')
