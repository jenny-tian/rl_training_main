"""S2（docs/lite3-online-s2-20261127/PLAN.md）的判定和汇总：按规则对照探路选目标、训练链的判定、脚本化检查、覆盖面。

用法：python reference_online_s2.py probe                 汇总规则对照探路，按方案第 2 节选目标，写 PROBE.json
      python reference_online_s2.py chains [--partial]    入选目标上 mrs 的判定（方案第 4 节），写 RESULTS_*.json 和 SUMMARY.json
      python reference_online_s2.py top                   脚本化检查的初筛和确认（方案第 5 节），写 TOP.json
前面几个函数不依赖记录文件，有测试。读记录时各目标、各变体用子进程分别汇总（运行入口在导入时由环境变量确定目标和变体）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

# 方案第 2 节的候选，每一类由易到难。
CLASSES = dict(boxes=('boxes_07', 'boxes_08'), rough=('rough_09', 'rough_10', 'rough_11'), slope=('slope_up_55', 'slope_up_57', 'slope_up_60'),
               slope_rough_up=('slope_rough_up_x11', 'slope_rough_up_x12'), slope_rough_down=('slope_rough_down_x11', 'slope_rough_down_x12'),
               stairs_down=('stairs_down_09', 'stairs_down_10'))
# 没有候选符合时沿用 S2′ 的目标和结果，不再训练。
FALLBACK = dict(slope_rough_down='slope_rough_down_x10', stairs_down='stairs_down_085')
RULE_CEILING, FROZEN_CEILING, FLOOR, PREFERRED_FROZEN = 204, 168, 24, 75   # 每个候选 240 个回合
TOP_GROUPS = ('none', 'hind_half', 'hind_full', 'all_half', 'h15', 'h15_hind_half')
EPSILON = 1e-9


def eligible(frozen, rule):
    """方案第 2 节的三个条件：规则对照没有到顶，原步态不太容易，两者至少一个不低于 10%。"""
    return rule <= RULE_CEILING and frozen <= FROZEN_CEILING and max(frozen, rule) >= FLOOR


def select_targets(counts):
    """counts[候选] = (原步态成功数, 规则对照成功数)。每一类取符合条件的候选里原步态成功数最接近 75 的一个，并列时取靠前（较容易）的。"""
    result = {}
    for name, candidates in CLASSES.items():
        good = [c for c in candidates if eligible(*counts[c])]
        if good:
            result[name] = dict(target=min(good, key=lambda c: abs(counts[c][0] - PREFERRED_FROZEN)), train=True)
        else:
            result[name] = dict(target=FALLBACK.get(name), train=False)
    return result


def groups_to_confirm(success):
    """初筛里成功数最高的一组动作（none 除外）；它带 h15 时，另外加上不带 h15 的组里最高的一组。并列时取登记顺序靠前的。"""
    if set(success) != set(TOP_GROUPS):
        raise ValueError('screening results for all six groups required')
    order = [g for g in TOP_GROUPS if g != 'none']
    best = max(order, key=lambda g: success[g])
    if not best.startswith('h15'): return [best]
    return [best, max([g for g in order if not g.startswith('h15')], key=lambda g: success[g])]


def confirmed(delta):
    """确认起点上这一组减 none 的配对汇总：增益不低于 10 个百分点且区间下限大于 0。"""
    return bool(delta['mean_session_delta'] >= .10 - EPSILON and delta['ci95'][0] > 0)


def coverage(effective):
    """effective：各目标是否满足有效（没有目标时是 None）。粗糙坡面要上行、下行都满足。六类里至少四类满足。"""
    classes = {name: bool(effective[name]) for name in ('stairs_up', 'stairs_down', 'boxes', 'rough', 'slope')}
    classes['slope_rough'] = bool(effective['slope_rough_up']) and bool(effective['slope_rough_down'])
    count = sum(classes.values())
    return dict(classes=classes, effective=count, passed=count >= 4)


def probe_counts(directory, candidate):
    """一个候选的规则对照探路结果。directory 是 logs/reference_six/online_directed_rule_20261126；Updated 组就是规则对照。"""
    import collections
    from pathlib import Path
    records = []
    for part in range(4):
        d = Path(directory) / candidate / 'directed' / f'target_0_{part:02d}'
        if json.loads((d / 'report.json').read_text())['status'] != 'passed':
            raise ValueError(f'{candidate} part {part} did not pass')
        records += json.loads((d / 'episodes.json').read_text())['records']
    arms = {arm: [r for r in records if r['arm'] == arm] for arm in ('Frozen', 'Updated')}
    if len(arms['Frozen']) != len(arms['Updated']):
        raise ValueError(f'{candidate}: unpaired records')
    outcomes = {arm: dict(collections.Counter(r['outcome'] for r in rows)) for arm, rows in arms.items()}
    progress = {arm: sum(r['mean_progress'] for r in rows) / len(rows) for arm, rows in arms.items()}
    return dict(n=len(arms['Frozen']), frozen=outcomes['Frozen'].get('success', 0), rule=outcomes['Updated'].get('success', 0),
                frozen_outcomes=outcomes['Frozen'], rule_outcomes=outcomes['Updated'], frozen_progress=progress['Frozen'], rule_progress=progress['Updated'])


def probe():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    rows = {c: probe_counts(root / 'logs/reference_six/online_directed_rule_20261126', c) for names in CLASSES.values() for c in names}
    if any(r['n'] != 240 for r in rows.values()): raise ValueError('every candidate needs 240 paired episodes')
    chosen = select_targets({c: (r['frozen'], r['rule']) for c, r in rows.items()})
    late = lambda o: o.get('stuck', 0) + o.get('timeout', 0)
    print('| 类 | 候选 | 原步态 | 规则对照 | 原步态的失败（姿态 / 没到终点 / 越界） | 规则对照的失败 | 是否符合 |\n|---|---|---|---|---|---|---|')
    for name, candidates in CLASSES.items():
        for c in candidates:
            r = rows[c]; f, u = r['frozen_outcomes'], r['rule_outcomes']
            mark = '入选' if chosen[name]['target'] == c and chosen[name]['train'] else '符合' if eligible(r['frozen'], r['rule']) else '不符合'
            print(f"| {name} | `{c}` | {r['frozen']} | {r['rule']} | {f.get('posture_failure', 0)} / {late(f)} / {f.get('lateral_exit', 0)} | "
                  f"{u.get('posture_failure', 0)} / {late(u)} / {u.get('lateral_exit', 0)} | {mark} |")
    print(json.dumps(chosen, ensure_ascii=False))
    (root / 'docs/lite3-online-s2-20261127/PROBE.json').write_text(json.dumps(dict(candidates=rows, selection=chosen), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    dict(probe=probe)[sys.argv[1]]()
