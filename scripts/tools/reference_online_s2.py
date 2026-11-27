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


def standing_failed(directory, candidate):
    """这个候选有没有评估分片在起步前的站立准备阶段就有机器人终止（入口的保护会中止这个分片）。"""
    from pathlib import Path
    errors = sorted((Path(directory) / candidate / 'directed').glob('target_0_*/error.json'))
    return any('standing preparation terminated' in json.loads(e.read_text())['error'] for e in errors)


def probe():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    logs = root / 'logs/reference_six/online_directed_rule_20261126'
    # 站立准备阶段就有终止的候选不能用在线入口评估，更不能训练，记为不可用，不参加选目标（方案补充一节）。
    unusable = [c for names in CLASSES.values() for c in names if standing_failed(logs, c)]
    rows = {c: probe_counts(logs, c) for names in CLASSES.values() for c in names if c not in unusable}
    if any(r['n'] != 240 for r in rows.values()): raise ValueError('every candidate needs 240 paired episodes')
    chosen = select_targets({c: (rows[c]['frozen'], rows[c]['rule']) if c in rows else (0, 0) for names in CLASSES.values() for c in names})
    late = lambda o: o.get('stuck', 0) + o.get('timeout', 0)
    print('| 类 | 候选 | 原步态 | 规则对照 | 原步态的失败（姿态 / 没到终点 / 越界） | 规则对照的失败 | 是否符合 |\n|---|---|---|---|---|---|---|')
    for name, candidates in CLASSES.items():
        for c in candidates:
            if c in unusable:
                print(f'| {name} | `{c}` | — | — | 站立准备阶段有终止，评估中止 | — | 不可用 |'); continue
            r = rows[c]; f, u = r['frozen_outcomes'], r['rule_outcomes']
            mark = '入选' if chosen[name]['target'] == c and chosen[name]['train'] else '符合' if eligible(r['frozen'], r['rule']) else '不符合'
            print(f"| {name} | `{c}` | {r['frozen']} | {r['rule']} | {f.get('posture_failure', 0)} / {late(f)} / {f.get('lateral_exit', 0)} | "
                  f"{u.get('posture_failure', 0)} / {late(u)} / {u.get('lateral_exit', 0)} | {mark} |")
    print(json.dumps(chosen, ensure_ascii=False))
    (root / 'docs/lite3-online-s2-20261127/PROBE.json').write_text(json.dumps(dict(candidates=rows, unusable=unusable, selection=chosen), ensure_ascii=False, indent=1))


def verdict(*, gain, late_minus_early, minus_rule, five):
    """方案第 4 节的前四条。三个配对汇总都带 mean_session_delta 和 ci95；five 是保持评估的五项。"""
    result = dict(effective=bool(gain['mean_session_delta'] >= .10 - EPSILON and gain['ci95'][0] > 0),
                  not_worse=bool(late_minus_early['mean_session_delta'] >= -.05 - EPSILON),
                  retention=bool(five['posture_increase_at_most_2pp'] and five['every_family_loss_at_most_5pp'] and five['flat_velocity_ratio_at_most_1p10']),
                  learning_contributes=bool(minus_rule['ci95'][0] > 0))
    result['passed'] = all(result.values())
    return result


def chain_one(partial):
    """一个入选目标上 mrs 的汇总（子进程，目标和变体由环境变量给出）。"""
    import numpy as np
    from reference_online_c1a2 import MARKS, OUTPUT, RUNS, TARGET, VARIANT, protocol
    from reference_online_c1a_results import paired_summary
    from reference_online_c1aea_results import against_fixed, frozen_reproduced
    from reference_online_select_results import done, evaluation, read, retention
    from reference_stairs_eval import ROOT, save
    assert VARIANT == 'mrs'
    logs = ROOT / 'logs/reference_six'
    pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())
    count = lambda v, arm, key='success': v[arm]['outcomes'].get(key, 0)
    marks = [m for m in MARKS[1:] if done(OUTPUT, m)]
    if not marks or (not partial and len(marks) != len(MARKS) - 1): return print(TARGET, 'mrs not finished')
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    result = dict(target=TARGET, method='mrs', marks=marks, versus_frozen={str(m): paired_summary(rows[m]) for m in marks})
    train = read(OUTPUT, 'train')['records']
    result['training'] = {str(m): {arm: dict(attempts=sum(r['arm'] == arm and r['control_step'] <= m for r in train) / 60,
        success=sum(r['arm'] == arm and r['control_step'] <= m and r['outcome'] == 'success' for r in train)) for arm in ('Frozen', 'Updated')} for m in marks}
    rule = evaluation(logs / 'online_directed_rule_20261126' / TARGET / 'directed', 0)
    result['rule_versus_frozen'] = paired_summary(rule)
    result['minus_rule'] = {str(m): against_fixed(rows[m], rule) for m in marks}
    result['frozen_reproduced'] = frozen_reproduced(rows[marks[-1]], rule)
    r2 = logs / RUNS['r2'] / TARGET
    shared = [m for m in marks if done(r2, m)]
    result['r2_versus_frozen'] = {str(m): paired_summary(evaluation(r2, m)) for m in shared}
    result['minus_r2'] = {str(m): against_fixed(rows[m], evaluation(r2, m)) for m in shared}
    if 10000 in marks and 1250 in marks: result['late_minus_early'] = against_fixed(rows[10000], rows[1250])
    if not partial:
        result.update(retention(OUTPUT, protocol(), result['versus_frozen']['10000']))
        result['registered'] = verdict(gain=result['versus_frozen']['10000'], late_minus_early=result['late_minus_early'],
                                       minus_rule=result['minus_rule']['10000'], five=result['five_criteria'])
        save(ROOT / f'docs/lite3-online-s2-20261127/RESULTS_mrs_{TARGET}.json', result)
    x = result['rule_versus_frozen']
    print(f"{TARGET}: Frozen {count(x, 'Frozen')} rule {count(x, 'Updated')} frozen identical {result['frozen_reproduced']['identical']}/{result['frozen_reproduced']['n']}")
    for m in marks:
        v, t = result['versus_frozen'][str(m)], result['training'][str(m)]
        line = (f" {m:>5}: mrs {count(v, 'Updated')} gain {pp(v)} posture {count(v, 'Frozen', 'posture_failure')} -> {count(v, 'Updated', 'posture_failure')}"
                f" attempts {t['Updated']['attempts']:.1f} | minus rule {pp(result['minus_rule'][str(m)])}")
        if str(m) in result['minus_r2']: line += f" | r2 {count(result['r2_versus_frozen'][str(m)], 'Updated')} minus r2 {pp(result['minus_r2'][str(m)])}"
        print(line)
    if 'late_minus_early' in result: print('   10000 minus 1250:', pp(result['late_minus_early']))
    if 'registered' in result:
        print('   five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3),
              'families', {f: round(100 * v['mean_session_delta'], 1) for f, v in result['retention_families'].items()})
        print('   registered', result['registered'])


def chains(partial):
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    docs = root / 'docs/lite3-online-s2-20261127'
    chosen = json.loads((docs / 'PROBE.json').read_text())['selection']
    for name, row in chosen.items():
        if row['train']:
            subprocess.run([sys.executable, __file__, 'chains', *(['--partial'] if partial else [])], check=not partial,
                           env=dict(os.environ, S2_CHILD='1', C1A2_TARGET=row['target'], C1A2_VARIANT='mrs'))
    if partial: return
    previous = root / 'docs/lite3-online-adapter2-20261126'
    files = dict(stairs_up=previous / 'RESULTS_mrs_stairs_up_07.json')
    for name, row in chosen.items():
        files[name] = (docs if row['train'] else previous) / f"RESULTS_mrs_{row['target']}.json" if row['target'] else None
    effective, table = {}, {}
    for name, path in files.items():
        if path is None: effective[name] = None; continue
        r = json.loads(path.read_text()); v = r['versus_frozen']['10000']
        effective[name] = bool(v['mean_session_delta'] >= .10 - EPSILON and v['ci95'][0] > 0)
        table[name] = dict(target=r['target'], source=path.parent.name, frozen=v['Frozen']['outcomes'].get('success', 0), mrs=v['Updated']['outcomes'].get('success', 0),
                           gain=v['mean_session_delta'], ci95=v['ci95'], effective=effective[name], registered=r.get('registered'))
    summary = dict(targets=table, coverage=coverage(effective))
    (docs / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary['coverage'], ensure_ascii=False))


def top_one():
    """脚本化检查的初筛和确认（子进程；入口按 none 这一组导入，只用它的路径）。"""
    import numpy as np
    from reference_online_c1a2 import OUTPUT
    from reference_online_c1a_results import paired_summary
    from reference_online_c1aea_results import against_fixed, frozen_reproduced
    from reference_online_select_results import done, evaluation, read
    from reference_stairs_eval import ROOT, save
    base = OUTPUT.parent
    pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())
    rows = {g: evaluation(base / g, 0) for g in TOP_GROUPS}
    stall = evaluation(ROOT / 'logs/reference_six/online_c1b_stall_rate04_20261124/stairs_up_07', 10000)
    check = frozen_reproduced(rows['none'], stall)
    same = sum(a['outcome'] == b['outcome'] for a, b in zip(sorted((r for r in rows['none'] if r['arm'] == 'Updated'), key=lambda r: (r['session'], r['repeat'])),
                                                            sorted((r for r in stall if r['arm'] == 'Updated'), key=lambda r: (r['session'], r['repeat']))))
    result = dict(wiring_check=dict(frozen_identical=check['identical'], n=check['n'], updated_same_outcome=same), screening={})
    print('wiring check: Frozen identical', check['identical'], '/', check['n'], 'Updated same outcome', same, '/ 240')
    for g in TOP_GROUPS:
        v = paired_summary(rows[g]); d = against_fixed(rows[g], rows['none'])
        steps = [read(base / g, f'target_0_{p:02d}', 'top.json') for p in range(4)]
        result['screening'][g] = dict(versus_frozen=v, minus_none=d, mean_abs_request_per_step=sum(x['mean_abs_request_per_step'] * x['policy_steps'] for x in steps) / sum(x['policy_steps'] for x in steps))
        print(f" {g:14s} Frozen {v['Frozen']['outcomes'].get('success', 0)} Updated {v['Updated']['outcomes'].get('success', 0)} outcomes {v['Updated']['outcomes']} minus none {pp(d)}")
    success = {g: result['screening'][g]['versus_frozen']['Updated']['outcomes'].get('success', 0) for g in TOP_GROUPS}
    result['to_confirm'] = groups_to_confirm(success)
    print('to confirm:', result['to_confirm'])
    if all(done(base / f'confirm_{g}', 0) for g in ['none'] + result['to_confirm']):
        none = evaluation(base / 'confirm_none', 0)
        result['confirmation'] = dict(none=paired_summary(none))
        for g in result['to_confirm']:
            other = evaluation(base / f'confirm_{g}', 0); d = against_fixed(other, none)
            result['confirmation'][g] = dict(versus_frozen=paired_summary(other), minus_none=d, confirmed=confirmed(d))
            print(f" confirm {g}: none {d['B_success']} group {d['D_success']} delta {pp(d)} confirmed {confirmed(d)} outcomes {result['confirmation'][g]['versus_frozen']['Updated']['outcomes']}")
    save(ROOT / 'docs/lite3-online-s2-20261127/TOP.json', result)


if __name__ == '__main__':
    if os.environ.get('S2_CHILD'): chain_one('--partial' in sys.argv) if sys.argv[1] == 'chains' else top_one()
    elif sys.argv[1] == 'probe': probe()
    elif sys.argv[1] == 'chains': chains('--partial' in sys.argv)
    else:
        assert sys.argv[1] == 'top'
        subprocess.run([sys.executable, __file__, 'top'], check=True, env=dict(os.environ, S2_CHILD='1', C1A2_TARGET='stairs_up_07', C1A2_VARIANT='r2', C1A2_TOP='none'))
