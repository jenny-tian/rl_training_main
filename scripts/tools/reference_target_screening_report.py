"""S0 目标筛选的统计：读各进程的 report.json，按方案第 4 节判定，输出 Markdown 表和 summary.json。

用法：python scripts/tools/reference_target_screening_report.py [运行名 ...]
默认读 screen 和三个摩擦运行。
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import reference_target_screening as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / 'logs/reference_six/target_screening_20261125'
DEFAULT = ('screen', 'screen_friction_1020', 'screen_friction_0512', 'screen_friction_0206', 'screen_round2', 'screen_round3')
LABEL = dict(eligible='符合', too_easy='太容易', too_hard='太难', excluded='排除')


def load(runs):
    records, checks = collections.defaultdict(list), []
    for run in runs:
        report = json.loads((ROOT / run / 'report.json').read_text())
        checks.append(dict(run=run, status=report['status'], episodes=report['episodes'], start_failures=report['start_failures'],
                           optimizer_updates=report['optimizer_updates'], actor_unchanged=report['actor_unchanged'],
                           model_sha256=report['model_sha256']))
        for record in report['records']:
            records[record['condition']].append(record)
    return records, checks


def fmt(value, digits=2, scale=1.):
    return '—' if value is None else f'{value * scale:.{digits}f}'


def main(runs):
    records, checks = load(runs)
    stats = {name: S.candidate_stats(rows) for name, rows in records.items()}
    chains = {family: [[n for n in chain if n in stats] for chain in groups] for family, groups in S.CHAINS.items()}
    chains = {family: [c for c in groups if c] for family, groups in chains.items() if any(groups)}
    selection = S.select(stats, chains)
    order = [n for groups in chains.values() for chain in groups for n in chain] + [n for n in S.REFERENCE if n in stats]
    order += [n for n in stats if n not in order]  # 第二轮的中间参数候选
    print('| 候选 | 成功 | 姿态失败 | 停住 | 超时 | 横向越界 | 起步失败 | 平均推进 | 判定 |\n|---|---|---|---|---|---|---|---|---|')
    for name in order:
        s = stats[name]
        kind = '对照' if name in S.REFERENCE else LABEL[S.classify(s)]
        print(f"| {name} | {s['success']}/{s['n']} | {s['posture_failure']} | {s['stuck']} | {s['timeout']} | {s['lateral_exit']} | "
              f"{s['start_failure']} | {s['mean_progress']:.0%} | {kind} |")
    print('\n| 候选 | 回合 | 支撑脚滑动 % | 支撑脚速度 m/s | 前腿受阻 % | 后腿受阻 % | 摆动脚碰撞 % | 力矩到限 % | 最大倾角 ° |\n|---|---|---|---|---|---|---|---|---|')
    diagnostics = {}
    for name in order:
        diagnostics[name] = dict(all=S.diagnostic_means(records[name]), failures=S.diagnostic_means(records[name], failures_only=True))
        for label, d in (('全部', diagnostics[name]['all']), ('失败', diagnostics[name]['failures'])):
            print(f"| {name}（{label}） | {d['episodes']} | {fmt(d['stance_slip_ratio'], 1, 100)} | {fmt(d['stance_foot_speed'], 3)} | "
                  f"{fmt(d['swing_stall_ratio_front'], 1, 100)} | {fmt(d['swing_stall_ratio_rear'], 1, 100)} | {fmt(d['swing_hit_ratio'], 1, 100)} | "
                  f"{fmt(d['torque_limit_ratio'], 1, 100)} | {fmt(d['max_tilt_deg'], 1)} |")
    print()
    for family, result in selection.items():
        print(f"{family}: 入选 {result['chosen']}，中间参数待筛 {result['bracket']}")
    print()
    for check in checks:
        print(check)
    (ROOT / 'summary.json').write_text(json.dumps(dict(checks=checks, stats=stats, selection=selection, diagnostics=diagnostics),
                                                  ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or DEFAULT)
