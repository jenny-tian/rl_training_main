"""把按会话分开训练的几部分（train_00 …）合并成 train/，评估任务和统计脚本从 train/ 读取。

合并的内容：各快照的学习器参数（按会话顺序拼接）、回合记录、报告。逐步轨迹留在各部分的目录里，不复制。
"""
from __future__ import annotations

import json

from reference_online_c1a2 import MARKS, OUTPUT, TRAIN_PARTS
from reference_stairs_eval import save, sha


def merge_heads(loaded):
    """loaded：各部分同一个快照的内容。返回 60 个会话 × (Frozen, Updated) 的参数列表。"""
    sessions = [s for c in loaded for s in c['sessions']]
    if sessions != list(range(60)): raise ValueError('parts do not cover sessions 0..59 in order')
    for c in loaded:
        if len(c['heads']) != 2 * len(c['sessions']): raise ValueError('two heads per session required')
        if any(h['updated'] != bool(k % 2) for k, h in enumerate(c['heads'])): raise ValueError('heads must alternate Frozen, Updated')
        if any(c[key] != loaded[0][key] for key in ('control_steps', 'base_sha256', 'noise_sha256', 'kind')):
            raise ValueError('parts disagree on step, base policy or noise table')
    return [h for c in loaded for h in c['heads']]


def merge():
    import torch
    assert TRAIN_PARTS > 1
    parts = [OUTPUT / f'train_{p:02d}' for p in range(TRAIN_PARTS)]
    reports = [json.loads((d / 'report.json').read_text()) for d in parts]
    if any(r['status'] != 'passed' or r['control_steps'] != 60000 or not r['base_unchanged'] or not r['frozen_unchanged'] for r in reports):
        raise ValueError('every training part must have passed')
    out = OUTPUT / 'train'
    out.mkdir(parents=True, exist_ok=False)
    names = ['initial.pt'] + [f'at_{m}.pt' for m in MARKS[1:]] + ['final.pt']
    for name in names:
        loaded = [torch.load(d / name, map_location='cpu', weights_only=True) for d in parts]
        torch.save(dict(heads=merge_heads(loaded), control_steps=loaded[0]['control_steps'], base_sha256=loaded[0]['base_sha256'],
                        noise_sha256=loaded[0]['noise_sha256'], kind='train'), out / name)
    records = []
    for p, d in enumerate(parts):
        for r in json.loads((d / 'episodes.json').read_text())['records']:
            records.append(dict(r, part=p))
    if {r['session'] for r in records} != set(range(60)): raise ValueError('records do not cover all sessions')
    save(out / 'episodes.json', dict(kind='train', control_steps=60000, records=records))
    save(out / 'report.json', dict(status='passed', name='train', kind='train', merged_parts=[d.name for d in parts],
        environments=sum(r['environments'] for r in reports), control_steps=60000,
        interactions=sum(r['interactions'] for r in reports), complete_episodes=sum(r['complete_episodes'] for r in reports),
        adapter_updates=sum(r['adapter_updates'] for r in reports), session_updates=[u for r in reports for u in r['session_updates']],
        base_optimizer_updates=0, base_unchanged=True, frozen_unchanged=True, noise_sha256=reports[0]['noise_sha256'],
        part_report_sha256=[sha(d / 'report.json') for d in parts], part_episodes_sha256=[sha(d / 'episodes.json') for d in parts],
        snapshot_sha256={name: sha(out / name) for name in names}, missing_rays=sum(r.get('missing_rays', 0) for r in reports)))
    print(f'MERGED {len(parts)} parts: episodes={len(records)}, updates={sum(r["adapter_updates"] for r in reports)}', flush=True)
    return 0
