"""Verify and compare frozen profiles; render every saved rollout locally on CPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import html
import multiprocessing
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['LP_NUM_THREADS'] = '2'
import mujoco
import numpy as np

import reference_capability_mujoco as baseline
from reference_capability_mujoco_deployment import (OUTPUT, DOCS, MODELS, SCENES, RADII,
    read, save, sha, verify_sources, load_npz, episode_path, make_model, terrain_kind,
    schedule, window_stats, command, angles, observation, ContactScorer, geometric_clear,
    termination, state_hash)
from reference_capability_mujoco_report import LABELS, OUTCOMES, decode


def validate(record):
    folder = episode_path(record, OUTPUT)
    trace = load_npz(folder/'trace.npz'); initial = load_npz(folder/'initial.npz')
    original = load_npz(episode_path(record)/'initial.npz')
    assert sha(folder/'trace.npz') == record['trace_sha256']
    assert record['model_sha256'] == sha(MODELS[record['policy']][0])
    for key in original:
        np.testing.assert_array_equal(initial[key], original[key])
    assert state_hash(SimpleNamespace(**initial)) == record['initial_state_sha256']
    assert all(np.isfinite(v).all() for v in trace.values())
    seconds = (5. if record['scene'] == 'stand' else 6.) if record['smoke'] else 20.
    intervals = schedule(seconds)[:len(trace['time'])]
    np.testing.assert_array_equal(trace['time'], np.array([end/1000 for _, end in intervals]))
    assert trace['time'][-1] == record['duration_s']
    assert intervals[-1][1] == record['physics_steps']
    scorer = ContactScorer(make_model(terrain_kind(record['scene'])))
    route = record['scene'] not in ('stand', 'turn_right', 'arc_left')
    scores = [None]*3; dwell = 0.
    for i, (begin, end) in enumerate(intervals):
        qpos = trace['qpos'][i-1] if i else initial['qpos']
        qvel = trace['qvel'][i-1] if i else initial['qvel']
        prev = trace['action'][i-1] if i else np.zeros(12)
        cmd = command(record['scene'], begin/1000, angles(qpos[3:7])[2])
        np.testing.assert_array_equal(cmd, trace['command'][i])
        np.testing.assert_array_equal(observation(SimpleNamespace(qpos=qpos, qvel=qvel), cmd, prev), trace['observation'][i])
        supports, _ = scorer.score(trace['qpos'][i], trace['qvel'][i], trace['torque'][i])
        np.testing.assert_array_equal(supports, trace['support_radii'][i])
        assert supports[1] == trace['support'][i]
        clear = geometric_clear(record['scene'], trace['qpos'][i, :3], trace['feet'][i], record['initial_xyz']) if route else False
        dwell = dwell+(end-begin)/1000 if clear else 0.
        for j, support in enumerate(supports):
            result = termination(trace['rpy'][i], trace['gravity'][i],
                trace['qpos'][i, 1]-record['initial_xyz'][1], route, clear, dwell,
                support, bool(trace['torso'][i]))
            if result and scores[j] is None:
                scores[j] = dict(outcome=result, duration_s=end/1000,
                    lateral_m=float(trace['qpos'][i, 1]-record['initial_xyz'][1]))
        assert scores[1] is None or i == len(intervals)-1
    np.testing.assert_array_equal(np.clip(trace['raw_action'], -100., 100.), trace['action'])
    assert np.max(np.abs(trace['torque'])) <= 30.
    for j, radius in enumerate(RADII):
        if scores[j] is None:
            scores[j] = dict(outcome=('survived' if not route or record['smoke'] else 'timeout')
                if record['duration_s'] == seconds else 'censored',
                duration_s=record['duration_s'], lateral_m=record['lateral_m'])
        assert scores[j] == record['sensitivity'][str(radius)]
    assert record['outcome'] == scores[1]['outcome']
    assert record['windows'] == [window_stats(trace, lo, hi) for lo, hi in ((4., 8.), (10., 14.), (16., 20.))]
    return trace


def analyze():
    verify_sources(read(OUTPUT/'plan.json'))
    current = read(OUTPUT/'complete.json')['records']
    smoke = read(OUTPUT/'smoke.json')
    assert smoke['status'] == 'passed' and len(current) == 36 and len(smoke['records']) == 4
    for record in current+smoke['records']:
        validate(record)
    prior = read(baseline.OUTPUT/'complete.json')['records']
    audit = read(OUTPUT/'baseline_contact_validation.json')['records']
    old = []
    for record, corrected in zip(prior, audit, strict=True):
        score = corrected['sensitivity']['0.03']
        trace = load_npz(episode_path(record)/'trace.npz')
        keep = trace['time'] <= score['duration_s']+1e-9
        trace = {k: v[keep] for k, v in trace.items()}
        row = dict(record, **score)
        row['progress_m'] = float((-1 if record['scene'] == 'stairs_down_5cm' else 1)*
            (trace['qpos'][-1, 0]-record['initial_xyz'][0]))
        row['windows'] = [window_stats(trace, lo, hi) for lo, hi in ((4., 8.), (10., 14.), (16., 20.))]
        row['sensitivity'] = corrected['sensitivity']
        old.append(row)
    groups = []
    for profile, records in (('20ms_24_36Nm', old), ('12ms_30Nm', current)):
        for scene in SCENES:
            for name in MODELS:
                rows = [r for r in records if r['scene'] == scene and r['policy'] == name]
                groups.append(dict(profile=profile, scene=scene, policy=name, episodes=len(rows),
                    outcomes={label: sum(r['outcome'] == label for r in rows) for label in OUTCOMES},
                    mean_duration_s=float(np.mean([r['duration_s'] for r in rows])),
                    mean_success_duration_s=float(np.mean([r['duration_s'] for r in rows if r['outcome'] == 'success']))
                        if any(r['outcome'] == 'success' for r in rows) else None,
                    mean_lateral_m=float(np.mean([r['lateral_m'] for r in rows])),
                    max_abs_lateral_m=float(max(abs(r['lateral_m']) for r in rows)),
                    max_tilt_deg=float(max(r['max_tilt_deg'] for r in rows)),
                    max_torque_saturation_fraction=max(r['torque_saturation_fraction'] for r in rows),
                    torso_contact_samples=sum(r['torso_contact_samples'] for r in rows),
                    windows=[dict(start_s=lo, end_s=hi, complete_episodes=sum(r['windows'][idx]['complete'] for r in rows),
                        **{metric: float(np.mean([r['windows'][idx][metric] for r in rows]))
                            if all(r['windows'][idx]['complete'] for r in rows) else None
                            for metric in ('vx_mean', 'yaw_mean', 'yaw_mae')})
                        for idx, (lo, hi) in enumerate(((4., 8.), (10., 14.), (16., 20.)))]))
    comparisons = []
    for a, b in zip(old, current, strict=True):
        assert all(a[k] == b[k] for k in ('policy', 'scene', 'repeat', 'initial_state_sha256'))
        comparisons.append(dict(policy=a['policy'], scene=a['scene'], repeat=a['repeat'],
            old_outcome=a['outcome'], new_outcome=b['outcome'], old_time_s=a['duration_s'], new_time_s=b['duration_s'],
            time_difference_s=b['duration_s']-a['duration_s'], old_lateral_m=a['lateral_m'], new_lateral_m=b['lateral_m']))
    summary = dict(status='verified', formal_episodes=36, smoke_episodes=4, optimizer_updates=0,
        primary_radius_m=.03, baseline_corrected_without_rerun=True, groups=groups, comparisons=comparisons,
        new_radius_outcome_differences=[dict(policy=r['policy'], scene=r['scene'], repeat=r['repeat'], sensitivity=r['sensitivity'])
            for r in current if len({s['outcome'] for s in r['sensitivity'].values()}) > 1],
        new_radius_event_differences=[dict(policy=r['policy'], scene=r['scene'], repeat=r['repeat'], sensitivity=r['sensitivity'])
            for r in current if len({(s['outcome'], s['duration_s']) for s in r['sensitivity'].values()}) > 1],
        report_source_sha256=sha(__file__))
    save(DOCS/'summary.json', summary)
    with (DOCS/'episodes.csv').open('w') as stream:
        fields = list(comparisons[0]); writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(comparisons)
    for name in ('complete.json', 'smoke.json', 'baseline_contact_validation.json'):
        shutil.copy2(OUTPUT/name, DOCS/name)
    save(DOCS/'verification.json', dict(status='passed', formal_episodes=36, smoke_episodes=4,
        finite_traces=True, commands_observations_recomputed=True, all_supports_and_outcomes_recomputed=True,
        identical_cross_profile_initial_states=True, all_source_hashes_unchanged=True,
        offline_scoring_physics_steps=0, optimizer_updates=0, report_source_sha256=sha(__file__)))
    print('ANALYSIS VERIFIED: all 40 traces, commands, observations, supports, outcomes and initial states', flush=True)


def video_indices(times):
    """Sample at shared 100 ms display times, keeping terminal state at the end."""
    targets = np.arange(int(np.ceil(times[-1]*10)))/10
    return np.minimum(np.searchsorted(times, targets, side='left'), len(times)-1).tolist()+[len(times)-1]


def render_one(record):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    name, scene, repeat = record['policy'], record['scene'], record['repeat']
    key = f'{name}_{scene}_{repeat}'
    target = DOCS/'media'/key; target.mkdir(parents=True, exist_ok=False)
    trace = load_npz(episode_path(record, OUTPUT)/'trace.npz')
    assert sha(episode_path(record, OUTPUT)/'trace.npz') == record['trace_sha256']
    model = make_model(terrain_kind(scene)); data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=640)
    camera = mujoco.MjvCamera(); camera.lookat[:] = [1.85, 0., .15]
    camera.distance, camera.azimuth, camera.elevation = 5.6, 125., -32.
    if scene in ('arc_left', 'turn_right'):
        camera.lookat[:] = [record['initial_xyz'][0], 0., .15]; camera.distance = 4.5
    option = mujoco.MjvOption(); option.geomgroup[1] = 0
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    indices = video_indices(trace['time'])
    writer = imageio.get_writer(target/'rollout.mp4', fps=10, codec='libx264', quality=7,
        macro_block_size=2, ffmpeg_params=['-threads', '2'])
    try:
        for frame_index, tick in enumerate(indices):
            data.qpos[:] = trace['qpos'][tick]; data.qvel[:] = trace['qvel'][tick]
            mujoco.mj_forward(model, data)
            np.testing.assert_array_equal(data.qpos, trace['qpos'][tick])
            renderer.update_scene(data, camera=camera, scene_option=option)
            raw = renderer.render(); assert raw.shape == (360, 640, 3) and raw[80:335].std() > 2
            frame = Image.fromarray(raw); draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 76), fill=(15, 23, 35))
            cmd, vel = trace['command'][tick], trace['velocity'][tick]
            delta = trace['qpos'][tick, :3]-record['initial_xyz']
            lines = [f'{name} | {scene} | start {repeat} | 12ms / 30Nm | ~1x',
                f't={trace["time"][tick]:.3f}s vx cmd/actual={cmd[0]:.2f}/{vel[0]:.2f} yaw={cmd[2]:.2f}/{vel[5]:.2f}',
                f'dx={delta[0]:.2f}m dy={delta[1]:.2f}m support(3cm)={trace["support"][tick]}',
                f'ENDED: {record["outcome"]}; terminal pose held' if tick == len(trace['time'])-1 else 'running']
            for row, text in enumerate(lines):
                draw.text((8, 3+18*row), text, font=font, fill='white')
            writer.append_data(np.asarray(frame))
            for label, index in (('first', 0), ('middle', len(indices)//2), ('terminal', len(indices)-1)):
                if frame_index == index:
                    frame.save(target/f'{label}.png')
    finally:
        writer.close(); renderer.close()
    result = dict(policy=name, scene=scene, repeat=repeat, fps=10, frames=len(indices),
        simulation_duration_s=record['duration_s'], rendered_state_indices=indices,
        renderer='OSMesa CPU', render_physics_steps=0, render_policy_steps=0,
        source_sha256=sha(__file__), mp4_sha256=sha(target/'rollout.mp4'))
    save(target/'video.json', result)
    print('RENDERED', key, flush=True)
    return result


def pair(scene, repeat):
    import imageio.v2 as imageio
    folders = [DOCS/'media'/f'{name}_{scene}_{repeat}' for name in MODELS]
    infos = [read(p/'video.json') for p in folders]
    target = DOCS/'media'/f'comparison_{scene}_{repeat}'; target.mkdir(parents=True, exist_ok=False)
    readers = [imageio.get_reader(p/'rollout.mp4') for p in folders]
    writer = imageio.get_writer(target/'comparison.mp4', fps=10, codec='libx264', quality=7,
        macro_block_size=2, ffmpeg_params=['-threads', '2'])
    last = [None, None]; frames = max(info['frames'] for info in infos)
    try:
        for tick in range(frames):
            for side in (0, 1):
                if tick < infos[side]['frames']:
                    last[side] = readers[side].get_next_data()
            combined = np.concatenate(last, axis=1); writer.append_data(combined)
            for label, index in (('first', 0), ('middle', frames//2), ('terminal', frames-1)):
                if tick == index:
                    imageio.imwrite(target/f'{label}.png', combined)
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    result = dict(scene=scene, repeat=repeat, fps=10, frames=frames, shorter_side_holds_terminal=True,
        mp4_sha256=sha(target/'comparison.mp4'), screenshot_note='Taken at the same display frame; terminal sides may be held.')
    save(target/'comparison.json', result)
    return result


def render():
    verify_sources(read(OUTPUT/'plan.json'))
    records = read(OUTPUT/'complete.json')['records']
    assert len(records) == 36
    if (OUTPUT/'render_started.json').exists():
        raise FileExistsError('Render already started; inspect saved evidence before retry')
    save(OUTPUT/'render_started.json', dict(source_sha256=sha(__file__), physics_steps=0))
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        singles = list(pool.map(render_one, records))
    pairs = [pair(scene, repeat) for scene in SCENES for repeat in range(3)]
    save(OUTPUT/'render_complete.json', dict(status='completed', singles=singles, comparisons=pairs,
        render_physics_steps=0, render_policy_steps=0, optimizer_updates=0))


def finish():
    from PIL import Image, ImageDraw, ImageFont
    verify_sources(read(OUTPUT/'plan.json'))
    assert read(DOCS/'summary.json')['report_source_sha256'] == sha(__file__)
    rendered = read(OUTPUT/'render_complete.json')
    assert len(rendered['singles']) == 36 and len(rendered['comparisons']) == 18
    videos = sorted((DOCS/'media').rglob('*.mp4')); assert len(videos) == 54
    with ThreadPoolExecutor(max_workers=2) as pool:
        infos = list(pool.map(decode, videos))
    for path, info in zip(videos, infos):
        comparison = path.name == 'comparison.mp4'
        meta = read(path.parent/('comparison.json' if comparison else 'video.json'))
        assert int(info['nb_frames']) == meta['frames'] and sha(path) == meta['mp4_sha256']
        assert (info['width'], info['height']) == ((1280, 360) if comparison else (640, 360))
        if not comparison:
            assert meta['source_sha256'] == sha(__file__)
    records = read(OUTPUT/'complete.json')['records']; cards = []
    for scene in SCENES:
        for repeat in range(3):
            rows = [r for r in records if r['scene'] == scene and r['repeat'] == repeat]
            detail = '；'.join(f'{r["policy"]}：{OUTCOMES[r["outcome"]]}，{r["duration_s"]:.3f} 秒' for r in rows)
            base = f'media/comparison_{scene}_{repeat}'
            cards.append(f'<article><h2>{LABELS[scene]} · 起点 {repeat}</h2><p>{html.escape(detail)}</p>'
                f'<video controls preload="none" poster="{base}/middle.png" src="{base}/comparison.mp4"></video></article>')
    (DOCS/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Lite3 12 ms 冻结对照</title>
<style>body{background:#101722;color:#e5edf7;font:16px/1.65 system-ui;margin:0}main{max-width:1300px;margin:auto;padding:24px}
article{background:#192536;padding:20px;margin:24px 0;border-radius:10px}video{width:100%}a{color:#8bc9ff}</style>
<main><h1>12 毫秒 / 30 Nm：旧模型与 B</h1><p>左为 zero10100，右为 B10200。全部 18 组固定起点，视频和截图只在本地。</p>
<p>5 cm 台阶、1–6 cm 粗糙地面；静态地形足心 3 cm 支撑判定。视频由保存状态渲染，不推进物理。
约 1 倍速，采样时间误差不超过 12 ms，末帧有不足 0.1 秒的取整。提前结束的一侧保持末姿态，标为 ENDED，不代表继续站稳。
转向存活不等于跟踪达标。此入口复用原部署周期和力矩上限，未运行完整 C++/UDP/站起状态机。</p>
<p><a href="RESULTS.md">阶段报告</a> · <a href="summary.json">完整数据</a> ·
<a href="../lite3-capability-mujoco-20261116/index.html">上一阶段 20 ms 视频（保留原始支撑判定，文字报告已修正）</a></p>'''+''.join(cards)+'</main></html>\n')
    sheet = Image.new('RGB', (1280, 6*220), '#101722'); draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    for idx, scene in enumerate(SCENES):
        draw.text((8, idx*220+4), f'{scene} | start 0 | zero10100 left / B right | 12ms,30Nm', font=font, fill='white')
        sheet.paste(Image.open(DOCS/'media'/f'comparison_{scene}_0/middle.png').resize((1280, 180)), (0, idx*220+30))
    sheet.save(DOCS/'overview.png')
    shutil.copy2(OUTPUT/'render_complete.json', DOCS/'render_complete.json')
    verification = read(DOCS/'verification.json')
    verification.update(videos_decoded=54, video_hashes_frames_dimensions_verified=True,
        render_physics_steps=0, render_policy_steps=0, media_local_only=True)
    save(DOCS/'verification.json', verification)
    save(DOCS/'raw_manifest.json', [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(OUTPUT.rglob('*')) if p.is_file()])
    save(DOCS/'artifact_manifest.json', [dict(path=str(p.relative_to(DOCS)), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(DOCS.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json'])
    print('FINISHED: 54 videos decoded and hashed; media local only', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('analyze', 'render', 'finish'))
    args = parser.parse_args()
    {'analyze': analyze, 'render': render, 'finish': finish}[args.phase]()
