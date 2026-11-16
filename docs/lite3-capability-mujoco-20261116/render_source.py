"""Render saved MuJoCo states without physics, then audit and summarize all pairs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import html
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess

# Software rendering consumes no GPU and never runs another policy episode.
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['LP_NUM_THREADS'] = '2'
import mujoco
import numpy as np

from reference_capability_mujoco import (OUTPUT, DOCS, MODELS, SCENES, read, save, sha,
    make_model, terrain_kind, observation, command, angles, verify_sources, PERIOD)

LABELS = {'flat': '平地直行', 'stairs_up_5cm': '5 cm 上台阶', 'stairs_down_5cm': '5 cm 下台阶',
    'rough': '粗糙地面直行', 'turn_right': '中速右转', 'arc_left': '边走边左转'}
OUTCOMES = {'success': '完成', 'survived': '存活至结束', 'lateral_exit': '横向越界',
    'orientation_failure': '姿态失败', 'timeout': '超时'}


def load(path):
    with np.load(path) as archive:
        return {k: archive[k] for k in archive.files}


def frame_indices(n):
    values = list(range(0, n, 5))
    if values[-1] != n-1:
        values.append(n-1)
    return values


def validate_record(path, record):
    from types import SimpleNamespace
    data, initial = load(path/'trace.npz'), load(path/'initial.npz')
    assert sha(path/'trace.npz') == record['trace_sha256']
    assert all(np.isfinite(v).all() for v in data.values())
    n = len(data['time'])
    assert n == round(record['duration_s']/PERIOD)
    assert np.allclose(data['time'], (np.arange(n)+1)*PERIOD, rtol=0, atol=1e-12)
    for tick in range(n):
        previous_pos = data['qpos'][tick-1] if tick else initial['qpos']
        previous_vel = data['qvel'][tick-1] if tick else initial['qvel']
        previous_action = data['action'][tick-1] if tick else np.zeros(12)
        expected_command = command(record['scene'], tick*PERIOD, angles(previous_pos[3:7])[2])
        assert np.array_equal(data['command'][tick], expected_command)
        expected_obs = observation(SimpleNamespace(qpos=previous_pos, qvel=previous_vel), expected_command, previous_action)
        assert np.array_equal(expected_obs, data['observation'][tick])
    assert np.array_equal(np.clip(data['raw_action'], -100, 100), data['action'])
    return data


def render_one(record):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    name, scene, repeat = record['policy'], record['scene'], record['repeat']
    key = f'{name}_{scene}_{repeat}'
    source = OUTPUT/'episodes'/key
    target = DOCS/'media'/key
    target.mkdir(parents=True, exist_ok=False)
    trace = validate_record(source, record)
    model = make_model(terrain_kind(scene)); data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1.85, 0., .15]
    camera.distance, camera.azimuth, camera.elevation = 5.6, 125., -32.
    if scene in ('arc_left', 'turn_right'):
        camera.lookat[:] = [record['initial_xyz'][0], 0., .15]
        camera.distance = 4.5
    option = mujoco.MjvOption()
    option.geomgroup[1] = 0  # hide collision overlays, retain original visual mesh
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    indices = frame_indices(len(trace['time']))
    writer = imageio.get_writer(target/'rollout.mp4', fps=10, codec='libx264', quality=7,
        macro_block_size=2, ffmpeg_params=['-threads', '2'])
    maximum_error = 0.
    try:
        for frame_index, tick in enumerate(indices):
            data.qpos[:] = trace['qpos'][tick]; data.qvel[:] = trace['qvel'][tick]
            mujoco.mj_forward(model, data)  # kinematics only: no mj_step, policy or optimization
            maximum_error = max(maximum_error, float(np.max(np.abs(data.qpos-trace['qpos'][tick]))))
            renderer.update_scene(data, camera=camera, scene_option=option)
            raw = renderer.render()
            assert raw.shape == (360, 640, 3) and raw[80:335].std() > 2
            frame = Image.fromarray(raw)
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 76), fill=(15, 23, 35))
            cmd, vel = trace['command'][tick], trace['velocity'][tick]
            delta = trace['qpos'][tick, :3]-record['initial_xyz']
            tilt = np.degrees(np.arccos(np.clip(-trace['gravity'][tick, 2], -1., 1.)))
            ended = tick == len(trace['time'])-1
            lines = [f'{name} | {scene} | start {repeat} | MuJoCo | ~1x',
                f't={trace["time"][tick]:.2f}s  vx cmd/actual={cmd[0]:.2f}/{vel[0]:.2f} m/s  yaw={cmd[2]:.2f}/{vel[5]:.2f} rad/s',
                f'dx={delta[0]:.2f} m  dy={delta[1]:.2f} m  tilt={tilt:.1f} deg',
                f'ENDED: {record["outcome"]}; terminal pose held' if ended else 'running']
            for i, line in enumerate(lines):
                draw.text((7, 3+i*18), line, font=font, fill='#ffd077' if ended and i == 3 else 'white')
            draw.rectangle((0, 339, 640, 360), fill=(15, 23, 35))
            draw.text((7, 341), 'Saved states | 20 ms policy | 24/36 Nm | rendering without physics', font=font, fill='white')
            writer.append_data(np.asarray(frame))
            for label, wanted in (('first', 0), ('middle', len(indices)//2), ('terminal', len(indices)-1)):
                if frame_index == wanted:
                    frame.save(target/f'{label}.png')
        for _ in range(10):
            writer.append_data(np.asarray(frame))
    finally:
        writer.close(); renderer.close()
    result = dict(policy=name, scene=scene, repeat=repeat, fps=10, frames=len(indices)+10,
        sample_steps=[t+1 for t in indices], terminal_hold_s=1., trace_sha256=record['trace_sha256'],
        maximum_qpos_error=maximum_error, render_physics_steps=0, render_policy_steps=0,
        renderer='OSMesa CPU', source_sha256=sha(__file__), mp4_sha256=sha(target/'rollout.mp4'))
    save(target/'video.json', result)
    print('RENDERED', key, flush=True)
    return result


def probe(path):
    import json
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames', '-of', 'json', str(path)], text=True))['streams'][0]


def pair(scene, repeat):
    import imageio.v2 as imageio
    from PIL import Image
    folders = [DOCS/'media'/f'{name}_{scene}_{repeat}' for name in MODELS]
    infos = [read(p/'video.json') for p in folders]
    target = DOCS/'media'/f'comparison_{scene}_{repeat}'
    target.mkdir(parents=True, exist_ok=False)
    readers = [imageio.get_reader(p/'rollout.mp4') for p in folders]
    writer = imageio.get_writer(target/'comparison.mp4', fps=10, codec='libx264', quality=7,
        macro_block_size=2, ffmpeg_params=['-threads', '2'])
    last = [None, None]
    frames = max(i['frames'] for i in infos)
    try:
        for tick in range(frames):
            for side in (0, 1):
                if tick < infos[side]['frames']:
                    last[side] = readers[side].get_next_data()
            writer.append_data(np.concatenate(last, axis=1))
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    for label in ('first', 'middle', 'terminal'):
        imageio.imwrite(target/f'{label}.png', np.concatenate([np.asarray(Image.open(p/f'{label}.png')) for p in folders], axis=1))
    result = dict(scene=scene, repeat=repeat, fps=10, frames=frames, shorter_side_holds_terminal=True,
        mp4_sha256=sha(target/'comparison.mp4'), paired_simulation_starts=True,
        screenshot_note='Middle/terminal screenshots use each individual clip midpoint/end; on-screen times may differ.')
    save(target/'comparison.json', result)
    return result


def render():
    complete = read(OUTPUT/'complete.json'); verify_sources(read(OUTPUT/'plan.json'))
    assert complete['status'] == 'completed' and len(complete['records']) == 36
    if (OUTPUT/'render_started.json').exists():
        raise FileExistsError('Inspect existing render evidence before any retry')
    save(OUTPUT/'render_started.json', dict(source_sha256=sha(__file__), backend='osmesa', physics_steps=0))
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        singles = list(pool.map(render_one, complete['records']))
    pairs = [pair(scene, repeat) for scene in SCENES for repeat in range(3)]
    save(OUTPUT/'render_complete.json', dict(status='completed', singles=singles, comparisons=pairs,
        render_physics_steps=0, render_policy_steps=0, optimizer_updates=0))
    print('RENDER COMPLETE: 36 single videos, 18 comparisons; no physics steps', flush=True)


def decode(path):
    subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-f', 'null', '-'],
        check=True, capture_output=True)
    return probe(path)


def report():
    from PIL import Image, ImageDraw, ImageFont
    plan = read(OUTPUT/'plan.json'); verify_sources(plan)
    complete, render_complete = read(OUTPUT/'complete.json'), read(OUTPUT/'render_complete.json')
    assert complete['formal_episodes'] == 36 and complete['smoke_episodes'] == 4
    assert complete['optimizer_updates'] == 0 and render_complete['render_physics_steps'] == 0
    records = complete['records']
    for r in records:
        source = OUTPUT/'episodes'/f'{r["policy"]}_{r["scene"]}_{r["repeat"]}'
        validate_record(source, r)
    for scene in SCENES:
        for repeat in range(3):
            assert len({r['initial_state_sha256'] for r in records if r['scene'] == scene and r['repeat'] == repeat}) == 1
    videos = sorted((DOCS/'media').rglob('*.mp4'))
    assert len(videos) == 54
    with ThreadPoolExecutor(max_workers=2) as executor:
        probes = list(executor.map(decode, videos))
    for path, info in zip(videos, probes):
        comparison = path.name == 'comparison.mp4'
        meta = read(path.parent/('comparison.json' if comparison else 'video.json'))
        assert int(info['nb_frames']) == meta['frames'] and sha(path) == meta['mp4_sha256']
        assert (info['width'], info['height']) == ((1280, 360) if comparison else (640, 360))
    groups = []
    for scene in SCENES:
        for name in MODELS:
            rows = [r for r in records if r['scene'] == scene and r['policy'] == name]
            values = {k: sum(r['outcome'] == k for r in rows) for k in OUTCOMES}
            windows = []
            for index in range(3):
                shared = [r for r in rows if r['windows'][index]['complete'] and next(o for o in records
                    if o['scene'] == scene and o['repeat'] == r['repeat'] and o['policy'] != name)['windows'][index]['complete']]
                windows.append(dict(window=rows[0]['windows'][index]['start_s'], paired_n=len(shared),
                    **{key: float(np.mean([r['windows'][index][key] for r in shared])) if shared else None
                        for key in ('vx_mean', 'yaw_mean', 'yaw_mae')}))
            groups.append(dict(scene=scene, policy=name, episodes=len(rows), **values,
                mean_duration_s=float(np.mean([r['duration_s'] for r in rows])),
                max_tilt_deg=max(r['max_tilt_deg'] for r in rows), common_complete_windows=windows))
    summary = dict(status='completed', formal_episodes=36, smoke_episodes=4, optimizer_updates=0,
        sources_unchanged=True, paired_initial_states=True, groups=groups, records=records,
        profile=plan['profile'], period_s=PERIOD, renderer='OSMesa CPU', videos=54, comparisons=18,
        total_episode_wall_s=sum(r['elapsed_s'] for r in records)+sum(r['elapsed_s'] for r in read(OUTPUT/'smoke.json')['records']),
        limitations=plan['training_differences'], report_source_sha256=sha(__file__))
    save(DOCS/'summary.json', summary)
    with (DOCS/'episodes.csv').open('w') as stream:
        fields = ['policy', 'scene', 'repeat', 'outcome', 'duration_s', 'progress_m', 'lateral_m',
            'max_tilt_deg', 'mean_vx_after_2s', 'torque_saturation_fraction', 'torso_contact_samples', 'initial_state_sha256']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in records)
    for filename in ('plan.json', 'smoke.json', 'complete.json', 'render_complete.json'):
        shutil.copy2(OUTPUT/filename, DOCS/('execution_plan.json' if filename == 'plan.json' else filename))
    cards = []
    for scene in SCENES:
        for repeat in range(3):
            rows = [next(r for r in records if (r['scene'], r['repeat'], r['policy']) == (scene, repeat, name)) for name in MODELS]
            detail = '；'.join(f'{r["policy"]}：{OUTCOMES[r["outcome"]]}，{r["duration_s"]:.2f} 秒' for r in rows)
            base = f'media/comparison_{scene}_{repeat}'
            cards.append(f'<article><h2>{LABELS[scene]} · 起点 {repeat}</h2><p>{html.escape(detail)}</p>'
                f'<video controls preload="none" poster="{base}/middle.png" src="{base}/comparison.mp4"></video></article>')
    (DOCS/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Lite3 MuJoCo 冻结对比</title>
<style>body{background:#101722;color:#e5edf7;font:16px/1.65 system-ui;margin:0}main{max-width:1300px;margin:auto;padding:24px}
article{background:#192536;padding:20px;margin:24px 0;border-radius:10px}video{width:100%}h2{font-size:22px}</style>
<main><h1>旧模型与 B：MuJoCo 冻结验证</h1><p>左：zero10100；右：B10200。全部 18 个配对起点均保留。视频和截图只在本地。</p>
<p>使用部署机器人资产，独立适配为 20 ms 策略周期、24/36 Nm 力矩限幅。台阶为 5 cm，粗糙地面高度 1–6 cm。
这不是原 C++/UDP 部署入口的原样运行，也不是 Isaac Lab 评测。每组仅 3 个固定起点。</p>
<p>画面由保存状态渲染，渲染不推进物理。提前结束的一侧保持终止姿态，末帧标注 ENDED；停留不表示继续站稳。
普通片段约 1 倍速，末采样间隔可能更短，以画面时间为准。转向的存活不等于跟踪达标。</p>'''+''.join(cards)+'</main></html>\n')
    sheet = Image.new('RGB', (1280, 6*220), '#101722'); draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    for index, scene in enumerate(SCENES):
        draw.text((8, index*220+4), f'{scene} | start 0 | zero10100 left / B right', font=font, fill='white')
        sheet.paste(Image.open(DOCS/'media'/f'comparison_{scene}_0/terminal.png').resize((1280, 180)), (0, index*220+30))
    sheet.save(DOCS/'overview.png')
    save(DOCS/'raw_manifest.json', [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(OUTPUT.rglob('*')) if p.is_file()])
    save(DOCS/'verification.json', dict(status='passed', formal_episodes=36, smoke_episodes=4, source_hashes_unchanged=True,
        native_actor_parity=plan['actor_parity'], all_commands_and_observations_recomputed=True,
        all_traces_finite=True, identical_paired_initial_states=True, videos_decoded=54,
        dimensions_and_frames_verified=True, render_physics_steps=0, optimizer_updates=0,
        report_source_sha256=sha(__file__)))
    save(DOCS/'artifact_manifest.json', [dict(path=str(p.relative_to(DOCS)), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(DOCS.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json'])
    print('REPORT VERIFIED: 36 formal, 4 smoke, 54 videos, 18 comparisons', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('render', 'report'))
    args = parser.parse_args()
    render() if args.phase == 'render' else report()
