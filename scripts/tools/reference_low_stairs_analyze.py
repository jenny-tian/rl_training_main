"""Analyze the approved low-stair matrix without changing any policy weights."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

import numpy as np

from reference_stairs_eval import ROOT, MODELS, load_actor, save, sha
from reference_low_stairs_evaluate import TRAINING, OUTPUT, MODEL, EXPECTED_SHA, read, verify_training, verify_run
from reference_core500_evaluate import same_initial
from reference_core500_analyze import paired_turn_rows, compare_video, write_csv
from reference_turn_analyze import analyze as analyze_turn
from reference_isaac_stairs_analyze import analyze as analyze_stairs

OUT = ROOT / 'docs/lite3-low-stairs-results-20261114'


def adapter_check(checkpoint):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    torch.set_num_threads(1)
    torch.manual_seed(20261114)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    with np.load(OUTPUT / 'turn_nominal/initial.npz') as initial:
        actual = torch.from_numpy(initial['observation'].copy())
    inputs = torch.cat((torch.randn(2048, 45), actual))
    obs = TensorDict({'policy': inputs}, batch_size=[len(inputs)])
    cfg = read_agent()
    native = MLPModel(obs, {'actor': ['policy']}, 'actor', 12, **cfg).eval()
    native.load_state_dict(state['actor_state_dict'], strict=True)
    MODELS[MODEL] = checkpoint
    adapted = load_actor(MODEL)
    with torch.inference_mode():
        difference = (native(obs) - adapted(inputs)).abs().max().item()
    assert difference == 0.
    sources = read(TRAINING / 'report.json')['sources']
    assert all(sha(row['path']) == row['sha256'] for row in sources)
    return dict(checkpoint=str(checkpoint), sha256=sha(checkpoint), updates=10000, adam_steps=200000,
                actor_adapter_samples=len(inputs), actor_adapter_max_difference=difference,
                training_sources_unchanged=len(sources), parameters_finite=True)


def read_agent():
    import yaml
    cfg = yaml.safe_load((TRAINING / 'params/agent.yaml').read_text())['actor']
    cfg.pop('class_name')
    assert cfg['obs_normalization'] is False
    return cfg


def training_metrics():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event = EventAccumulator(str(TRAINING / 'training'), size_guidance={'scalars': 0})
    event.Reload()
    tags = ['Train/mean_reward', 'Train/mean_episode_length', 'Episode_Termination/bad_orientation_2',
            'Episode_Termination/time_out', 'Metrics/base_velocity/error_vel_xy',
            'Metrics/base_velocity/error_vel_yaw', 'Episode_Reward/stair_progress',
            'Episode_Reward/stair_landing', 'Episode_Reward/stair_clearance']
    metrics = {}
    for tag in tags:
        values = event.Scalars(tag)
        assert len(values) == 10000 and values[0].step == 0 and values[-1].step == 9999
        array = np.array([v.value for v in values])
        assert np.isfinite(array).all()
        metrics[tag] = dict(first100=float(array[:100].mean()), previous5336=float(array[5236:5336].mean()),
                            last100=float(array[-100:].mean()), last1000=float(array[-1000:].mean()))
    return metrics


def stairs_heights(report):
    rows = []
    for height in (0., .02, .03, .05):
        records = [r for r in report['records'] if r['height'] == height]
        counts = {name: sum(r['outcome'] == name for r in records)
                  for name in ('success', 'stuck', 'orientation_failure', 'lateral_exit', 'timeout')}
        assert len(records) == 60 and sum(counts.values()) == 60
        rows.append(dict(profile=report['profile'], direction=report['direction'], height=height, starts=60,
                         first_riser=sum(r['best_risers'] >= 1 for r in records),
                         best_risers=max(r['best_risers'] for r in records), **counts))
    return rows


def check_log(directory):
    lines = (directory / 'runtime.log').read_text(errors='replace').splitlines()
    bad = [line for line in lines if re.search(
        r'\[Error\]|Traceback|switching to software|GPU solver failed|GPU Bp pipeline failed|Contacts have been dropped', line, re.I)]
    assert not bad, (directory, bad)
    return dict(run=str(directory), runtime_errors=0, physics_fallback_messages=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    complete = read(OUTPUT / 'complete.json')
    assert complete['status'] == 'passed' and complete['candidate_episodes'] == 1600
    checkpoint = verify_training()
    turns, windows, pairs, pairing, heights, groups, episodes, rewards, checks, logs = [], [], [], [], [], [], [], [], [], []
    raw = set()
    for entry in complete['matrix']:
        directory = Path(entry['candidate'])
        report = verify_run(directory, entry['kind'], entry['profile'], entry['direction'])
        logs.append(check_log(directory))
        if entry['kind'] == 'turn':
            _, data, rows, grouped = analyze_turn(directory)
            turns.extend(grouped)
            windows.extend(rows)
            for baseline_name, baseline in (
                ('original10000', ROOT / f'logs/reference_six/code_turn_audit_20261112/update10000_{entry["profile"]}'),
                ('core10500', ROOT / f'logs/reference_six/core500_evaluation_20261112/core10500_turn_{entry["profile"]}'),
            ):
                equal = same_initial(directory, baseline)
                pairing.append(dict(baseline=baseline_name, profile=entry['profile'], initial_equal=equal,
                                    candidate=str(directory), source=str(baseline)))
                if equal:
                    _, old_data, old_rows, _ = analyze_turn(baseline)
                    assert all(np.isfinite(v).all() for v in old_data.values())
                    pairs.extend(dict(baseline=baseline_name, **r)
                                 for r in paired_turn_rows(old_rows, rows, entry['profile']))
                    raw.update(p for p in baseline.iterdir() if p.is_file())
        else:
            _, _, data, ep, rw, _ = analyze_stairs(directory)
            episodes.extend(ep)
            rewards.extend(rw)
            heights.extend(stairs_heights(report))
            groups.extend(dict(profile=entry['profile'], direction=entry['direction'],
                               **{k: v for k, v in g.items() if k != 'outcomes'},
                               **{name: g['outcomes'].get(name, 0) for name in
                                  ('success', 'stuck', 'orientation_failure', 'lateral_exit', 'timeout')})
                          for g in report['groups'])
        assert all(np.isfinite(v).all() for v in data.values())
        checks.append(dict(run=str(directory), first_episodes_only=True, arrays_finite=True,
                           episodes=report['episodes'], actor_unchanged=report['actor_unchanged']))
        raw.update(p for p in directory.iterdir() if p.is_file())
        for name in ('report.json', 'config.json', 'execution.json'):
            shutil.copy2(directory / name, out / f'{directory.name}_{name}')
    assert sum(c['episodes'] for c in checks) == 1600 and len(episodes) == 960
    for name, rows in [('stairs_heights', heights), ('stairs_groups', groups), ('stairs_episodes', episodes),
                       ('stairs_reward_windows', rewards), ('turn_groups', turns), ('turn_windows', windows)]:
        write_csv(out / f'{name}.csv', rows)
    if pairs:
        write_csv(out / 'turn_paired.csv', pairs)
    video_checks = {}
    for video in complete['videos']:
        directory = Path(video['candidate'])
        verify_run(directory, video['kind'], video['profile'], video['direction'], video['video_env'])
        logs.append(check_log(directory))
        dest = out / 'clips' / video['name']
        compare_video(directory, Path(video['formal']), dest)
        video_checks[video['name']] = read(dest / 'replay_validation.json')
        raw.update(p for p in directory.iterdir() if p.is_file())
    adapter = adapter_check(checkpoint)
    metrics = training_metrics()
    save(out / 'training_metrics.json', metrics)
    save(out / 'summary.json', dict(status='passed', status_scope='artifact validation, not policy acceptance',
        candidate_episodes=1600, candidate_sha256=EXPECTED_SHA, stairs_heights=heights, stairs_groups=groups,
        turn_groups=turns, turn_paired=pairs, turn_initial_pairing=pairing,
        rollout_checks=checks, matrix=complete['matrix'], videos=complete['videos'],
        caveats=['Single training seed; fixed test starts; Isaac Lab only, not deployment qualification.',
                 'Common reference rewards are diagnostic scores and do not participate in action selection.',
                 'Turn comparisons include only paired complete windows; failure counts retain every start.',
                 'No paired older-policy low-stair evaluation; historical stair results are context only.']))
    save(out / 'verification.json', dict(status='passed', checkpoint=adapter, formal_episodes=1600,
        video_replays_excluded_from_formal_count=True, video_checks=video_checks, runtime_logs=logs,
        turn_pairing=pairing, no_training_performed=True))
    for name in ('execution.json', 'reward_override_audit.json', 'low_stairs_terrain_audit.json'):
        shutil.copy2(TRAINING / name, out / f'training_{name}')
    raw.update(p for p in OUTPUT.iterdir() if p.is_file())
    raw.update([TRAINING / 'report.json', TRAINING / 'execution.json', checkpoint])
    save(out / 'raw_manifest.json', [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(raw)])
    plot(out, heights, turns, pairs)
    print('LOW STAIRS ANALYSIS COMPLETE: 1600 formal episodes, four verified video replays', flush=True)


def plot(out, heights, turns, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, profile in enumerate(('nominal', 'training_randomized')):
        for col, direction in enumerate(('up', 'down')):
            selected = [next(r for r in heights if (r['profile'], r['direction'], r['height']) ==
                             (profile, direction, h)) for h in (.02, .03, .05)]
            ax = axes[row, col]
            ax.bar(np.arange(3) - .17, [r['first_riser'] for r in selected], .34, label='First riser')
            ax.bar(np.arange(3) + .17, [r['success'] for r in selected], .34, label='Full course')
            ax.set_xticks(range(3), ['2 cm', '3 cm', '5 cm'])
            ax.set_ylim(0, 65)
            ax.set_title(f'{profile} / {direction}')
            ax.set_ylabel('Completed / 60 starts')
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / 'stairs_completion.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cases = ('low_left', 'low_right', 'turn_left', 'turn_right', 'fast_left', 'fast_right')
    for row, profile in enumerate(('nominal', 'training_randomized')):
        for col, baseline in enumerate(('original10000', 'core10500')):
            ax = axes[row, col]
            selected = [next((r for r in pairs if (r['baseline'], r['profile'], r['case'], r['window']) ==
                              (baseline, profile, case, 'steady')), None) for case in cases]
            if all(r is not None for r in selected):
                ax.plot(range(6), [r['baseline_actual'] for r in selected], marker='o', label=baseline)
                ax.plot(range(6), [r['candidate_actual'] for r in selected], marker='o', label=MODEL)
                labels = [f'{label}\nn={r["n"]}' for label, r in zip(
                    ['L .2', 'R .2', 'L .5', 'R .5', 'L .8', 'R .8'], selected)]
                ax.set_xticks(range(6), labels)
            ax.plot(range(6), [.2, -.2, .5, -.5, .8, -.8], 'k--', label='command')
            ax.set_ylabel('Paired mean yaw velocity, 4-8 s (rad/s)')
            ax.set_title(f'{profile} / {baseline}')
            ax.grid(alpha=.2)
            ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'turn_tracking.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
