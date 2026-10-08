"""Compare the fixed retention coverage ablation with its paid paired controls."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    evidence = {}

    def read(path):
        evidence[str(path)] = sha256_file(path)
        return json.loads(Path(path).read_text())

    plan = read('docs/lite3_retention_coverage_plan_20261008.json')
    control = read(plan['control_result'])
    assert sha256_file(plan['control_result']) == plan['control_result_sha256']
    candidate = read('docs/lite3_retention_coverage_pilot_result_20261008.json')
    training = read('docs/lite3_retention_coverage_training_audit_20261008.json')
    dataset = read('docs/lite3_retention_coverage_dataset_audit_20261008.json')
    for result in (control, candidate, training, dataset):
        assert result['status'] == 'verified'
        for path, digest in result['evidence'].items():
            assert sha256_file(path) == digest
    original_run = read(Path(plan['reused_baseline_run']) / 'report.json')
    previous_run = read('artifacts/prefix_imitation_20261008/learner2247-preserve10/pilot-evaluation/report.json')
    current_run = read(Path(plan['pilot_run']) / 'report.json')
    runs = [('Original PPO2063', original_run, 'base'), ('Preservation10', previous_run, 'trial'),
            ('Added retention coverage', current_run, 'trial')]
    diagnostics = {}
    for name, run, model in runs:
        diagnostics[name] = {}
        for condition in ('course', 'recovery'):
            phase = run['phases'][condition + '-' + model]
            assert phase['initial_sha256'] == original_run['phases'][condition + '-base']['initial_sha256']
            episodes = [e for e in phase['episodes'] if
                        (e['terrain'] != 'flat' if condition == 'course' else e['case'] == 26)]
            diagnostics[name][condition] = dict(
                initial_sha256=phase['initial_sha256'],
                by_terrain={terrain: dict(
                    success=sum(e['success'] for e in episodes if e['terrain'] == terrain),
                    rms_m=statistics.mean(e['body_lateral_rms_m'] for e in episodes if e['terrain'] == terrain),
                    peak_m=max(e['max_body_lateral_m'] for e in episodes if e['terrain'] == terrain))
                    for terrain in sorted({e['terrain'] for e in episodes})},
                worst_episode=max(episodes, key=lambda e: e['max_body_lateral_m']),
                failures=[e for e in episodes if not e['success']])
    stage = dataset['collection_controls'] + candidate['controls']
    assert dataset['collection_controls'] <= plan['max_collection_controls']
    assert candidate['controls'] <= plan['max_pilot_controls']
    costs = dict(collection=dataset['collection_controls'], pilot=candidate['controls'],
                 supervised_physics=0, stage_through_pilot=stage, new_uncertain=0,
                 cumulative_charged_through_pilot=plan['cumulative_before']['charged'] + stage,
                 cumulative_confirmed_through_pilot=plan['cumulative_before']['confirmed'] + stage,
                 historical_uncertain=plan['cumulative_before']['uncertain'])
    new, old = candidate['results']['trial'], control['results']['trial']
    delta = {key: new[key] - old[key] for key in
             ('course_total', 'course_rms_m', 'course_peak_m', 'case26_recovery', 'case26_rms_m', 'case26_peak_m')}
    summary = dict(status='verified', diagnostics=diagnostics, costs=costs, paired_delta_vs_preserve10=delta,
        gates=candidate['gates'], pilot_passed=candidate['pilot_passed'],
        final_checkpoint=training['final_checkpoint'], offline_validation_by_origin=training['offline_validation_by_origin'],
        evidence=evidence, new_physics=0,
        scope='Fixed data coverage ablation at learner2247/eval2251. Same TRAIN case26, no new original7-source validation in this summary or independent training replication.')
    write_json_atomic('docs/lite3_retention_coverage_comparison_20261008.json', summary)
    values = [candidate['results']['base'], old, new]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    labels = ['Original', 'Weight 10', '+ Coverage']
    colors = ['#65758B', '#2A8C82', '#B27C48']
    specs = [('course_total', 1, 'Ordinary full-course completion (of 64)', None),
             ('course_rms_m', 100, 'Ordinary rough mean episode RMS (cm)', values[0]['course_rms_m'] * 105),
             ('course_peak_m', 100, 'Ordinary rough peak lateral error (cm)', values[0]['course_peak_m'] * 100),
             ('case26_peak_m', 100, 'TRAIN case26 recovery peak lateral error (cm)', 70)]
    for ax, (key, scale, title, gate) in zip(axes.flat, specs):
        plotted = [v[key] * scale for v in values]
        if key == 'case26_peak_m':
            plotted[0] = float('nan')
            ax.text(0, 5, '0/24\nearly failures', ha='center')
        ax.bar(labels, plotted, color=colors, width=.6)
        for i, value in enumerate(plotted):
            if value == value:
                ax.text(i, value, f'{value:.2f}' if scale == 100 else str(int(value)), ha='center', va='bottom')
        if gate is not None:
            ax.axhline(gate, color='#B44747', ls='--', label=f'Gate {gate:.2f}')
            ax.legend(frameon=False)
        ax.set_ylim(0, max([v for v in plotted if v == v] + [gate or 0]) * 1.25)
        ax.set_title(title)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"New ordinary-state coverage: pilot {'PASSED' if candidate['pilot_passed'] else 'FAILED'}\nAutonomous case26 recovery: previous {old['case26_recovery']}/24, new {new['case26_recovery']}/24; same seed 2251", fontsize=14)
    target = Path('docs/figures/lite3-retention-coverage-20261008')
    for suffix in ('.png', '.svg'):
        fig.savefig(target.with_suffix(suffix), dpi=160)
    svg = target.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
