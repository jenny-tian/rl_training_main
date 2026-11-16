"""Locked, bounded current-model target/headroom protocol. No optimizer."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from reference_stairs_eval import ROOT, save, sha

OUTPUT = ROOT / 'logs/reference_six/online_targets_20261116'
DOCS = ROOT / 'docs/lite3-online-targets-20261116'
MODEL_PATHS = {
    'zero10100': ROOT / 'logs/reference_six/low_stairs_zero_preservation_20261115/train_001_100/model_updates_10100.pt',
    'B': ROOT / 'logs/reference_six/stair_allocation_20261116/B/train_001_100/model_updates_10200.pt',
}
HASHES = {'zero10100': '290d94746c467b5cd21ef74c47a908011521c1f145da35ba7b94f1b92d9b7497',
          'B': 'aba3a52e362fbb6f6c860bb8db384c12a84d13f6f39b53cae16465eaf2a84b79'}
CONDITIONS = ('stairs_05', 'stairs_06', 'stairs_07', 'stairs_08', 'boxes', 'rocky')


def read(path):
    return json.loads(Path(path).read_text())


def provenance():
    from reference_low_stairs_evaluate import verify_training, TRAINING, EXPECTED_SHA
    from reference_low_stairs_zero_preservation_evaluate import checked_training
    from reference_stair_allocation import parent_state
    verify_training()
    assert checked_training() == (MODEL_PATHS['zero10100'], HASHES['zero10100'])
    parent_state()
    import torch
    scratch = read(TRAINING / 'report.json')
    terrain_audit = read(TRAINING / 'low_stairs_terrain_audit.json')
    assert scratch['checkpoint_loaded'] is None and scratch['from_scratch']
    assert terrain_audit['step_height_range_m'] == [.02, .05]
    stages = []
    for name, path in MODEL_PATHS.items():
        assert sha(path) == HASHES[name]
        report = read(path.parent / 'report.json')
        assert report['status'] == 'passed' and report['optimizer_updates'] == 100
        assert report['output_sha256'] == HASHES[name]
        assert report['parent_sha256'] == (EXPECTED_SHA if name == 'zero10100' else HASHES['zero10100'])
        cfg = read(path.parent / 'configuration.json')['environment']
        terrain = cfg['scene']['terrain']['terrain_generator']
        for key in ('stairs_up', 'stairs_down'):
            assert terrain['sub_terrains'][key]['step_height_range'] == [.02, .05]
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert state['actor_state_dict']['mlp.0.weight'].shape[1] == 45
        assert state['critic_state_dict']['mlp.0.weight'].shape[1] == 393
        stages.append(dict(model=name, checkpoint=str(path), sha256=sha(path),
            parent_sha256=report['parent_sha256'], extra_updates=100,
            terrain=terrain, configuration_sha256=sha(path.parent / 'configuration.json')))
    return dict(status='passed', scratch_sha256=EXPECTED_SHA, scratch_updates=10000,
        scratch_checkpoint_loaded=None, scratch_terrain_audit=terrain_audit, stages=stages,
        stair_range_m=[.02, .05], heldout_heights_m=[.06, .07, .08],
        slope_coefficient_range=[0., .4], slope_max_deg=21.80140948635181,
        rough_noise_range_m=[.01, .06], friction_range=[.5, 1.],
        caveat='Six families describe procedural generators, not disjoint height sets. Boxes/rocky use new geometry families; rocky amplitudes overlap trained roughness. Old high-stair checkpoints are not ancestors.')


def protocol():
    return dict(schema=1, approval='User approved NEXT_STAGE_PROPOSAL.md, 2026-11-16',
        models=HASHES, conditions=list(CONDITIONS), policy_dt=.02, physics_dt=.005,
        nominal=True, native_actuator_delay=[0, 1], command=[.4, 0., 0.],
        startup='50 zero-action preparation steps, then 0.5 s pause and 0.5 s linear ramp',
        episode_steps=1000, smoke_runs_max=4, smoke_steps_per_run_max=600,
        smoke_steps_include_preparation=True, smoke_models=['B', 'zero10100'],
        baseline_n=32, grid_n=8, confirmation_n=16,
        budgets=dict(baseline=384, grid=1200, confirmation=64, total=1648),
        seeds=dict(environment=20261116, reset_table=20261116, boxes=101, rocky=101),
        namespaces=dict(smoke=10, baseline=11, grid=12, confirmation=13),
        start_ranges=dict(x=[.20, .30], y=[-.10, .10], yaw=[-.03, .03]),
        initial_velocity=0., grid_values=[-2., -1., 0., 1., 2.], bases=['sag', 'roll', 'hipx'],
        residual_limits=[.24, .20, .20] * 4, rate_limit=.02,
        geometry=dict(tile_size=[8., 8.], stair_tread=.3, stair_platform=3., stair_border=1.,
            boxes=dict(flat_length=3., lane_width=2., grid_width=.4, grid_height_range=[.08, .08], seed=101),
            rocky=dict(flat_length=3., transition_length=.5, lane_width=2., rough_amplitude=.08,
                grid_spacing=.1, rock_count=24, rock_fraction=.58, radius_x=[.12, .32], radius_y=[.10, .28],
                height_range=[.045, .10], seed=101),
            difficulty_range=[.5, .5], boxes_goal_x=5.5, rocky_goal_x=5.,
            stairs_goal='last ray-audited positive riser + 0.30 m'),
        outcomes=dict(priority=['posture_failure', 'lateral_exit', 'success', 'stuck_or_timeout'],
            lateral_abs_y_max=1., success_hold_s=.26, support_feet_min=2, torso_force_max_N=1.,
            success_clearance_m=[.18, .55], tilt_max_deg=44.,
            stairs_last_foot_x_margin=.05, stairs_last_foot_z_margin=-.01,
            progress='maximum forward root displacement divided by fixed route length',
            sustained_progress_min=.5, stuck_last5s_displacement_m=.15),
        target_selection=dict(min_success=7, max_success=19, max_targets=2,
            priority='one ascending-height eligible stair, one boxes-then-rocky eligible novel family; then remaining eligible in condition order',
            exclude_zero_and_progress_below=.5),
        grid_rank=['success descending', 'posture failures ascending', 'mean progress descending', 'basis/code order'],
        confirmation=dict(min_extra_success=4, posture_failures_must_not_increase=True),
        stop_on=['interface failure', 'unclear provenance', 'no confirmed headroom'],
        training=False, optimizer_updates=0, media='local only')


def lock():
    target = OUTPUT / 'lock.json'
    if target.exists():
        existing = read(target)
        assert existing['protocol'] == protocol()
        for row in existing['sources']:
            assert sha(row['path']) == row['sha256'], row['path']
        return existing
    audit = provenance()
    paths = [Path(__file__), Path(__file__).with_name('reference_online_interface.py'),
             Path(__file__).with_name('reference_online_collect.py')]
    package = ROOT / 'source/rl_training/rl_training'
    paths += [package / rel for rel in ('ea_lpl/basis.py', 'ea_lpl/residual_bridge.py', 'ea_lpl/reset_table.py',
        'online_adaptation/safety.py', 'online_adaptation/types.py', 'source_scan/bridge.py',
        'tasks/manager_based/locomotion/velocity/terrains/heldout_boxes.py',
        'tasks/manager_based/locomotion/velocity/terrains/rocky_road_track.py')]
    value = dict(protocol=protocol(), provenance=audit,
                 sources=[dict(path=str(p), sha256=sha(p)) for p in paths])
    save(target, value)
    save(DOCS / 'LOCK.json', value)
    return value


def rows(kind, condition=None, winner=None):
    p = protocol()
    count = {'smoke': 2, 'baseline': 32, 'grid': 8, 'confirmation': 16}[kind]
    codes = [dict(basis='sag', u=[0., 0.], code='zero')]
    if kind == 'grid':
        codes = [dict(basis=b, u=list(u), code=f'{b}:{u[0]:g},{u[1]:g}')
            for b in p['bases'] for u in itertools.product(p['grid_values'], repeat=2)]
    if kind == 'confirmation':
        if winner is None or winner['u'] == [0., 0.]:
            raise ValueError('confirmation requires a nonzero winner')
        codes += [winner]
    return [dict(env_id=i, **r) for i, r in enumerate(
        dict(condition=c, slot=s, namespace=p['namespaces'][kind], **code)
        for c in ([condition] if condition else CONDITIONS) for code in codes for s in range(count))]


def choose_targets(reports):
    table = []
    for c in CONDITIONS:
        group = [r for r in reports['B']['records'] if r['condition'] == c]
        assert len(group) == 32
        success = sum(r['outcome'] == 'success' for r in group)
        progress = sum(r['progress_fraction'] for r in group) / 32
        table.append(dict(condition=c, completed=success, mean_progress=progress,
            eligible=c != 'stairs_05' and 7 <= success <= 19 and progress >= .5))
    eligible = [r['condition'] for r in table if r['eligible']]
    stairs = [c for c in eligible if c.startswith('stairs')]
    novel = [c for c in eligible if not c.startswith('stairs')]
    chosen = stairs[:1] + novel[:1]
    chosen += [c for c in eligible if c not in chosen][:2-len(chosen)]
    return dict(table=table, selected=chosen)


if __name__ == '__main__':
    lock()
