"""Offline summaries and pre-registered selections; never launches simulation."""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from reference_online_current import OUTPUT, DOCS, completed, registration
from reference_online_targets import CONDITIONS, choose_targets, protocol, read
from reference_stairs_eval import save, sha


def summarize(records):
    n = len(records)
    counts = Counter(r['outcome'] for r in records)
    active = sum(r['safety_counts']['active_steps'] for r in records)
    return dict(n=n, **{k: counts[k] for k in ('success', 'posture_failure', 'lateral_exit', 'stuck', 'timeout')},
        mean_progress=float(np.mean([r['progress_fraction'] for r in records])),
        active_steps=active,
        safety_fractions={k: sum(r['safety_counts'][k] for r in records) / active
                          for k in ('projected', 'paused', 'base_projection', 'zero_changed')},
        mean_learned_effect_norm=sum(r['mean_learned_effect_norm'] * r['safety_counts']['active_steps']
                                    for r in records) / active)


def initial_pairing(first, second, first_ids=None, second_ids=None):
    """Compare full prepared states, not merely the shared reset RNG keys."""
    with np.load(first / 'initial.npz') as a, np.load(second / 'initial.npz') as b:
        ai = np.arange(len(a['keys'])) if first_ids is None else np.asarray(first_ids)
        bi = np.arange(len(b['keys'])) if second_ids is None else np.asarray(second_ids)
        errors = {}
        for key in ('root', 'joint_pos', 'joint_vel', 'observation'):
            x, y = a[key][ai].copy(), b[key][bi].copy()
            if key == 'root':
                x[:, :3] -= a['origins'][ai]
                y[:, :3] -= b['origins'][bi]
            errors[key] = float(np.max(np.abs(x - y)))
        same_keys = bool(np.array_equal(a['keys'][ai], b['keys'][bi]))
        same_delays = bool(np.array_equal(a['delays'][:, ai], b['delays'][:, bi]))
    return dict(pairs=len(ai), same_keys=same_keys, same_delays=same_delays,
                max_abs_errors=errors, exact=same_keys and same_delays and max(errors.values()) == 0.)


def baseline():
    reports = {m: completed(OUTPUT / f'baseline_{m}') for m in ('B', 'zero10100')}
    pairing = initial_pairing(OUTPUT / 'baseline_B', OUTPUT / 'baseline_zero10100')
    if not pairing['exact']:
        raise ValueError(f'baseline prepared states differ: {pairing}')
    assert read(OUTPUT / 'baseline_B/geometry.json') == read(OUTPUT / 'baseline_zero10100/geometry.json')
    selection = choose_targets(reports)
    selection['paired_initial_states'] = pairing
    selection['summary'] = {m: {c: summarize([r for r in report['records'] if r['condition'] == c])
                               for c in CONDITIONS} for m, report in reports.items()}
    selection['report_hashes'] = {m: sha(OUTPUT / f'baseline_{m}/report.json') for m in reports}
    save(OUTPUT / 'selection.json', selection)
    save(DOCS / 'SELECTION.json', selection)
    print(selection)


def grid(condition):
    directory = OUTPUT / f'grid_B_{condition}'
    report = completed(directory)
    groups = {}
    for record in report['records']:
        groups.setdefault(record['code'], []).append(record)
    table = [dict(code=code, basis=rs[0]['basis'], u=rs[0]['u'], order=i, **summarize(rs))
             for i, (code, rs) in enumerate(groups.items())]
    assert len(table) == 75 and all(r['n'] == 8 for r in table)
    reference_ids = [r['env_id'] for r in next(iter(groups.values()))]
    pairing = initial_pairing(directory, directory, reference_ids * 75,
                              [r['env_id'] for rs in groups.values() for r in rs])
    if not pairing['exact']:
        raise ValueError(f'grid prepared states differ: {pairing}')
    ranked = sorted(table, key=lambda r: (-r['success'], r['posture_failure'], -r['mean_progress'], r['order']))
    best = ranked[0]
    value = dict(condition=condition, ranked=ranked, winner=best, nonzero_winner=best['u'] != [0., 0.],
                 paired_initial_states=pairing,
                 report_sha256=sha(OUTPUT / f'grid_B_{condition}/report.json'))
    save(DOCS / f'GRID_{condition}.json', value)
    if value['nonzero_winner']:
        save(OUTPUT / f'winner_{condition}.json', {k: best[k] for k in ('code', 'basis', 'u')})
    print(value['winner'])


def confirmation(condition):
    directory = OUTPUT / f'confirmation_B_{condition}'
    report = completed(directory)
    zero = [r for r in report['records'] if r['code'] == 'zero']
    fixed = [r for r in report['records'] if r['code'] != 'zero']
    assert len(zero) == len(fixed) == 16
    a, b = summarize(zero), summarize(fixed)
    pairing = initial_pairing(directory, directory, [r['env_id'] for r in zero], [r['env_id'] for r in fixed])
    if not pairing['exact']:
        raise ValueError(f'confirmation prepared states differ: {pairing}')
    assert [r['slot'] for r in zero] == [r['slot'] for r in fixed]
    value = dict(condition=condition, zero=a, fixed=b, paired_initial_states=pairing,
        extra_success=b['success'] - a['success'],
        confirmed=b['success'] - a['success'] >= protocol()['confirmation']['min_extra_success']
                  and b['posture_failure'] <= a['posture_failure'],
        paired_outcomes=dict(both_success=sum(x['outcome'] == y['outcome'] == 'success' for x, y in zip(zero, fixed)),
            only_zero=sum(x['outcome'] == 'success' and y['outcome'] != 'success' for x, y in zip(zero, fixed)),
            only_fixed=sum(x['outcome'] != 'success' and y['outcome'] == 'success' for x, y in zip(zero, fixed))),
        report_sha256=sha(directory / 'report.json'))
    save(DOCS / f'CONFIRMATION_{condition}.json', value)
    print(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('baseline', 'grid', 'confirmation'))
    parser.add_argument('--condition', choices=CONDITIONS)
    args = parser.parse_args()
    registration()
    if args.stage == 'baseline':
        baseline()
    else:
        if args.condition is None:
            parser.error('condition required')
        globals()[args.stage](args.condition)
