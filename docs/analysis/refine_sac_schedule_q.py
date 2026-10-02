"""Decompose saved Bellman errors without rerunning networks or simulation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def classify_events(events):
    """Only explicit performance degradation identifies a rollback in this audit."""
    return {key: [e for e in events if predicate(e['reason'])] for key, predicate in {
        'performance_rollbacks': lambda reason: reason == 'performance_degradation',
        'fall_resets': lambda reason: reason == 'fall_reset',
        'other_events': lambda reason: reason not in ('performance_degradation', 'fall_reset'),
    }.items()}


def decompose(q1, q2, target, selected, terminal):
    arrays = [np.asarray(x).reshape(-1) for x in (q1, q2, target, selected, terminal)]
    q1, q2, target, selected, terminal = arrays
    if len({len(x) for x in arrays}) != 1 or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('finite arrays of equal length required')
    if not all(np.isin(x, [0, 1]).all() for x in (selected, terminal)):
        raise ValueError('binary masks required')
    selected, terminal = selected.astype(bool), terminal.astype(bool)
    error2 = (np.minimum(q1.astype(np.float64), q2.astype(np.float64))-target.astype(np.float64))**2
    n, t = selected & ~terminal, selected & terminal
    total = float(error2[selected].sum())
    return {'samples': int(selected.sum()), 'terminal_samples': int(t.sum()),
            'nonterminal_samples': int(n.sum()), 'squared_error_sum': total,
            'terminal_squared_error_sum': float(error2[t].sum()),
            'terminal_squared_error_fraction': float(error2[t].sum()/total) if total else None,
            'nonterminal_rmse': float(np.sqrt(error2[n].mean())) if n.any() else None,
            'all_rmse': float(np.sqrt(error2[selected].mean())) if selected.any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    start, cpu = time.perf_counter(), time.process_time()
    p = json.loads(Path(args.protocol).read_text())
    if p['schema'] != 'schedule_q_refinement_v1':
        raise ValueError('unsupported protocol')
    for rec in p['inputs']:
        if sha256_file(rec['path']) != rec['sha256']:
            raise ValueError('input hash mismatch')
    source = Path(p['inputs'][0]['path'])
    old = json.loads(source.read_text())
    if old['status'] != 'complete':
        raise ValueError('requires completed parent audit')
    allowed = {rec['path']: rec['sha256'] for rec in p['inputs']}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    result = {'schema': p['schema'], 'protocol_sha256': sha256_file(args.protocol),
              'script_sha256': sha256_file(__file__), 'snapshots': {}, 'limits': p['limits'],
              'new_physics_steps': 0, 'new_optimizer_updates': 0, 'new_network_forwards': 0}
    for label, snapshot in old['snapshots'].items():
        events = snapshot.get('control_events', snapshot.get('rollbacks', []))
        item = {'events': classify_events(events), 'operators': {}}
        for operator, metrics in snapshot['operators'].items():
            path = source.parent/f'{label}-{operator}.npz'
            if allowed[str(path)] != metrics['rows_sha256']:
                raise ValueError('row file differs from parent report')
            with np.load(path, allow_pickle=False) as rows:
                item['operators'][operator] = {name: decompose(rows['q1'], rows['q2'], rows['target_mean'],
                    rows[f'mask_{name}'], rows['mask_terminal']) for name in ('all', 'common_prefix', 'own_postwarmup')}
        result['snapshots'][label] = item
    for rec in p['inputs']:
        if sha256_file(rec['path']) != rec['sha256']:
            raise ValueError('input changed during audit')
    result.update(status='complete', wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu)
    write_json_atomic(out/'report.json', result)
    write_json_atomic(out/'protocol.json', p)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
