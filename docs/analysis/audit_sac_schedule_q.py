"""Read-only, hash-locked final-snapshot SAC audit (no simulator or optimizer)."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.soft_value_diagnostics import fixed_soft_rows, summarize_soft_rows
from rl_training.online_adaptation.types import TransitionBatch


def checked(record):
    path = Path(record['path'])
    if sha256_file(path) != record['sha256']:
        raise ValueError(f'input hash changed: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    if p['schema'] != 'lite3_schedule_q_audit_v1':
        raise ValueError('unsupported protocol')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started, cpu_started = time.perf_counter(), time.process_time()
    report = {'status': 'running', 'protocol_sha256': sha256_file(args.protocol),
              'script_sha256': sha256_file(__file__), 'new_physics_steps': 0,
              'new_optimizer_updates': 0, 'snapshots': {}, 'limits': p['limits']}
    write_json_atomic(out/'protocol.json', p)
    write_json_atomic(out/'report.json', report)
    states = {}
    for label, rec in p['snapshots'].items():
        ledger = json.loads(checked(rec['ledger']).read_text())
        if ledger['latest_checkpoint']['sha256'] != rec['checkpoint']['sha256']:
            raise ValueError('not final committed checkpoint')
        states[label] = torch.load(checked(rec['checkpoint']), map_location='cpu', weights_only=False)['experiment_state']
        r = states[label]['agent']['replay']
        if not (r['size'] == 5000 and r['capacity'] == 5000 and r['cursor'] == 0
                and ledger['spent']['adaptation'] == 5000):
            raise ValueError('requires exactly first fill, no overwritten rows')
    prefix = p['common_prefix_rows']
    for terrain in (101, 107):
        a, b = (states[f'{arm}-{terrain}']['agent']['replay'] for arm in ('reference', 'actor025'))
        for f in fields(TransitionBatch):
            if not torch.equal(a[f.name][:prefix], b[f.name][:prefix]):
                raise ValueError(f'common prefix differs: {terrain}/{f.name}')
    report['common_prefix_all_fields_equal'] = True
    for label, state in states.items():
        rec = p['snapshots'][label]
        u, r = state['agent']['updater'], state['agent']['replay']
        if (state['training_contract']['entropy_coordinates'] != 'normalized_tanh_action_v1'
                or u['critic_action_semantics'] != 'requested_residual_before_safety_v1'
                or state['agent']['config'].n_step != 1):
            raise ValueError('incompatible SAC contract')
        batch = TransitionBatch(**{f.name: r[f.name][:r['size']].float() if r[f.name].is_floating_point()
                                  else r[f.name][:r['size']] for f in fields(TransitionBatch)})
        torch.testing.assert_close(batch.bootstrap_mask, 1-batch.terminated, rtol=0, atol=0)
        actor = ResidualActor(r['frame_dim'], r['action_dim'], action_limits=u['actor']['action_limits'])
        critic, target = (TwinCritic(r['frame_dim'], r['action_dim']) for _ in range(2))
        for model, weights in ((actor, u['actor']), (critic, u['critic']), (target, u['target_critic'])):
            model.load_state_dict(weights, strict=True)
            model.eval().requires_grad_(False)
        idx = torch.arange(r['size'])
        terminal = batch.terminated.reshape(-1).bool()
        masks = {'all': idx >= 0, 'common_prefix': idx < prefix, 'own_postwarmup': idx >= prefix,
                 'nonzero_request': batch.requested_residual_action.ne(0).any(-1), 'terminal': terminal,
                 'success_terminal': terminal & batch.reward.reshape(-1).eq(305),
                 'failure_terminal': terminal & batch.reward.reshape(-1).eq(-8)}
        stable = state['stable_manager']['stable_state']
        item = {'checkpoint_sha256': rec['checkpoint']['sha256'], 'final_version': u['version'],
                'stable_version': None if stable is None else stable['version'],
                'alpha': float(u['log_alpha'].exp()), 'operators': {}, 'rollbacks': []}
        for log in rec['logs']:
            records = [json.loads(line) for line in checked(log).read_text().splitlines()]
            for i, row in enumerate(records):
                if row.get('rollback_reason'):
                    item['rollbacks'].append({'log': log['path'], 'step': row['step'],
                        'reason': row['rollback_reason'], 'logged_control_version': row['actor_version'],
                        'accepted_actor_updates': row['actor_updates'],
                        'same_step_update_version': (row.get('last_update') or {}).get('version'),
                        'next_control_version': records[i+1]['actor_version'] if i+1 < len(records) else None})
            if records:
                item['cumulative_accepted_actor_updates'] = records[-1]['actor_updates']
        for operator in p['operators']:
            rows = fixed_soft_rows(actor, critic, target if operator == 'target_critic' else critic, batch,
                                   alpha=item['alpha'], residual_gain=p['residual_gain'], draws=p['draws'],
                                   batch_size=p['batch_size'], seed=p['seed'])
            path = out/f'{label}-{operator}.npz'
            np.savez_compressed(path, **{k: v.numpy() for k, v in rows.items()},
                                **{f'mask_{k}': v.numpy() for k, v in masks.items()})
            item['operators'][operator] = {'rows_sha256': sha256_file(path),
                'strata': {key: summarize_soft_rows(rows, mask) for key, mask in masks.items()}}
        report['snapshots'][label] = item
        write_json_atomic(out/'report.json', report)
        print(label, 'complete', flush=True)
    for rec in p['snapshots'].values():
        for record in [rec['checkpoint'], rec['ledger'], *rec['logs']]:
            checked(record)
    report.update(status='complete', wall_seconds=time.perf_counter()-started,
                  cpu_seconds=time.process_time()-cpu_started, threads=1)
    write_json_atomic(out/'report.json', report)


if __name__ == '__main__':
    main()
