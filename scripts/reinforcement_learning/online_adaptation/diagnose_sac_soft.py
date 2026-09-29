"""Audit preregistered frozen SAC snapshots on identical occupied replay rows."""

import argparse
from dataclasses import fields
from pathlib import Path
import time

import numpy as np
import torch

from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.soft_value_diagnostics import fixed_soft_rows, summarize_soft_rows
from rl_training.online_adaptation.types import TransitionBatch
from scripts.reinforcement_learning.online_adaptation.confirm_value_history import checked_path, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    p = read_json(args.protocol)
    if p['schema'] != 'lite3_sac_fixed_soft_diagnostic_v1':
        raise ValueError('unsupported soft diagnostic protocol')
    operator_target = p.get('operator_target', 'target_critic')
    if operator_target not in ('target_critic', 'online_critic'):
        raise ValueError('unsupported bootstrap operator')
    if 'parent_report' in p:
        checked_path(p['parent_report'])
    manifest = read_json(checked_path(p['session_manifest']))
    root = Path(p['session_manifest']['path']).parent.parent
    ledger = read_json(root / 'session_summary.json')
    snapshots = {}
    for label, record in p['snapshots'].items():
        if ledger['trials'][record['trial_id']]['status'] != record['ledger_status']:
            raise ValueError('original ledger status changed')
        snapshots[label] = torch.load(checked_path(record), map_location='cpu', weights_only=False)['experiment_state']
    before, after = snapshots['before']['agent'], snapshots['after']['agent']
    for key in ('actor',):
        for name, value in before['updater'][key].items():
            if not torch.equal(value, after['updater'][key][name]):
                raise ValueError('actor changed between snapshots')
    if not torch.equal(before['updater']['log_alpha'], after['updater']['log_alpha']):
        raise ValueError('temperature changed between snapshots')
    r = before['replay']
    n = r['size']
    if r['cursor'] != n or after['replay']['cursor'] != after['replay']['size']:
        raise ValueError('diagnostic requires an unwrapped identical replay prefix')
    batch = TransitionBatch(**{f.name: r[f.name][:n].clone() for f in fields(TransitionBatch)})
    for field in fields(batch):
        if not torch.equal(getattr(batch, field.name), after['replay'][field.name][:n]):
            raise ValueError('replay prefix differs between snapshots')
    # Match ReplayBuffer.sample: stored half-precision observations enter networks as float32.
    batch = TransitionBatch(**{f.name: (getattr(batch, f.name).float() if getattr(batch, f.name).is_floating_point()
                                       else getattr(batch, f.name)) for f in fields(batch)})
    torch.testing.assert_close(batch.discount, torch.full_like(batch.discount, p['gamma']), rtol=0, atol=1e-7)
    torch.testing.assert_close(batch.bootstrap_mask, 1-batch.terminated, rtol=0, atol=0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': p['schema'], 'status': 'running', 'protocol_sha256': sha256_file(args.protocol),
              'original_source_sha256': manifest['worker_configuration']['source_sha256'],
              'new_control_transitions': 0, 'new_optimizer_updates': 0,
              'samples': n, 'residual_gain': p['residual_gain'], 'operator_target': operator_target, 'snapshots': {},
              'coverage': {'terminated': int(batch.terminated.sum()), 'truncated': int(batch.truncated.sum()),
                           'zero_request_rows': int((batch.requested_residual_action == 0).all(dim=1).sum())},
              'limits': 'shared training replay and bootstrapped targets, not heldout soft-Q calibration'}
    write_json_atomic(output/'protocol.json', p)
    write_json_atomic(output/'report.json', report)
    torch.set_num_threads(1)
    for label, snapshot in snapshots.items():
        agent = snapshot['agent']
        u = agent['updater']
        contract = snapshot['training_contract']
        if (contract != snapshots['before']['training_contract']
                or contract['entropy_coordinates'] != p['entropy_coordinates']
                or agent['config'].n_step != 1 or u['version'] != 0
                or u['actor_optimizer']['state'] or u['alpha_optimizer']['state']
                or u['critic_action_semantics'] != 'requested_residual_before_safety_v1'):
            raise ValueError('incompatible fixed-policy checkpoint')
        actor = ResidualActor(r['frame_dim'], r['action_dim'], action_limits=u['actor']['action_limits'])
        critic, target = TwinCritic(r['frame_dim'], r['action_dim']), TwinCritic(r['frame_dim'], r['action_dim'])
        for model, state in ((actor, u['actor']), (critic, u['critic']), (target, u['target_critic'])):
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
        alpha = float(u['log_alpha'].exp())
        started = time.perf_counter()
        rows = fixed_soft_rows(actor, critic, target if operator_target == 'target_critic' else critic,
                              batch, alpha=alpha, residual_gain=p['residual_gain'],
                              draws=p['draws'], batch_size=p['batch_size'], seed=p['seed'])
        seconds = time.perf_counter()-started
        np.savez_compressed(output/f'{label}_rows.npz', **{k:v.numpy() for k,v in rows.items()})
        metrics = {'all': summarize_soft_rows(rows),
                   'terminal': summarize_soft_rows(rows, batch.terminated.bool()),
                   'truncated': summarize_soft_rows(rows, batch.truncated.bool())}
        report['snapshots'][label] = {'metrics': metrics, 'seconds': seconds, 'alpha': alpha,
            'saved_transient_gain': agent['residual_gain'], 'warmup': snapshot['target_warmup'],
            'critic_optimizer_steps': sorted({int(v['step']) for v in u['critic_optimizer']['state'].values()}),
            'checkpoint_sha256': p['snapshots'][label]['sha256'], 'rows_sha256': sha256_file(output/f'{label}_rows.npz')}
        write_json_atomic(output/'report.json', report)
        print(label, metrics, flush=True)
    for record in p['snapshots'].values():
        checked_path(record)
    report['status'] = 'complete'
    write_json_atomic(output/'report.json', report)


if __name__ == '__main__':
    main()
