"""Append fresh learner-state ordinary labels without consuming pilot trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_training.lite3_joint_conventions import LITE3_SAFE_ACTION_LOWER, LITE3_SAFE_ACTION_UPPER
from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(plan['collection_run'])
    report = json.loads((root / 'report.json').read_text())
    protocol = json.loads(Path(plan['collection_protocol']).read_text())
    audit = json.loads(Path(plan['collection_audit']).read_text())
    assert report['status'] == 'complete' and audit['status'] == 'verified'
    assert report['protocol_sha256'] == plan['collection_protocol_sha256'] == sha256_file(plan['collection_protocol'])
    assert protocol['purpose'] == 'fresh_training_retention_coverage' and protocol['conditions'] == ['course']
    assert audit['evidence'][str(root / 'report.json')] == sha256_file(root / 'report.json')
    assert sha256_file(plan['parent_dataset']) == plan['parent_dataset_sha256']
    assert sha256_file(plan['parent_dataset_audit']) == plan['parent_dataset_audit_sha256']
    assert sha256_file(plan['base_checkpoint']) == plan['base_sha256']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    actor = FrozenPPOExpert(torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=False)['actor_state_dict'])
    with np.load(plan['parent_dataset'], allow_pickle=False) as old:
        arrays = {key: value.copy() for key, value in old.items()}
    for group in ('correction', 'post_recovery'):
        for split in ('train', 'validation'):
            y = arrays[f'{group}_{split}_obs'][:, 9] * plan['route_lateral_scale_m']
            assert (np.abs(y) > plan['ordinary_max_abs_y_m']).all()
    folder = root / 'course-trial'
    with np.load(folder / 'initial.npz', allow_pickle=False) as initial:
        terrain = initial['terrain'].copy()
        assert (initial['case'] == -1).all()
    phase = report['phases']['course-trial']
    assert [e['env_id'] for e in phase['episodes']] == list(range(protocol['num_envs']))
    evidence = {str(path): sha256_file(path) for path in (
        Path(args.plan), root / 'report.json', Path(plan['collection_audit']),
        Path(plan['parent_dataset']), Path(plan['parent_dataset_audit']))}
    new = {}
    counts = dict(active_rows=0, ordinary_domain_rows=0, excluded_rows=0, eligible_rows_from_failed_episodes=0)
    failures = np.array([not e['success'] for e in phase['episodes']])
    for file in sorted((folder / 'trajectory').glob('chunk-*.npz')):
        evidence[str(file)] = sha256_file(file)
        assert audit['phases']['course-trial']['trace_sha256'][file.name] == evidence[str(file)]
        with np.load(file, allow_pickle=False) as trace:
            obs = trace['policy_observation']
            assert obs.shape[-1] == 48 and np.isfinite(obs).all()
            valid = trace['active'] & (np.abs(obs[..., 9] * plan['route_lateral_scale_m']) <= plan['ordinary_max_abs_y_m'])
            counts['active_rows'] += int(trace['active'].sum())
            counts['ordinary_domain_rows'] += int(valid.sum())
            counts['eligible_rows_from_failed_episodes'] += int((valid & failures[None, :]).sum())
            for group, mask in (('flat', terrain < 2), ('rough101', (terrain >= 2) & (terrain < 5)), ('rough107', terrain >= 5)):
                for split in ('train', 'validation'):
                    selected = valid & mask[None, :] & ((np.arange(len(terrain))[None, :] % 4 == 0) == (split == 'validation'))
                    new.setdefault(f'{group}_{split}_obs', []).append(obs[selected].copy())
    counts['excluded_rows'] = counts['active_rows'] - counts['ordinary_domain_rows']
    new = {key: np.concatenate(value) for key, value in new.items()}
    low = np.array(LITE3_SAFE_ACTION_LOWER, np.float32)
    high = np.array(LITE3_SAFE_ACTION_UPPER, np.float32)
    with torch.no_grad():
        for key, obs in list(new.items()):
            assert len(obs) > 0
            predictions = [actor(torch.from_numpy(obs[i:i + 4096])).numpy() for i in range(0, len(obs), 4096)]
            new[key.removesuffix('_obs') + '_target'] = np.clip(np.concatenate(predictions), low, high)
    for key, value in new.items():
        arrays[key] = np.concatenate((arrays[key], value))
    np.savez_compressed(output / 'dataset.npz', **arrays)
    write_json_atomic(output / 'report.json', dict(
        status='complete', protocol_sha256=sha256_file(args.plan), dataset_sha256=sha256_file(output / 'dataset.npz'),
        parent_dataset_sha256=plan['parent_dataset_sha256'], label_actor_sha256=plan['base_sha256'],
        new_counts={key: len(value) for key, value in new.items() if key.endswith('_obs')},
        counts={key: len(value) for key, value in arrays.items() if key.endswith('_obs')},
        selection_counts=counts, collection_controls=report['ledger']['confirmed'], new_physics=0, evidence=evidence,
        scope='Fresh ordinary learner states only; original expert mean labels are behavioral references, not verified corrective successes. Recovery labels unchanged; no pilot or original held-out source labels.'))


if __name__ == '__main__':
    main()
