"""Independently reconstruct added stream labels and verify the fixed actor fit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from rl_training.lite3_joint_conventions import LITE3_SAFE_ACTION_LOWER, LITE3_SAFE_ACTION_UPPER
from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--training')
    parser.add_argument('--training-protocol')
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = json.loads(Path(args.plan).read_text())
    folder = Path(args.dataset)
    report = json.loads((folder / 'report.json').read_text())
    assert report['status'] == 'complete' and report['protocol_sha256'] == sha256_file(args.plan)
    assert report['dataset_sha256'] == sha256_file(folder / 'dataset.npz')
    for path, digest in report['evidence'].items():
        assert sha256_file(path) == digest
    assert sha256_file(plan['parent_dataset']) == plan['parent_dataset_sha256']
    assert sha256_file(plan['base_checkpoint']) == report['label_actor_sha256'] == plan['base_sha256']
    root = Path(plan['collection_run'])
    rollout = json.loads((root / 'report.json').read_text())
    protocol = json.loads(Path(plan['collection_protocol']).read_text())
    audit = json.loads(Path(plan['collection_audit']).read_text())
    assert rollout['status'] == 'complete' and audit['status'] == 'verified'
    assert rollout['protocol_sha256'] == plan['collection_protocol_sha256'] == sha256_file(plan['collection_protocol'])
    assert audit['evidence'][str(root / 'report.json')] == sha256_file(root / 'report.json')
    assert protocol['purpose'] == 'fresh_training_retention_coverage' and protocol['conditions'] == ['course']
    assert 'action_prefix' not in protocol
    assert rollout['ledger']['confirmed'] == rollout['ledger']['charged'] == report['collection_controls'] <= plan['max_collection_controls']
    assert rollout['ledger']['uncertain_transitions'] == 0
    actor = FrozenPPOExpert(torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=False)['actor_state_dict'])
    old = np.load(plan['parent_dataset'], allow_pickle=False)
    dataset = np.load(folder / 'dataset.npz', allow_pickle=False)
    assert set(old.files) == set(dataset.files)
    phase = rollout['phases']['course-trial']
    states, active = [], []
    for file in sorted((root / 'course-trial/trajectory').glob('chunk-*.npz')):
        assert sha256_file(file) == audit['phases']['course-trial']['trace_sha256'][file.name]
        with np.load(file, allow_pickle=False) as trace:
            states.append(trace['policy_observation'])
            active.append(trace['active'])
    states, active = np.concatenate(states), np.concatenate(active)
    eligible = active & (abs(states[:, :, 9]) * plan['route_lateral_scale_m'] <= plan['ordinary_max_abs_y_m'])
    failures = np.array([not e['success'] for e in phase['episodes']])
    assert report['selection_counts'] == dict(active_rows=int(active.sum()), ordinary_domain_rows=int(eligible.sum()),
        excluded_rows=int(active.sum() - eligible.sum()), eligible_rows_from_failed_episodes=int((eligible & failures[None, :]).sum()))
    replayed = 0
    for group in plan['batch_groups']:
        for split in ('train', 'validation'):
            key = f'{group}_{split}'
            count = len(old[key + '_obs'])
            assert dataset[key + '_obs'].shape == (report['counts'][key + '_obs'], 48)
            assert dataset[key + '_target'].shape == (len(dataset[key + '_obs']), 12)
            for suffix in ('obs', 'target'):
                np.testing.assert_array_equal(dataset[key + '_' + suffix][:count], old[key + '_' + suffix])
            if group in ('correction', 'post_recovery'):
                assert len(dataset[key + '_obs']) == count
                assert (abs(dataset[key + '_obs'][:, 9]) * plan['route_lateral_scale_m'] > plan['ordinary_max_abs_y_m']).all()
                continue
            ids = np.array([e['env_id'] for e in phase['episodes'] if
                e['terrain'] == {'flat': 'flat', 'rough101': '101', 'rough107': '107'}[group]
                and ((e['env_id'] % 4 == 0) == (split == 'validation'))])
            mask = eligible & np.isin(np.arange(states.shape[1]), ids)[None, :]
            expected = states[mask]
            np.testing.assert_array_equal(dataset[key + '_obs'][count:], expected)
            assert len(expected) == report['new_counts'][key + '_obs']
            with torch.no_grad():
                for i in range(0, len(expected), 4096):
                    target = actor(torch.from_numpy(expected[i:i + 4096])).numpy()
                    target = np.clip(target, np.array(LITE3_SAFE_ACTION_LOWER, np.float32), np.array(LITE3_SAFE_ACTION_UPPER, np.float32))
                    np.testing.assert_allclose(dataset[key + '_target'][count + i:count + i + len(target)], target, atol=1e-6, rtol=1e-6)
                    replayed += len(target)
    result = dict(status='verified', collection_controls=report['collection_controls'], new_physics=0,
        new_label_rows_replayed=replayed, new_counts=report['new_counts'], counts=report['counts'], selection_counts=report['selection_counts'],
        evidence={str(path): sha256_file(path) for path in (Path(args.plan), folder / 'report.json', Path(plan['collection_audit']))})
    if args.training:
        training = Path(args.training)
        fitted = json.loads((training / 'report.json').read_text())
        t = json.loads(Path(args.training_protocol).read_text())
        assert fitted['status'] == 'complete' and fitted['updates'] == plan['updates'] == t['updates']
        assert fitted['optimizer_steps'] == [plan['updates']] and fitted['new_physics'] == 0
        assert fitted['protocol_sha256'] == sha256_file(args.training_protocol)
        assert fitted['dataset_sha256'] == report['dataset_sha256']
        assert t['dataset_report_sha256'] == sha256_file(folder / 'report.json')
        assert t['base_sha256'] == plan['base_sha256'] and Path(t['base_checkpoint']).resolve() == Path(plan['base_checkpoint']).resolve()
        for key in ('updates', 'learning_rate', 'max_grad_norm', 'batch_groups', 'preservation_weight'):
            assert t[key] == plan[key]
        assert t['seed'] == plan['learner_seed'] and fitted['preservation_weight'] == plan['preservation_weight']
        assert sha256_file(training / 'final.pt') == fitted['final_checkpoint']['sha256']
        payload = torch.load(training / 'final.pt', map_location='cpu', weights_only=False)
        assert set(payload) == {'schema', 'actor_state_dict', 'training_protocol_sha256'}
        assert payload['training_protocol_sha256'] == fitted['protocol_sha256']
        model = FrozenPPOExpert(payload['actor_state_dict'])
        for label, network in (('before', actor), ('after', model)):
            with torch.no_grad():
                for group in plan['batch_groups']:
                    for split in ('train', 'validation'):
                        x, y = dataset[f'{group}_{split}_obs'], dataset[f'{group}_{split}_target']
                        total = sum(float(F.mse_loss(network(torch.from_numpy(x[i:i + 4096])), torch.from_numpy(y[i:i + 4096]), reduction='sum')) for i in range(0, len(x), 4096))
                        np.testing.assert_allclose(total / y.size, fitted[label][split][group], atol=1e-6, rtol=2e-4)
        changed = [k for k, value in model.state_dict().items() if not torch.equal(value, actor.state_dict()[k])]
        assert changed == fitted['changed_tensors'] and len(changed) == 8
        result.update(training_updates=fitted['updates'], final_checkpoint=fitted['final_checkpoint'],
                      before=fitted['before'], after=fitted['after'], changed_tensors=changed)
        by_origin = {}
        with torch.no_grad():
            for group in plan['batch_groups']:
                key = f'{group}_validation'
                count = len(old[key + '_obs'])
                by_origin[group] = {}
                for label, region in (('original', slice(0, count)), ('added', slice(count, None))):
                    x, y = dataset[key + '_obs'][region], dataset[key + '_target'][region]
                    if len(x):
                        total = sum(float(F.mse_loss(model(torch.from_numpy(x[i:i + 4096])), torch.from_numpy(y[i:i + 4096]), reduction='sum')) for i in range(0, len(x), 4096))
                        by_origin[group][label] = dict(rows=len(x), after_mse=total / y.size)
        result['offline_validation_by_origin'] = by_origin
        for path in (training / 'report.json', Path(args.training_protocol)):
            result['evidence'][str(path)] = sha256_file(path)
    result['scope'] = 'Same TRAIN case26 data plus fresh ordinary learner-state labels, whole-stream offline split; no original7-source qualification or independent learning replication.'
    write_json_atomic(args.output, result)


if __name__ == '__main__':
    main()
