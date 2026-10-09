"""Reconstruct masks, reset ages, frozen action targets and actual reward charges."""
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
    for name in ('protocol', 'run', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--training-audit')
    args = parser.parse_args()
    root = Path(args.run)
    p = json.loads(Path(args.protocol).read_text())
    r = json.loads((root/'report.json').read_text())
    assert r['status'] == 'complete' and r['protocol_sha256'] == sha256_file(args.protocol)
    assert sha256_file(p['base_checkpoint']) == p['base_sha256']
    training = bool(args.training_audit)
    options = p['recovery_anchor'] if training else p['anchor_options']
    evidence = {str(f): sha256_file(f) for f in (Path(args.protocol), root/'report.json')}
    if training:
        audit = json.loads(Path(args.training_audit).read_text())
        assert audit['status'] == 'verified' and audit['evidence'][str(root/'report.json')] == sha256_file(root/'report.json')
        evidence[args.training_audit] = sha256_file(args.training_audit)
        assert r['warmstart']['distribution_initialized_from_mean_only'] and r['recovery_anchor_model_unchanged']
        assert r['ppo_updates'] == p['iterations']
        assert r['optimizer_steps'] == [p['iterations']*p['ppo']['num_learning_epochs']*p['ppo']['num_mini_batches']]
    else:
        assert r['anchor_model_unchanged'] and r['recovery_anchor_timing_and_mask_passed']
    torch.set_num_threads(4)
    teacher = FrozenPPOExpert(torch.load(p['base_checkpoint'], map_location='cpu', weights_only=False)['actor_state_dict'])
    age = np.zeros(p['num_envs'], np.int64)
    history = np.zeros((p['num_envs'], 12), np.float32)
    total = labels = replay = resets = nonzero = 0
    case_rows = dict(ordinary=0, early=0, tail=0)
    lo, hi = np.array(LITE3_SAFE_ACTION_LOWER, np.float32), np.array(LITE3_SAFE_ACTION_UPPER, np.float32)
    assert p['num_envs'] == 64 and p['recovery_resets']['fixed_train_case_count'] == 12
    expected_cases = np.full(64, -1, np.int64)
    expected_cases[40:52] = 26
    traces = {}
    for file in sorted((root/'trajectory').glob('chunk-*.npz')):
        traces[file.name] = sha256_file(file)
        with np.load(file, allow_pickle=False) as d:
            obs = d['policy_observation']
            np.testing.assert_array_equal(d['recovery_case'], np.broadcast_to(expected_cases, d['recovery_case'].shape))
            mask = (d['recovery_case'] < 0) | (d['episode_steps'] < options['early_steps'])
            np.testing.assert_array_equal(d['recovery_anchor_mask'], mask)
            emitted = np.clip(d['requested_action'], lo, hi)
            np.testing.assert_array_equal(d['emitted_action'], emitted)
            target = d['recovery_anchor_target']
            assert not np.any(target[~mask])
            with torch.no_grad():
                selected = obs[mask]
                expected = np.concatenate([np.clip(teacher(torch.from_numpy(selected[i:i+4096])).numpy(), lo, hi) for i in range(0, len(selected), 4096)])
            np.testing.assert_allclose(target[mask], expected, atol=3e-5, rtol=1e-5)
            replay += len(selected)
            cost = np.where(mask, ((emitted-target)**2).mean(-1), 0)
            np.testing.assert_allclose(d['recovery_anchor_cost'], cost, atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(d['recovery_anchor_applied_cost'], cost, atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(d['recovery_anchor_reward'], -options['weight']*cost*.02, atol=1e-6, rtol=1e-5)
            labels += int(mask.sum())
            nonzero += int((cost > 0).sum())
            ordinary = d['recovery_case'] < 0
            case_rows['ordinary'] += int(ordinary.sum())
            case_rows['early'] += int((mask & ~ordinary).sum())
            case_rows['tail'] += int((~mask).sum())
            for i in range(len(obs)):
                np.testing.assert_array_equal(d['episode_steps'][i], age)
                np.testing.assert_allclose(obs[i, :, -12:], history, atol=1e-6, rtol=0)
                np.testing.assert_allclose(obs[i, :, 6:9], np.broadcast_to([.3,0,0],(64,3)), atol=1e-7, rtol=0)
                if total:
                    resets += int(((age == 0) & (expected_cases >= 0)).sum())
                done = d['done'][i].astype(bool)
                history = emitted[i].copy()
                history[done] = 0
                age = np.where(done, 0, age+1)
                total += p['num_envs']
    assert total == r['ledger']['confirmed'] == r['ledger']['charged']
    assert r['ledger']['uncertain_transitions'] == 0 and all(v>0 for v in case_rows.values()) and resets > 0 and nonzero > 0
    if training:
        assert labels == r['recovery_anchor_label_rows']
    else:
        assert dict(**case_rows, repeated_recovery_resets=resets, nonzero_cost_rows=nonzero) == r['observed']
    result = dict(status='verified', controls=total, labels_replayed=replay, mask_rows=case_rows,
        repeated_recovery_resets=resets, nonzero_cost_rows=nonzero, trace_sha256=traces, evidence=evidence,
        scope='Training-only soft reward constraint; ordinary states and first12 recovery controls, no inference-time teacher or clock.')
    if training:
        assert sha256_file(root/'final.pt') == r['final_checkpoint']['sha256']
        result.update(final_checkpoint=r['final_checkpoint'], training_updates=r['ppo_updates'])
    write_json_atomic(args.output, result)


if __name__ == '__main__':
    main()
