"""Measure final actor history use on existing TRAIN-source replay states only.

Masking a feature at fixed states is an offline sensitivity measurement; it is
not a closed-loop evaluation or a substitute for the separately trained control.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan_path = Path('docs/lite3_route_history_plan_20261010.json')
    audit_path = Path('docs/lite3_route_history_history_update_audit_20261010.json')
    plan, audit = [json.loads(p.read_text()) for p in (plan_path, audit_path)]
    assert audit['status'] == 'verified' and audit['history_enabled']
    model = audit['final_checkpoint']
    assert sha256_file(model['path']) == model['sha256']
    assert sha256_file(plan['dataset']) == plan['dataset_sha256']
    payload = torch.load(model['path'], map_location='cpu', weights_only=False)
    actor = FrozenPPOExpert(payload['actor_state_dict'])
    rows = {}
    with np.load(plan['dataset'], allow_pickle=False) as data, torch.no_grad():
        for group in ('correction', 'flat', 'rough101', 'rough107', 'post_recovery'):
            for split in ('train', 'validation'):
                key = group + '_' + split + '_obs'
                states = data[key]
                changes = []
                for start in range(0, len(states), 4096):
                    x = torch.from_numpy(states[start:start+4096].copy())
                    masked = x.clone()
                    masked[:, 12] = 0
                    changes.append((actor(x) - actor(masked)).numpy())
                delta = np.concatenate(changes)
                per_row = np.sqrt(np.mean(delta**2, axis=1))
                rows[group + '_' + split] = dict(
                    rows=len(states), history_nonzero_rows=int(np.count_nonzero(states[:, 12])),
                    history_absolute_mean=float(np.abs(states[:, 12]).mean()),
                    raw_action_rms_change=float(np.sqrt(np.mean(delta**2))),
                    raw_action_maximum_absolute_change=float(np.abs(delta).max()),
                    per_row_raw_action_rms_median=float(np.median(per_row)),
                    per_row_raw_action_rms_p95=float(np.quantile(per_row, .95)))
    evidence = {str(p): sha256_file(p) for p in
                (plan_path, audit_path, Path(plan['dataset']), Path(model['path']))}
    write_json_atomic(args.output, dict(
        status='verified', groups=rows, new_controls=0, optimizer_steps=0,
        evidence=evidence,
        scope='Final history actor on unchanged TRAIN-source replay states, with its inserted input retained versus zeroed. Includes original whole-stream offline validation partition; no pilot or original seven-source states. Raw mean-action sensitivity before clipping, not behavior quality, retraining, or a closed-loop counterfactual.'))


if __name__ == '__main__':
    main()
