"""CPU-only review of archived real PPO evidence; no new simulator or update."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'source/rl_training'), str(ROOT/'scripts/tools')]
import torch
from source_live_ppo_validation import validate, state_digest
from rl_training.source_scan.moving_diagnostic import restore_tensor

DEFAULT_CHECKPOINT = Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt')


def digest(path, *, compressed=False):
    result = hashlib.sha256()
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as stream:
        for chunk in iter(lambda: stream.read(1048576), b''): result.update(chunk)
    return result.hexdigest()


def verify(checkpoint):
    torch.set_num_threads(1)
    provenance = json.loads((HERE/'provenance.json').read_text())
    for name, entry in provenance['copies'].items():
        path = HERE/name
        if entry.get('lossless_gzip'):
            assert digest(path) == entry['archive_sha256']
            assert digest(path, compressed=True) == entry['source_sha256']
        else:
            assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']
    raw = HERE/'raw'
    plan = json.loads((raw/'run_manifest.json').read_text())
    execution = json.loads((raw/'execution.json').read_text())
    approved = json.loads((raw/'lite3_source_live_ppo_run1_20261102.approved.json').read_text())
    proposal = json.loads((raw/'lite3_source_live_ppo_run1_20261102.proposal.json').read_text())
    assert approved == plan and approved['user_approved'] is True and proposal['user_approved'] is False
    assert all(approved[key] == value for key, value in proposal.items() if key not in ('user_approved', 'status'))
    for section in ('input_sha256', 'dependency_sha256'):
        for name, expected in plan[section].items():
            path = ROOT/name if section == 'input_sha256' else Path(name)
            assert digest(path) == expected, name
    assert digest(checkpoint) == plan['checkpoint_sha256']
    acceptance = validate(raw, plan, execution, checkpoint=checkpoint)
    assert acceptance['accepted'] and not acceptance['synthetic_cpu_fixture'], acceptance
    receipt = json.loads((raw/'budget_receipt.json').read_text())
    ledger = json.loads((raw/'budget_ledger_after.json').read_text())
    entries = [row for row in ledger['entries'] if row['run'] == 1]
    assert entries == [receipt] and receipt['state'] == 'accepted'
    assert receipt['reserved_transitions'] == receipt['attempted_upper_bound'] == receipt['returned_transitions'] == receipt['stored_transitions'] == 384
    assert receipt['completed_updates'] == 2
    report = json.loads((raw/'result.json').read_text())
    controls = [json.loads(line) for line in (raw/'result.steps.jsonl').read_text().splitlines()]
    models = [torch.load(raw/f'model_{i}.pt', map_location='cpu', weights_only=True) for i in (0, 1, 2)]
    changes = []
    for i in (1, 2):
        row = {'update': i}
        for kind in ('actor', 'critic'):
            before, after = models[i-1][kind], models[i][kind]
            row[kind+'_changed_tensors'] = sum(not torch.equal(before[k], after[k]) for k in before)
            row[kind+'_l2_delta'] = math.sqrt(sum(float((after[k].double()-before[k].double()).square().sum()) for k in before))
            row[kind+'_sha256'] = state_digest(after)
        row['losses'] = report['training_snapshot']['update_receipts'][i-1]['losses']
        changes.append(row)
    starts = restore_tensor(report['initial_positions_w'])[:, 0].clone()
    maxima = {'reset_relative_forward_progress_m': 0., 'link_displacement_per_5ms_m': 0.,
              'link_speed_mps': 0., 'finite_difference_link_acceleration_mps2': 0.}
    minimum_margin = float('inf')
    with gzip.open(raw/'result.events.jsonl.gz', 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            if event['phase'] == 'reset_checked':
                ids = event['ids']; starts[ids] = restore_tensor(event['rebase_positions_w'])[ids, 0]
            if event['phase'] != 'motion_substep': continue
            positions = restore_tensor(event['body_positions_w'])
            maxima['reset_relative_forward_progress_m'] = max(maxima['reset_relative_forward_progress_m'], float((positions[:, 0, 0]-starts[:, 0]).max()))
            for field, key in (('displacements_m', 'link_displacement_per_5ms_m'),
                ('body_link_velocities_w', 'link_speed_mps'),
                ('finite_difference_acceleration_mps2', 'finite_difference_link_acceleration_mps2')):
                maxima[key] = max(maxima[key], float(restore_tensor(event[field]).norm(dim=-1).max()))
            lower, upper = torch.tensor(event['safe_lower_xy_w']), torch.tensor(event['safe_upper_xy_w'])
            minimum_margin = min(minimum_margin, float(torch.minimum(positions[..., :2]-lower[:, None], upper[:, None]-positions[..., :2]).min()))
    samples = [json.loads(line) for line in (raw/'gpu_samples.jsonl').read_text().splitlines()]
    card_rows = [next(line for line in sample['gpu'].splitlines() if line.startswith('0,')).split(',') for sample in samples]
    clip_rows = sum(any(a != b for a, b in zip(raw_action, base_action))
        for row in controls for raw_action, base_action in zip(row['raw_action'], row['base_action']))
    log = (raw/'runtime.log').read_text(errors='replace')
    assert json.loads((HERE/'cleanup_check.json').read_text())['process_group_empty'] is True
    assert not torch.cuda.is_initialized()
    return {'status': 'archived_real_ppo_evidence_verified', 'executed_revision': plan['base_revision'],
        'acceptance': acceptance, 'parameter_changes': changes,
        'raw_action_clipped_rows': clip_rows, 'additional_projection_rows': sum(bool(v[0]) for r in controls for v in r['projected']),
        'paused_rows': sum(bool(v[0]) for r in controls for v in r['paused']),
        'forced_resets': [{'step': row['step'], 'ids': row['forced_ids'], 'terminal_steps': row['terminal']['steps'],
                          'episode_counters_after': row['episode_counters']} for row in controls if row['forced_ids']],
        'final_episode_counters': controls[-1]['episode_counters'], 'exposure': report['exposure'],
        'descriptive_motion_extrema': maxima, 'minimum_sampled_envelope_margin_m': minimum_margin,
        'card_level_resource_samples': {'count': len(samples), 'used_peak_mib': max(int(r[3].strip().split()[0]) for r in card_rows),
            'free_min_mib': min(int(r[4].strip().split()[0]) for r in card_rows)},
        'log_markers': {'foot_visual_reference_warnings': log.count('Unresolved reference prim path'),
            'gpu_enumeration_skip_warnings': log.count('Skipping NVIDIA GPU due CUDA being in bad state'),
            'python_tracebacks': log.count('Traceback (most recent call last)')},
        'real_transitions_charged': 384, 'production_inputs': len(plan['input_sha256']),
        'installed_dependencies': len(plan['dependency_sha256']), 'new_simulation_launches': 0,
        'new_ppo_updates': 0, 'training_ready': False, 'g3_g4_qualified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    print(json.dumps(verify(parser.parse_args().checkpoint), indent=2))
