"""独立种子的父模型能力样本；不使用正式验收轨迹。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_regression_probe import PARENT, HASHES

SEED = 20261116
OUTPUT = ROOT / 'logs/reference_six/low_stairs_guarded_trial_20261115'
BANK = OUTPUT / 'training_bank.npz'


def collect(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0',
        kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
    import reference_turn_diagnose as protocol
    import reference_low_stairs_regression_probe as probe
    from reference_eval_checkpoint import register_checkpoint
    protocol.SEED = SEED
    probe.CANDIDATE = PARENT
    probe.HASHES = dict(parent=HASHES['parent'], candidate=HASHES['parent'])
    args.kind, args.policy, args.intervention = 'turn', 'parent', 'native'
    args.model, args.repeats, args.steps, args.video_env = 'regression_parent', 20, 400, None
    register_checkpoint(args.model, PARENT, HASHES['parent'])
    probe.collect(args, app)
    save(args.output / 'bank_provenance.json', dict(seed=SEED, parent_sha256=HASHES['parent'],
        training=False, optimizer_updates=0, diagnostic_bank_used=False, source_sha256=sha(__file__)))


def build():
    import torch
    from reference_low_stairs_preservation import capability_groups, GROUPS
    bank = {k: [] for k in ('observation', 'parent_action', 'group', 'env_id', 'tick', 'profile')}
    sources = []
    for p, profile in enumerate(('nominal', 'training_randomized')):
        directory = OUTPUT / f'bank_{profile}'
        report = json.loads((directory / 'report.json').read_text())
        provenance = json.loads((directory / 'bank_provenance.json').read_text())
        assert provenance['seed'] == SEED and report['model_sha256'] == HASHES['parent']
        assert json.loads((directory / 'execution.json').read_text())['status'] == 'completed'
        config = json.loads((directory / 'config.json').read_text())
        assert config['cfg']['seed'] == config['cfg']['scene']['terrain']['terrain_generator']['seed'] == SEED
        with np.load(directory / 'diagnostic.npz') as data, np.load(directory / 'trace.npz') as trace, np.load(directory / 'initial.npz') as initial:
            q = initial['root'][:, 3:7]
            tilt0 = np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))
            tilt = np.concatenate((tilt0[None], np.arccos(np.clip(-trace['gravity'][:-1, :, 2], -1, 1))))
            groups = capability_groups(torch.from_numpy(data['observation'][..., 6:9]), torch.from_numpy(tilt)).numpy()
            survived = np.array([r['outcome'] == 'survived' for r in report['records']])
            valid = trace['active'] & survived[None] & (tilt < np.deg2rad(30)) & (groups >= 0)
            valid &= (np.arange(len(groups))[:, None] % 5 == 0)
            rng = np.random.default_rng(SEED + p)
            for group in range(len(GROUPS)):
                pairs = np.argwhere(valid & (groups == group))
                pairs = pairs[rng.permutation(len(pairs))[:2048]]
                if not len(pairs):
                    continue
                t, e = pairs.T
                for key in ('observation', 'parent_action'):
                    bank[key].append(data[key][t, e])
                for key, value in dict(group=np.full(len(t), group), env_id=e, tick=t, profile=np.full(len(t), p)).items():
                    bank[key].append(value)
        sources.extend(dict(path=str(directory / name), sha256=sha(directory / name))
                       for name in ('diagnostic.npz', 'trace.npz', 'initial.npz', 'report.json', 'config.json', 'bank_provenance.json'))
    bank = {k: np.concatenate(v) for k, v in bank.items()}
    assert set(bank['group']) == set(range(5)) and np.isfinite(bank['observation']).all()
    if BANK.exists():
        raise FileExistsError(BANK)
    np.savez_compressed(BANK, **bank)
    report = dict(status='passed', seed=SEED, parent_sha256=HASHES['parent'], bank_sha256=sha(BANK),
        groups={name: int((bank['group'] == g).sum()) for g, name in enumerate(GROUPS)},
        source_files=sources, source_sha256=sha(__file__), diagnostic_bank_used=False,
        formal_evaluation_seeds=[20261114, 20261112], optimizer_updates=0)
    save(OUTPUT / 'training_bank.json', report)
    print(json.dumps(report['groups']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--profile', choices=('nominal', 'training_randomized'))
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.build:
        return build()
    if not args.profile:
        parser.error('需要指定 profile')
    args.output = OUTPUT / f'bank_{args.profile}'
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=1, output=args.output, minimum_free_mib=10240)
    try:
        collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc(); save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
