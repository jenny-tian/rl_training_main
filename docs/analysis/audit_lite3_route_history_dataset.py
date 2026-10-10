"""Reproduce the frozen dataset audit from row provenance and raw past frames."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path('artifacts/route_history_20261010/dataset')
    report = json.loads((root / 'report.json').read_text())
    for name, expected in report['evidence'].items():
        assert sha256_file(name) == expected, name
    assert sha256_file(root / 'dataset.npz') == report['dataset_sha256']
    assert sha256_file(root / 'row_provenance.npz') == report['provenance_sha256']
    raw = []
    for name in report['source_order']:
        source = Path(name)
        metadata = json.loads((source / 'report.json').read_text())
        assert len(metadata['phases']) == 1
        phase = source / next(iter(metadata['phases']))
        pieces = {'policy_observation': [], 'done': []}
        for path in sorted((phase / 'trajectory').glob('chunk-*.npz')):
            with np.load(path, allow_pickle=False) as chunk:
                for key in pieces:
                    pieces[key].append(chunk[key])
        raw.append({key: np.concatenate(values) for key, values in pieces.items()})
    rows = resets = 0
    old_path = Path('artifacts/retention_coverage_20261008/dataset/dataset.npz')
    assert sha256_file(old_path) == report['parent_dataset_sha256']
    with np.load(root / 'dataset.npz', allow_pickle=False) as dataset, \
            np.load(root / 'row_provenance.npz', allow_pickle=False) as provenance, \
            np.load(old_path, allow_pickle=False) as old:
        assert set(dataset.files) == set(old.files)
        assert set(provenance.files) == {key for key in old.files if key.endswith('_obs')}
        for key in dataset.files:
            if not key.endswith('_obs'):
                np.testing.assert_array_equal(dataset[key], old[key])
                continue
            values, origins = dataset[key], provenance[key]
            assert values.shape == (len(old[key]), 49) and origins.shape == (len(values), 4)
            np.testing.assert_array_equal(np.concatenate((values[:, :12], values[:, 13:]), -1), old[key])
            assert np.all((origins[:, 2] % 4 == 0) == ('_validation_' in key))
            assert np.all((origins[:, 0] >= 0) & (origins[:, 0] < len(raw)))
            for index, source in enumerate(raw):
                select = origins[:, 0] == index
                time, env, declared_reset = origins[select, 1:].T.astype(int)
                assert np.all((time >= 0) & (time < len(source['done'])))
                assert np.all((env >= 0) & (env < source['done'].shape[1]))
                current = source['policy_observation'][time, env]
                previous_time = np.maximum(time - 1, 0)
                reset = (time == 0) | source['done'][previous_time, env].astype(bool)
                np.testing.assert_array_equal(declared_reset, reset)
                np.testing.assert_array_equal(current, old[key][select])
                previous_y = np.where(reset, current[:, 9], source['policy_observation'][previous_time, env, 9])
                feature = np.clip((current[:, 9] - previous_y) * np.float32(1.65) / np.float32(.02), -1., 1.) / np.float32(.3)
                np.testing.assert_array_equal(values[select, 12], feature)
                resets += int(reset.sum())
            rows += len(values)
    assert rows == report['total_rows'] == 174282 and resets == report['reset_rows'] == 152
    write_json_atomic(args.output, dict(
        status='verified', rows=rows, reset_rows=resets, new_controls=0,
        original_observations_and_targets_bitwise_preserved=True,
        past_frame_from_raw_before_filtering=True, whole_stream_split_preserved=True,
        evidence={str(root / name): sha256_file(root / name)
                  for name in ('dataset.npz', 'report.json', 'row_provenance.npz')}))


if __name__ == '__main__':
    main()
