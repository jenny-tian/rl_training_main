from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/tools'))
from reference_online_failure_capture import capture_failure
from reference_online_targets import read


def test_failure_capture_preserves_existing_rows_without_calling_environment(tmp_path):
    def collect():
        tick = 2
        previous = torch.ones(2, 12)
        buffers = {'root': [np.full((2, 3), i) for i in range(2)]}
        records = [dict(outcome='success'), None]
        raise ValueError('synthetic interface failure')
    try:
        collect()
    except ValueError as error:
        capture_failure(error, tmp_path)
    report = read(tmp_path / 'failure_capture.json')
    assert report['recorded_control_steps'] == 2 and report['scalars']['tick'] == 2
    assert report['partial_records'][0]['outcome'] == 'success'
    assert not report['simulation_rerun']
    with np.load(tmp_path / 'failure_trace.npz') as data:
        assert np.array_equal(data['previous'], np.ones((2, 12)))
        assert data['buffers.root'].shape == (2, 2, 3)


def test_startup_error_is_reported_without_inventing_control_steps(tmp_path):
    try:
        raise ValueError('startup failure')
    except ValueError as error:
        capture_failure(error, tmp_path)
    assert read(tmp_path / 'failure_capture.json')['status'] == 'no_collection_frame'
    assert not (tmp_path / 'failure_trace.npz').exists()
