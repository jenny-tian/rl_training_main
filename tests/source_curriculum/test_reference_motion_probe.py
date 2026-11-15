import json

import numpy as np
import pytest

from reference_low_stairs_motion_probe import exact_replay


@pytest.mark.parametrize('field',['report','config','initial','trace'])
def test_recorder_rejects_any_original_state_or_protocol_change(tmp_path,field):
    paths=[tmp_path/x for x in ('new','old')]
    for p in paths:
        p.mkdir()
        for name in ('report','config'):(p/f'{name}.json').write_text(json.dumps({'key':1}))
        for name in ('initial','trace'):np.savez(p/f'{name}.npz',root=np.array([1.,2.]))
    assert exact_replay(*paths)['status']=='passed'
    if field in ('report','config'):(paths[0]/f'{field}.json').write_text(json.dumps({'key':2}))
    else:np.savez(paths[0]/f'{field}.npz',root=np.array([1.,2.000001]))
    with pytest.raises(AssertionError):exact_replay(*paths)


def test_unknown_additional_source_is_rejected(tmp_path):
    paths=[tmp_path/x for x in ('new','old')]
    for p in paths:
        p.mkdir();(p/'report.json').write_text('{}')
        (p/'config.json').write_text('{}')
    (paths[0]/'config.json').write_text(json.dumps({'sources':[{'path':'unknown.py','sha256':'a'*64}]}))
    with pytest.raises(AssertionError):exact_replay(*paths)
