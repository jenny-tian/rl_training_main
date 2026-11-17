"""Lost supervisor status must not turn incomplete or altered artifacts valid."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
import reference_online_c1b_recover as recovery
from reference_stairs_eval import save,sha
from reference_online_targets import read


@pytest.fixture
def directory(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery.pilot,'OUTPUT',tmp_path)
    monkeypatch.setattr(recovery.psutil,'pid_exists',lambda pid:False)
    directory=tmp_path/'train'; directory.mkdir()
    save(directory/'execution.json',dict(status='running',pid=42))
    (directory/'trace.npz').write_bytes(b'synthetic trace')
    (directory/'final.pt').write_bytes(b'synthetic final')
    save(directory/'report.json',dict(status='passed',control_steps=20000,base_unchanged=True,
        frozen_unchanged=True,base_optimizer_updates=0,trace_sha256=sha(directory/'trace.npz'),
        final_sha256=sha(directory/'final.pt')))
    save(directory/'execution_recovery.json',dict(status='artifacts_complete_exit_status_unavailable',
        original_execution_sha256=sha(directory/'execution.json'),exit_code=None,
        artifacts=[dict(path=str(directory/f),sha256=sha(directory/f)) for f in ('trace.npz','final.pt','report.json')]))
    return directory


def test_complete_recovery_retains_unknown_exit_status(directory):
    assert recovery.recovered_completed(directory)['status']=='passed'
    assert read(directory/'execution.json')['status']=='running'
    assert read(directory/'execution_recovery.json')['exit_code'] is None


@pytest.mark.parametrize('name',['trace.npz','final.pt','execution.json'])
def test_altered_artifact_rejected(directory,name):
    if name=='execution.json': save(directory/name,dict(status='running',pid=43))
    else: (directory/name).write_bytes(b'altered')
    with pytest.raises(AssertionError): recovery.recovered_completed(directory)


def test_incomplete_run_rejected(directory):
    value=read(directory/'execution_recovery.json'); value['status']='failed_or_incomplete'
    save(directory/'execution_recovery.json',value)
    with pytest.raises(AssertionError): recovery.recovered_completed(directory)
