import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rl_training.source_scan.smoke_artifacts import (
    SmokeArtifactWriter, artifact_paths, atomic_json, finalize_and_close, validate_smoke_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT/'tests/source_scan/fixtures/smoke_persistence_child.py'


def child(tmp_path, mode='application_exit'):
    output = tmp_path/'result.json'
    process = subprocess.run([sys.executable, str(FIXTURE), str(output), mode],
                             env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                             capture_output=True, text=True, timeout=20)
    assert not process.stderr, process.stderr
    return output, process.returncode


def review(output, returncode=0):
    return validate_smoke_artifacts(output, returncode=returncode, expected_attempt_id='cpu-fixture',
                                    expected_revision='a'*40, expected_checkpoint_sha256='b'*64)


@pytest.mark.parametrize('mode', ['application_exit', 'close_returns'])
def test_complete_result_and_all_rows_survive_real_process_exit_during_close(tmp_path, mode):
    output, returncode = child(tmp_path, mode)
    assert returncode == 0
    result = review(output, returncode)
    assert result['accepted'], result
    assert result['record_count'] == 24
    assert result['application_close_returned'] == (mode == 'close_returns')
    paths = artifact_paths(output)
    report = json.loads(output.read_text())
    assert report['status'] == 'passed'
    assert json.loads(paths['steps'].read_text().splitlines()[-1])['terminal']['observations']['critic'][0] == [2.]*3


@pytest.mark.parametrize('mode', ['missing', 'starting_zero', 'environment_exit', 'environment_error',
                                 'journal_close_error', 'midway_exit'])
def test_parent_rejects_incomplete_or_failed_evidence_even_when_exit_is_zero(tmp_path, mode):
    output, returncode = child(tmp_path, mode)
    result = review(output, returncode)
    assert result['accepted'] is False and result['reasons']
    if mode == 'midway_exit':
        assert returncode == 17
        report = json.loads(output.read_text())
        assert report['status'] == 'running' and report['simulation_control_steps'] == 5
        assert len(artifact_paths(output)['steps'].read_text().splitlines()) == 5
    elif mode == 'environment_error':
        assert returncode == 0
        assert json.loads(output.read_text())['status'] == 'failed'
        assert json.loads(artifact_paths(output)['cleanup'].read_text())['environment'] == 'failed'


@pytest.mark.parametrize('fault', ['row_missing', 'row_nonfinite', 'snapshot_missing', 'timeout_is_failure',
                                  'exposure_missing', 'wrong_attempt', 'wrong_revision', 'wrong_checkpoint',
                                  'nonzero_exit', 'truncated_jsonl', 'stale_report', 'wrong_seed',
                                  'wrong_attribution', 'unreset_column_change'])
def test_corrupt_or_contradictory_artifacts_never_pass(tmp_path, fault):
    output, returncode = child(tmp_path)
    paths = artifact_paths(output)
    report = json.loads(output.read_text())
    rows = [json.loads(line) for line in paths['steps'].read_text().splitlines()]
    if fault == 'row_missing':
        rows.pop()
    elif fault == 'row_nonfinite':
        rows[4]['raw_action'][0][0] = float('nan')
    elif fault == 'snapshot_missing':
        rows[-1]['terminal']['observations'].pop('critic')
    elif fault == 'timeout_is_failure':
        rows[-1]['terminal']['terminated'][0] = True
    elif fault == 'exposure_missing':
        report.pop('exposure')
    elif fault == 'wrong_attempt':
        report['attempt_id'] = 'some-other-attempt'
    elif fault == 'wrong_revision':
        report['base_revision'] = 'c'*40
    elif fault == 'wrong_checkpoint':
        report['checkpoint_sha256'] = 'c'*64
    elif fault == 'nonzero_exit':
        returncode = 23
    elif fault == 'stale_report':
        report['status'] = 'starting'
    elif fault == 'wrong_seed':
        report['geometry_seed'] = 10001
    elif fault == 'wrong_attribution':
        report['exposure']['flat']['transitions'] -= 1
        report['exposure']['multi_scale_rough']['transitions'] += 1
    elif fault == 'unreset_column_change':
        rows[3]['terrain_columns_after'][1] = 2
    journal = ''.join(json.dumps(row)+'\n' for row in rows)
    if fault == 'truncated_jsonl':
        journal = journal[:-20]
    paths['steps'].write_text(journal)
    # Refresh the digest so schema validation is independently exercised, not
    # only a digest mismatch detecting all mutations above.
    import hashlib
    report['steps_sha256'] = hashlib.sha256(paths['steps'].read_bytes()).hexdigest()
    atomic_json(output, report)
    result = review(output, returncode)
    assert result['accepted'] is False and result['reasons'], fault


def test_invalid_atomic_update_preserves_previous_complete_file(tmp_path, monkeypatch):
    output = tmp_path/'report.json'
    atomic_json(output, {'status': 'running'})
    before = output.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(output, {'invalid': float('nan')})
    assert output.read_bytes() == before
    def failed_replace(*args):
        raise OSError('injected interrupted replace')
    monkeypatch.setattr(os, 'replace', failed_replace)
    with pytest.raises(OSError):
        atomic_json(output, {'status': 'passed'})
    assert output.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))


def test_existing_attempt_is_never_overwritten(tmp_path):
    output, _ = child(tmp_path)
    before = {p: p.read_bytes() for p in artifact_paths(output).values()}
    with pytest.raises(FileExistsError):
        SmokeArtifactWriter(output, {'attempt_id': 'another-attempt'})
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_disk_failure_still_attempts_both_cleanup_callbacks(tmp_path, monkeypatch):
    writer = SmokeArtifactWriter(tmp_path/'result.json', {'attempt_id': 'cpu-fixture'})
    report = {'attempt_id': 'cpu-fixture', 'status': 'passed'}
    closed = []
    def failed_persist(*args):
        raise OSError('injected disk full')
    monkeypatch.setattr(writer, 'persist', failed_persist)
    with pytest.raises(OSError):
        finalize_and_close(writer, report, environment_close=lambda: closed.append('env'),
                           application_close=lambda: closed.append('app'))
    assert closed == ['env', 'app']
    assert not review(tmp_path/'result.json')['accepted']


def test_actual_smoke_entrypoint_persists_failure_before_fake_app_exits_zero(tmp_path):
    # Exercise main()'s actual finally path. The injected launcher never imports
    # Isaac and the forced gym import failure occurs before Torch or any scene.
    output = tmp_path/'result.json'
    script = r'''
import builtins, os, runpy, sys, types
class Launcher:
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument('--device', default='cpu')
    def __init__(self, args):
        self.app = types.SimpleNamespace(close=lambda: os._exit(0))
sys.modules['isaaclab'] = types.ModuleType('isaaclab')
sys.modules['isaaclab.app'] = types.SimpleNamespace(AppLauncher=Launcher)
original_import = builtins.__import__
def cpu_import(name, *args, **kwargs):
    if name == 'gymnasium':
        raise RuntimeError('CPU injected startup failure before scene construction')
    return original_import(name, *args, **kwargs)
builtins.__import__ = cpu_import
sys.argv = [sys.argv[1], '--checkpoint', 'unused-cpu-fixture', '--output', sys.argv[2], '--attempt-id', 'cpu-fixture']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    process = subprocess.run([sys.executable, '-c', script, str(ROOT/'scripts/tools/source_training_integration_smoke.py'), str(output)],
                             env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=20)
    assert process.returncode == 0
    report = json.loads(output.read_text())
    assert report['status'] == 'failed'
    assert 'CPU injected startup failure' in report['error']
    assert report['phase'] == 'result_before_shutdown'
    assert not review(output)['accepted']


def test_supervisor_validate_cli_rejects_historical_zero_exit_without_modifying_archive():
    import hashlib
    archive = ROOT/'docs/lite3-source-training-integration-smoke-20261102/raw/result.json'
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    process = subprocess.run([sys.executable, str(ROOT/'scripts/tools/source_training_integration_supervisor.py'),
                              'validate', '--output', str(archive), '--returncode', '0',
                              '--attempt-id', 'old-attempt', '--revision', 'ae19a40015c81e1c772168530761124188abfe24',
                              '--checkpoint-sha256', 'b'*64], capture_output=True, text=True, timeout=20)
    assert process.returncode == 1
    assert json.loads(process.stdout)['accepted'] is False
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before


def test_supervisor_validate_cli_accepts_complete_synthetic_fixture(tmp_path):
    output, _ = child(tmp_path)
    process = subprocess.run([sys.executable, str(ROOT/'scripts/tools/source_training_integration_supervisor.py'),
                              'validate', '--output', str(output), '--returncode', '0',
                              '--attempt-id', 'cpu-fixture', '--revision', 'a'*40,
                              '--checkpoint-sha256', 'b'*64], capture_output=True, text=True, timeout=20)
    assert process.returncode == 0
    assert json.loads(process.stdout)['accepted'] is True


def test_supervisor_rejects_residual_group_and_unapproved_run_before_any_launch(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('probe_supervisor', ROOT/'scripts/tools/source_training_integration_supervisor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output, _ = child(tmp_path)
    result = module.validate_existing(output, returncode=0, attempt_id='cpu-fixture', revision='a'*40,
                                      checkpoint_sha256='b'*64, process_group_empty=False)
    assert not result['accepted'] and 'child_process_group_remains' in result['reasons']
    manifest = tmp_path/'unapproved.json'
    manifest.write_text(json.dumps({'user_approved': False}))
    def forbid_launch(*args, **kwargs):
        raise AssertionError('CPU stage must not launch a probe or query a GPU')
    monkeypatch.setattr(module.subprocess, 'Popen', forbid_launch)
    monkeypatch.setattr(module.subprocess, 'check_output', forbid_launch)
    with pytest.raises(ValueError, match='unapproved'):
        module.run_approved(SimpleNamespace(manifest=manifest))
