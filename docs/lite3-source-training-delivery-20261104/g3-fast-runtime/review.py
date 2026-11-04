"""Verify the failed real run archive; this does not qualify physical evidence."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def main(folder):
    folder = folder.resolve()
    checked = 0
    for line in (folder / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (folder / name).resolve()
        assert path.is_relative_to(folder) and digest(path) == expected, name
        checked += 1
    load = lambda name: json.loads((folder / name).read_text())
    index = load('raw/runtime_artifacts.json')
    bulk = 0
    for row in index['files']:
        p = Path(row['original_path']) if row['repository_path'] is None else folder / row['repository_path']
        assert p.stat().st_size == row['size_bytes'] and digest(p) == row['sha256'], str(p)
        bulk += row['repository_path'] is None
    summary = load('raw/actual_record_summary.json')
    execution = load('raw/runtime/execution.json')
    acceptance = load('raw/runtime/acceptance.json')
    report = load('raw/runtime/result.json')
    assert summary['complete_control_records'] == 1884
    assert summary['readonly_transitions_with_complete_control_records'] == 226080
    assert summary['incomplete_final_control_record_bytes'] == 0
    assert summary['ordinary_terminal_batches'] == 77 and summary['ended_episodes'] == 78
    assert summary['best_flat_record']['forward_progress_m'] == 3.9067344665527344
    assert summary['candidate_flat_completions'] == []
    assert summary['new_ppo_updates'] == summary['new_training_transitions'] == 0
    assert summary['trained_checkpoints_found'] == []
    assert execution['timed_out'] and execution['returncode'] == 124
    assert execution['process_group_empty'] and execution['validation_process_group_empty']
    assert execution['validation_returncode'] == 1
    assert not acceptance['accepted'] and not acceptance['training_ready']
    assert report['status'] == 'running' and report['simulation_control_steps'] == 0
    assert load('raw/approved_supplement.json')['user_approved'] is True
    assert set(load('raw/original_slots_after.json')['jobs']) == {'g2', 'g3'}
    assert set(load('raw/supplement_slots_after.json')['jobs']) == {'g3'}
    assert digest(folder/'raw/production_ledger_after.json') == 'b1d224ba20a93c93c99189d700aeb145d506259f56462eb506def64f8f888189'
    assert load('raw/checkpoint_after.json')['original_checkpoint_unchanged']
    assert load('raw/gzip_container_check.json')['complete_gzip_container']
    return dict(archive_integrity_verified=True, hashed_files=checked, bulk_files_verified=bulk,
        actual_supplemental_GPU_jobs=1, complete_controls=1884, readonly_transitions=226080,
        G3_accepted=False, physical_evidence_independently_qualified=False,
        gzip_container_complete=True, production_ledger_unchanged=True,
        new_ppo_updates=0, new_model_delivered=False,
        scope='Failed-run archive and saved-control summary; no runtime qualification')


if __name__ == '__main__':
    print(json.dumps(main(Path(__file__).resolve().parent), indent=2))
