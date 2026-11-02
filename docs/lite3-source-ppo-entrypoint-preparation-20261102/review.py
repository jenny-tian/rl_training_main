"""Offline archive checks only; never re-execute synthetic updates or Isaac."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def verify():
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    for name, metadata in provenance['copied_files'].items():
        data = (ROOT / 'raw' / name).read_bytes()
        assert len(data) == metadata['bytes']
        assert hashlib.sha256(data).hexdigest() == metadata['sha256']
    suites = ET.parse(ROOT / 'raw/registered_regression.xml').getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in suites) == 433
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    assert json.loads((ROOT / 'raw/cpu_regression_execution.json').read_text())['returncode'] == 2
    inputs = None
    for device in ('cpu', 'cuda'):
        summary = json.loads((ROOT / f'raw/{device}/summary.json').read_text())
        assert summary['status'] == 'passed' and summary['synthetic'] is True
        assert summary['simulation_steps'] == summary['real_source_training_transitions'] == summary['real_source_ppo_updates'] == 0
        assert summary['synthetic_ppo_updates'] == 6 and summary['synthetic_transitions'] == 1152
        assert len(summary['runs']) == 3
        assert len({r['actor_initial_sha256'] for r in summary['runs']}) == 1
        assert len({r['critic_initial_sha256'] for r in summary['runs']}) == 3
        if inputs is None: inputs = summary['input_sha256']
        assert inputs == summary['input_sha256']
        for run in summary['runs']:
            assert run['actor_initial_sha256'] != run['actor_final_sha256']
            assert run['critic_initial_sha256'] != run['critic_final_sha256']
            assert run['completed_updates'] == 2 and run['synthetic_transitions'] == 384
            assert run['backward_checks'] == run['optimizer_steps_completed'] == 40
            assert run['optimizer_state_entries'] == 17
            assert all(math.isfinite(value) for row in run['losses'] for value in row.values())
            report = json.loads((ROOT / f'raw/{device}/run_{run["run"]}/run_report.json').read_text())
            assert report['status'] == 'completed' and report['data_kind'] == 'synthetic'
            assert report['stored_transitions'] == report['returned_transitions'] == report['attempted_transition_upper_bound'] == 384
            assert sum(v['transitions'] for v in report['exposure'].values()) == 384
            assert len(report['update_receipts']) == 2
            assert report['update_receipts'][-1]['actor_sha256'] == run['actor_final_sha256']
            assert report['real_source_training_transitions'] == report['real_source_ppo_updates'] == 0
            for update, digest in run['checkpoint_sha256'].items():
                assert provenance['synthetic_checkpoint_inventory'][f'{device}/run_{run["run"]}/actor_update_{update}.pt']['sha256'] == digest
    for name, digest in inputs.items():
        assert hashlib.sha256((ROOT.parents[1] / name).read_bytes()).hexdigest() == digest, name
    execution = json.loads((ROOT / 'raw/cuda_synthetic_execution.json').read_text())
    assert execution['returncode'] == 0 and execution['process_group_empty'] is True and execution['timed_out'] is False
    assert json.loads((ROOT / 'scope.json').read_text())['user_approved'] is True
    assert json.loads((ROOT / 'next_stage.json').read_text())['user_approved'] is False
    return {'status': 'archive_verified', 'registered_regression_passed': 433,
            'full_source_directory_collection_passed': False,
            'cpu_synthetic_updates': 6, 'cuda_synthetic_updates': 6,
            'real_source_updates': 0, 'simulation_launches': 0,
            'production_inputs': len(inputs), 'copied_files': len(provenance['copied_files'])}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
