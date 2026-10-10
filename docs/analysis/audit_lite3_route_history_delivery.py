"""Verify the delivered route-history evidence, physical ledger and file links."""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--figures-reviewed', action='store_true')
    args = parser.parse_args()
    assert args.figures_reviewed, 'View both generated figures before delivery'
    output = Path(args.output).resolve()
    hashes, visited, evidence = {}, set(), {}
    trace_checks = 0

    def digest(path):
        path = Path(path).resolve()
        if path not in hashes:
            hashes[path] = sha256_file(path)
        return hashes[path]

    def check(path, expected):
        assert digest(path) == expected, str(path)

    def read(path):
        return json.loads(Path(path).read_text())

    def visit(path):
        nonlocal trace_checks
        path = Path(path).resolve()
        if path in visited or path.suffix != '.json':
            return
        visited.add(path)
        data = read(path)
        if not isinstance(data, dict):
            return
        for name, expected in data.get('evidence', {}).items():
            check(name, expected)
            visit(name)
        roots = [Path(name).parent for name in data.get('evidence', {})
                 if Path(name).name == 'report.json']

        def trace(mapping, phase=None):
            nonlocal trace_checks
            if not mapping:
                return
            candidates = []
            for root in roots:
                candidates.append(root / 'trajectory')
                if phase and phase != 'root':
                    candidates.append(root / phase / 'trajectory')
                candidates.extend(root.glob('*/trajectory'))
            name, expected = next(iter(mapping.items()))
            matches = {p.resolve() for p in candidates
                       if (p / name).is_file() and digest(p / name) == expected}
            assert matches, (str(path), phase, name)
            for folder in matches:
                for name, expected in mapping.items():
                    check(folder / name, expected)
                    trace_checks += 1

        if 'trace_sha256' in data:
            trace(data['trace_sha256'])
        for phase, item in data.get('phases', {}).items():
            if 'trace_sha256' in item:
                trace(item['trace_sha256'], phase)
        for phase, mapping in data.get('phase_trace_sha256', {}).items():
            trace(mapping, phase)

    documents = [
        Path('docs/lite3-route-history-result-2026-10-10.md'),
        Path('docs/lite3-route-history-runbook-2026-10-10.md'),
        Path('docs/lite3-project-summary-2026-10-08.md'),
        Path('docs/lite3-next-experiments-plan-2026-10-06.md'),
        Path('docs/lite3-flowdagger-implementation-2026-10-03.md'),
        Path('docs/lite3_experiment_registry.md'),
    ]
    files = list(Path('docs').glob('lite3_route_history_*.json'))
    files += list(Path('docs').glob('lite3_route_history_*.txt'))
    files += list(Path('docs/figures').glob('lite3-route-history-20261010.*'))
    files += list(Path('docs/figures').glob('lite3-route-history-recovery-20261010.*'))
    files += documents
    files += [Path(__file__), Path('tests/online_adaptation/test_policy_route_history.py')]
    files += [Path('docs/analysis') / name for name in (
        'build_lite3_route_history_dataset.py', 'audit_lite3_route_history_dataset.py',
        'audit_lite3_route_history.py',
        'audit_lite3_history_training.py', 'audit_lite3_history_pair.py',
        'summarize_lite3_route_history.py', 'assemble_lite3_condition_evaluations.py',
        'audit_lite3_prefix_imitation_pilot.py', 'audit_lite3_recovery_teacher.py',
        'probe_lite3_velocity_observations.py', 'diagnose_lite3_recovery_lateral.py',
        'diagnose_lite3_frozen_return_velocity.py', 'diagnose_lite3_route_history_use.py')]
    for path in files:
        if path.resolve() != output:
            evidence[str(path)] = digest(path)
            visit(path)

    plan = read('docs/lite3_route_history_plan_20261010.json')
    result = read('docs/lite3_route_history_result_20261010.json')
    # Passing the pilot requires a separately completed conditional validation.
    # This delivery command covers the registered stop after a failed pilot.
    assert result['status'] == 'verified' and not result['history_pilot_passed']
    runs = []
    for arm in plan['arms']:
        runs += [Path(f'artifacts/route_history_20261010/{arm}-smoke')]
        runs += [Path(f'artifacts/route_history_20261010/{arm}/{part}')
                 for part in ('train', 'course-evaluation', 'recovery-evaluation')]
    actual = {p.parent.resolve() for p in Path('artifacts/route_history_20261010').rglob('ledger.json')
              if p.parent.name != 'pilot-combined'}
    assert actual == {p.resolve() for p in runs} and len(actual) == 8
    total, sources, ledgers = 0, None, {}
    for root in runs:
        report, ledger = read(root / 'report.json'), read(root / 'ledger.json')
        assert report['status'] == ledger['status'] == 'complete'
        assert report['ledger'] == ledger
        assert ledger['pending'] is None and ledger['uncertain_transitions'] == 0
        assert ledger['charged'] == ledger['confirmed'] <= ledger['max_transitions']
        total += ledger['confirmed']
        ledgers[str(root)] = ledger['confirmed']
        current = report.get('source_files', report.get('sources'))
        if sources is None:
            sources = current
        assert current == sources and len(current) == 177
        for name, expected in current.items():
            check(name, expected)
        if 'final_checkpoint' in report:
            check(report['final_checkpoint']['path'], report['final_checkpoint']['sha256'])
        for model in report.get('models', {}).values():
            check(model['path'], model['actual_sha256'])
        for name in ('report.json', 'ledger.json'):
            evidence[str(root / name)] = digest(root / name)
    for arm in plan['arms']:
        combined = read(f'artifacts/route_history_20261010/{arm}/pilot-combined/report.json')
        execution = combined['execution']
        assert execution['processes'] == 2
        assert execution['repeated_or_discarded_controls'] == execution['interrupted_processes'] == 0
        assert {p['condition'] for p in execution['parts']} == {'course', 'recovery'}
    assert total == result['costs']['stage_through_pilot'] <= plan['max_new_controls']
    assert result['costs']['cumulative_charged'] == plan['cumulative_before']['charged'] + total
    assert result['costs']['cumulative_confirmed'] == plan['cumulative_before']['confirmed'] + total
    assert result['costs']['cumulative_charged'] - result['costs']['cumulative_confirmed'] == 64
    pretrain = read('docs/lite3_route_history_pretrain_checks_20261010.json')
    assert pretrain['unique_unit_tests'] == 56
    assert '55 passed' in Path('docs/lite3_route_history_tests_20261010.txt').read_text()
    assert '1 passed' in Path('docs/lite3_route_history_rsl_integration_test_20261010.txt').read_text()
    dataset = read('docs/lite3_route_history_dataset_audit_20261010.json')
    assert dataset['rows'] == 174282 and dataset['reset_rows'] == 152
    reproduced = Path('artifacts/route_history_20261010/dataset/reproduced-audit.json')
    assert reproduced.read_bytes() == Path('docs/lite3_route_history_dataset_audit_20261010.json').read_bytes()
    evidence[str(reproduced)] = digest(reproduced)
    links = 0
    for doc in documents:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#', 1)[0].strip('<>')
            assert (doc.parent / target).exists(), (str(doc), target)
            links += 1
    write_json_atomic(args.output, dict(
        status='verified', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        physical_runs_completed=8, physical_ledgers=ledgers, new_controls=total,
        new_uncertain=0, historical_uncertain=64, conditional_validation_started=False,
        no_completed_condition_repeated=True, runtime_files_checked=len(sources),
        runtime_sources_matching_all_completed_segments=True,
        recursive_json_evidence_files=len(visited), unique_hashed_files=len(hashes),
        trace_digest_checks=trace_checks, local_links_checked=links,
        unit_tests_passed=56, unit_tests_failed=0, dataset_rows=174282,
        dataset_audit_reproduced_byte_for_byte=True,
        figures_visually_checked=args.figures_reviewed,
        evidence=evidence))


if __name__ == '__main__':
    main()
