"""Combine disjoint completed conditions after a documented setup interruption.

Original raw reports/ledgers are immutable. This output explicitly represents
two processes; it never marks the original interrupted process complete.
"""
import argparse
import copy
import json
import os
from pathlib import Path

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('addendum', 'original-run', 'continuation-run', 'output', 'audit'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    evidence = {}
    def read(path):
        evidence[str(path)] = sha256_file(path)
        return json.loads(Path(path).read_text())
    addendum = read(args.addendum)
    protocol = read(addendum['original_protocol'])
    continuation = read(addendum['continuation_protocol'])
    interruption = read(addendum['interruption_audit'])
    for key in ('original_protocol', 'continuation_protocol', 'interruption_audit'):
        assert sha256_file(addendum[key]) == addendum[key+'_sha256']
    assert interruption['status'] == 'verified_interrupted_run' and interruption['uncertain'] == 0
    for path, digest in interruption['evidence'].items():
        assert sha256_file(path) == digest
        evidence[path] = digest
    ignored = {'conditions', 'max_controls', 'registered_at_utc', 'scope'}
    assert set(protocol) == set(continuation)
    assert all(protocol[k] == continuation[k] for k in set(protocol)-ignored)
    assert protocol['conditions'] == ['course', 'recovery'] and continuation['conditions'] == ['recovery']
    assert continuation['max_controls'] == addendum['remaining_condition_cap'] == 128000
    oldroot, newroot = Path(args.original_run), Path(args.continuation_run)
    old, new = read(oldroot/'report.json'), read(newroot/'report.json')
    oldledger, newledger = read(oldroot/'ledger.json'), read(newroot/'ledger.json')
    assert old['status'] == 'running' and new['status'] == newledger['status'] == 'complete'
    assert old['protocol_sha256'] == addendum['original_protocol_sha256']
    assert new['protocol_sha256'] == addendum['continuation_protocol_sha256']
    assert old['sources'] == new['sources'] and old['models'] == new['models']
    for path, digest in new['sources'].items():
        assert sha256_file(path) == digest
    assert set(old['phases']) == {'course-trial'} and set(new['phases']) == {'recovery-trial'}
    assert not (oldroot/'recovery-trial').exists()
    assert oldledger['charged'] == oldledger['confirmed'] == interruption['confirmed'] == addendum['completed_controls']
    for ledger in (oldledger, newledger):
        assert ledger['pending'] is None and ledger['charged'] == ledger['confirmed']
    total = oldledger['confirmed'] + newledger['confirmed']
    assert total <= addendum['maximum_combined_controls'] <= protocol['max_controls']
    phases = {}
    for root, report, ledger in ((oldroot, old, oldledger), (newroot, new, newledger)):
        for name, phase in report['phases'].items():
            assert phase['status'] == 'complete' and phase['confirmed_controls'] == ledger['confirmed']
            assert read(root/name/'report.json') == phase
            manifest = read(root/name/'trajectory/manifest.json')
            assert manifest['status'] == 'complete'
            assert manifest['steps']*protocol['num_envs'] == phase['confirmed_controls']
            phases[name] = copy.deepcopy(phase)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for name, root in [('course-trial', oldroot), ('recovery-trial', newroot)]:
        (out/name).symlink_to(os.path.relpath((root/name).resolve(), out.resolve()), target_is_directory=True)
    combined = dict(schema='lite3_assembled_evaluation_v1', status='complete',
        protocol_sha256=addendum['original_protocol_sha256'], sources=new['sources'], models=new['models'], phases=phases,
        ledger=dict(schema='lite3_vector_step_ledger_v1', status='complete', num_envs=protocol['num_envs'],
            charged=total, confirmed=total, pending=None, uncertain_transitions=0, max_transitions=protocol['max_controls'],
            metadata=dict(assembled_from_disjoint_completed_conditions=True)),
        execution=dict(processes=2, original_run_status='interrupted_after_course', original_raw_records_preserved=True,
            ordinary_controls_reused=oldledger['confirmed'], continued_recovery_controls=newledger['confirmed'],
            repeated_or_discarded_controls=0, original_run=str(oldroot), continuation_run=str(newroot)),
        evidence=evidence,
        scope='Explicit derived report for disjoint ordinary and recovery conditions in separate processes after scene-setup interruption; same model/seed/protocol fields, no repeated completed condition or additional budget.')
    write_json_atomic(out/'report.json', combined)
    write_json_atomic(out/'ledger.json', combined['ledger'])
    write_json_atomic(args.audit, dict(status='verified', controls=total, process_count=2,
        complete_conditions=['course-trial', 'recovery-trial'], extra_physics=0, uncertain=0,
        original_raw_records_preserved=True, evidence=dict(evidence, **{str(out/'report.json'): sha256_file(out/'report.json')})))
    print(json.dumps(combined['execution'], indent=2))


if __name__ == '__main__':
    main()
