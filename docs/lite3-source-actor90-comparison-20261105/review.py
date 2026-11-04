"""Verify archived failed-run facts without starting Isaac Sim or allocating CUDA."""
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(folder):
    folder = folder.resolve()
    checked = 0
    for line in (folder / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (folder / name).resolve()
        assert path.is_relative_to(folder) and digest(path) == expected, name
        checked += 1
    load = lambda name: json.loads((folder / name).read_text())
    for item in load('raw/runtime_artifacts.json')['files']:
        path = folder / item['repository_path']
        assert path.stat().st_size == item['size_bytes'] and digest(path) == item['sha256']
    summary = load('deliverables/status.json')
    execution = load('raw/runtime/execution.json')
    assert not load('raw/runtime/acceptance.json')['accepted']
    assert not load('raw/runtime/receipt.json')['accepted']
    assert execution['process_group_empty'] and execution['returncode'] == 0
    assert load('raw/compare.supervisor_receipt.json')['returncode'] == 1
    assert not (folder / 'raw/runtime/result.cleanup.json').exists()
    rows = [json.loads(line) for line in (folder / 'raw/runtime/result.steps.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(1, 2001))
    assert all(row['actor'] == 'old' and row['actor_step'] == row['step'] and row['finite'] is True for row in rows)
    with (folder / 'deliverables/execution_steps.csv').open(newline='') as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == len(rows)
    assert all(int(out['step']) == row['step'] and out['actor'] == row['actor'] for out, row in zip(exported, rows))
    assert load('raw/runtime/result.json')['actors'] == []
    assert load('raw/runtime/agent_stop.json')['stalled_seconds_before_stop'] > 120
    before, after = load('raw/slots_before.json')['jobs'], load('raw/slots_after.json')['jobs']
    assert set(after) - set(before) == {'compare'} and all(after[key] == value for key, value in before.items())
    assert digest(folder / 'raw/budget_before.json') == digest(folder / 'raw/budget_after.json') == summary['budget_sha256']
    assert digest(folder / 'raw/numeric_acceptance_before.json') == digest(folder / 'raw/numeric_acceptance_after.json')
    assert load('raw/numeric_acceptance_after.json')['accepted'] is False
    assert summary['new_training_transitions'] == summary['ppo_updates'] == summary['new_actor_controls'] == 0
    assert summary['recorded_diagnostic_transitions'] == 120 * len(rows)
    assert summary['paired_comparison_available'] is False and summary['automatic_retry'] is False
    return dict(archive_integrity_verified=True,hashed_files=checked,diagnostic_accepted=False,
        old_actor_controls=len(rows),new_actor_controls=0,paired_comparison_available=False,
        new_training_transitions=0,budget_unchanged=True,process_group_empty=True,
        formal_selection_qualified=False)


if __name__ == '__main__':
    print(json.dumps(main(Path(__file__).resolve().parent), indent=2))
