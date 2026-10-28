"""Read-only evidence review; writes only a separate review JSON, never runs simulation."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

archive = Path(__file__).resolve().parent
root = archive.parents[1]
sys.path.insert(0, str(root / 'scripts/tools'))
from source_shared_report_validation import shared_report_passed
from source_parallel_terrain_startup_supervisor import find_collision_cooking_failure, find_fatal_exception

read = lambda name: json.loads((archive / name).read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest, execution, outcome, result = map(read, ['manifest.json', 'execution.json', 'outcome.json', 'result.json'])
assert manifest['revision'] == '2134d87b9db03263a6aef7335088c8ef98bc0f94'
assert manifest['check'] == 'shared_bank_g2'
assert shared_report_passed(result, reset_lifecycle=True)
assert outcome['status'] == 'diagnostic_smoke_passed' and execution['exit_code'] == 0
assert execution['code_hashes_unchanged'] and outcome['code_hashes_unchanged']
for d in (execution, outcome):
    assert all(d[k] is None for k in ('stop_reason', 'failure_marker', 'collision_cooking_failure', 'fatal_exception', 'finalization_error'))
assert all(sha(root / p) == h for p, h in manifest['source_file_sha256'].items())
assert all(sha(Path(p)) == h for p, h in manifest['dependency_file_sha256'].items())
assets = json.loads((root / 'docs/lite3-usd-collision-geometry-audit-20261027.json').read_text())['assets']
assert all(sha(root / x['path']) == x['sha256'] for x in assets.values())
logs = {}
for name in ('console.log', 'kit.log', 'isaaclab.log'):
    text = (archive / name).read_text(errors='replace')
    errors = [l for l in text.splitlines() if '[Error]' in l or 'Traceback (most recent call last)' in l]
    assert not errors and find_collision_cooking_failure(text) is None and find_fatal_exception(text) is None
    logs[name] = {'explicit_errors': errors, 'warnings': text.count('[Warning]')}
contact = [json.loads(l) for l in (archive / 'result.shared.jsonl').read_text().splitlines()]
journal = [json.loads(l) for l in (archive / 'result.shared.resets.jsonl').read_text().splitlines()]
assert contact[-1]['phase'] == 'complete'
assert sum(x.get('phase') == 'tensor_contact_sample' for x in contact) == 80
checked = [x for x in journal if x.get('phase') == 'reset_checked']
compared = [x for x in journal if x.get('phase') == 'assignment_comparison']
assert len(checked) == len(compared) == 91
assert [x['ids'] for x in checked[:-1]] == [[0], [1], [0, 1]] * 30
assert all(x['mode'] == 'direct' for x in checked[:-1])
assert checked[-1]['ids'] == [0] and checked[-1]['mode'] == 'automatic_timeout'
assert all(x['untouched_buffers_equal'] for x in checked)
assert all(x['actual_assignments'] == x['expected_assignments'] for x in compared)
tickets = journal[0]['initial_assignments'] + [t for x in checked for t in x['assignments']]
assert len(tickets) == 123 and [t['assignment_index'] for t in tickets] == list(range(123))
assert all(t['row'] == 0 and t['block_index'] == i // 120 and t['index_in_block'] == i % 120 for i, t in enumerate(tickets))
source = ast.parse((root / 'source/rl_training/rl_training/source_scan/parallel_terrain_allocation.py').read_text())
counts = ast.literal_eval(next(n.value for n in source.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCE_RESET_BLOCK_COUNTS' for t in n.targets)))
assert dict(Counter(t['subterrain'] for t in tickets[:120])) == counts
assert result['reset_checks']['blocks'][0]['counts'] == counts
assert dict(Counter(t['subterrain'] for t in tickets[120:])) == result['reset_checks']['blocks'][1]['counts']
auto, = [x for x in journal if x.get('phase') == 'automatic_reset_output']
assert auto['terminated'] == [False, False] and auto['truncated'] == [True, False]
assert journal[-1]['phase'] == 'post_automatic_reset_step' and journal[-1]['passed'] is True
assert not (archive / 'test_processes_after.txt').read_text().strip()
gpu = (archive / 'gpu_after.csv').read_text().splitlines()[0].split(', ')
assert gpu[1] not in (archive / 'gpu_compute_after.csv').read_text()
review = dict(
    reviewed_at_utc=datetime.now(timezone.utc).isoformat(), status='shared_bank_g2_2env_passed',
    runtime_revision=manifest['revision'], supervisor_exit_code=0, child_exit_code=execution['exit_code'],
    run_count_this_phase=1, automatic_retry=False, runtime_source_modified=False,
    source_hashes_verified=len(manifest['source_file_sha256']), dependency_hashes_verified=len(manifest['dependency_file_sha256']),
    assets_verified=len(assets), assignment_count=len(tickets), direct_reset_count=120,
    audited_reset_batches=len(checked), complete_blocks=1, tail_tickets=tickets[120:],
    automatic_reset_ids=[0], post_automatic_reset_step_passed=True,
    nonempty_observation_history_tested=bool(result['reset_checks']['history_buffers']),
    contact_substeps=80, contact_control_steps=20, total_control_steps=22,
    ground_contact_points=result['ground_contact_points'], cross_environment_contacts_in_initial_probe=0,
    contact_evidence_scope='initial 20 control steps only; excludes G2 final two control steps',
    elapsed_s=execution['elapsed_s'], environment_construction_seconds=result['environment_construction_seconds'],
    sampled_gpu_peak_mib=execution['sampled_gpu_peak_mib'], cpu_peak_rss_kib=result['cpu_peak_rss_kib'],
    gpu0_after=gpu, test_processes_remaining=False, log_review=logs,
    journal_phase_counts=dict(Counter(x.get('phase', 'initial') for x in journal)),
    policy_loaded=False, ppo_updates=0, full_motion_isolation_qualified=False, full_parallel_runtime_qualified=False,
)
(archive / 'post_run_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(review, ensure_ascii=False, indent=2))
