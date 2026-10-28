"""Compare the failed diagnostic and current repair using CPU-only fixtures."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'source/rl_training'))
spec = importlib.util.spec_from_file_location('lifecycle_regression_fixture', root / 'tests/source_scan/test_source_parallel_reset_lifecycle.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
old = types.ModuleType('failed_g2_diagnostic')
exec(compile(subprocess.check_output([
    'git', 'show', '7918d00:scripts/tools/source_parallel_reset_lifecycle.py'
], cwd=root, text=True), '<7918d00 diagnostic>', 'exec'), old.__dict__)
with tempfile.TemporaryDirectory(prefix='g2_offline_comparison_') as temporary:
    directory = Path(temporary)
    before = tests._lifecycle_env()
    try:
        old.run_lifecycle(before, 130001, directory / 'old.jsonl', lambda _: None)
    except AssertionError as error:
        assert str(error) == 'runtime assignment differs from deterministic replay'
        old_error = str(error)
    else:
        raise AssertionError('the regression fixture did not detect the original bug')
    after = tests._lifecycle_env()
    report = tests.m.run_lifecycle(after, 130001, directory / 'new.jsonl', lambda _: None)
    assert report['assignment_count'] == 363
    print(json.dumps({
        'simulator_started': False,
        'baseline_revision': '7918d00',
        'baseline_error': old_error,
        'baseline_attempted_batches': len(before.reset_calls),
        'repaired_completed_batches': report['reset_batch_count'],
        'repaired_assignment_count': report['assignment_count'],
        'repaired_complete_blocks': report['complete_blocks'],
        'repaired_auto_reset_ids': report['automatic_reset_ids'],
        'runtime_qualification_claim': False,
    }, indent=2))
