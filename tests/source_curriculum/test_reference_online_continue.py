from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/tools'))
from reference_online_continue import validate_budget


def test_prior_failed_smoke_still_consumes_one_of_four_attempts():
    attempts = [dict(kind='smoke', episodes=12, output=f'run{i}', inherited=i == 0) for i in range(3)]
    validate_budget(attempts, 'smoke', 12, Path('fourth'))
    attempts.append(dict(kind='smoke', episodes=12, output='fourth'))
    with pytest.raises(RuntimeError, match='four'):
        validate_budget(attempts, 'smoke', 12, Path('fifth'))


@pytest.mark.parametrize('kind,limit,count', [('baseline', 384, 192), ('grid', 1200, 600), ('confirmation', 64, 32)])
def test_original_formal_budget_cannot_expand(kind, limit, count):
    attempts = [dict(kind=kind, episodes=limit-count, output='first')]
    validate_budget(attempts, kind, count, Path('second'))
    with pytest.raises(RuntimeError, match='exceeded'):
        validate_budget(attempts, kind, count+1, Path('second'))


def test_same_output_cannot_be_retried():
    with pytest.raises(RuntimeError, match='already reserved'):
        validate_budget([dict(kind='smoke', episodes=12, output='used')], 'smoke', 12, Path('used'))
