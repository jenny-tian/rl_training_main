from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_stair_allocation_complete import schedule, execute_once


def previous():
    return dict(status='stopped_regression',matrix=[dict(variant=v,name='heading_up_nominal',episodes=240) for v in ('A','B')])


def test_completion_preserves_exact_remaining_budget_without_repeating_nominal_up():
    rows=schedule(previous())
    assert len(rows)==18 and len({(r['variant'],r['name']) for r in rows})==18
    assert all(r['name']!='heading_up_nominal' for r in rows)
    assert sum(r['episodes'] for r in rows)+960==5600
    assert sum(r['episodes'] for r in rows if r['kind']=='turn')==1280


def test_completion_rejects_unexpected_prior_matrix():
    p=previous();p['matrix'].pop()
    with pytest.raises(AssertionError):schedule(p)


def test_incomplete_existing_job_cannot_launch_again(tmp_path):
    with pytest.raises(RuntimeError,match='no automatic retry'):
        execute_once('unused.py',[],tmp_path)
