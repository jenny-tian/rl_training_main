"""Paired evaluation summaries count episodes, keeping session identity."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_online_c1b_results import paired_summary


def rows():
    outcomes=[('success','lateral_exit'),('timeout','success'),('posture_failure','success'),('success','success')]
    return [dict(session=i//2,repeat=i%2,condition='stairs_06',arm=arm,outcome=outcome,
                 mean_progress=.5,policy_steps=100) for i,pair in enumerate(outcomes)
            for arm,outcome in zip(('Frozen','Updated'),pair)]


def test_pairs_and_percentage_points_keep_same_repeat_in_different_sessions():
    result=paired_summary(rows())
    assert result['Frozen']['successes']==2 and result['Updated']['successes']==3
    assert result['success_delta']==.25 and result['posture_delta']==-.25
    assert result['gained_starts']==2 and result['lost_starts']==1
    assert result['Frozen']['one_episode_resolution']==.25


def test_missing_pair_rejected():
    with pytest.raises(AssertionError): paired_summary(rows()[:-1])


def test_duplicate_pair_rejected():
    records=rows()
    with pytest.raises(AssertionError,match='duplicate'): paired_summary(records+[records[0]])
