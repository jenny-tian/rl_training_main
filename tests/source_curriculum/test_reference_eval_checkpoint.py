"""Prevent accidental checkpoint substitution in frozen comparisons."""
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_eval_checkpoint import register_checkpoint
from reference_stairs_eval import MODELS
from reference_core500_analyze import paired_turn_rows,summarize_stairs


def test_custom_checkpoint_requires_its_recorded_hash(tmp_path):
    path=tmp_path/'policy.pt'; path.write_bytes(b'test checkpoint')
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    assert register_checkpoint('core10500_test',path,expected)==expected
    assert MODELS.pop('core10500_test')==path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        register_checkpoint('core10500_test',path,expected)


@pytest.mark.parametrize('name,path,digest',[
    ('update10000','arbitrary.pt','0'*64),('reference','arbitrary.pt','0'*64),
    ('custom',None,None),('custom','arbitrary.pt',None),('../label',None,None),
])
def test_bad_or_ambiguous_checkpoint_selection_is_rejected(name,path,digest):
    with pytest.raises(ValueError): register_checkpoint(name,path,digest)


def test_pairing_does_not_count_failed_windows_as_good_tracking():
    old=[dict(case='turn_left',window='steady',repeat=i,complete_window=True,yaw_actual=.2,yaw_mae=.3) for i in range(3)]
    new=[dict(case='turn_left',window='steady',repeat=i,complete_window=i!=1,yaw_actual=.4,yaw_mae=.1 if i!=1 else 0.) for i in range(3)]
    row=paired_turn_rows(old,new,'nominal')[0]
    assert row['n']==2 and row['improved']==2 and row['worsened']==0
    assert row['mae_delta']==pytest.approx(-.2)


def test_stair_summary_keeps_first_step_separate_from_complete_success():
    records=[]
    for height in (0.,.05,.1,.15):
        records.extend([dict(height=height,best_risers=1,outcome='orientation_failure'),
                        dict(height=height,best_risers=0,outcome='stuck')])
    rows=summarize_stairs(dict(records=records),'candidate','nominal','up')
    assert all(r['starts']==2 and r['first_riser']==1 and r['success']==0 for r in rows)
    assert all(r['orientation_failure']==1 for r in rows)
