"""统计口径：首次支撑未得分不等于整回合未得分，资格通过也不等于发奖。"""
import numpy as np

from reference_low_stairs_reward_state_analyze import supported_levels,pending_events
from reference_low_stairs_reward_observer import LANDING_RULES


def case():
    shape=(12,1,4);zeros=lambda:np.zeros(shape,np.float32)
    d={k:zeros() for k in ('ground','land_airtime_at_check','land_displacement','land_body_progress','land_peak_at_check',
                          'land_before_pending_z','land_before_dwell')}
    for k in ('support','wall','land_awarded','land_raw_touch','land_qualifies','land_after_pending'):
        d[k]=np.zeros(shape,bool)
    for k in ('upright','enabled'):d[k]=np.ones((12,1),bool)
    for k in LANDING_RULES:d['land_rule_'+k]=np.ones(shape,bool)
    d['support'][:,:,1:]=True
    r=dict(env_id=0,height=.05,speed=.32,outcome='success')
    return r,d,{'active':np.ones((12,1),bool)}


def test_later_credit_distinct_from_first_support_span():
    r,d,tr=case()
    d['ground'][:,0,0]=.05
    d['support'][2:5,0,0]=True;d['support'][8:,0,0]=True
    d['land_raw_touch'][[2,8],0,0]=True;d['land_awarded'][10,0,0]=True
    rows=supported_levels(r,d,tr,'test',0.)
    assert len(rows)==1
    assert not rows[0]['credited_by_support_end'] and rows[0]['credited_by_episode_end']
    assert rows[0]['raw_touch_in_first_span'] and rows[0]['raw_touch_lag_from_support_begin_s']==0.


def test_qualification_cancelled_when_other_supports_disappear():
    r,d,tr=case()
    d['support'][:,0,0]=True;d['support'][4,0,1:]=False
    d['land_qualifies'][2,0,0]=True;d['land_after_pending'][2:4,0,0]=True
    d['land_before_dwell'][4,0,0]=.04
    rows=pending_events(r,d,tr,'test')
    assert len(rows)==1 and rows[0]['result']=='cancelled'
    assert rows[0]['reasons']=='fewer_than_two_supports'
    assert rows[0]['touch_s']==.06 and rows[0]['end_s']==.1
