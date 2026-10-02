import numpy as np
import pytest
from docs.analysis.audit_flat_anchor_training import check_cost


def test_anchor_audit_rejects_rough_targets_and_wrong_cost():
    action=np.ones((2,3,12),dtype=np.float32);target=np.zeros_like(action)
    mask=np.array([[True,False,False],[True,False,False]])
    cost=mask.astype(np.float32)
    check_cost(action,target,mask,cost)
    wrong=cost.copy();wrong[0,0]=0
    with pytest.raises(AssertionError):check_cost(action,target,mask,wrong)
    target[0,1,0]=1
    with pytest.raises(AssertionError):check_cost(action,target,mask,cost)
