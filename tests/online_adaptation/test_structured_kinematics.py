import math
import numpy as np
import pytest
pytest.importorskip('pxr.Usd')
from docs.analysis.audit_structured_kinematics import forward


def test_forward_rotation_convention_with_known_quarter_turn():
    joints=[]
    for i,leg in enumerate(('FL','FR','HL','HR')):
        parent,child=np.eye(4),np.eye(4)
        parent[0,3]=i
        child[0,3]=-1
        joints.append({'name':leg,'parent':'/Lite3/TORSO','child':f'/Lite3/{leg}_FOOT',
                       'revolute':True,'axis':'Z','parent_anchor':parent.tolist(),'child_anchor':child.tolist()})
    feet=forward(joints,{leg:math.pi/2 for leg in ('FL','FR','HL','HR')})
    for i,leg in enumerate(('FL','FR','HL','HR')):
        np.testing.assert_allclose(feet[leg],[i,1,0],atol=1e-12)


def test_disconnected_graph_is_rejected():
    with pytest.raises(ValueError,match='disconnected'):
        forward([{'parent':'missing','child':'also_missing'}],{})
