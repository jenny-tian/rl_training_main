import copy
from docs.analysis.audit_route_transfer import qualify


def test_all_registered_terrain_and_each_independent_policy_must_pass():
    row=dict(candidate=5,base=5,complete=True,safe=True,retention=True)
    good={str(t):[dict(row),dict(row,candidate=4)] for t in range(4)}
    assert qualify(good)
    for field,value in [('candidate',3),('complete',False),('safe',False),('retention',False)]:
        bad=copy.deepcopy(good);bad['2'][1][field]=value;assert not qualify(bad)
    assert not qualify({k:v for k,v in good.items() if k!='3'})
