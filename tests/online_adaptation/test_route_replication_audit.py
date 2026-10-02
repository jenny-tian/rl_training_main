import copy
from docs.analysis.audit_route_replication import qualify


def test_independent_replication_requires_each_run_gain_retention_and_nine_of_ten():
    row=dict(candidate=5,base=0,complete=True,retention=True,safe=True)
    good={'101':[dict(row),dict(row,candidate=4)],'107':[dict(row),dict(row)]}
    assert qualify(good)
    assert not qualify({})
    for field,value in [('candidate',3),('base',4),('retention',False),('complete',False),('safe',False)]:
        bad=copy.deepcopy(good);bad['101'][1][field]=value
        assert not qualify(bad)
