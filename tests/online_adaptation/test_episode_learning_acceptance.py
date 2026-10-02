from copy import deepcopy
from docs.analysis.audit_episode_learning import replication_gate


def records():
    good={'complete':True,'retention_pass':True,'evaluation_safe':True,'uncertain_steps':0,
          'candidate_completions':4,'base_completions':3}
    return {'101':deepcopy(good),'107':deepcopy(good)}


def test_second_seed_requires_four_successes_on_both_terrains_and_learning_gain():
    t=records();assert replication_gate(t)
    t['101']['candidate_completions']=3;assert not replication_gate(t)
    t=records()
    for s in t.values():s['base_completions']=s['candidate_completions']
    assert not replication_gate(t)


def test_missing_unsafe_incomplete_or_retention_failure_cannot_open_gate():
    for key,value in [('complete',False),('retention_pass',False),('evaluation_safe',False),('uncertain_steps',1)]:
        t=records();t['107'][key]=value;assert not replication_gate(t)
    t=records();del t['107'];assert not replication_gate(t)
