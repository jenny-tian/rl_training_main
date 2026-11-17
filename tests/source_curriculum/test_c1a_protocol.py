from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/tools'))
from reference_online_c1a import MARKS,jobs,protocol,rows
from reference_online_c1a_collect import TraceWriter
from reference_online_c1a_env import current_protocol
from reference_online_c1a_results import paired_summary
import reference_online_c1b_env_repaired as legacy_environment
import reference_online_c1b_env_reset_api as legacy_reset


def test_v2_session_counts_and_all_budgets_include_physics_steps():
    schedule = jobs(); budgets = protocol()['budgets']
    assert len(rows('train'))==120 and protocol()['pairs']==60
    assert len(schedule)==47 and len({j['name'] for j in schedule})==47
    for kind in ('smoke','train','target','retention'):
        assert sum(j['reserved_steps'] for j in schedule if j['kind']==kind)==budgets[kind]
    assert sum(j['reserved_steps'] for j in schedule)==budgets['total']==12888000
    assert max(j['environments'] for j in schedule)==120
    assert protocol()['primary_mark']==20000 and MARKS[-1]==60000


@pytest.mark.parametrize('kind,parts,repeats',[('target',4,4),('retention',20,20)])
def test_partitions_cover_every_pair_once_per_start_with_adjacent_arms(kind,parts,repeats):
    all_rows = [r for part in range(parts) for r in rows(kind,part=part)]
    keys = [(r['session'],r['condition'],r['repeat'],r['arm']) for r in all_rows]
    assert len(set(keys))==len(keys)==60*2*repeats
    for session in range(60): assert sum(r['session']==session for r in all_rows)==2*repeats
    for part in range(parts):
        part_rows = rows(kind,part=part)
        for a,b in zip(part_rows[::2],part_rows[1::2]):
            assert a['arm']=='Frozen' and b['arm']=='Updated' and a['slot']==b['slot']
            assert {k:v for k,v in a.items() if k not in ('arm','env_id')}=={k:v for k,v in b.items() if k not in ('arm','env_id')}


def test_new_namespaces_and_smoke_keys_do_not_overlap_previous_experiment():
    from reference_online_c1b import protocol as previous
    assert not set(protocol()['namespaces'].values()) & set(previous()['namespaces'].values())
    first = {r['slot'] for r in rows('smoke',smoke_index=1)}
    second = {r['slot'] for r in rows('smoke',smoke_index=2)}
    assert not first & second
    with pytest.raises(ValueError): rows('target',part=4)


def test_reused_environment_protocol_is_restored_on_failure():
    originals = legacy_environment.protocol,legacy_reset.protocol
    with pytest.raises(RuntimeError):
        with current_protocol():
            assert legacy_environment.protocol()['algorithm']=='C1-a'
            assert legacy_reset.protocol()['namespaces']['train']==31
            raise RuntimeError('synthetic initialization failure')
    assert (legacy_environment.protocol,legacy_reset.protocol)==originals


def test_trace_chunks_preserve_boundary_samples_and_reject_nonfinite(tmp_path):
    writer = TraceWriter(tmp_path)
    for t in range(1001): writer.append(dict(phase=np.array([t%3]),code=np.array([[t,0.]],float)))
    writer.flush()
    assert [(c['start'],c['stop']) for c in writer.chunks]==[(0,1000),(1000,1001)]
    with np.load(tmp_path/writer.chunks[1]['file']) as archive: assert archive['code'][0,0,0]==1000
    with pytest.raises(ValueError): writer.append(dict(phase=np.array([1]),code=np.array([[np.nan,0.]])))
    assert writer.steps==1001


def test_statistics_use_session_pairing_and_reject_duplicates():
    records = []
    for session in range(3):
        for repeat in range(4):
            for arm in ('Frozen','Updated'):
                records.append(dict(session=session,repeat=repeat,condition='stairs_06',arm=arm,
                    outcome='success' if repeat<(1 if arm=='Frozen' else 2) else 'timeout',
                    xy_velocity_error_sum=1.,policy_steps=100))
    result = paired_summary(records,pairs=3)
    assert result['session_deltas']==[.25]*3 and result['ci95']==[.25,.25]
    with pytest.raises(ValueError): paired_summary(records+[records[0]],pairs=3)
    with pytest.raises(ValueError): paired_summary(records[:-1],pairs=3)
    with pytest.raises(ValueError): paired_summary(records[:-8],pairs=3)
