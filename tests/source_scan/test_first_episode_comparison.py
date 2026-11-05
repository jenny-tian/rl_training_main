"""Terminal/reset and paired-horizon checks using small recorded-array fixtures."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'source/rl_training'))
from rl_training.source_scan.first_episode_comparison import first_episodes, compare_episodes

GATES = dict(max_paired_velocity_rms_error_ratio=1.5,min_paired_reward_ratio=0.8,max_clipped_joint_fraction=0.01)


def fixture(ends=(3,5), completed=(True,False)):
    length = 6
    n = len(ends)
    arrays = dict(done=np.zeros((length,n),bool),state=np.zeros((length,n,14),np.float32),
        rewards=np.ones((length,n,1),np.float32),actions=np.zeros((length,n,5),np.float32),
        columns=np.tile(np.arange(n),(length,1)))
    arrays['state'][:,:,0] = 0.3
    arrays['state'][:,:,3] = 0.04
    assignments = [dict(env_id=i,column=i,subterrain='terrain_'+str(i)) for i in range(n)]
    events = []
    for i,end in enumerate(ends):
        arrays['done'][end-1,i] = True
        arrays['state'][end-1,i,5] = 9.2 if completed[i] else 4.0
        events.append(dict(env_id=i,control_step=end,episode_step=end,hold=50 if completed[i] else 0,
            completed=completed[i],terms=['source_training_completion' if completed[i] else 'source_terrain_tile_bounds']))
    return arrays,events,assignments


def test_first_episode_keeps_terminal_sample_and_ignores_later_success():
    arrays,events,assignments = fixture()
    arrays['state'][3:,0,3] = 10000
    arrays['rewards'][3:,0] = 10000
    arrays['columns'][3:,0] = 10
    arrays['done'][5,0] = True
    events.append(dict(env_id=0,control_step=6,episode_step=3,hold=50,completed=True,terms=['source_training_completion']))
    result = first_episodes(arrays,events,assignments)
    assert len(result) == 2 and result[0]['steps'] == 3
    assert result[0]['forward_progress_m'] == pytest.approx(9.2)
    assert result[0]['velocity_rms_error_mps'] == pytest.approx(0.2)
    assert result[0]['reward_per_control'] == pytest.approx(0.02)
    assert result[1]['completed'] is False and result[1]['bounds_ending'] is True


@pytest.mark.parametrize('fault',['missing','shifted','hold','column','unfinished'])
def test_invalid_first_episode_evidence_is_rejected(fault):
    arrays,events,assignments = fixture()
    if fault == 'missing': events.pop(0)
    if fault == 'shifted': events[0]['episode_step'] -= 1
    if fault == 'hold': events[0]['hold'] = 49
    if fault == 'column': arrays['columns'][1,0] = 99
    if fault == 'unfinished': arrays['done'][:,1] = False
    with pytest.raises(ValueError):
        first_episodes(arrays,events,assignments)


def test_paired_window_excludes_warmup_and_both_unpaired_tails():
    a,ae,assign = fixture(ends=(3,5),completed=(True,True))
    b,be,_ = fixture(ends=(5,3),completed=(True,True))
    a['state'][0,:,3] = b['state'][0,:,3] = 100
    a['state'][3:,1,3] = b['state'][3:,0,3] = 100
    a['rewards'][0] = b['rewards'][0] = 100
    a['rewards'][3:,1] = b['rewards'][3:,0] = 100
    result = compare_episodes(first_episodes(a,ae,assign),first_episodes(b,be,assign),a,b,GATES,warmup=1)
    assert result['paired_controls'] == 4
    for metrics in result['paired_metrics'].values():
        assert metrics['velocity_rms_error_mps'] == pytest.approx(0.2)
        assert metrics['reward_per_control'] == pytest.approx(0.02)
    assert result['passed'] is True


def test_equal_total_completion_cannot_hide_a_condition_regression():
    a,ae,assign = fixture(completed=(True,False))
    b,be,_ = fixture(completed=(False,True))
    result = compare_episodes(first_episodes(a,ae,assign),first_episodes(b,be,assign),a,b,GATES,warmup=1)
    assert result['checks']['total_completion'] is True
    assert result['checks']['condition_completion'] is False
    assert result['passed'] is False


def test_short_pair_is_reported_as_excluded_without_reset_data():
    a,ae,assign = fixture(ends=(1,5))
    episodes = first_episodes(a,ae,assign)
    result = compare_episodes(episodes,deepcopy(episodes),a,a,GATES,warmup=1)
    assert result['excluded_environment_ids'] == [0]
    assert result['paired_controls'] == 4
    with pytest.raises(ValueError,match='no paired samples'):
        compare_episodes(episodes,episodes,a,a,GATES,warmup=5)


def test_compressed_archive_reads_do_not_scale_with_environment_count(tmp_path,monkeypatch):
    arrays,events,assign = fixture(ends=(3,)*20,completed=(True,)*20)
    path = tmp_path/'recording.npz'
    np.savez_compressed(path,**arrays)
    reads = []
    original = np.lib.npyio.NpzFile.__getitem__
    def counted(self,key):
        reads.append(key)
        return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',counted)
    expected = first_episodes(arrays,events,assign)
    with np.load(path) as archived:
        actual = first_episodes(archived,events,assign)
        assert actual == expected
        assert len(reads) <= 5
        reads.clear()
        compared = compare_episodes(actual,actual,archived,archived,GATES,warmup=1)
        assert compared == compare_episodes(actual,actual,arrays,arrays,GATES,warmup=1)
        assert len(reads) <= 6
