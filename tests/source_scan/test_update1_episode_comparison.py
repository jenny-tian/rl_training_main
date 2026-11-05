"""Check prefix identity and unbiased common windows without simulator access."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'source/rl_training'))
from rl_training.source_scan.update1_episode_comparison import LABELS, short_prefix_checks, three_actor_windows


def recordings():
    episodes, arrays = {}, {}
    for label, ends in zip(LABELS, ((4, 6), (6, 4), (5, 5))):
        episodes[label] = [dict(env_id=i, column=i, terrain='condition_'+str(i), steps=end, completed=True)
                           for i, end in enumerate(ends)]
        state = np.zeros((7, 2, 14), dtype=np.float32)
        state[:, :, 0] = .3
        state[:, :, 3] = .04
        state[:, :, 9] = -.4
        state[:, :, 12] = -.1
        arrays[label] = dict(state=state, rewards=np.ones((7, 2, 2)))
    return episodes, arrays


def test_three_way_window_excludes_all_unmatched_tails_and_warmup():
    episodes, arrays = recordings()
    for a in arrays.values():
        a['state'][0] = 100
        a['rewards'][0] = 100
        a['state'][4:] = 100
        a['rewards'][4:] = 100
    result = three_actor_windows(episodes, arrays, warmup=1)
    assert result['paired_transitions_per_actor'] == 6
    assert result['excluded_environment_ids'] == []
    for value in result['metrics'].values():
        assert value['transitions'] == 6
        assert value['velocity_rms_error_mps'] == pytest.approx(.2)
        assert value['body_vx_mps'] == pytest.approx(.3)
        assert value['reward_per_control'] == pytest.approx(.04)
        assert value['lateral_rms_m'] == pytest.approx(.4)
        assert value['yaw_rms_deg'] == pytest.approx(np.degrees(.1))
    assert result['pairs'][0]['end_control'] == 4


def test_common_window_includes_terminal_sample():
    episodes, arrays = recordings()
    arrays['update1']['state'][3, 0, 9] = -1
    result = three_actor_windows(episodes, arrays, warmup=1)
    assert result['pairs'][0]['endpoints']['update1']['lateral_m'] == -1
    assert result['pairs'][0]['metrics']['update1']['mean_abs_lateral_m'] == pytest.approx(.6)


def test_assignment_mismatch_is_rejected():
    episodes, arrays = recordings()
    episodes['update2'][0]['column'] = 10
    with pytest.raises(ValueError, match='assignments differ'):
        three_actor_windows(episodes, arrays, warmup=1)


def test_short_environment_is_reported_without_using_reset_data():
    episodes, arrays = recordings()
    episodes['update1'][0]['steps'] = 1
    result = three_actor_windows(episodes, arrays, warmup=1)
    assert result['excluded_environment_ids'] == [0]
    assert result['paired_transitions_per_actor'] == 3
    episodes['update1'][1]['steps'] = 1
    with pytest.raises(ValueError, match='no common samples'):
        three_actor_windows(episodes, arrays, warmup=1)


def test_short_prefix_compares_registered_fields_exactly():
    a = dict(state=np.zeros((402, 2, 14)), rewards=np.zeros((402, 2, 3)),
             columns=np.zeros((402, 2)), done=np.zeros((402, 2), bool))
    ref = {key: value[:400].copy() for key, value in a.items()}
    ref['state'] = ref['state'][:, :, :9].copy()
    a['state'][:, :, 9:] = 100
    a['state'][400:] = 100
    assert all(short_prefix_checks(a, ref).values())
    a['state'][399, 1, 8] = 1e-9
    assert short_prefix_checks(a, ref)['state'] is False
    shortened = deepcopy(ref)
    shortened['done'] = shortened['done'][:-1]
    with pytest.raises(ValueError, match='complete registered short'):
        short_prefix_checks(a, shortened)
