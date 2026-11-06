import numpy as np
import pytest

from rl_training.ea_lpl.episodes import episode_records


def synthetic(done_steps):
    steps, n = 10, len(done_steps)
    done = np.zeros((steps, n), dtype=bool)
    events, assignments = [], []
    for env, ends in enumerate(done_steps):
        start = 0
        for end in ends:
            done[end - 1, env] = True
            events.append(dict(env_id=env, control_step=end, episode_step=end - start, hold=0,
                               completed=False, terms=["source_terrain_tile_bounds"]))
            start = end
        assignments.append(dict(env_id=env, column=0, subterrain="flat"))
    state = np.zeros((steps, n, 15))
    state[:, :, 5] = np.arange(steps)[:, None]
    arrays = dict(done=done, state=state, rewards=np.zeros((steps, n, 3)), actions=np.zeros((steps, n, 5)),
                  columns=np.zeros((steps, n), dtype=np.int64))
    return arrays, events, assignments


def test_records_split_by_done_flags():
    arrays, events, assignments = synthetic([[4, 9], [6, 10]])
    records = episode_records(arrays, events, assignments, episodes_per_env=2)
    assert [(r["env_id"], r["episode_index"], r["steps"]) for r in records] == [(0, 0, 4), (0, 1, 5), (1, 0, 6), (1, 1, 4)]
    assert records[1]["forward_progress_m"] == 8.0
    assert records[0]["bounds_ending"] and not records[0]["completed"]


def test_missing_episode_or_shifted_event_is_rejected():
    arrays, events, assignments = synthetic([[4, 9], [6]])
    with pytest.raises(ValueError):
        episode_records(arrays, events, assignments, episodes_per_env=2)
    events[0] = dict(events[0], control_step=3)
    with pytest.raises(ValueError):
        episode_records(arrays, events, assignments, episodes_per_env=1)
