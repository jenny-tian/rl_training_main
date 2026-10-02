import copy
import json

import numpy as np
import pytest

from rl_training.online_adaptation.episode_search import EpisodeSearch


def results(pending, reasons=None):
    reasons = reasons or ['completion', 'physical_failure', 'task_timeout', 'lateral_violation']
    return [{'id': p['id'], 'terminal_reason': r, 'normalized_progress': v,
             'safety_interventions': 0, 'charged_steps': 1000}
            for p,r,v in zip(pending, reasons, [.7,.9,.8,.6])]


def test_resume_preserves_pending_and_next_generation_without_extra_rng():
    original = EpisodeSearch(31)
    pending = original.ask()
    saved = json.loads(json.dumps(original.state_dict()))
    restored = EpisodeSearch.from_state_dict(saved)
    assert original.ask() == restored.ask() == pending
    assert pending[0]['code'] == [0., 0.]
    for search in (original, restored):
        search.tell(results(pending))
    assert original.ask() == restored.ask()
    assert original.state_dict() == restored.state_dict()


def test_completion_outranks_progress_and_variance_has_floor():
    s = EpisodeSearch(7)
    pending = s.ask()
    s.tell(list(reversed(results(pending))))
    assert s.history[0]['elite_ids'] == ['0:0', '0:1']
    np.testing.assert_allclose(s.mean, np.mean([pending[0]['code'], pending[1]['code']], axis=0))
    assert (s.std >= .1).all()


def test_truncated_and_aborted_attempts_retained_without_becoming_elites():
    s = EpisodeSearch(1)
    p = s.ask()
    r = results(p, ['collector_cutoff', 'interrupted', 'physical_failure', 'lateral_violation'])
    s.tell(r)
    assert s.history[0]['elite_ids'] == ['0:2', '0:3']
    assert s.history[0]['results'] == r
    assert sum(x['charged_steps'] for x in s.history[0]['results']) == 4000


def test_no_completed_pair_does_not_move_search_distribution():
    s = EpisodeSearch(2)
    p = s.ask()
    s.tell(results(p, ['interrupted']*4))
    assert s.mean.tolist() == [0., 0.]
    assert s.std.tolist() == [.3, .3]
    assert s.generation == 1 and s.history[0]['elite_ids'] == []


@pytest.mark.parametrize('corrupt', ['duplicate', 'partial', 'nan', 'no_cost'])
def test_invalid_tell_is_atomic(corrupt):
    s = EpisodeSearch(9)
    r = results(s.ask())
    before = s.state_dict()
    if corrupt == 'duplicate': r[1]['id'] = r[0]['id']
    elif corrupt == 'partial': r.pop()
    elif corrupt == 'nan': r[0]['normalized_progress'] = float('nan')
    else: r[0]['charged_steps'] = 0
    with pytest.raises(ValueError): s.tell(r)
    assert s.state_dict() == before


def test_external_pending_and_saved_objects_cannot_mutate_search():
    s = EpisodeSearch(8)
    pending = s.ask()
    before = s.state_dict()
    pending[0]['code'][0] = 99
    saved = s.state_dict()
    saved['mean'][0] = 99
    assert s.state_dict() == before


@pytest.mark.parametrize('key,value', [('mean',[2.,0.]), ('std',[0.,.3]), ('generation',-1)])
def test_corrupt_checkpoint_rejected(key, value):
    state = EpisodeSearch(4).state_dict()
    state[key] = value
    with pytest.raises(ValueError): EpisodeSearch.from_state_dict(state)


def test_history_and_distribution_cannot_disagree_on_resume():
    s = EpisodeSearch(13)
    s.tell(results(s.ask()))
    state = s.state_dict()
    state['history'][0]['elite_ids'] = ['0:2', '0:3']
    with pytest.raises(ValueError, match='history'):
        EpisodeSearch.from_state_dict(state)
    state = s.state_dict()
    state['mean'][0] += .01
    with pytest.raises(ValueError, match='distribution'):
        EpisodeSearch.from_state_dict(state)
