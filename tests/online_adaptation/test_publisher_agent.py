import torch

from rl_training.online_adaptation.agent import OnlineAdaptationAgent
from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher


def test_publisher_swaps_complete_versions():
    actor = torch.nn.Linear(2, 1, bias=False)
    publisher = DoubleBufferedActorPublisher(actor)
    updated = torch.nn.Linear(2, 1, bias=False)
    updated.weight.data.fill_(3.0)
    publisher.publish(updated, version=4)
    published, version = publisher.current()
    assert version == 4
    assert torch.equal(published.weight, updated.weight)


def test_agent_does_not_request_update_before_minimum_replay():
    assert not OnlineAdaptationAgent.should_update(replay_size=255, min_replay_size=256, paused=False)
    assert OnlineAdaptationAgent.should_update(replay_size=256, min_replay_size=256, paused=False)
    assert not OnlineAdaptationAgent.should_update(replay_size=256, min_replay_size=256, paused=True)


def test_vectorized_learning_pauses_only_when_all_environments_are_paused():
    assert not OnlineAdaptationAgent.all_environments_paused(torch.tensor([[False], [True]]))
    assert OnlineAdaptationAgent.all_environments_paused(torch.tensor([[True], [True]]))
