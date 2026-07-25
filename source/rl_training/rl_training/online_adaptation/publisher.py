from __future__ import annotations

import copy
import threading

from torch import nn


class DoubleBufferedActorPublisher:
    def __init__(self, actor: nn.Module):
        self._actors = [copy.deepcopy(actor), copy.deepcopy(actor)]
        for published_actor in self._actors:
            published_actor.eval()
            published_actor.requires_grad_(False)
        self._active_index = 0
        self._version = 0
        self._state_lock = threading.Lock()
        self._writer_lock = threading.Lock()

    def publish(self, training_actor: nn.Module, version: int) -> None:
        with self._writer_lock:
            with self._state_lock:
                inactive_index = 1 - self._active_index
            inactive_actor = self._actors[inactive_index]
            inactive_actor.load_state_dict(training_actor.state_dict())
            inactive_actor.eval()
            with self._state_lock:
                self._active_index = inactive_index
                self._version = int(version)

    def current(self) -> tuple[nn.Module, int]:
        with self._state_lock:
            return self._actors[self._active_index], self._version
