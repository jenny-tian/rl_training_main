"""Attribute control-compute latency without changing the deadline clock."""

import gc
import time


class ControlLatencyProbe:
    def __init__(self):
        self.active = False
        self.gc_start = None
        self.events = []
        self.stages = {}
        self.callback = self._gc_event
        gc.callbacks.append(self.callback)

    def _gc_event(self, phase, info):
        now = time.perf_counter()
        if phase == "start":
            self.gc_start = now
        elif self.active and self.gc_start is not None:
            self.events.append((info["generation"], max(0., now - max(self.gc_start, self.started))))
            self.gc_start = None

    def start(self, started):
        self.started = self.marked = started
        self.events.clear()
        self.stages.clear()
        self.cpu_started = time.thread_time()
        self.active = True

    def mark(self, name):
        now = time.perf_counter()
        self.stages[name] = (now - self.marked) * 1000.
        self.marked = now

    def finish(self):
        wall_ms = (time.perf_counter() - self.started) * 1000.
        cpu_ms = (time.thread_time() - self.cpu_started) * 1000.
        self.active = False
        return {"wall_before_fallback_ms": wall_ms, "thread_cpu_ms": cpu_ms,
                "gc_ms": sum(duration for _, duration in self.events) * 1000.,
                "gc_generations": [generation for generation, _ in self.events],
                "host_stage_ms": dict(self.stages)}

    def close(self):
        self.active = False
        if self.callback in gc.callbacks:
            gc.callbacks.remove(self.callback)
