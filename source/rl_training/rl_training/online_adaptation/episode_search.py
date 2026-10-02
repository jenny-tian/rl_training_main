"""Deterministic, critic-free episode search; physics accounting belongs to its caller.

A generation must contain four finished attempts. Interrupted/collector-cutoff
attempts are retained but cannot become elites. Codes never update within an
episode. This module neither selects final deployment nor grants free rollouts.
"""
from __future__ import annotations

import copy
import math

import numpy as np


class EpisodeSearch:
    SCHEMA = 'bounded_episode_elites_v1'
    TERMINALS = frozenset({'completion', 'physical_failure', 'lateral_violation',
                          'task_timeout', 'safety_failure', 'qualification_failure'})

    def __init__(self, seed: int, code_dim: int = 2):
        if type(seed) is not int or seed < 0 or type(code_dim) is not int or code_dim < 1:
            raise ValueError('nonnegative integer seed and positive code dimension required')
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros(code_dim)
        self.std = np.full(code_dim, .3)
        self.generation = 0
        self.pending = None
        self.history = []

    def ask(self):
        """Repeated calls return the same pending codes without consuming RNG."""
        if self.pending is None:
            codes = np.clip(self.rng.normal(self.mean, self.std, (4, len(self.mean))), -1., 1.)
            if self.generation == 0:
                codes[0] = 0.
            self.pending = [{'id': f'{self.generation}:{i}', 'code': code.tolist()} for i, code in enumerate(codes)]
        return copy.deepcopy(self.pending)

    def tell(self, results):
        """Commit all four attempts atomically, including failed/interrupted ones.

        Each result provides id, terminal_reason, normalized_progress in [0,1],
        safety_interventions, and charged_steps. The caller must bind these to
        the physical ledger; this kernel cannot verify a claimed rollout.
        """
        if self.pending is None:
            raise ValueError('no pending generation')
        if not isinstance(results, list) or len(results) != 4:
            raise ValueError('all four attempt results required')
        expected = {p['id'] for p in self.pending}
        if {r['id'] for r in results} != expected:
            raise ValueError('missing, duplicate or foreign attempt id')
        by_id = {r['id']: copy.deepcopy(r) for r in results}
        ordered = [by_id[p['id']] for p in self.pending]
        for r in ordered:
            progress = r['normalized_progress']
            if (isinstance(progress, bool) or not isinstance(progress, (int, float))
                    or not math.isfinite(progress) or not 0 <= progress <= 1):
                raise ValueError('progress must be finite and normalized')
            for key in ('safety_interventions', 'charged_steps'):
                if type(r[key]) is not int or r[key] < 0:
                    raise ValueError(f'{key} must be a nonnegative integer')
            if not isinstance(r['terminal_reason'], str):
                raise ValueError('terminal reason required')
            if r['terminal_reason'] in self.TERMINALS and r['charged_steps'] == 0:
                raise ValueError('finished physical episodes require charged steps')
        eligible = [i for i,r in enumerate(ordered) if r['terminal_reason'] in self.TERMINALS]
        eligible.sort(key=lambda i: (ordered[i]['terminal_reason'] == 'completion',
                                     ordered[i]['normalized_progress'],
                                     -ordered[i]['safety_interventions'], -i), reverse=True)
        elites = eligible[:2] if len(eligible) >= 2 else []
        before = {'mean': self.mean.tolist(), 'std': self.std.tolist()}
        if elites:
            codes = np.asarray([self.pending[i]['code'] for i in elites])
            self.mean = codes.mean(0)
            self.std = np.maximum(codes.std(0), .1)
        self.history.append({'generation': self.generation, 'attempts': copy.deepcopy(self.pending),
                             'results': ordered, 'elite_ids': [self.pending[i]['id'] for i in elites],
                             'before': before, 'after': {'mean': self.mean.tolist(), 'std': self.std.tolist()}})
        self.pending = None
        self.generation += 1

    def state_dict(self):
        return copy.deepcopy({'schema': self.SCHEMA, 'mean': self.mean.tolist(), 'std': self.std.tolist(),
                              'generation': self.generation, 'rng': self.rng.bit_generator.state,
                              'pending': self.pending, 'history': self.history})

    @classmethod
    def from_state_dict(cls, state):
        """Restore distribution, pending attempts and RNG without issuing new trials."""
        if state.get('schema') != cls.SCHEMA:
            raise ValueError('unsupported search checkpoint')
        mean, std = np.asarray(state['mean']), np.asarray(state['std'])
        if (mean.ndim != 1 or len(mean) < 1 or std.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(std).all()
                or (abs(mean) > 1).any() or (std < .1).any()):
            raise ValueError('invalid search distribution')
        if type(state['generation']) is not int or state['generation'] != len(state['history']):
            raise ValueError('generation/history mismatch')
        obj = cls(0, len(mean))
        obj.mean, obj.std = mean.copy(), std.copy()
        obj.rng.bit_generator.state = copy.deepcopy(state['rng'])
        obj.generation = state['generation']
        obj.history = copy.deepcopy(state['history'])
        pending = copy.deepcopy(state['pending'])
        if pending is not None:
            if len(pending) != 4 or [p['id'] for p in pending] != [f'{obj.generation}:{i}' for i in range(4)]:
                raise ValueError('invalid pending attempt IDs')
            codes = np.asarray([p['code'] for p in pending])
            if codes.shape != (4, len(mean)) or not np.isfinite(codes).all() or (abs(codes) > 1).any():
                raise ValueError('invalid pending codes')
        obj.pending = pending
        replay = cls(0, len(mean))
        for entry in obj.history:
            attempts = entry['attempts']
            codes = np.asarray([p['code'] for p in attempts])
            if (len(attempts) != 4 or [p['id'] for p in attempts] != [f'{replay.generation}:{i}' for i in range(4)]
                    or codes.shape != (4, len(mean)) or not np.isfinite(codes).all() or (abs(codes) > 1).any()):
                raise ValueError('invalid historical attempts')
            replay.pending = copy.deepcopy(attempts)
            replay.tell(entry['results'])
            if replay.history[-1] != entry:
                raise ValueError('history does not reproduce elite update')
        if not np.array_equal(replay.mean, obj.mean) or not np.array_equal(replay.std, obj.std):
            raise ValueError('distribution differs from recorded history')
        return obj
