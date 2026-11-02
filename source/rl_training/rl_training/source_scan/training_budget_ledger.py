"""Shared, crash-conservative Source interaction reservations across worktrees."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

from .frozen_actor import CHECKPOINT_SHA256
from .smoke_artifacts import atomic_json

SCHEMA = 'lite3_source_ppo_budget_ledger_v1'
CEILINGS = {'numerics': 262144, 'midpoint': 1048576, 'pilot': 2097152}


def canonical_budget_path(root):
    common = subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                                     cwd=root, text=True, timeout=10).strip()
    return Path(common) / 'source-ppo-ledgers/lite3-l1-v1.json'


class BudgetLedger:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def read(self):
        if not self.path.exists():
            return {'schema': SCHEMA, 'checkpoint_sha256': CHECKPOINT_SHA256,
                    'ceilings': CEILINGS, 'entries': []}
        data = json.loads(self.path.read_text())
        if (data.get('schema') != SCHEMA or data.get('checkpoint_sha256') != CHECKPOINT_SHA256
                or data.get('ceilings') != CEILINGS or not isinstance(data.get('entries'), list)):
            raise ValueError('invalid Source budget ledger')
        seen = set()
        for entry in data['entries']:
            if (entry.get('attempt_id') in seen or type(entry.get('run')) is not int or entry['run'] not in (1, 2, 3)
                    or type(entry.get('reserved_transitions')) is not int or entry['reserved_transitions'] <= 0
                    or entry.get('state') not in ('reserved', 'accepted', 'completed', 'failed')):
                raise ValueError('invalid or duplicate budget reservation')
            seen.add(entry['attempt_id'])
        for run in (1, 2, 3):
            if self.charged(run, data) > CEILINGS['pilot']:
                raise ValueError('Source seed budget overdrawn')
        return data

    def reference(self):
        self.read()
        return {'path': str(self.path), 'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest()
                if self.path.exists() else None}

    @staticmethod
    def charged(run, data):
        return sum(row['reserved_transitions'] for row in data['entries'] if row['run'] == run)

    def assert_fresh(self, run, expected_reference):
        if expected_reference != self.reference():
            raise ValueError('budget ledger differs from approved snapshot')
        if self.charged(run, self.read()):
            raise ValueError('Source seed already has reserved interactions; a revised cumulative budget lock is required')

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('another Source budget writer is active') from None
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def reserve(self, *, run, attempt_id, transitions, purpose, output_dir, revision, expected_reference):
        if type(run) is not int or run not in (1, 2, 3) or not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError('invalid budget identity')
        if type(transitions) is not int or not 0 < transitions <= CEILINGS['numerics']:
            raise ValueError('invalid budget reservation size')
        with self._locked():
            self.assert_fresh(run, expected_reference)
            data = self.read()
            if any(row['attempt_id'] == attempt_id for row in data['entries']):
                raise ValueError('budget attempt already reserved')
            row = {'attempt_id': attempt_id, 'run': run, 'purpose': purpose,
                   'reserved_transitions': transitions, 'state': 'reserved',
                   'output_dir': str(Path(output_dir).resolve()), 'base_revision': revision,
                   'automatic_refund': False}
            data['entries'].append(row)
            atomic_json(self.path, data)
            return dict(row)

    def finish(self, attempt_id, *, accepted, attempted, returned, stored, updates):
        with self._locked():
            data = self.read()
            rows = [row for row in data['entries'] if row['attempt_id'] == attempt_id]
            if len(rows) != 1:
                raise ValueError('missing budget reservation')
            row = rows[0]
            if (any(type(x) is not int or x < 0 for x in (attempted, returned, stored, updates))
                    or not stored <= returned <= attempted <= row['reserved_transitions']):
                raise ValueError('observed interactions exceed budget reservation')
            row.update(state='accepted' if accepted else 'failed', attempted_upper_bound=attempted,
                       returned_transitions=returned, stored_transitions=stored, completed_updates=updates)
            atomic_json(self.path, data)
            return dict(row)

    def remaining(self, run):
        charged = self.charged(run, self.read())
        return {name: {'ceiling': ceiling, 'reserved': charged,
                      'remaining': ceiling - charged,
                      'remaining_120x24_updates': max(0, (ceiling - charged) // 2880)}
                for name, ceiling in CEILINGS.items()}
