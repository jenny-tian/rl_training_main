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
                    'ceilings': dict(CEILINGS), 'entries': []}
        data = json.loads(self.path.read_text())
        if (data.get('schema') != SCHEMA or data.get('checkpoint_sha256') != CHECKPOINT_SHA256
                or data.get('ceilings') != CEILINGS or not isinstance(data.get('entries'), list)):
            raise ValueError('invalid Source budget ledger')
        seen = set()
        for entry in data['entries']:
            if (not isinstance(entry, dict) or not isinstance(entry.get('attempt_id'), str) or not entry['attempt_id']
                    or not isinstance(entry.get('purpose'), str) or entry.get('automatic_refund') is not False
                    or entry.get('attempt_id') in seen or type(entry.get('run')) is not int or entry['run'] not in (1, 2, 3)
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

    def numerics_plan(self, run):
        """Plan a fresh numeric trial while retaining the accepted diagnostic charge."""
        if type(run) is not int or run not in (1, 2, 3):
            raise ValueError('invalid Source run')
        data = self.read()
        prior = [row for row in data['entries'] if row['run'] == run]
        if prior and (len(prior) != 1 or prior[0]['purpose'] != 'bounded_live_ppo_integration'
                or prior[0]['state'] != 'accepted' or prior[0]['reserved_transitions'] != 384
                or prior[0].get('returned_transitions') != 384 or prior[0].get('stored_transitions') != 384
                or prior[0].get('completed_updates') != 2):
            raise ValueError('Source numeric trial requires unused history or one accepted 384-transition diagnostic')
        charged = self.charged(run, data)
        nodes = {}
        for name, ceiling in CEILINGS.items():
            updates = max(0, (ceiling-charged)//2880)
            nodes[name] = {'ceiling': ceiling, 'remaining_updates': updates,
                'planned_transitions': updates*2880, 'cumulative_transitions': charged+updates*2880,
                'unused_transitions': ceiling-charged-updates*2880}
        if not nodes['numerics']['remaining_updates']:
            raise ValueError('numeric budget cannot fund a whole rollout')
        return {'run': run, 'num_envs': 120, 'rollout_length': 24,
                'baseline_charged_transitions': charged, 'nodes': nodes,
                'initialization': 'registered_actor_fresh_critic_optimizer',
                'diagnostic_weights_reused': False, 'resume': False}

    def reserve_numerics(self, *, run, attempt_id, output_dir, revision, expected_reference, expected_plan):
        """Reserve the complete approved trial atomically, without refund or resume."""
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError('missing numeric attempt identity')
        with self._locked():
            if expected_reference != self.reference():
                raise ValueError('budget ledger differs from approved snapshot')
            plan = self.numerics_plan(run)
            if expected_plan != plan:
                raise ValueError('cumulative budget differs from approved plan')
            data = self.read()
            if any(row['attempt_id'] == attempt_id for row in data['entries']):
                raise ValueError('numeric attempt already reserved')
            node = plan['nodes']['numerics']
            row = {'attempt_id': attempt_id, 'run': run, 'purpose': 'source_numerics_cumulative',
                   'reserved_transitions': node['planned_transitions'], 'state': 'reserved',
                   'output_dir': str(Path(output_dir).resolve()), 'base_revision': revision,
                   'automatic_refund': False, 'baseline_charged_transitions': plan['baseline_charged_transitions'],
                   'node_ceiling': node['ceiling'], 'max_updates': node['remaining_updates'],
                   'num_envs': 120, 'rollout_length': 24, 'initialization': plan['initialization']}
            data['entries'].append(row)
            atomic_json(self.path, data)
            return dict(row)

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

    def reserve_adaptation_trial(self, *, run, attempt_id, output_dir, revision,
                                 expected_reference, expected_plan):
        """Append the agreed two-update correction trial without changing old ceilings.

        This one-off increment retains all earlier charges, including failures.
        A different baseline, profile or trial size requires a new explicit plan.
        """
        from .adaptation_profile import SOURCE_ADAPTATION_PROFILE

        plan = dict(profile=SOURCE_ADAPTATION_PROFILE, num_envs=120, rollout_length=24,
                    max_updates=2, reserved_transitions=5760,
                    baseline_charged_transitions=259584, cumulative_ceiling=265344,
                    initialization='registered_actor_checkpoint_std_fresh_critic_optimizer')
        if type(run) is not int or run != 1 or not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError('invalid adaptation trial identity')
        if (expected_plan != plan or any(type(expected_plan[k]) is not type(v) for k, v in plan.items())):
            raise ValueError('adaptation trial differs from the agreed incremental plan')
        with self._locked():
            if expected_reference != self.reference():
                raise ValueError('budget ledger differs from approved snapshot')
            data = self.read()
            if any(row['state'] == 'reserved' for row in data['entries']):
                raise ValueError('another Source reservation is unfinished')
            if any(row['attempt_id'] == attempt_id for row in data['entries']):
                raise ValueError('adaptation attempt already reserved')
            if self.charged(run, data) != plan['baseline_charged_transitions']:
                raise ValueError('adaptation baseline charge changed; no retry or refund')
            row = dict(plan, attempt_id=attempt_id, run=run, purpose='source_adaptation_two_update_trial',
                       state='reserved', automatic_refund=False,
                       output_dir=str(Path(output_dir).resolve()), base_revision=revision)
            data['entries'].append(row)
            atomic_json(self.path, data)
            return dict(row)

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
        if type(accepted) is not bool: raise ValueError('budget result must be boolean')
        with self._locked():
            data = self.read()
            rows = [row for row in data['entries'] if row['attempt_id'] == attempt_id]
            if len(rows) != 1:
                raise ValueError('missing budget reservation')
            row = rows[0]
            if (any(type(x) is not int or x < 0 for x in (attempted, returned, stored, updates))
                    or not stored <= returned <= attempted <= row['reserved_transitions']):
                raise ValueError('observed interactions exceed budget reservation')
            if updates > row.get('max_updates', 2 if row['purpose'] == 'bounded_live_ppo_integration' else 91):
                raise ValueError('observed updates exceed reservation')
            changes = dict(state='accepted' if accepted else 'failed', attempted_upper_bound=attempted,
                           returned_transitions=returned, stored_transitions=stored, completed_updates=updates)
            if row['state'] != 'reserved' and any(row.get(key) != value for key, value in changes.items()):
                raise ValueError('budget result already finalized')
            row.update(changes)
            atomic_json(self.path, data)
            return dict(row)

    def remaining(self, run):
        charged = self.charged(run, self.read())
        return {name: {'ceiling': ceiling, 'reserved': charged,
                      'remaining': ceiling - charged,
                      'remaining_120x24_updates': max(0, (ceiling - charged) // 2880)}
                for name, ceiling in CEILINGS.items()}
