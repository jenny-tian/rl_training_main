"""Fail-closed contact evidence for the bounded shared-bank probe (no Kit imports)."""

from collections import Counter
from contextlib import contextmanager
import math
import operator
import re


def classify_contact(path0, path1):
    """Classify collider paths; unknown ownership is validated by the collector."""
    def owner(path):
        match = re.match(r"^/World/envs/env_(\d+)/Robot/", path)
        return int(match[1]) if match else None

    owners = (owner(path0), owner(path1))
    if all(i is not None for i in owners) and owners[0] != owners[1]:
        return "cross_environment", None
    for i, path in ((owners[0], path1), (owners[1], path0)):
        if i is not None and (path == '/World/ground' or path.startswith('/World/ground/')):
            return "ground", i
    return "other", None


def _integer(value):
    if isinstance(value, bool):
        raise ValueError('boolean in integer contact field')
    return operator.index(value)


def _point(data):
    result = {name: [float(x) for x in getattr(data, name)] for name in ('position', 'normal', 'impulse')}
    result['separation'] = float(data.separation)
    if any(len(result[name]) != 3 for name in ('position', 'normal', 'impulse')) or not all(
        math.isfinite(x) for x in [*result['position'], *result['normal'], *result['impulse'], result['separation']]
    ):
        raise ValueError('malformed or nonfinite contact point')
    return result


class ContactCapture:
    """Keep callback telemetry separate; only validated immediate points count."""

    def __init__(self, *, stage_id, reporting_bodies, decode_path, event_types, record,
                 start_step, decimation, control_steps):
        self.stage_id = _integer(stage_id)
        self.bodies = frozenset(reporting_bodies)
        self.decode_path, self.record = decode_path, record
        self.event_types = event_types
        self.start_step = _integer(start_step)
        self.decimation, self.control_steps = decimation, control_steps
        self.poll_steps, self.control_poll_counts, self.errors = [], [], []
        self.counts = {channel: Counter(calls=0, headers=0, data=0) for channel in ('poll', 'callback')}
        self.ground = {channel: Counter() for channel in self.counts}
        self.cross_contacts = []
        self.runtime_checked = False
        self.positive_controls_validated = False

    def runtime(self, interface, get_disabled):
        attached = _integer(interface.get_attached_stage())
        disabled = get_disabled()
        self.record({'phase': 'contact_runtime', 'stage_id': self.stage_id, 'attached_stage': attached,
                     'disable_contact_processing': disabled})
        if self.stage_id <= 0 or attached != self.stage_id or disabled is not False:
            raise AssertionError('contact runtime stage mismatch or processing disabled/unknown')
        self.runtime_checked = True

    def _owner(self, actor, collider):
        if actor in self.bodies and (collider == actor or collider.startswith(actor + '/')):
            match = re.match(r'^/World/envs/env_([01])/Robot/', actor)
            if match:
                return int(match[1])
        if all(p == '/World/ground' or p.startswith('/World/ground/') for p in (actor, collider)):
            return 'ground'
        raise ValueError(f'unknown actor/collider ownership: {actor}, {collider}')

    def consume(self, channel, headers, data, sim_step):
        counts = self.counts[channel]
        counts['calls'] += 1
        counts['headers'] += len(headers)
        counts['data'] += len(data)
        self.record({'phase': 'contact_batch', 'channel': channel, 'sim_step': sim_step,
                     'headers': len(headers), 'data': len(data), 'call': counts['calls']})
        # Validate and copy the complete borrowed buffer before accepting any positives.
        points = [_point(p) for p in data]
        used, positives = set(), Counter()
        for header in headers:
            if _integer(header.stage_id) != self.stage_id:
                raise ValueError('contact header stage mismatch')
            event = self.event_types.get(header.type)
            if event not in ('found', 'persist', 'lost'):
                raise ValueError('unknown contact event type')
            actors = [str(self.decode_path(p)) for p in (header.actor0, header.actor1)]
            paths = [str(self.decode_path(p)) for p in (header.collider0, header.collider1)]
            owners = [self._owner(a, c) for a, c in zip(actors, paths)]
            kind, owner = classify_contact(*paths)
            if owners[0] in (0, 1) and owners[1] in (0, 1) and owners[0] != owners[1]:
                self.cross_contacts.append({'channel': channel, 'sim_step': sim_step, 'colliders': paths})
                self.record({'phase': 'cross_environment_contact', **self.cross_contacts[-1]})
                raise AssertionError('cross-environment contact reported')
            offset, count = _integer(header.contact_data_offset), _integer(header.num_contact_data)
            if offset < 0 or count < 0 or offset + count > len(points):
                raise ValueError('contact data range out of bounds')
            if event == 'lost' and count:
                raise ValueError('lost contact unexpectedly contains points')
            indices = set(range(offset, offset + count))
            if used & indices:
                raise ValueError('overlapping contact data ranges')
            used.update(indices)
            self.record({'phase': 'contact', 'channel': channel, 'sim_step': sim_step, 'event': event,
                         'stage_id': self.stage_id, 'actors': actors, 'colliders': paths, 'kind': kind,
                         'offset': offset, 'points': points[offset:offset + count]})
            if kind == 'ground' and event in ('found', 'persist'):
                positives[owner] += count
        if len(used) != len(points):
            raise ValueError('unreferenced contact data')
        self.ground[channel].update(positives)

    def error(self, error, channel):
        item = {'channel': channel, 'error': repr(error)}
        self.errors.append(item)
        self.record({'phase': 'contact_error', **item})

    def callback(self, headers, data, sim_step):
        # Kit may swallow Python callback exceptions. The next post-physics hook
        # re-raises before another physics substep can run.
        try:
            self.consume('callback', headers, data, sim_step)
        except Exception as error:
            self.error(error, 'callback')

    def poll(self, env, interface, get_disabled):
        step = _integer(env._sim_step_counter)
        self.record({'phase': 'contact_poll_attempt', 'sim_step': step})
        self.runtime(interface, get_disabled)
        if step != self.start_step + len(self.poll_steps) + 1:
            raise AssertionError('nonconsecutive contact poll step')
        headers, data = interface.get_contact_report()
        self.consume('poll', headers, data, step)
        self.poll_steps.append(step)
        if self.errors:
            raise AssertionError(f'contact callback failed: {self.errors}')

    def check_control(self, env, control_step):
        if (control_step != len(self.control_poll_counts) + 1
                or len(self.poll_steps) != control_step * self.decimation
                or env._sim_step_counter != self.start_step + len(self.poll_steps)):
            raise AssertionError('incomplete contact polls in control step')
        self.control_poll_counts.append(self.decimation)

    def finish(self):
        if (len(self.poll_steps) != self.control_steps * self.decimation
                or len(self.control_poll_counts) != self.control_steps or self.errors or self.cross_contacts):
            raise AssertionError('incomplete or failed contact capture')
        if not all(self.ground['poll'][i] > 0 for i in (0, 1)):
            raise AssertionError('contact-report positive control missing: both robots must touch ground')
        self.positive_controls_validated = True

    def summary(self):
        return {'schema': 'shared_contact_capture_v1', 'stage_id': self.stage_id,
                'runtime_checked': self.runtime_checked, 'start_sim_step': self.start_step,
                'decimation': self.decimation, 'control_steps': self.control_steps,
                'poll_steps': list(self.poll_steps), 'control_poll_counts': list(self.control_poll_counts),
                'channels': {k: dict(v) for k, v in self.counts.items()},
                'ground_contact_points': dict(self.ground['poll']),
                'callback_ground_contact_points': dict(self.ground['callback']),
                'cross_environment_contacts': len(self.cross_contacts), 'errors': list(self.errors),
                'positive_controls_validated': self.positive_controls_validated}


@contextmanager
def capture_substeps(env, interface, capture, get_disabled):
    """Temporarily wrap one empty recorder instance, without adding recorder terms."""
    manager = env.recorder_manager
    name = 'record_post_physics_decimation_step'
    had_override = name in vars(manager)
    previous = vars(manager).get(name)
    original = getattr(manager, name)
    subscription = None
    installed = False
    try:
        capture.runtime(interface, get_disabled)
        if len(manager.active_terms) or env.cfg.decimation != capture.decimation:
            raise AssertionError('contact probe requires an empty recorder and expected decimation')

        def after_physics():
            capture.poll(env, interface, get_disabled)
            return original()

        subscription = interface.subscribe_contact_report_events(
            lambda headers, data: capture.callback(headers, data, env._sim_step_counter))
        if subscription is None:
            raise AssertionError('contact report subscription missing')
        setattr(manager, name, after_physics)
        installed = True
        yield capture
        capture.finish()
    except Exception as error:
        capture.error(error, 'capture')
        raise
    finally:
        if installed:
            if had_override:
                setattr(manager, name, previous)
            else:
                delattr(manager, name)
        # carb.Subscription releases its subscription when the holder is dropped.
        subscription = None
        capture.record({'phase': 'contact_summary', **capture.summary()})


def contact_summary_passed(summary, ground_contacts, *, control_steps=20, decimation=4):
    """Require both positive controls and every physics substep in the result file."""
    if not isinstance(summary, dict):
        return False
    start = summary.get('start_sim_step')
    channels = summary.get('channels')
    if type(start) is not int or start < 0 or not isinstance(channels, dict):
        return False
    for name in ('poll', 'callback'):
        counts = channels.get(name)
        if not isinstance(counts, dict) or any(type(counts.get(k)) is not int or counts[k] < 0
                                               for k in ('calls', 'headers', 'data')):
            return False
    points = summary.get('ground_contact_points')
    if not isinstance(points, dict) or not isinstance(ground_contacts, dict):
        return False
    if not all(type(points.get(str(i), points.get(i))) is int
               and points.get(str(i), points.get(i, 0)) > 0
               and points.get(str(i), points.get(i)) == ground_contacts.get(str(i), ground_contacts.get(i))
               for i in (0, 1)):
        return False
    total = control_steps * decimation
    return (summary.get('schema') == 'shared_contact_capture_v1'
            and type(summary.get('stage_id')) is int and summary['stage_id'] > 0
            and summary.get('runtime_checked') is True
            and summary.get('positive_controls_validated') is True
            and summary.get('decimation') == decimation and summary.get('control_steps') == control_steps
            and summary.get('poll_steps') == list(range(start + 1, start + total + 1))
            and summary.get('control_poll_counts') == [decimation] * control_steps
            and channels['poll']['calls'] == total
            and channels['poll']['headers'] >= 2 and channels['poll']['data'] >= 2
            and summary.get('errors') == [] and summary.get('cross_environment_contacts') == 0)
