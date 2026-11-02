"""CPU-only synthetic child; no torch, Isaac, checkpoint or GPU imports."""
from pathlib import Path
import os
import sys

from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.smoke_artifacts import SmokeArtifactWriter, finalize_and_close

output, mode = Path(sys.argv[1]), sys.argv[2]
if mode == 'missing':
    os._exit(0)
report = {'attempt_id': 'cpu-fixture', 'base_revision': 'a'*40, 'checkpoint_sha256': 'b'*64,
          'status': 'starting', 'num_envs': 2, 'simulation_control_steps': 0, 'ppo_updates': 0,
          'training_ready': False, 'g3_g4_qualified': False, 'diagnostic_only': True,
          'injected_timeout': True, 'critic_width': 3, 'synthetic_cpu_fixture': True,
          'ppo_seed': 130001, 'geometry_seed': 30000, 'sampler_seed': 131001,
          'noise_rng_seed': 132001, 'reset_state_id': 40000, 'reset_randomized': False,
          'shared_bank_rows': 1, 'stochastic_std': 1.0}
writer = SmokeArtifactWriter(output, report)
if mode == 'starting_zero':
    writer.close()
    os._exit(0)
for step in range(1, 25):
    last = step == 24
    terminal = None
    if last:
        terminal = {'ids': [0], 'steps': [2000], 'hold': [0], 'completed': [False],
                    'terminated': [False, False], 'truncated': [True, False],
                    'observations': {'policy': [[0.]*45 for _ in range(2)], 'critic': [[2.]*3 for _ in range(2)]},
                    'terms': {name: [name == 'source_training_timeout'] for name in (
                        'source_training_timeout', 'source_training_completion', 'source_training_safety',
                        'source_terrain_tile_bounds', 'illegal_contact', 'bad_orientation_2')}}
    row = {'step': step, 'done': [1, 0] if last else [0, 0],
           'time_outs': [True, False] if last else [False, False],
           'episode_counters': [0, 24] if last else [step, step], 'terminal': terminal,
           'terrain_columns_before': [0, 1], 'terrain_columns_after': [14, 1] if last else [0, 1],
           'projected': [[False], [False]], 'paused': [[False], [False]]}
    row.update({name: [[step/100.]*12 for _ in range(2)] for name in (
        'raw_action', 'base_action', 'requested_action', 'executed_action')})
    writer.append_step(row)
    report.update(status='running', simulation_control_steps=step, total_transitions=step*2)
    writer.persist(report)
    if mode == 'midway_exit' and step == 5:
        os._exit(17)
report.update(status='passed', exposure={name: {
    'episode_assignments': 1 if name in ('flat', 'multi_scale_rough', 'friction_alternating_regions') else 0,
    'transitions': 24 if name in ('flat', 'multi_scale_rough') else 0} for name in SOURCE_RESET_BLOCK_COUNTS})


def environment_close():
    if mode == 'environment_error':
        raise RuntimeError('injected environment cleanup failure')
    if mode == 'environment_exit':
        os._exit(0)


def application_close():
    if mode != 'close_returns':
        os._exit(0)


if mode == 'journal_close_error':
    def failed_close():
        raise OSError('injected journal close failure')
    writer.close = failed_close
finalize_and_close(writer, report, environment_close=environment_close, application_close=application_close)
