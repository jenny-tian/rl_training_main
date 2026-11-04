"""CPU preparation of scale qualification and bounded-numerics readiness."""
from __future__ import annotations

from copy import deepcopy

from .frozen_actor import CHECKPOINT_SHA256


def selected_scale_proposal():
    n, bodies, ground = 120, 2040, 24
    pairs = bodies*(bodies+ground)
    frame_bytes = pairs*16 + bodies*12  # 3 float32 forces, uint32 count, net force.
    return {'schema': 'lite3_selected_scale_qualification_proposal_v1',
        'status': 'proposed_not_executable', 'user_approved': False,
        'num_envs': n, 'shared_bank_rows': 1, 'device': 'cuda:0',
        'registered_cells': 15, 'ppo_training_updates': 0,
        'policy': {'checkpoint_sha256': CHECKPOINT_SHA256, 'read_only': True,
                   'noise_std': 1., 'noise_seed': 132001, 'command_mps': [.3, 0., 0.]},
        'seeds': {'environment': 130001, 'geometry': 30000, 'sampler': 131001,
                  'reset_state_id': 40000, 'reset_randomized': False},
        'startup_ladder': [16, 32, 60, 120], 'separate_approval_per_attempt': True,
        'retry': False, 'resume': False, 'automatic_scale_increase': False,
        'minimum_free_vram_mib': 12288, 'sharing_allowed': True,
        'cleanup_grace_s': 15,
        'stages': {
            'G1': {'status': 'not_run', 'control_steps_per_scale': 1, 'startup_timeout_s': 120,
                   'wall_timeout_s': 240, 'policy_loaded': False,
                   'requires': 'all_smaller_shared_bank_startup_scales_passed'},
            'G2': {'status': 'not_run', 'num_envs': n, 'control_steps': 24,
                   'diagnostic_transitions': 2880, 'wall_timeout_s': 240,
                   'reset_batches': [[0], [1, 17, 119], list(range(0, 120, 2)), list(range(120))],
                   'requires': 'G1_120_passed', 'full_ticket_blocks_required': 1},
            'G3': {'status': 'not_run', 'control_steps': 2000, 'physics_substeps': 8000,
                   'diagnostic_transitions': 240000, 'startup_timeout_s': 120, 'wall_timeout_s': 900,
                   'requires': 'G1_G2_and_sparse_capture_implementation_qualified',
                   'all_cells_observed': True, 'flat_full_progress_and_hold_evidence_required': True,
                   'all_old_policy_cells_must_succeed': False,
                   'ordinary_task_endings': 'record_reset_and_continue',
                   'integrity_errors': 'stop_and_retain_fault_evidence',
                   'continuous_time_safety_claim': False},
            'G4_collection': {'status': 'not_run', 'warmup_control_steps': 48,
                   'measured_control_steps': 240, 'diagnostic_transitions': 34560,
                   'wall_timeout_s': 300, 'minimum_transitions_per_second': 600.,
                   'reset_p95_limit_s': 1., 'heavy_contact_journal': False,
                   'production_safety_and_finite_checks': True, 'ppo_updates': 0,
                   'end_to_end_training_throughput_qualified': False},
            'G4_end_to_end': {'status': 'pending_bounded_numerics',
                   'minimum_transitions_per_second': 600.,
                   'measurement': 'collection_plus_PPO_and_durable_production_logging_after_scene_start',
                   'candidate_pilot_steady_budget_s': 3600,
                   'threshold_rationale': '600 transitions/s exceeds 2097152/3600; startup is reported separately'}},
        'evidence_strategy': {'sensor_count': bodies, 'filter_count': bodies+ground, 'pair_count': pairs,
            'dense_frame_bytes': frame_bytes, 'dense_G3_bytes_before_JSON': frame_bytes*8000,
            'encoding': 'all_nonzero_count_or_force_pairs_with_implicit_zero_pairs',
            'every_frame_checks': ['full_matrix_finite', 'all_cross_environment_pairs_zero',
                'counts_offsets_capacity', 'detailed_force_normal_reconstruction', 'pair_sum_equals_net',
                'all_body_tile_ownership', 'scanner_hit_ownership_and_age', 'selective_reset_invariance'],
            'sparse_force_only_pairs_must_be_included': True,
            'contact_capacity_candidate': 1048576, 'raw_journal_byte_limit': 2147483648,
            'record_raw_points_and_pairs': True, 'record_full_fault_snapshot': True,
            'independent_CPU_reconstruction_required': True,
            'implementation_status': 'cpu_implemented_runtime_pending'},
        'metrics': ['startup_s', 'scene_build_s', 'first_step_s', 'process_RSS_peak_bytes',
                    'card_and_process_GPU_memory', 'control_steps_per_second', 'transitions_per_second',
                    'reset_latency_p50_s', 'reset_latency_p95_s', 'GPU_temperature_and_utilization'],
        'diagnostic_interactions_are_training_data': False,
        'diagnostic_interactions_must_be_reported_separately': True,
        'development_or_sealed_feedback_allowed': False}


def validate_numerics_runtime_receipt(receipt, input_sha256, *, stage_revision=None):
    expected = {'schema': 'lite3_bounded_numerics_runtime_readiness_v1',
        'status': 'qualified_for_bounded_numerics', 'num_envs': 120, 'shared_bank_rows': 1,
        'device': 'cuda:0', 'input_sha256': input_sha256,
        'g0_passed': True, 'g1_passed': True, 'g2_passed': True, 'g3_passed': True,
        'g4_collection_passed': True, 'g4_end_to_end_status': 'pending_bounded_numerics'}
    if stage_revision:
        from .delivery_revision import validate_revision
        validate_revision(stage_revision)
        expected.update(schema='lite3_bounded_numerics_limited_readiness_v2',g3_passed=False,
            limited_motion_evidence_passed=True,stage_revision=stage_revision)
    for key, value in expected.items():
        if type(receipt.get(key)) is not type(value) or receipt[key] != value:
            raise ValueError('bounded numeric runtime receipt mismatch: '+key)
    if receipt.get('minimum_transitions_per_second') != 600.:
        raise ValueError('throughput threshold differs from proposed contract')
    if (type(receipt.get('measured_collection_transitions_per_second')) not in (int, float)
            or not 600 <= receipt['measured_collection_transitions_per_second'] < float('inf')):
        raise ValueError('collection throughput has not met the registered target')
    return receipt


def numerics_readiness_template(input_sha256):
    return {'schema': 'lite3_bounded_numerics_runtime_readiness_v1', 'status': 'not_run',
        'num_envs': 120, 'shared_bank_rows': 1, 'device': 'cuda:0',
        'input_sha256': deepcopy(input_sha256), 'g0_passed': False, 'g1_passed': False,
        'g2_passed': False, 'g3_passed': False, 'g4_collection_passed': False,
        'g4_end_to_end_status': 'pending_bounded_numerics',
        'minimum_transitions_per_second': 600., 'measured_collection_transitions_per_second': None}
