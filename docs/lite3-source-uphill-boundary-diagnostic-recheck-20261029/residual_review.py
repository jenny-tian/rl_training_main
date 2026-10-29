"""Quantify the archived residual without running a policy or simulator."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.tools.source_uphill_drift_audit import analyze


def review():
    previous = ROOT / 'docs/lite3-source-policy-prequalification-uphill-l1-20261029/samples.jsonl'
    current = OUT / 'samples.jsonl'
    assert current.read_bytes() == previous.read_bytes()
    report, arrays = analyze()
    t, vy = arrays['time'], arrays['world'][:, 1]
    left = float(np.sum(vy[:-1] * np.diff(t)))
    right = float(np.sum(vy[1:] * np.diff(t)))
    trap = (left + right) / 2
    observed = report['observed_lateral_displacement_m']
    quat = sum(report['body_xyz_world_y_contribution_m'])
    return {
        'schema': 'lite3_uphill_residual_explanation_v1', 'simulation_started': False,
        'samples_sha256': hashlib.sha256(current.read_bytes()).hexdigest(),
        'current_and_previous_samples_byte_identical': True,
        'integration_interval_s': [float(t[0]), float(t[-1])],
        'sampling_period_s': 0.02, 'physics_period_s': 0.005,
        'observed_displacement_m': observed,
        'quaternion_velocity_integral_m': quat,
        'direct_world_velocity_trapezoid_m': trap,
        'direct_minus_quaternion_integral_m': trap - quat,
        'reported_residual_m': observed - quat,
        'direct_world_velocity_residual_m': observed - trap,
        'residual_percent_of_displacement': 100 * (observed - quat) / observed,
        'left_rectangle_residual_m': observed - left,
        'right_rectangle_residual_m': observed - right,
        'interpretation': 'The 0.35 mm difference is a position-versus-sampled-velocity integration residual. Rotation roundoff accounts for only about 1.13e-8 m of it.',
        'limitations': [
            'Both quantities start at the same first sample. The absent reset-to-first-sample interval is not an explanation for this residual.',
            'The 50 Hz trapezoidal rule approximates the evolving velocity; it does not reproduce all 200 Hz physics updates or solver position corrections.',
            'Rectangle-rule differences demonstrate quadrature sensitivity; they do not measure the exact trapezoid error.',
            'Sampling, integration/constraint-solver updates and state precision cannot be separately quantified from these logs.',
            'The reference-point audit must be read separately: torso authored COM offset is zero, unlike the triggered thigh. No actual runtime COM readback was captured.',
            'A small kinematic residual does not isolate the causal source of drift or establish prediction safety.',
        ],
    }


if __name__ == '__main__':
    with (OUT / 'residual_explanation.json').open('x') as stream:
        json.dump(review(), stream, indent=2, allow_nan=False)
        stream.write('\n')
