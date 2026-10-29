"""Independent rotation examples and conservation across the archived sections."""

import numpy as np

from scripts.tools.source_uphill_drift_audit import analyze, lateral_contributions


def test_full_rotation_accounts_for_vertical_velocity_on_a_rolled_body():
    half = np.sqrt(0.5)
    quaternions = np.array([[half, 0, 0, half], [half, half, 0, 0]])
    velocity = np.array([[2., 3., 4.], [2., 3., 4.]])
    np.testing.assert_allclose(lateral_contributions(quaternions, velocity), [[2., 0., 0.], [0., 0., -4.]], atol=1e-14)


def test_section_accounting_conserves_the_observed_archived_displacement():
    report, _ = analyze()
    assert [s["section"] for s in report["segments"]] == ["entry_flat", "entry_transition", "slope"]
    parts = report["segments"]
    np.testing.assert_allclose(sum(s["lateral_displacement_m"] for s in parts), report["observed_lateral_displacement_m"], atol=1e-12)
    np.testing.assert_allclose(np.sum([s["body_xyz_contribution_m"] for s in parts], axis=0), report["body_xyz_world_y_contribution_m"], atol=1e-12)
    assert report["terminal_phase"] == "pre_reset"
    assert abs(report["integration_residual_m"]) < 0.005
