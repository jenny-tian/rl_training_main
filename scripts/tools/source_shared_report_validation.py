"""Shared-bank result validation without importing Kit or the task package."""

from source_shared_tensor_contact import tensor_summary_passed

SHARED_CONTROL_STEPS = 20


def shared_report_passed(report):
    """Fail closed if a success file omits any required shared-bank coverage."""
    if not isinstance(report, dict):
        return False
    contacts = report.get('ground_contact_points', {})
    resets = report.get('reset_checks', {})
    if not isinstance(contacts, dict) or not isinstance(resets, dict):
        return False
    return (
        report.get('status') == 'shared_bank_smoke_passed'
        and report.get('num_envs') == 2 and report.get('bank_rows') == 1
        and report.get('zero_action_steps') == SHARED_CONTROL_STEPS
        and report.get('overlapping_spawn_checked') is True
        and report.get('cross_environment_contacts') == 0
        and all(type(contacts.get(str(i), contacts.get(i))) is int
                and contacts.get(str(i), contacts.get(i, 0)) > 0 for i in (0, 1))
        and tensor_summary_passed(report.get('contact_capture'), contacts)
        and report.get('vertical_occlusion_probe_passed') is True
        and resets.get('assignment_count') == 8 and resets.get('reset_batch_count') == 5
        and resets.get('direct_reset_count') == 6
        and report.get('policy_loaded') is False and report.get('ppo_updates') == 0
    )
