"""Shared-bank result validation without importing Kit or the task package."""

from source_shared_tensor_contact import tensor_summary_passed

SHARED_CONTROL_STEPS = 20


def shared_report_passed(report, *, reset_lifecycle=False):
    """Fail closed if a success file omits any required shared-bank coverage."""
    if not isinstance(report, dict):
        return False
    contacts = report.get('ground_contact_points', {})
    resets = report.get('reset_checks', {})
    if not isinstance(contacts, dict) or not isinstance(resets, dict):
        return False
    num_envs = report.get('num_envs')
    if type(num_envs) is not int or num_envs not in (2, 8) or (num_envs == 8 and not reset_lifecycle):
        return False
    return (
        report.get('status') == ('shared_bank_g2_passed' if reset_lifecycle else 'shared_bank_smoke_passed')
        and report.get('bank_rows') == 1
        and report.get('zero_action_steps') == SHARED_CONTROL_STEPS
        and (not reset_lifecycle or report.get('total_zero_action_steps') == SHARED_CONTROL_STEPS + 2)
        and report.get('overlapping_spawn_checked') is True
        and report.get('cross_environment_contacts') == 0
        and all(type(contacts.get(str(i), contacts.get(i))) is int
                and contacts.get(str(i), contacts.get(i, 0)) > 0 for i in range(num_envs))
        and isinstance(report.get('contact_capture'), dict)
        and isinstance(report['contact_capture'].get('tensor_topology'), dict)
        and report['contact_capture']['tensor_topology'].get('num_envs', 2) == num_envs
        and tensor_summary_passed(report.get('contact_capture'), contacts)
        and report.get('vertical_occlusion_probe_passed') is True
        and (num_envs != 8 or report.get('vertical_occlusion_environment_ids') == list(range(1, num_envs)))
        and (shared_lifecycle_passed(resets) if reset_lifecycle else (
            resets.get('assignment_count') == 8 and resets.get('reset_batch_count') == 5
            and resets.get('direct_reset_count') == 6))
        and report.get('policy_loaded') is False and report.get('ppo_updates') == 0
    )


def shared_lifecycle_passed(report):
    """Require complete G2 coverage for the selected shared-bank size."""
    if not isinstance(report, dict):
        return False
    n = report.get('num_envs')
    if type(n) is not int:
        return False
    if n == 2:
        expected = dict(assignment_count=123, complete_blocks=1, direct_reset_count=120,
                        reset_batch_count=91, automatic_reset_ids=[0], tail_ticket_count=3)
    elif n == 8:
        expected = dict(assignment_count=129, complete_blocks=1, direct_reset_count=120,
                        reset_batch_count=33, automatic_reset_ids=[7], tail_ticket_count=9)
    else:
        return False
    return (report.get('status') == 'shared_reset_lifecycle_passed'
            and report.get('bank_rows') == 1
            and all(report.get(key) == value for key, value in expected.items())
            and report.get('diagnostic_timeout_injected') is True
            and report.get('post_automatic_reset_step_checked') is True
            and report.get('physics_control_steps') == 2
            and (n != 8 or report.get('batch_coverage') == {
                'singleton': True, 'noncontiguous': True, 'unordered_input': True,
                'large_batch': True, 'highest_env_id': True}))
