"""B2 allocation, persistent turn sampling, and reset-safe contact evidence."""
import torch
import pytest

from rl_training.source_curriculum.allocation import b2_column_counts
from rl_training.source_curriculum.block import BlockContactDiagnostics, apply_pure_turn


@pytest.mark.parametrize("n", [120, 640, 4096])
def test_balanced_groups(n):
    counts = b2_column_counts(n)
    assert len(counts) == 20 and sum(counts) == n and min(counts) > 0
    assert abs(counts[4] - .2 * n) <= 1 and abs(counts[5] - .2 * n) <= 1
    assert abs(counts[9] - .1 * n) <= 1
    assert abs(sum(counts[10:13]) - .2 * n) <= 1
    if n == 4096:
        assert [counts[4], counts[5], counts[9], sum(counts[10:13])] == [819, 819, 410, 819]
        others = [v for i, v in enumerate(counts) if i not in (4, 5, 9, 10, 11, 12)]
        assert len(others) == 14 and sum(others) == 1229 and set(others) == {87, 88}


def test_turn_has_both_signs_no_translation_heading_or_standing():
    command = torch.ones(1000, 3)
    heading = torch.ones(1000, dtype=torch.bool)
    standing = heading.clone()
    ids = torch.arange(100, 900)
    generator = torch.Generator().manual_seed(42)
    for _ in range(3):
        apply_pure_turn(command, heading, standing, ids, generator=generator)
        assert not command[ids, :2].any() and not heading[ids].any() and not standing[ids].any()
        assert (command[ids, 2].abs() >= .2).all() and (command[ids, 2].abs() <= .8).all()
        assert (command[ids, 2] > 0).any() and (command[ids, 2] < 0).any()
        assert (command[:100] == 1).all() and heading[:100].all()


def test_contact_snapshot_precedes_reset_and_only_failures_count():
    diag = BlockContactDiagnostics(2, ["FL_SHANK", "BASE"], "cpu")
    diag.observe(torch.tensor([[20., 0.], [0., 10.]]))
    diag.observe(torch.tensor([[8., 0.], [0., 0.]]))
    ids = torch.arange(2)
    diag.finish(ids, torch.tensor([5, 4]), torch.tensor([True, False]))
    diag.reset(ids)
    assert diag.last_terminal.tolist() == [[8., 0.], [0., 0.]]
    assert diag.last_peak.tolist() == [[20., 0.], [0., 10.]]
    assert diag.failure_contacts.sum() == 1 and diag.failure_contacts[5, 0] == 1
    assert not diag.peak.any()
