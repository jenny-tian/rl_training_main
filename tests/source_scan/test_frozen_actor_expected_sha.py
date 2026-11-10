"""CPU checks for explicit checkpoint identities and the preserved default pin."""
import hashlib

import pytest
import torch

from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256, LAYER_DIMS, load_frozen_source_actor


def test_explicit_wrong_hash_is_rejected_before_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"tiny fake checkpoint")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("must verify SHA before loading"))
    with pytest.raises(ValueError, match="^checkpoint hash differs from the expected SHA256$"):
        load_frozen_source_actor(checkpoint, expected_sha256="a" * 64)


def test_default_checkpoint_hash_error_is_preserved(tmp_path):
    checkpoint = tmp_path / "not_model_1998.pt"
    checkpoint.write_bytes(b"tiny fake checkpoint")
    for kwargs in ({}, {"expected_sha256": CHECKPOINT_SHA256}):
        with pytest.raises(ValueError, match=r"^checkpoint hash differs from the registered model_1998\.pt$"):
            load_frozen_source_actor(checkpoint, **kwargs)


@pytest.mark.parametrize("sha", [None, 123, "A" * 64, "a" * 63, "a" * 65, "g" * 64, "a" * 64 + "\n"])
def test_expected_hash_requires_64_lowercase_hex_characters(tmp_path, sha):
    with pytest.raises(ValueError, match="expected_sha256 must be 64 lowercase hex characters"):
        load_frozen_source_actor(tmp_path / "does_not_exist.pt", expected_sha256=sha)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_explicit_matching_hash_loads_only_finite_frozen_actor(tmp_path, nonfinite):
    state = {"distribution.log_std_param": torch.zeros(12)}
    for i, (n_in, n_out) in enumerate(zip(LAYER_DIMS, LAYER_DIMS[1:])):
        state[f"mlp.{2 * i}.weight"] = torch.zeros(n_out, n_in)
        state[f"mlp.{2 * i}.bias"] = torch.zeros(n_out)
    if nonfinite:
        state["mlp.0.weight"][0, 0] = float("nan")
    checkpoint = tmp_path / "candidate.pt"
    torch.save({"actor_state_dict": state}, checkpoint)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if nonfinite:
        with pytest.raises(FloatingPointError, match="non-finite actor tensor"):
            load_frozen_source_actor(checkpoint, expected_sha256=sha)
    else:
        actor = load_frozen_source_actor(str(checkpoint), expected_sha256=sha)
        assert not actor.training and not any(p.requires_grad for p in actor.parameters())
        torch.testing.assert_close(actor(torch.ones(2, 45)), torch.zeros(2, 12))
