import pytest
import torch

from rl_training.online_adaptation.teacher_steering import (
    TEACHER_DECODER_SCHEMA,
    fit_low_rank_teacher_decoder,
    load_teacher_decoder,
    save_teacher_decoder,
    supervise_latent_actor,
)
from rl_training.online_adaptation.networks import LatentResidualActor


def test_teacher_decoder_fit_is_bounded_and_reproducible(tmp_path):
    torch.manual_seed(4)
    limits = torch.tensor([0.24, 0.20, 0.20] * 4)
    basis = torch.randn(12, 3)
    codes = torch.tanh(torch.randn(64, 3))
    residual = (codes @ basis.T).tanh() * limits * 0.2

    fit = fit_low_rank_teacher_decoder(residual, limits, code_dim=3)

    assert fit.decoder_matrix.shape == (12, 3)
    assert fit.code_targets.shape == (64, 3)
    assert torch.all(fit.reconstruction.abs() <= limits + 1e-6)
    assert torch.all(fit.code_targets.abs() <= 1.0)
    assert fit.explained_variance > 0.5

    path = tmp_path / "teacher_decoder.pt"
    save_teacher_decoder(path, fit, limits, source_metadata={"split": "train"})
    restored = load_teacher_decoder(path, action_limits=limits, code_dim=3)
    torch.testing.assert_close(restored, fit.decoder_matrix)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    assert artifact["schema"] == TEACHER_DECODER_SCHEMA
    assert artifact["source_metadata"]["split"] == "train"


def test_teacher_decoder_rejects_out_of_bounds_teacher_and_mismatched_artifact(tmp_path):
    limits = torch.ones(4)
    with pytest.raises(ValueError, match="exceeds"):
        fit_low_rank_teacher_decoder(torch.full((8, 4), 1.1), limits, code_dim=2)

    fit = fit_low_rank_teacher_decoder(torch.zeros(8, 4), limits, code_dim=2)
    path = tmp_path / "teacher_decoder.pt"
    save_teacher_decoder(path, fit, limits)
    with pytest.raises(ValueError, match="action limits"):
        load_teacher_decoder(path, action_limits=torch.ones(4) * 2, code_dim=2)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    artifact["code_targets"] = torch.full_like(artifact["code_targets"], 2.0)
    corrupted = tmp_path / "corrupted.pt"
    torch.save(artifact, corrupted)
    with pytest.raises(ValueError, match="code targets"):
        load_teacher_decoder(corrupted, action_limits=limits, code_dim=2)
    with pytest.raises(FileExistsError):
        save_teacher_decoder(path, fit, limits)


def test_supervised_latent_actor_fits_bounded_teacher_codes_without_decoder_gradients():
    torch.manual_seed(3)
    limits = torch.tensor([0.24, 0.20, 0.20] * 4)
    actor = LatentResidualActor(frame_dim=59, action_dim=12, code_dim=2, action_limits=limits)
    history = torch.randn(32, 4, 59)
    frame = torch.randn(32, 59)
    base = torch.randn(32, 12)
    targets = torch.tanh(torch.randn(32, 2))
    result = supervise_latent_actor(
        actor, history, frame, base, targets, epochs=2, batch_size=16, learning_rate=2.0e-3
    )
    assert result["samples"] == 32
    assert result["final_code_mse"] >= 0.0
    assert actor.decoder_matrix.grad is None
    with pytest.raises(ValueError, match="code_targets width"):
        supervise_latent_actor(actor, history, frame, base, torch.zeros(32, 3))
