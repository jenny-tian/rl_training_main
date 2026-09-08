import argparse
from types import SimpleNamespace

import pytest

from scripts.reinforcement_learning.rsl_rl.cli_args import (
    add_motion_prior_args,
    update_motion_prior_env_cfg,
)


def test_motion_prior_cli_defaults_to_on_when_path_is_supplied():
    parser = argparse.ArgumentParser()
    add_motion_prior_args(parser)
    args = parser.parse_args(["--motion_prior_path", "/tmp/ref.npz"])
    cfg = SimpleNamespace(motion_prior_path=None, motion_prior_mode="off", motion_prior_weight=1.0, motion_prior_decay_steps=100)

    update_motion_prior_env_cfg(cfg, args)

    assert cfg.motion_prior_path == "/tmp/ref.npz"
    assert cfg.motion_prior_mode == "on"


def test_motion_prior_cli_rejects_enabled_mode_without_reference():
    cfg = SimpleNamespace(motion_prior_path=None, motion_prior_mode="off", motion_prior_weight=1.0, motion_prior_decay_steps=100)
    args = SimpleNamespace(motion_prior_path=None, motion_prior_mode="decay", motion_prior_weight=None, motion_prior_decay_steps=None)
    with pytest.raises(ValueError, match="requires --motion_prior_path"):
        update_motion_prior_env_cfg(cfg, args)


def test_motion_prior_cli_leaves_unrelated_config_unchanged():
    cfg = SimpleNamespace(other="value")
    args = SimpleNamespace(motion_prior_path=None, motion_prior_mode=None, motion_prior_weight=None, motion_prior_decay_steps=None)
    assert update_motion_prior_env_cfg(cfg, args) is cfg


def test_motion_prior_cli_reconfigures_terms_after_hydra_post_init():
    class Config(SimpleNamespace):
        def configure_motion_prior_terms(self):
            self.terms_configured = True

    cfg = Config(
        motion_prior_path=None,
        motion_prior_mode="off",
        motion_prior_weight=1.0,
        motion_prior_decay_steps=100,
        terms_configured=False,
    )
    args = SimpleNamespace(
        motion_prior_path="/tmp/ref.npz",
        motion_prior_mode="on",
        motion_prior_weight=None,
        motion_prior_decay_steps=None,
    )

    update_motion_prior_env_cfg(cfg, args)

    assert cfg.terms_configured is True
