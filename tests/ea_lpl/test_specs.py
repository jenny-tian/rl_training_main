import pytest

from rl_training.ea_lpl.specs import EQUAL8, p0_specs, validate_spec


def test_p0_specs_are_valid_and_paired():
    specs = {spec["label"]: validate_spec(spec) for spec in p0_specs()}
    assert set(specs) == {"p0_e0n_l1", "p0_e0p_l1", "p0_e1p_l1", "p0_e1p_l2", "p0_e0p_l2"}
    assert specs["p0_e0n_l1"]["columns"] is None and specs["p0_e0n_l1"]["reset_table_seed"] is None
    for name in ("p0_e0p_l1", "p0_e1p_l1", "p0_e1p_l2", "p0_e0p_l2"):
        assert specs[name]["columns"] == list(EQUAL8)
        assert specs[name]["slots"] == list(range(120))
    assert specs["p0_e1p_l1"]["heading_hold"] and not specs["p0_e0p_l1"]["heading_hold"]


def test_spec_rejections():
    spec = dict(p0_specs()[1])
    with pytest.raises(ValueError):
        validate_spec(dict(spec, namespace=9))
    with pytest.raises(ValueError):
        validate_spec(dict(spec, columns=None, episodes_per_env=2))
    with pytest.raises(ValueError):
        validate_spec(dict(spec, max_episode_steps=1000))
    with pytest.raises(ValueError):
        validate_spec(dict(spec, level="L3"))
