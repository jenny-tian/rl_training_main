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
        validate_spec(dict(spec, level="L4"))


from rl_training.ea_lpl.headroom import grid_codes, grid_rows, pack_specs  # noqa: E402


def test_residual_code_specs_validate():
    targets = [dict(level="L1", column=4, terrain="slope_cross_slope_left")]
    specs = pack_specs(grid_rows(targets), label_prefix="p1_grid", namespace=2)
    assert len(specs) == 5
    for spec in specs:
        validate_spec(spec)
        assert len(spec["codes"]["basis"]) == 120
    names = [n for spec in specs for n in spec["codes"]["code_name"] if n != "pad"]
    assert len(names) == 73 * 8 and names.count("zero") == 8


def test_code_specs_need_fixed_columns_and_bounded_codes():
    spec = pack_specs(grid_rows([dict(level="L1", column=4, terrain="x")]), label_prefix="p", namespace=2)[0]
    with pytest.raises(ValueError):
        validate_spec(dict(spec, columns=None))
    bad = dict(spec, codes=dict(spec["codes"], u=[[5.0, 0.0]] * 120))
    with pytest.raises(ValueError):
        validate_spec(bad)
