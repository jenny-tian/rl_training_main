from rl_training.ea_lpl.headroom import grid_codes


def test_grid_has_one_zero_and_24_codes_per_basis():
    codes = grid_codes()
    assert len(codes) == 73 and codes[0]["code_name"] == "zero"
    for name in ("sag", "roll", "hipx"):
        assert sum(c["basis"] == name and c["code_name"] != "zero" for c in codes) == 24
    assert len({c["code_name"] for c in codes}) == 73
