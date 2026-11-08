from rl_training.ea_lpl.headroom import grid_codes


def test_grid_has_one_zero_and_24_codes_per_basis():
    codes = grid_codes()
    assert len(codes) == 73 and codes[0]["code_name"] == "zero"
    for name in ("sag", "roll", "hipx"):
        assert sum(c["basis"] == name and c["code_name"] != "zero" for c in codes) == 24
    assert len({c["code_name"] for c in codes}) == 73


from rl_training.ea_lpl.headroom import (choose_basis, confirmation_candidates, confirmation_result, ea_decision,
                                         screen_table)


def row(code_name, basis, completed, progress=5.0, safety=False, column=4, slot=0):
    return dict(level="L1", column=column, terrain="slope_cross_slope_left", code_name=code_name, basis=basis,
                u=[1.0, 0.0], slot=slot, completed=completed, safety_ending=safety, forward_progress_m=progress)


def test_screen_and_candidates():
    rows = [row("zero", "sag", s < 1, slot=s) for s in range(8)]
    rows += [row("roll:+1,+0", "roll", s < 4, slot=s) for s in range(8)]
    rows += [row("roll:+2,+0", "roll", s < 2, slot=s) for s in range(8)]
    rows += [row("pad", "sag", True, slot=1000)]
    table = screen_table(rows)
    assert {r["code_name"]: r["completed"] for r in table} == {"zero": 1, "roll:+1,+0": 4, "roll:+2,+0": 2}
    assert [c["code_name"] for c in confirmation_candidates(table)] == ["roll:+1,+0"]


def test_confirmation_and_basis_choice():
    table = [dict(level="L1", column=4, terrain="t", code_name="zero", basis="sag", u=[0.0, 0.0], n=16,
                  completed=3, safety=0, progress_mean_m=5.0),
             dict(level="L1", column=4, terrain="t", code_name="roll:+1,+0", basis="roll", u=[1.0, 0.0], n=16,
                  completed=8, safety=1, progress_mean_m=7.0),
             dict(level="L1", column=4, terrain="t", code_name="sag:+1,+0", basis="sag", u=[1.0, 0.0], n=16,
                  completed=6, safety=3, progress_mean_m=6.0)]
    confirmed = confirmation_result(table)
    flags = {r["code_name"]: r["confirmed"] for r in confirmed}
    assert flags == {"roll:+1,+0": True, "sag:+1,+0": False}
    choice = choose_basis(confirmed)
    assert choice["basis"] == "roll" and choice["targets"][0]["oracle_u"] == [1.0, 0.0]
    assert choose_basis([dict(r, confirmed=False) for r in confirmed]) is None


def test_ea_decision_thresholds():
    decision = ea_decision(dict(p_proj=0.01, p_guard=0.02),
                           dict(sag=dict(informative_fraction_macro=0.25, median_prediction_error=0.2)),
                           basis_name="sag")
    assert decision == dict(c1a_go=False, c1b_go=True)
