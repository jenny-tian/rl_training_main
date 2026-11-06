import json

from scripts.tools.ea_lpl_p0_summary import resolve_run


def write(folder, status):
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({"status": status}))


def test_resolve_run_prefers_first_passed_attempt(tmp_path):
    write(tmp_path / "p0_e1p_l1", "failed")
    write(tmp_path / "p0_e1p_l1_retry1", "passed")
    write(tmp_path / "p0_e0n_l1", "passed")
    assert resolve_run(tmp_path, "p0_e1p_l1") == tmp_path / "p0_e1p_l1_retry1"
    assert resolve_run(tmp_path, "p0_e0n_l1") == tmp_path / "p0_e0n_l1"
    assert resolve_run(tmp_path, "p0_e0p_l2") is None


def test_resolve_run_returns_none_when_every_attempt_failed(tmp_path):
    write(tmp_path / "p0_e1p_l2", "failed")
    write(tmp_path / "p0_e1p_l2_retry1", "failed")
    assert resolve_run(tmp_path, "p0_e1p_l2") is None
