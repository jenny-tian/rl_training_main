"""Independently reconstruct the FlowDAgger pilot from persisted step chunks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.flowdagger import ConditionalFlow, NoisePolicy


def audit_phase(folder, protocol):
    report = json.loads((folder / "report.json").read_text())
    assert report["status"] == "complete"
    initial = np.load(folder / "initial_state.npz", allow_pickle=False)
    manifest = json.loads((folder / "trajectory/manifest.json").read_text())
    assert manifest["status"] == "complete"
    width = protocol["num_envs"]
    active = np.ones(width, dtype=bool)
    last_history = np.zeros((width, 12), dtype=np.float32)
    peak = np.zeros(width); peak_from_start = np.zeros(width); squares = np.zeros(width); samples = np.zeros(width)
    successes = np.zeros(width, dtype=bool)
    ends = np.zeros(width, dtype=int)
    intervention_count = 0
    steps = 0
    hashes = {}
    for chunk in manifest["chunks"]:
        file = folder / "trajectory" / chunk["file"]
        hashes[chunk["file"]] = sha256_file(file)
        with np.load(file, allow_pickle=False) as data:
            obs = data["policy_observation"]
            assert np.isfinite(obs).all()
            np.testing.assert_allclose(obs[:, :, 6:9], np.broadcast_to([.3, 0., 0.], obs[:, :, 6:9].shape), rtol=0., atol=1.e-7)
            chosen = np.where(data["intervention"][:, :, None], data["teacher_requested_action"], data["student_requested_action"])
            expected_action = clip_lite3_policy_action(torch.from_numpy(chosen)).numpy()
            np.testing.assert_array_equal(data["emitted_action"], expected_action)
            for row in range(len(obs)):
                np.testing.assert_array_equal(data["active"][row], active)
                np.testing.assert_allclose(obs[row, :, -12:], last_history, atol=1.e-6, rtol=0.)
                np.testing.assert_array_equal(data["done"][row].astype(bool),
                    data["success"][row] | data["failure"][row] | data["deadline"][row])
                lateral = data["post_root_state"][row, :, 1] - initial["origins"][:, 1]
                offset = data["post_root_state"][row, :, 1] - initial["root_state"][:, 1]
                peak = np.maximum(peak, np.where(active, abs(lateral), 0.))
                peak_from_start = np.maximum(peak_from_start, np.where(active, abs(offset), 0.))
                squares += np.where(active, lateral**2, 0.)
                samples += active
                ending = active & data["done"][row].astype(bool)
                successes[ending] = data["success"][row, ending]
                ends[ending] = steps + 1
                intervention_count += int(data["intervention"][row].sum())
                assert not (data["intervention"][row] & ~active).any()
                active &= ~data["done"][row].astype(bool)
                last_history = data["emitted_action"][row].copy()
                last_history[data["done"][row].astype(bool)] = 0.
                steps += 1
    assert steps * width == report["confirmed_controls"]
    assert intervention_count == report["interventions"]
    if report["mode"] != "collect":
        assert intervention_count == 0
    for episode in report["episodes"]:
        i = episode["env_id"]
        assert bool(successes[i]) == episode["success"]
        assert (ends[i] or steps) == episode["steps"]
        np.testing.assert_allclose(peak[i], episode["max_body_lateral_m"], atol=1.e-6)
        np.testing.assert_allclose(peak_from_start[i], episode["max_body_offset_from_start_m"], atol=1.e-6)
        np.testing.assert_allclose(np.sqrt(squares[i]/samples[i]), episode["body_lateral_rms_m"], atol=1.e-5)
    return dict(status="verified", transitions=steps*width, interventions=intervention_count,
                max_body_lateral_m=float(peak.max()), median_trial_peak_m=float(np.median(peak)),
                max_body_offset_from_start_m=float(peak_from_start.max()),
                mean_trial_body_rms_m=float(np.sqrt(squares/samples).mean()),
                by_terrain=report["by_terrain"], trace_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.run)
    report = json.loads((directory / "report.json").read_text())
    p = json.loads(Path(args.protocol).read_text())
    assert report["status"] == "complete"
    assert report["protocol_sha256"] == sha256_file(args.protocol)
    phases = [f"round-{i:02d}" for i in range(p["rounds"])] + [f"evaluation-{m}" for m in ("teacher", "flow", "adapted")]
    results = {name: audit_phase(directory / name, p) for name in phases}
    assert sum(r["transitions"] for r in results.values()) == report["ledger"]["confirmed"]
    assert report["ledger"]["uncertain_transitions"] == 0
    assert report["initial_states_paired"] and report["frozen_flow_and_teacher_unchanged"] and report["evaluation_actor_unchanged"]
    assert report["actor_changed"]
    assert sha256_file(directory / "final.pt") == report["final_checkpoint_sha256"]
    # Reconstruct deployment from the standalone saved networks on CPU. The
    # expert is never loaded in this check, so it cannot mask a teacher fallback.
    torch.set_num_threads(4)
    payload = torch.load(directory / "final.pt", map_location="cpu", weights_only=False)
    flow = ConditionalFlow(**payload["flow_config"]).eval().requires_grad_(False)
    flow.load_state_dict(payload["flow"])
    actor = NoisePolicy(bound=payload["noise_bound"]).eval().requires_grad_(False)
    actor.load_state_dict(payload["actor"])
    replay_checks = 0
    for mode in ("flow", "adapted"):
        files = sorted((directory / f"evaluation-{mode}" / "trajectory").glob("chunk-*.npz"))
        for file in (files[0], files[len(files)//2], files[-1]):
            with np.load(file, allow_pickle=False) as data:
                active = data["active"].reshape(-1)
                obs = torch.from_numpy(data["policy_observation"].reshape(-1, 48)[active][::64][:32])
                recorded = torch.from_numpy(data["student_requested_action"].reshape(-1, 12)[active][::64][:32])
                noise = torch.zeros((len(obs), 12)) if mode == "flow" else actor(obs)
                torch.testing.assert_close(flow(obs, noise), recorded, atol=3.e-5, rtol=1.e-5)
                replay_checks += len(obs)
    audit = dict(status="verified", run=str(directory.resolve()), protocol_sha256=sha256_file(args.protocol),
                 phases=results, ledger=report["ledger"], final_checkpoint_sha256=report["final_checkpoint_sha256"],
                 cpu_deployment_replay_rows=replay_checks, replay_without_teacher=True,
                 scope="development evaluation on paired initial states; no independent training replication or new-terrain claim")
    write_json_atomic(args.output, audit)
    print(json.dumps({k: {n:v for n,v in r.items() if n != "trace_sha256"} for k,r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
