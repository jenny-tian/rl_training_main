"""Distil a route48 Gaussian PPO prior into a conditional rectified flow.

No new physics and no evaluation trajectories are consumed. The protocol pins
the expert, training chunks, split, optimization budget and final-only choice.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl_training.online_adaptation.flowdagger import ConditionalFlow, FrozenPPOExpert, gaussian_distillation_loss, invert_euler
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.lite3_joint_conventions import clip_lite3_policy_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    if p["schema"] != "lite3_flowdagger_pilot_v1":
        raise ValueError("unknown protocol")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", protocol_sha256=sha256_file(args.protocol), new_physics_controls=0, updates=0)
    write_json_atomic(out / "report.json", report)
    started = time.perf_counter()
    try:
        torch.set_num_threads(4)
        torch.manual_seed(p["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if sha256_file(p["teacher_checkpoint"]) != p["teacher_sha256"]:
            raise ValueError("expert checksum mismatch")
        teacher_state = torch.load(p["teacher_checkpoint"], map_location="cpu", weights_only=False)["actor_state_dict"]
        expert = FrozenPPOExpert(teacher_state).to(args.device)
        expert_initial = {k: v.clone() for k, v in expert.state_dict().items()}
        training, validation, sources = [], [], {}
        for entry in p["training_chunks"]:
            path = Path(entry["path"])
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("training data checksum mismatch")
            sources[str(path)] = entry["sha256"]
            with np.load(path, allow_pickle=False) as chunk:
                obs = chunk["policy_observation"][::p["pretrain"]["time_stride"]]
            if obs.shape[1:] != (128, 48) or not np.isfinite(obs).all():
                raise ValueError("training observation schema mismatch")
            # Entire environment streams are disjoint, even across chunk/episode boundaries.
            held = np.arange(128) % 4 == 0
            training.append(obs[:, ~held].reshape(-1, 48))
            validation.append(obs[:, held].reshape(-1, 48))
        train_obs = torch.from_numpy(np.concatenate(training)).to(args.device)
        valid_obs = torch.from_numpy(np.concatenate(validation)).to(args.device)
        valid_obs = valid_obs[::max(1, len(valid_obs) // 4096)][:4096]
        report.update(training_rows=len(train_obs), validation_rows=len(valid_obs), training_sources=sources,
                      validation_split="all env IDs divisible by4 held out across every training chunk")
        flow = ConditionalFlow(steps=p["flow"]["steps"]).to(args.device)
        flow.initialize_from_expert(expert)
        flow_initial = {k: v.clone() for k, v in flow.state_dict().items()}
        optimizer = torch.optim.Adam(flow.parameters(), lr=p["pretrain"]["learning_rate"])
        gen = torch.Generator(device=args.device).manual_seed(p["seed"] + 1)
        std = teacher_state["distribution.log_std_param"].exp().to(args.device)
        for update in range(p["pretrain"]["updates"]):
            index = torch.randint(len(train_obs), (p["pretrain"]["batch_size"],), generator=gen, device=args.device)
            obs = train_obs[index]
            with torch.no_grad():
                mean = expert(obs)
            noise = torch.randn(mean.shape, generator=gen, device=args.device)
            t = torch.rand((len(obs), 1), generator=gen, device=args.device)
            loss, fm, center = gaussian_distillation_loss(flow, obs, mean, std, noise, t, p["pretrain"]["center_weight"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite flow objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            report["updates"] = update + 1
            if (update + 1) % 250 == 0:
                event = dict(update=update + 1, flow_loss=float(fm), center_loss=float(center), seconds=time.perf_counter()-started)
                with (out / "updates.jsonl").open("a") as stream:
                    stream.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
                write_json_atomic(out / "report.json", report)
        flow.eval().requires_grad_(False)
        with torch.no_grad():
            target = expert(valid_obs)
            zero = torch.zeros_like(target)
            predicted = flow(valid_obs, zero)
            error = clip_lite3_policy_action(predicted) - clip_lite3_policy_action(target)
            inverse = invert_euler(flow, valid_obs, target, **p["inversion"])
            state_dependency = (flow(valid_obs, torch.ones_like(target)) - predicted).abs().mean()
        report.update(emitted_action_rmse=float(error.square().mean().sqrt()),
                      emitted_action_max_error=float(error.abs().max()),
                      inversion_accepted=int(inverse.accepted.sum()), inversion_total=len(target),
                      inverse_max_error=float(inverse.max_error.max()),
                      latent_response_mean=float(state_dependency),
                      flow_changed=any(not torch.equal(v, flow.state_dict()[k]) for k, v in flow_initial.items()),
                      expert_unchanged=all(torch.equal(v, expert.state_dict()[k]) for k, v in expert_initial.items()))
        checkpoint = dict(schema="lite3_conditional_flow_v1", config=flow.config, flow=flow.state_dict(),
                          optimizer=optimizer.state_dict(), protocol_sha256=report["protocol_sha256"],
                          expert_sha256=p["teacher_sha256"], updates=report["updates"])
        torch.save(checkpoint, out / "final.pt")
        report.update(checkpoint_sha256=sha256_file(out / "final.pt"), seconds=time.perf_counter()-started)
        report["engineering_gate_passed"] = (report["emitted_action_rmse"] <= p["pretrain"]["max_validation_rmse"]
            and report["inversion_accepted"] == report["inversion_total"] and report["flow_changed"]
            and report["expert_unchanged"] and float(state_dependency) > .01)
        report["status"] = "complete"
        write_json_atomic(out / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "training_sources"}), flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error), seconds=time.perf_counter()-started)
        write_json_atomic(out / "report.json", report)
        raise


if __name__ == "__main__":
    main()
