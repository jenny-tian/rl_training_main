"""Episode-held-out capacity diagnostic; training replay only, no new physics."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn

from rl_training.online_adaptation.flowdagger import ConditionalFlow, NoisePolicy, balanced_batch, preservation_loss
from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


class WideNoisePolicy(nn.Module):
    def __init__(self, original, copies, generator):
        super().__init__()
        self.bound = original.bound
        self.mlp = nn.Sequential(nn.Linear(48, 256*copies), nn.ELU(),
                                 nn.Linear(256*copies, 128*copies), nn.ELU(), nn.Linear(128*copies, 12))
        with torch.no_grad():
            self.mlp[0].weight.copy_(original.mlp[0].weight.repeat_interleave(copies, 0))
            self.mlp[0].bias.copy_(original.mlp[0].bias.repeat_interleave(copies))
            self.mlp[2].weight.copy_(original.mlp[2].weight.repeat_interleave(copies, 0).repeat_interleave(copies, 1)/copies)
            self.mlp[2].bias.copy_(original.mlp[2].bias.repeat_interleave(copies))
            weight = original.mlp[4].weight.repeat_interleave(copies, 1)/copies
            # Zero-sum outgoing perturbations preserve the initial function but
            # let cloned units receive different gradients during learning.
            perturb = torch.randn((12, 128, copies), generator=generator)*.001
            perturb -= perturb.mean(-1, keepdim=True)
            self.mlp[4].weight.copy_(weight + perturb.reshape(12, -1))
            self.mlp[4].bias.copy_(original.mlp[4].bias)

    def forward(self, obs):
        return self.bound*torch.tanh(self.mlp(obs)/self.bound)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(607)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path("artifacts/flowdagger_20261004/pilot-v5/round-01/adapter.pt")
    prior_path = Path("artifacts/flowdagger_20261003/pilot-v2/final.pt")
    prior = torch.load(prior_path, map_location="cpu", weights_only=False)
    replay = torch.load(source, map_location="cpu", weights_only=False)["buffer"]
    report = dict(status="running", new_physics=0, source_sha256=sha256_file(source),
        prior_sha256=sha256_file(prior_path), seed=607, updates_per_model=1500,
        batch_size=512, lr=3.e-5, preservation_weight=25., models={},
        split="Each whole buffer episode at index0 modulo4 held out; no evaluation trajectories used",
        gate="Wide correction held-out emitted RMS <=0.8*compact, autonomous RMS <=compact+0.002; exact initial function retained")
    write_json_atomic(output/"report.json", report)
    train, valid = {}, {}
    for kind, episodes in replay.items():
        for dataset, remainder in ((train, False), (valid, True)):
            rows = [pair for index, pair in enumerate(episodes) if (index % 4 == 0) == remainder]
            dataset[kind] = tuple(torch.cat([row[i] for row in rows]).to(args.device) for i in (0, 1))
        valid[kind] = tuple(v[::max(1,len(v)//4096)][:4096] for v in valid[kind])
    flow = ConditionalFlow(**prior["flow_config"]).to(args.device).eval().requires_grad_(False)
    flow.load_state_dict(prior["flow"])
    original = NoisePolicy(bound=prior["noise_bound"]); original.load_state_dict(prior["actor"])
    model_builders = {"compact": lambda: NoisePolicy(bound=prior["noise_bound"]),
                      "wide": lambda: WideNoisePolicy(original, 4, torch.Generator().manual_seed(607))}
    start = time.perf_counter()
    for name, builder in model_builders.items():
        model = builder()
        if name == "compact":
            model.load_state_dict(prior["actor"])
        with torch.no_grad():
            probe = train["autonomous"][0][:128].cpu()
            error = float((model(probe)-original(probe)).abs().max())
            if error > 1.e-5:
                raise ValueError("widening changed initial policy")
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=report["lr"])
        generator = torch.Generator(device=args.device).manual_seed(611)
        for update in range(report["updates_per_model"]):
            obs, target = balanced_batch(train, report["batch_size"], generator)
            loss, noise_loss, action_loss = preservation_loss(flow, model, obs, target, weight=25.)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if (update+1) % 250 == 0:
                print(json.dumps(dict(model=name, update=update+1, noise_mse=float(noise_loss), action_mse=float(action_loss))), flush=True)
        metrics = dict(initial_max_error=error, parameters=sum(v.numel() for v in model.parameters()))
        with torch.no_grad():
            for kind, (obs, target) in valid.items():
                predicted = model(obs)
                deviation = clip_lite3_policy_action(flow(obs, predicted))-clip_lite3_policy_action(flow(obs, target))
                rms = deviation.square().mean(-1).sqrt()
                metrics[kind] = dict(rows=len(obs), noise_mse=float((predicted-target).square().mean()),
                    emitted_rms_mean=float(rms.mean()), emitted_rms_p99=float(torch.quantile(rms,.99)))
        report["models"][name] = metrics
        torch.save(dict(actor=model.state_dict(), hidden=[256,128] if name == "compact" else [1024,512]), output/f"{name}.pt")
        write_json_atomic(output/"report.json", report)
    a, b = report["models"]["compact"], report["models"]["wide"]
    report.update(status="complete", seconds=time.perf_counter()-start,
        wide_gate_passed=b["correction"]["emitted_rms_mean"] <= .8*a["correction"]["emitted_rms_mean"]
            and b["autonomous"]["emitted_rms_mean"] <= a["autonomous"]["emitted_rms_mean"]+.002)
    write_json_atomic(output/"report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
