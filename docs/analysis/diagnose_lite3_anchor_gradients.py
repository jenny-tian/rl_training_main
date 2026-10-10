"""Local TRAIN gradient geometry; never claim unavailable historical minibatches."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def vector(loss, parameters):
    return torch.cat([g.detach().flatten() for g in torch.autograd.grad(loss, parameters)])


def geometry(gradients):
    norms = {k: float(v.double().norm()) for k, v in gradients.items()}
    cosines = {}
    for a, x in gradients.items():
        for b, y in gradients.items():
            if a < b:
                cosines[a+' / '+b] = float(torch.dot(x.double(), y.double())/(norms[a]*norms[b])) if norms[a]*norms[b] else None
    return dict(norms=norms, cosine=cosines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'protocol', 'training-audit', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.run)
    p = json.loads(Path(args.protocol).read_text())
    r = json.loads((root/'report.json').read_text())
    audit = json.loads(Path(args.training_audit).read_text())
    assert r['status'] == 'complete' and audit['status'] == 'verified'
    assert r['protocol_sha256'] == sha256_file(args.protocol)
    assert audit['evidence'][str(root/'report.json')] == sha256_file(root/'report.json')
    assert p['recovery_resets']['split'] == 'train' and audit['validation_cases_used'] == 0
    assert p['ppo']['entropy_coef'] == 0 and not p.get('tail_anchor')
    paths = [Path(args.protocol), Path(args.training_audit), root/'report.json', root/'rsl/model_0.pt',
             root/'final.pt', Path(p['base_checkpoint']), Path(p['mean_anchor']['dataset'])]
    payload = torch.load(root/'rsl/model_0.pt', map_location='cpu', weights_only=False)
    assert payload['iter'] == 0
    assert {int(v['step']) for v in payload['optimizer_state_dict']['state'].values()} == {20}
    actor = FrozenPPOExpert(payload['actor_state_dict']).requires_grad_(True)
    critic = nn.Sequential(nn.Linear(51,512), nn.ELU(), nn.Linear(512,256), nn.ELU(), nn.Linear(256,128), nn.ELU(), nn.Linear(128,1))
    critic.load_state_dict({k[4:]:v for k,v in payload['critic_state_dict'].items() if k.startswith('mlp.')})
    critic.requires_grad_(False)
    trace = root/'trajectory/chunk-00001.npz'
    assert sha256_file(trace) == audit['trace_sha256'][trace.name]
    paths.append(trace)
    with np.load(trace, allow_pickle=False) as d:
        obs = torch.from_numpy(d['policy_observation'])
        critic_obs = torch.from_numpy(d['critic_observation'])
        actions = torch.from_numpy(d['requested_action'])
        rewards, done = torch.from_numpy(d['reward']), torch.from_numpy(d['done']).bool()
        with torch.no_grad():
            values = critic(critic_obs).squeeze(-1)
            last_value = critic(torch.from_numpy(d['next_critic_observation'][-1])).squeeze(-1)
            advantages = torch.zeros_like(values)
            running = 0
            for i in reversed(range(len(obs))):
                future = last_value if i == len(obs)-1 else values[i+1]
                running = rewards[i] + (~done[i])*p['ppo']['gamma']*future-values[i] + (~done[i])*p['ppo']['gamma']*p['ppo']['lam']*running
                advantages[i] = running
            advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
        cases = d['recovery_case'].copy()
    flat_obs, flat_actions = obs.flatten(0,1), actions.flatten(0,1)
    std = payload['actor_state_dict']['distribution.log_std_param'].exp()
    with torch.no_grad():
        means = actor(flat_obs)
        old_logprob = torch.distributions.Normal(means,std).log_prob(flat_actions).sum(-1)
        standardized = (flat_actions-means)/std
    logprob = torch.distributions.Normal(actor(flat_obs),std).log_prob(flat_actions).sum(-1)
    surrogate = -(advantages.flatten()*torch.exp(logprob-old_logprob)).mean()
    parameters = list(actor.mlp.parameters())
    ppo_gradient = vector(surrogate, parameters)
    reference = FrozenPPOExpert(torch.load(p['base_checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict'])
    spec = p['mean_anchor']
    assert sha256_file(spec['dataset']) == spec['dataset_sha256']
    with np.load(spec['dataset'], allow_pickle=False) as data:
        states = {k:torch.from_numpy(data[k+'_train_obs'].copy()) for k in spec['batch_groups']}
    generator = torch.Generator().manual_seed(spec['seed'])
    for _ in range(20):
        for k,n in spec['batch_groups'].items():torch.randint(len(states[k]),(n,),generator=generator)
    sampled = {k:states[k][torch.randint(len(states[k]),(n,),generator=generator)] for k,n in spec['batch_groups'].items()}
    total = sum(spec['batch_groups'].values())
    results = {}
    for name, model in [('after_update1',actor), ('final256',FrozenPPOExpert(torch.load(root/'final.pt',map_location='cpu',weights_only=False)['actor_state_dict']).requires_grad_(True))]:
        gradients = {}
        losses = {}
        for group,x in sampled.items():
            loss = spec['coefficient']*len(x)/total*(model(x)-reference(x)).square().mean()
            losses[group] = float(loss.detach())
            gradients[group] = vector(loss,list(model.mlp.parameters()))
        gradients['ordinary_sum'] = sum(gradients[k] for k in ('flat','rough101','rough107'))
        gradients['anchor_sum'] = gradients['ordinary_sum']+gradients['correction']
        if name == 'after_update1':gradients['ppo_full_rollout'] = ppo_gradient
        results[name] = dict(weighted_group_losses=losses, **geometry(gradients))
    write_json_atomic(args.output, dict(status='verified', checkpoints=results, rollout_controls=int(done.numel()),
        recovery_controls=int((cases>=0).sum()), sampled_rows=spec['batch_groups'],
        standardized_action_mean=float(standardized.mean()), standardized_action_std=float(standardized.std()),
        new_controls=0, policy_optimizer_steps=0,
        scope='CPU local Euclidean gradients on actor mean parameters only, before clipping/Adam. model_0 is after20 optimizer steps; next TRAIN rollout uses that behavior checkpoint. Full-rollout normalized GAE surrogate at ratio1, NOT the unavailable actual historical minibatch order. Replay uses update2 first draw, four groups128. Final checkpoint evaluates the SAME TRAIN replay sample, with no claim about final PPO gradient. No pilot or held-out observations.',
        evidence={str(x):sha256_file(x) for x in paths}))


if __name__ == '__main__':
    main()
