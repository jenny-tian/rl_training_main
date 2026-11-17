"""Current-filter fidelity and EA diagnostics from saved trajectories only."""
from __future__ import annotations

import argparse
from dataclasses import fields
import sys

import numpy as np
import torch

from reference_stairs_eval import ROOT, save, sha
from reference_online_current import OUTPUT, DOCS, completed, registration

sys.path.insert(0, str(ROOT / 'source/rl_training'))
from rl_training.ea_lpl.basis import BASIS_NAMES, basis
from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter
from rl_training.ea_lpl.current_shadow import finite_difference_jacobian, rollout_mean_action
from rl_training.ea_lpl.shadow import parameterization_jacobian, sensitivity, relative_deviation, request_from_codes
from rl_training.online_adaptation.safety import SafetyLimits
from rl_training.online_adaptation.types import SafetyState

LIMITS = SafetyLimits(residual_limits=torch.tensor([.24, .20, .20] * 4))


def load(directory):
    with np.load(directory / 'trace.npz') as values:
        keys = [k for k in values.files if k.startswith(('state.', 'memory.'))]
        keys += ['base_action', 'action', 'residual_request', 'active', 'terminated', 'truncated', 'learned_effect']
        return {k: torch.from_numpy(values[k]) for k in keys}


def state_at(data, t):
    return SafetyState(**{f.name: data[f'state.{f.name}'][t] for f in fields(SafetyState)})


def fidelity(data):
    live = CurrentModelSafetyFilter(LIMITS)
    action_error = memory_error = component_error = 0.
    for t in range(len(data['base_action'])):
        before = live.temporal_snapshot(data['base_action'][t])
        for k, value in before.items():
            memory_error = max(memory_error, float((value.to(torch.float32)
                - data[f'memory.{k}'][t].to(torch.float32)).abs().max()))
        decision = live.project(data['base_action'][t], data['residual_request'][t], state_at(data, t))
        action_error = max(action_error, float((decision.action - data['action'][t]).abs().max()))
        component_error = max(component_error, float((live.execution_components()['learned_effect']
                                                      - data['learned_effect'][t]).abs().max()))
        done = data['terminated'][t] | data['truncated'][t]
        if done.any():
            live.reset_temporal_state(torch.where(done)[0])
    result = dict(action_max_abs_error=action_error, memory_max_abs_error=memory_error,
                  learned_effect_max_abs_error=component_error, includes_unscored_resets=True)
    if max(action_error, memory_error, component_error) > 1.e-6:
        raise ValueError(f'current safety replay fidelity failure: {result}')
    return result


def switching(data, name, *, interval=10, sigma=.6, seed=0):
    """Counterfactual filter response on fixed physical states; no new rollout."""
    steps, n = data['base_action'].shape[:2]
    v = basis(name).expand(n, 12, 2)
    live = CurrentModelSafetyFilter(LIMITS)
    previous = torch.zeros(n, 2)
    rng = torch.Generator().manual_seed(seed)
    singles, macros, errors = [], [], []
    activated = total = 0
    for t in range(0, steps - interval + 1, interval):
        active = data['active'][t:t+interval].all(0)
        if not active.any():
            break
        bases = list(data['base_action'][t:t+interval])
        states = [state_at(data, tt) for tt in range(t, t+interval)]
        snapshot = live.temporal_snapshot(bases[0])
        code = sigma * torch.randn(n, 2, generator=rng)
        if t > 0:
            zero = torch.zeros(n, 2)
            one = sensitivity(finite_difference_jacobian(LIMITS, snapshot, bases[:1], states[:1], v, zero, .05))
            one0 = sensitivity(parameterization_jacobian(v, zero, .05))
            jac = finite_difference_jacobian(LIMITS, snapshot, bases, states, v, previous, sigma)
            macro0 = sensitivity(parameterization_jacobian(v, previous, sigma))
            actual = (rollout_mean_action(LIMITS, snapshot, bases, states, request_from_codes(v, code))
                      - rollout_mean_action(LIMITS, snapshot, bases, states, request_from_codes(v, previous)))
            predicted = torch.einsum('njk,nk->nj', jac, code-previous)
            err = (actual-predicted).norm(dim=-1) / actual.norm(dim=-1).clamp_min(1.e-9)
            singles.extend(relative_deviation(one, one0)[active].tolist())
            macros.extend(relative_deviation(sensitivity(jac), macro0)[active].tolist())
            errors.extend(err[active].tolist())
        request = request_from_codes(v, code)
        for base, state in zip(bases, states):
            live.project(base, request, state)
            effect = live.execution_components()['learned_effect']
            activated += int(((effect-request).abs().amax(-1) > 1.e-6)[active].sum())
            total += int(active.sum())
        previous = code
    macro_fraction = float(np.mean(np.asarray(macros) >= .3)) if macros else 0.
    median_error = float(np.median(errors)) if errors else None
    return dict(decisions=len(errors), informative_fraction_single=float(np.mean(np.asarray(singles) >= .3)) if singles else 0.,
        informative_fraction_macro=macro_fraction, median_prediction_error=median_error,
        projection_activation=activated / max(total, 1),
        c1b_ea_criterion_met=macro_fraction >= .20 and median_error is not None and median_error <= .30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--switching', action='store_true')
    args = parser.parse_args()
    registration()
    directory = OUTPUT / args.run
    report = completed(directory)
    torch.set_num_threads(1)
    data = load(directory)
    result = dict(run=args.run, safety_memory=CurrentModelSafetyFilter.SEMANTICS,
        report_sha256=sha(directory / 'report.json'), trace_sha256=sha(directory / 'trace.npz'),
        fidelity=fidelity(data), simulation_steps_added=0, optimizer_updates=0,
        counterfactual_caveat='Shadow switching holds physical states and base actions fixed; it is not closed-loop adaptation evidence.')
    mask = data['active'].clone(); mask[:10] = False
    mismatch = (data['learned_effect']-data['residual_request']).abs().amax(-1) > 1.e-6
    guards = data['state.nonfoot_contact'][..., 0] | (data['state.projected_gravity_xy'].norm(dim=-1) > .55)
    result['c1a'] = dict(samples=int(mask.sum()), p_proj=float(mismatch[mask].float().mean()),
                        p_guard=float(guards[mask].float().mean()))
    result['c1a']['ea_criterion_met'] = max(result['c1a']['p_proj'], result['c1a']['p_guard']) >= .05
    if args.switching:
        result['switching_parameters'] = dict(interval=10, sigma=.6, seed=0, delta_single=.05,
            relative_deviation_threshold=.3, macro_fraction_min=.20, median_error_max=.30)
        result['c1b'] = {}
        for condition in dict.fromkeys(r['condition'] for r in report['records']):
            ids = [r['env_id'] for r in report['records'] if r['condition'] == condition]
            subset = {k: v[:, ids] if v.shape[1] != 1 else v for k, v in data.items()}
            result['c1b'][condition] = {name: switching(subset, name) for name in BASIS_NAMES}
            print(condition, result['c1b'][condition], flush=True)
    save(DOCS / f'REPLAY_{args.run}.json', result)
    print(args.run, result['fidelity'], result['c1a'], flush=True)


if __name__ == '__main__':
    main()
