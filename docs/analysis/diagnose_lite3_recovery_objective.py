"""Audit phase-specific objectives on the immutable 49D TRAIN pair.

No simulation, optimization, pilot trajectories or invented reward components.
The body-only lateral cost is a lower bound on cost, not the recorded foot-aware
reward. Replay losses and environment rewards deliberately retain separate units.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


DT = .02
PHASES = (("0-0.24s", 0, 12), ("0.24-1s", 12, 50), ("1-2s", 50, 100),
          ("2-4s", 100, 200), ("4s+", 200, None))


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return dict(rows=0, mean=None, median=None, p95=None, maximum=None)
    assert np.isfinite(x).all()
    return dict(rows=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.quantile(x, .95)), maximum=float(x.max()))


def reconstruct(d, p):
    """Use reward-time/pre-reset geometry; never returned reset positions."""
    state, origin = d['heading_root_state'], d['heading_origins']
    np.testing.assert_array_equal(state, d['velocity_root_state'])
    np.testing.assert_array_equal(origin, d['velocity_origins'])
    y = state[..., 1] - origin[..., 1]
    pre_y = d['pre_root_state'][..., 1] - origin[..., 1]
    age, case = d['episode_steps'], d['recovery_case']
    anchor = (case < 0) | (age < p['recovery_anchor']['early_steps'])
    np.testing.assert_array_equal(d['recovery_anchor_mask'], anchor)
    cost = np.where(anchor, np.mean((d['emitted_action'] - d['recovery_anchor_target'])**2, -1), 0)
    np.testing.assert_allclose(cost, d['recovery_anchor_applied_cost'], atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(d['recovery_anchor_reward'], -p['recovery_anchor']['weight']*cost*DT, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(d['recovery_anchor_reward'][~anchor], 0)
    w, x, qy, z = np.moveaxis(state[..., 3:7], -1, 0)
    yaw = np.arctan2(2*(w*z+x*qy), 1-2*(qy*qy+z*z))
    heading = p['route_training']
    target_yaw = np.clip(np.arctan2(-y, np.full_like(y, heading['heading_lookahead_m'])),
                         -np.deg2rad(heading['heading_maximum_deg']), np.deg2rad(heading['heading_maximum_deg']))
    np.testing.assert_allclose(d['heading_reward'], heading['heading_weight']*(1-np.cos(yaw-target_yaw))*DT, atol=1e-6, rtol=1e-5)
    v = p['lateral_velocity']
    target_v = np.clip(-v['gain_per_second']*y, -v['maximum_m_s'], v['maximum_m_s'])
    np.testing.assert_allclose(d['velocity_reward'], v['weight']*(state[..., 8]-target_v)**2*DT, atol=1e-6, rtol=1e-5)
    failure, success, deadline = [d['task_'+k].astype(bool) for k in ('failure', 'success', 'deadline')]
    assert not np.any((failure & success) | (failure & deadline) | (success & deadline))
    np.testing.assert_array_equal(d['done'].astype(bool), failure | success | deadline)
    terminal = np.where(failure | deadline, -2., np.where(success, 5., 0.))
    # Only three reward components were recorded directly. The terminal term is
    # reconstructed from the verified event flags and the unchanged source.
    remainder = (d['reward'].astype(np.float64) - d['recovery_anchor_reward']
                 - d['heading_reward'] - d['velocity_reward'] - terminal)
    result = dict(abs_body_y_m=np.abs(y), outward_root_velocity_m_s=np.sign(y)*state[..., 8],
                  outward_origin_difference_m_s=(np.abs(y)-np.abs(pre_y))/DT,
                  outward_fraction=(np.sign(y)*state[..., 8] > 0).astype(float),
                  heading_target_saturated_fraction=(np.abs(target_yaw) >= np.deg2rad(heading['heading_maximum_deg'])-1e-6).astype(float),
                  velocity_target_saturated_fraction=(np.abs(target_v) >= v['maximum_m_s']-1e-6).astype(float),
                  anchor_active_fraction=anchor.astype(float), reward=d['reward'],
                  anchor_reward=d['recovery_anchor_reward'], heading_reward=d['heading_reward'],
                  velocity_reward=d['velocity_reward'], terminal_reward=terminal,
                  unallocated_reward=remainder,
                  body_lateral_penalty_magnitude_lower_bound=(-p['recovery_course_lateral_weight'])*y**2*DT)
    assert all(np.isfinite(x).all() for x in result.values())
    return result


def collect_episodes(age, done, success, failure, deadline, y, reward, gamma):
    """Retain every complete and right-censored recovery episode across chunks."""
    assert age.ndim == 2 and all(a.shape == age.shape for a in (done, success, failure, deadline, y, reward))
    np.testing.assert_array_equal(age[0], 0)
    np.testing.assert_array_equal(age[1:], np.where(done[:-1], 0, age[:-1]+1))
    episodes = []
    for slot in range(age.shape[1]):
        start = 0
        ends = np.flatnonzero(done[:, slot]).tolist()
        if not ends or ends[-1] != len(age)-1:
            ends.append(len(age)-1)
        for end in ends:
            if start > end:
                continue
            indices = slice(start, end+1)
            abs_y = abs(y[indices, slot])
            peak = int(np.argmax(abs_y))
            terminal = bool(done[end, slot])
            outcome = ('success' if success[end, slot] else 'failure' if failure[end, slot]
                       else 'deadline' if deadline[end, slot] else 'censored')
            assert terminal == (outcome != 'censored')
            episodes.append(dict(slot=slot, start_control=start, end_control=end,
                controls=end-start+1, outcome=outcome, peak_body_y_m=float(abs_y[peak]),
                peak_age_s=(peak+1)*DT, peak_global_step=start+peak,
                reward_sum=float(reward[indices, slot].sum(dtype=np.float64)),
                discounted_reward_sum=float(np.dot(gamma**np.arange(end-start+1), reward[indices, slot])),
                first12_reward_sum=float(reward[start:min(start+12, end+1), slot].sum(dtype=np.float64))))
            start = end+1
    assert sum(e['controls'] for e in episodes) == age.size
    return episodes


def temporal_weights(gamma, lam, rollout_steps):
    """Separate discounted-return weights, GAE trace weights and truncation."""
    q = gamma*lam
    return dict(gamma=gamma, lam=lam, gamma_lambda=q, rollout_steps=rollout_steps,
        rollout_seconds=rollout_steps*DT, discount_efold_seconds=-DT/math.log(gamma),
        gae_trace_efold_seconds=-DT/math.log(q),
        offsets={str(n): dict(seconds=n*DT, return_discount=gamma**n,
                              nominal_untruncated_gae_trace=q**n,
                              fraction_rollout_positions_with_direct_reward=(max(rollout_steps-n, 0)/rollout_steps))
                 for n in (12, 25, 31, 32, 50, 100, 200)},
        interpretation='Direct reward coefficient with value predictions fixed. Within a rollout it is (gamma*lambda)^lag until done; across rollout boundaries it is zero. Bootstrapped critic values can carry longer-term information. This is not a measured effective learning horizon or proof that the critic failed.')


def gradient_summary(records):
    groups = {}
    for group in ('anchor', 'correction', 'ordinary'):
        ratios = [r['norms'][group]/r['norms']['ppo'] for r in records if r['norms']['ppo'] > 0]
        cosines = [r['cosine'][group+'_vs_ppo'] for r in records if r['cosine'][group+'_vs_ppo'] is not None]
        groups[group] = dict(norm_ratio_to_ppo=stats(ratios), cosine=stats(cosines),
                             negative_cosine_fraction=float(np.mean(np.array(cosines) < 0)))
    return dict(records=len(records), groups=groups,
                weighted_replay_group_losses={g: stats([r['weighted_group_losses'][g] for r in records])
                                               for g in records[0]['weighted_group_losses']})


def return_ordering(episodes, gamma):
    """Describe observed outcomes; do not infer an achievable failure action."""
    result = {}
    for name, selected in [('all', episodes)] + [
            ('start_updates_'+str(lo+1)+'-'+str(lo+64),
             [e for e in episodes if lo*32 <= e['start_control'] < (lo+64)*32])
            for lo in range(0, 256, 64)]:
        slots = {}
        for slot in range(12):
            outcomes = {}
            for outcome in ('success', 'failure'):
                rows = [e for e in selected if e['slot'] == slot and e['outcome'] == outcome]
                outcomes[outcome] = stats([e['discounted_reward_sum'] for e in rows])
            slots[str(slot+40)] = outcomes
        comparable = [v for v in slots.values() if all(v[k]['rows'] for k in ('success', 'failure'))]
        success = [e for e in selected if e['outcome'] == 'success']
        failure = [e for e in selected if e['outcome'] == 'failure']
        result[name] = dict(
            success=stats([e['discounted_reward_sum'] for e in success]),
            failure=stats([e['discounted_reward_sum'] for e in failure]),
            slots=slots, comparable_slots=len(comparable),
            slots_success_mean_below_failure=sum(v['success']['mean'] < v['failure']['mean'] for v in comparable),
            fixed_trace_failure_rescore_minus5=stats([
                e['discounted_reward_sum']-3*gamma**(e['controls']-1) for e in failure]))
    return dict(windows=result,
        interpretation='Raw environment discounted returns, before replay loss and PPO/critic estimation. Stratified by environment slot and episode start update block; policies, reset randomness and sampled actions still differ. Minus5 only substitutes the terminal failure charge on the identical observed trajectory: G_new=G_old-3*gamma^(T-1). It is arithmetic sensitivity, not a trained policy, causal failure preference, or a change to any acceptance criterion.')


def analyze_arm(arm, plan, evidence):
    root = Path('artifacts/route_history_20261010')/arm/'train'
    prefix = 'docs/lite3_route_history_'+arm
    def read(path):
        evidence[str(path)] = sha256_file(path)
        return json.loads(Path(path).read_text())
    protocol_path = prefix+'_train_protocol_20261010.json'
    p, r = read(protocol_path), read(root/'report.json')
    assert r['status'] == 'complete' and r['protocol_sha256'] == evidence[protocol_path]
    assert p['recovery_resets']['split'] == 'train'
    assert p['route_history']['enabled'] == (arm == 'history')
    import yaml
    for name in ('env', 'agent'):
        path = root/(name+'.yaml')
        evidence[str(path)] = sha256_file(path)
        cfg = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        if name == 'env':
            assert float(cfg['sim']['dt'])*int(cfg['decimation']) == DT
            terminal = cfg['rewards']['course_terminal']
            assert terminal['func'] == 'rl_training.policy_course:course_terminal_reward'
            assert terminal['params'] == {} and float(terminal['weight']) == 1.
            assert float(cfg['rewards']['course_lateral']['weight']) == p['recovery_course_lateral_weight']
        else:
            assert int(cfg['num_steps_per_env']) == p['rollout_steps']
            assert all(float(cfg['algorithm'][k]) == p['ppo'][k] for k in ('gamma', 'lam'))
    audits = [read(prefix+'_'+name+'_audit_20261010.json') for name in ('training', 'heading', 'velocity', 'anchor', 'update')]
    for audit in audits:
        assert audit['status'] == 'verified'
        assert audit['evidence'][str(root/'report.json')] == evidence[str(root/'report.json')]
    assert audits[0]['exposed_training_cases'] == [26] and audits[0]['validation_cases_used'] == 0
    # Historical runtime source must still match before interpreting its terms.
    source_paths = ('source/rl_training/rl_training/policy_course.py',
                    'source/rl_training/rl_training/policy_route.py',
                    'source/rl_training/rl_training/policy_recovery_anchor.py',
                    'scripts/reinforcement_learning/train_lite3_policy.py')
    for path in source_paths:
        evidence[path] = sha256_file(path)
        assert evidence[path] == r['source_files'][path]
    manifest = read(root/'trajectory/manifest.json')
    files = sorted((root/'trajectory').glob('chunk-*.npz'))
    assert manifest['status'] == 'complete'
    assert [f.name for f in files] == [c['file'] for c in manifest['chunks']]
    assert len(files) == p['iterations'] == 256 and p['rollout_steps'] == 32
    expected_names = {f.name for f in files}
    for a in audits[:4]:
        assert set(a['trace_sha256']) == expected_names
    parts, ages, cases, all_steps, episode_arrays = {}, [], [], [], {}
    for number, path in enumerate(files):
        digest = sha256_file(path)
        assert all(a['trace_sha256'][path.name] == digest for a in audits[:4])
        evidence[str(path)] = digest
        with np.load(path, allow_pickle=False) as d:
            assert d['reward'].shape == (32, p['num_envs'])
            for k, v in reconstruct(d, p).items():
                parts.setdefault(k, []).append(v)
            ages.append(d['episode_steps'].copy())
            cases.append(d['recovery_case'].copy())
            all_steps.append(np.broadcast_to(np.arange(number*32, (number+1)*32)[:, None], d['reward'].shape))
            for k in ('done', 'task_success', 'task_failure', 'task_deadline', 'reward'):
                episode_arrays.setdefault(k, []).append(d[k][:, 40:52].copy())
            episode_arrays.setdefault('y', []).append((d['heading_root_state'][:, 40:52, 1]-d['heading_origins'][:, 40:52, 1]).copy())
    values = {k: np.concatenate(v) for k, v in parts.items()}
    age, case, steps = [np.concatenate(v) for v in (ages, cases, all_steps)]
    assert age.size == r['ledger']['confirmed'] == r['ledger']['charged'] == plan['training_controls_per_arm']
    assert r['ledger']['uncertain_transitions'] == 0 and int((case == 26).sum()) == audits[0]['recovery_controls']
    expected = np.full(64, -1); expected[40:52] = 26
    np.testing.assert_array_equal(case, np.broadcast_to(expected, case.shape))
    assert manifest['steps'] == len(age)
    windows = {}
    for label, window in [('all_updates', np.ones_like(case, bool)), ('last64_updates', steps >= 192*32)]:
        masks = {'ordinary': (case < 0) & window}
        for phase, lo, hi in PHASES:
            masks[phase] = (case == 26) & (age >= lo) & window
            if hi is not None:
                masks[phase] &= age < hi
        assert sum(m.sum() for m in masks.values()) == window.sum()
        windows[label] = {name: {k: stats(v[mask]) for k, v in values.items()} for name, mask in masks.items()}
    ep = {k: np.concatenate(v) for k, v in episode_arrays.items()}
    episodes = collect_episodes(age[:, 40:52], ep['done'].astype(bool), ep['task_success'], ep['task_failure'],
                               ep['task_deadline'], ep['y'], ep['reward'], p['ppo']['gamma'])
    counts = Counter(e['outcome'] for e in episodes)
    for name in ('success', 'failure', 'deadline'):
        assert counts[name] == r['recovery_episode_counts'][name]
    episode_summary = {}
    for outcome in ('success', 'failure', 'deadline', 'censored'):
        selected = [e for e in episodes if e['outcome'] == outcome]
        episode_summary[outcome] = dict(episodes=len(selected),
            peak_body_y_m=stats([e['peak_body_y_m'] for e in selected]),
            peak_age_s=stats([e['peak_age_s'] for e in selected]),
            controls=stats([e['controls'] for e in selected]),
            reward_sum=stats([e['reward_sum'] for e in selected]),
            discounted_reward_sum=stats([e['discounted_reward_sum'] for e in selected]),
            body_peak_above_70cm=sum(e['peak_body_y_m'] > .7 for e in selected))
    trace = root/'gradient_diagnostics.jsonl'
    evidence[str(trace)] = sha256_file(trace)
    assert evidence[str(trace)] == r['gradient_diagnostics']['sha256'] == audits[-1]['evidence'][str(trace)]
    gradients = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [x['update'] for x in gradients] == list(range(1, 257)) and all(x['minibatch'] == 0 for x in gradients)
    return dict(controls=age.size, recovery_controls=int((case == 26).sum()), windows=windows,
                recovery_episodes=episode_summary, episode_records=episodes,
                return_ordering=return_ordering(episodes, p['ppo']['gamma']),
                gradients={name: gradient_summary(g) for name, g in [('all_updates', gradients), ('last64_updates', gradients[-64:])]},
                temporal_weights=temporal_weights(p['ppo']['gamma'], p['ppo']['lam'], p['rollout_steps']),
                replay=dict(batch_groups=p['mean_anchor']['batch_groups'], group_weights=p['mean_anchor']['group_weights'],
                            coefficient=p['mean_anchor']['coefficient'], tail_replay_rows=audits[-1]['tail_replay_rows'],
                            validation_rows_used=audits[-1]['validation_rows_used_for_updates']))


def plot_report(report, figure):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    names = [x[0] for x in PHASES]
    for arm, color in [('zero', '#297b8e'), ('history', '#c87534')]:
        groups = report['arms'][arm]['windows']['last64_updates']
        for ax, field, scale, label in [(axes[0, 0], 'outward_origin_difference_m_s', 100, 'Origin outward drift (cm/s)'),
                                        (axes[0, 1], 'abs_body_y_m', 100, 'Mean absolute body offset (cm)')]:
            ax.plot(names, [groups[n][field]['mean']*scale for n in names], marker='o', label=arm, color=color)
            ax.set_ylabel(label)
        axes[1, 0].plot(names, [-groups[n]['velocity_reward']['mean'] for n in names], marker='o', color=color, label=arm+' velocity')
        axes[1, 0].plot(names, [-groups[n]['anchor_reward']['mean'] for n in names], marker='x', linestyle='--', color=color, label=arm+' online anchor')
    axes[1, 0].set_ylabel('Recorded penalty per control')
    temporal = report['arms']['zero']['temporal_weights']
    lags = np.arange(201)
    axes[1, 1].plot(lags*DT, temporal['gamma']**lags, label='Return discount gamma^lag')
    axes[1, 1].plot(lags*DT, temporal['gamma_lambda']**lags, linestyle='--', label='Untruncated GAE trace (reference)')
    axes[1, 1].axvline(.64, color='#888888', linestyle=':', label='Rollout duration (0.64 s)')
    axes[1, 1].set(xlabel='Delay (s)', ylabel='Weight', yscale='log')
    axes[0, 0].axhline(0, color='#888888', linewidth=.7)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Existing TRAIN pair: last 64 updates; phases have survivor bias\nGAE beyond rollout depends on learned value bootstrap')
    path = Path(figure); path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        fig.savefig(path.with_suffix(suffix), dpi=160)
    plt.close(fig)
    svg = path.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--figure', required=True)
    args = parser.parse_args()
    path = Path('docs/lite3_route_history_plan_20261010.json')
    plan = json.loads(path.read_text())
    evidence = {str(path): sha256_file(path), str(Path(__file__).relative_to(Path.cwd())): sha256_file(__file__)}
    arms = {arm: analyze_arm(arm, plan, evidence) for arm in ('zero', 'history')}
    root = Path(plan['dataset']).parent
    audit = json.loads(Path(plan['dataset_audit']).read_text())
    assert audit['status'] == 'verified' and sha256_file(plan['dataset_audit']) == plan['dataset_audit_sha256']
    evidence[plan['dataset_audit']] = sha256_file(plan['dataset_audit'])
    replay = {}
    # Provenance identifies source control ages before filtering. Load no targets
    # and no held-out streams, including the original offline validation split.
    provenance = root/'row_provenance.npz'
    assert sha256_file(provenance) == audit['evidence'][str(provenance)]
    evidence[str(provenance)] = sha256_file(provenance)
    with np.load(provenance, allow_pickle=False) as data:
        for group in ('correction', 'flat', 'rough101', 'rough107', 'post_recovery'):
            rows = data[group+'_train_obs']
            replay[group] = dict(rows=len(rows), source_indices=np.unique(rows[:, 0]).tolist(),
                                  minimum_control_age=int(rows[:, 1].min()), maximum_control_age=int(rows[:, 1].max()),
                                  sampled_in_this_pair=group != 'post_recovery')
    result = dict(status='verified', arms=arms, replay_train_provenance=replay, evidence=evidence,
        new_controls=0, optimizer_steps=0, pilot_rows_used=0, original_seven_source_rows_used=0,
        scope='All TRAIN controls of both changing-policy runs; all failures and right-censored episodes retained. Phase means weight controls, not independent sources. Last64-update summaries still mix changing policies. Actual logged heading/velocity/online-anchor rewards are reconstructed, terminal reward reconstructed from source and event flags; other rewards remain unallocated. Foot-aware lateral cost is NOT reconstructed. Its positive body-only penalty is a lower bound. First-minibatch replay gradients mix phases; no per-phase gradient or Adam-causality claim. No learning or new evaluation.')
    write_json_atomic(args.output, result)
    plot_report(result, args.figure)


if __name__ == '__main__':
    main()
