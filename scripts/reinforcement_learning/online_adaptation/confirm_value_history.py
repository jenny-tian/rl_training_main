"""Fit once on old training episodes, seal models, then confirm on fresh episodes."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl_training.online_adaptation.interaction_budget import digest
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.value_diagnostics import (
    append_past_velocity_integral, fit_value_network, make_history_value_network,
    make_value_network, time_features,
)
from scripts.reinforcement_learning.online_adaptation.diagnose_complete_value import (
    evaluate_episodes, load_episode,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_path(record):
    path = Path(record['path'])
    if sha256_file(path) != record['sha256']:
        raise ValueError(f"locked input changed: {path}")
    return path


def analysis_source_sha256():
    import rl_training.online_adaptation.value_diagnostics as value_module
    import scripts.reinforcement_learning.online_adaptation.diagnose_complete_value as episode_module
    paths = [Path(__file__), Path(value_module.__file__), Path(episode_module.__file__)]
    return digest({p.name: sha256_file(p) for p in paths})


def load_protocol(path):
    protocol = read_json(path)
    if protocol['schema'] != 'lite3_value_history_confirmation_v1':
        raise ValueError('unsupported history confirmation protocol')
    original = read_json(checked_path(protocol['original_protocol']))
    checked_path(protocol['original_analysis'])
    candidate = protocol['candidate']
    if (candidate['schema'] != 'policy48_past_body_velocity_integral2_v1'
            or candidate['velocity_columns'] != [45, 46]
            or candidate['original_input_dim'] != 48 or candidate['input_dim'] != 50
            or candidate['hidden_layers'] != [128, 128] or candidate['activation'] != 'ELU'
            or candidate['step_dt'] != .02 or candidate['distance_scale'] != 12.
            or protocol['fit'] != original['fit'] or protocol['gamma'] != original['gamma']
            or protocol['time_degree'] != original['time_degree']
            or protocol['base_sha256'] != original['base_sha256']):
        raise ValueError('candidate differs from locked single-change contract')
    for terrain, prior in zip(protocol['terrains'], original['terrains'], strict=True):
        if (terrain['seed'] != prior['seed'] or terrain['train_trials'] != prior['train_trials']
                or terrain['training_session_dir'] != prior['session_dir']
                or terrain['seen_validation_trials'] != prior['validation_trials']):
            raise ValueError('training split changed')
    return protocol, original


def checked_session(root, *, terrain, learner_seed, base_sha256, expected_trials):
    root = Path(root)
    state = read_json(root / 'session_summary.json')
    manifest = read_json(root / 'ledger/manifest.json')
    arguments = manifest['worker_configuration']['arguments']
    if (state['active'] is not None or state['pending'] is not None or state['uncertain_steps']
            or set(state['trials']) != set(expected_trials)):
        raise ValueError('missing, uncertain, or extra session trials; do not replace starts')
    if (manifest['algorithm'] != 'lora' or manifest['terrain_seed'] != terrain
            or manifest['learner_seed'] != learner_seed or manifest['base']['sha256'] != base_sha256
            or manifest['terrain_split'] != 'development' or manifest['seal'] is not None
            or arguments['task_deadline_steps'] != 2000 or arguments['initial_state'] != 'perturbed-v1'
            or arguments['target_warmup_steps'] != 0 or arguments['forward_speed'] != .3
            or arguments['terrain_stage'] != 'rough_4cm'):
        raise ValueError('collection identity differs from locked protocol')
    return state, manifest


def predict_model(model, *, history=False):
    model.eval()

    def predict(observations):
        # evaluate_episodes calls this separately for each episode: no cross-reset prefix.
        x = append_past_velocity_integral(observations) if history else observations
        with torch.no_grad():
            return model(torch.as_tensor(x, dtype=torch.float32)).numpy().reshape(-1)

    return predict


def fit_once(protocol_path, output):
    protocol, original = load_protocol(protocol_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': protocol['schema'], 'status': 'running',
              'protocol_sha256': sha256_file(protocol_path),
              'analysis_source_sha256': analysis_source_sha256(), 'terrains': {}}
    write_json_atomic(output / 'manifest.json', report)
    write_json_atomic(output / 'protocol.json', protocol)
    prior_report = read_json(protocol['original_analysis']['path'])
    for terrain in protocol['terrains']:
        # Seal before *any* confirmation collector has created its directory.
        if Path(terrain['confirmation_session_dir']).exists():
            raise ValueError('confirmation collection already exists; fit must precede collection')
    for terrain in protocol['terrains']:
        seed = terrain['seed']
        root = terrain['training_session_dir']
        state, manifest = checked_session(
            root, terrain=seed, learner_seed=original['learner_seed'], base_sha256=protocol['base_sha256'],
            expected_trials=terrain['train_trials'] + terrain['seen_validation_trials'])
        # Only metadata of the old validation split is inspected for seed disjointness.
        train = [load_episode(root, state['trials'][t], manifest, protocol['gamma'])
                 for t in terrain['train_trials']]
        prior = prior_report['terrains'][str(seed)]
        prior_episodes = {e['trial_id']: e for e in prior['episodes']}
        for episode in train:
            if episode['metadata']['input_sha256'] != prior_episodes[episode['metadata']['trial_id']]['input_sha256']:
                raise ValueError('original training evidence changed')
        x = np.concatenate([append_past_velocity_integral(e['x']) for e in train])
        y = np.concatenate([e['y'] for e in train])
        candidate = make_history_value_network(train[0]['initial_critic'])
        initial = make_value_network(48)
        initial.load_state_dict(train[0]['initial_critic'], strict=True)
        with torch.no_grad():
            a = candidate(torch.from_numpy(x)).numpy()
            b = initial(torch.from_numpy(x[:, :48])).numpy()
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
        baseline_path = checked_path(terrain['baseline_checkpoint'])
        baseline = torch.load(baseline_path, map_location='cpu', weights_only=False)
        if not baseline['diagnostic_only'] or baseline['protocol'] != original:
            raise ValueError('baseline is not the locked original diagnostic')
        started = time.perf_counter()
        fit = fit_value_network(candidate, x, y, **protocol['fit'])
        seconds = time.perf_counter() - started
        bundle_path = output / f'critic-{seed}.pt'
        torch.save({'schema': protocol['schema'], 'diagnostic_only': True,
                    'no_actor_or_online_session_checkpoint': True,
                    'protocol_sha256': report['protocol_sha256'],
                    'history_model': candidate.state_dict(), 'optimizer': fit['optimizer'],
                    'original_model': baseline['model'],
                    'constant': prior['constant'], 'time_coefficients': prior['time_coefficients']}, bundle_path)
        report['terrains'][str(seed)] = {
            'bundle': {'path': str(bundle_path), 'sha256': sha256_file(bundle_path)},
            'fit_seconds': seconds, 'optimizer_steps': fit['optimizer_steps'],
            'training_curve': fit['training_curve'],
            'initial_prediction_max_abs': float(np.max(np.abs(a-b))),
            'training_metrics': evaluate_episodes(train, predict_model(candidate, history=True)),
            'training_episodes': [e['metadata'] for e in train],
            'old_reset_seeds': [t['reset_seed'] for t in state['trials'].values()],
            'training_source_sha256': manifest['worker_configuration']['source_sha256']}
        write_json_atomic(output / 'manifest.json', report)
        print(seed, 'fixed fit complete', report['terrains'][str(seed)]['training_metrics']['pooled'], flush=True)
    report['status'] = 'complete'
    write_json_atomic(output / 'manifest.json', report)


def supported_by_confirmation(metrics):
    candidate = metrics['history_critic']
    ev = candidate['pooled']['explained_variance']
    return bool(ev is not None and ev > 0 and all(
        candidate['pooled']['rmse'] < metrics[name]['pooled']['rmse']
        and candidate['episode_equal_mean_rmse'] < metrics[name]['episode_equal_mean_rmse']
        for name in ('original_critic', 'constant', 'time_only')))


def confirm_once(protocol_path, fit_dir, output, fit_manifest_sha256):
    protocol, _ = load_protocol(protocol_path)
    fit_path = checked_path({'path': str(Path(fit_dir) / 'manifest.json'), 'sha256': fit_manifest_sha256})
    fitted = read_json(fit_path)
    if (fitted['status'] != 'complete' or fitted['protocol_sha256'] != sha256_file(protocol_path)
            or fitted['analysis_source_sha256'] != analysis_source_sha256()):
        raise ValueError('fit not complete or protocol/source changed after fit')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': protocol['schema'], 'status': 'running',
              'protocol_sha256': sha256_file(protocol_path), 'fit_manifest_sha256': fit_manifest_sha256,
              'analysis_source_sha256': analysis_source_sha256(), 'terrains': {}}
    write_json_atomic(output / 'report.json', report)
    steps = 0
    for terrain in protocol['terrains']:
        seed = terrain['seed']
        fit_result = fitted['terrains'][str(seed)]
        bundle = torch.load(checked_path(fit_result['bundle']), map_location='cpu', weights_only=False)
        if bundle['protocol_sha256'] != report['protocol_sha256'] or not bundle['diagnostic_only']:
            raise ValueError('bundle protocol identity mismatch')
        root = terrain['confirmation_session_dir']
        state, manifest = checked_session(
            root, terrain=seed, learner_seed=protocol['confirmation_learner_seed'],
            base_sha256=protocol['base_sha256'], expected_trials=terrain['confirmation_trials'])
        actual_seeds = [state['trials'][t]['reset_seed'] for t in terrain['confirmation_trials']]
        if (actual_seeds != terrain['confirmation_reset_seeds']
                or len(set(actual_seeds)) != len(actual_seeds)
                or set(actual_seeds) & set(fit_result['old_reset_seeds'])):
            raise ValueError('confirmation reset seeds changed or overlap old data')
        episodes = [load_episode(root, state['trials'][t], manifest, protocol['gamma'])
                    for t in terrain['confirmation_trials']]
        terrain_steps = sum(len(e['y']) for e in episodes)
        if terrain_steps != sum(state['spent'].values()):
            raise ValueError('confirmation cost differs from complete episode data')
        steps += terrain_steps
        if steps > protocol['max_additional_control_transitions']:
            raise ValueError('confirmation exceeded registered control budget')
        original_model, history_model = make_value_network(48), make_value_network(50)
        original_model.load_state_dict(bundle['original_model'], strict=True)
        history_model.load_state_dict(bundle['history_model'], strict=True)
        predictors = {
            'original_critic': predict_model(original_model),
            'history_critic': predict_model(history_model, history=True),
            'constant': lambda x: np.full(len(x), bundle['constant']),
            'time_only': lambda x: time_features(x, protocol['time_degree']) @ np.asarray(bundle['time_coefficients'])}
        metrics = {name: evaluate_episodes(episodes, predictor) for name, predictor in predictors.items()}
        report['terrains'][str(seed)] = {
            'episodes': [e['metadata'] for e in episodes], 'actual_control_transitions': terrain_steps,
            'source_sha256': manifest['worker_configuration']['source_sha256'],
            'session_manifest_sha256': sha256_file(Path(root) / 'ledger/manifest.json'),
            'metrics': metrics, 'diagnostic_supported': supported_by_confirmation(metrics)}
        write_json_atomic(output / 'report.json', report)
        print(seed, {name: m['pooled'] for name, m in metrics.items()}, flush=True)
    report.update(status='complete', actual_control_transitions=steps,
                  diagnostic_supported=all(t['diagnostic_supported'] for t in report['terrains'].values()),
                  online_route_qualified=False)
    write_json_atomic(output / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'confirm'))
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fit-dir')
    parser.add_argument('--fit-manifest-sha256')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'fit':
        fit_once(args.protocol, args.output)
    else:
        if not args.fit_dir or not args.fit_manifest_sha256:
            parser.error('confirmation requires fit directory and precollection manifest hash')
        confirm_once(args.protocol, args.fit_dir, args.output, args.fit_manifest_sha256)


if __name__ == '__main__':
    main()
