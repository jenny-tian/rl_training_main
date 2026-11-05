"""Recompute the diagnosis tables from archived samples, without a simulator."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'source/rl_training'))
from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES, LITE3_SAFE_ACTION_LOWER, LITE3_SAFE_ACTION_UPPER
from probe import OLD, NEW, HASHES, STEPS, NUM_ENVS, CASES, STATE_NAMES, ACTION_NAMES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name,value):
    (HERE/'deliverables'/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(name,rows):
    with (HERE/'deliverables'/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SavedConfigLoader(yaml.SafeLoader):
    pass


SavedConfigLoader.add_multi_constructor('tag:yaml.org,2002:python/',
    lambda loader,tag,node: loader.construct_sequence(node) if isinstance(node,yaml.SequenceNode)
    else loader.construct_mapping(node) if isinstance(node,yaml.MappingNode) else loader.construct_scalar(node))


def compare_reference(source_rewards,reward_path):
    ref=HERE/'raw/reference_project'
    manifest=json.loads((ref/'manifest.json').read_text())
    for entry in manifest['files']:
        assert digest(ref/entry['archive_name'])==entry['sha256']
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    def assignments(path):
        result={}
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Assign):
                target=ast.unparse(node.targets[0])
                if target.startswith('self.rewards.'):
                    result[target]=ast.unparse(node.value)
        return result
    old_functions=functions(ref/'rewards.py');current_functions=functions(reward_path)
    function_rows=[]
    for name,cfg in source_rewards.items():
        function=cfg['func'].split(':')[-1]
        if not cfg['func'].startswith('rl_training.'):
            status='external_library_not_compared'
        elif function not in old_functions:
            status='only_in_current_project'
        else:
            status='same_ast' if old_functions[function]==current_functions[function] else 'different_ast'
        function_rows.append(dict(term=name,function=function,status=status))
    old_assignments=assignments(ref/'rough_env_cfg.py')
    current_assignments=assignments(reward_path.parent.parent/'config/quadruped/deeprobotics_lite3/rough_env_cfg.py')
    assignment_rows=[dict(target=k,reference=old_assignments.get(k),current=current_assignments.get(k),
        same=old_assignments.get(k)==current_assignments.get(k)) for k in sorted(set(old_assignments)|set(current_assignments))]
    comparison=dict(reference_root=manifest['reference_root'],reference_revision=manifest['revision'],
        comparison_method='AST comparison; reference project was not imported or simulated',
        identical_rough_reward_assignments=sum(r['same'] for r in assignment_rows),
        assignments=assignment_rows,active_reward_functions=function_rows,
        reference_logs_directory_present=manifest['logs_directory_present'],
        curriculum_note='terrain_levels_vel updates gait_level; gait_level_curve only returns it for logging',
        suitability='reuse relevant reward definitions, adapt kernel/penalties/gating to fixed 0.3 m/s and the existing actor')
    dump('reference_project_comparison.json',comparison)


def main():
    torch.set_num_threads(1)
    (HERE/'deliverables').mkdir(exist_ok=True)
    raw=HERE/'raw/runtime'
    result=json.loads((raw/'result.json').read_text())
    execution=json.loads((raw/'execution.json').read_text())
    assert result['status']=='passed' and execution['exit_code']==0 and execution['deadline_failure'] is None
    assert result['diagnostic_transitions']==240000 and result['training_transitions']==result['ppo_updates']==0
    assert result['budget_before']==result['budget_after']==execution['budget_before']==execution['budget_after']
    assert not execution['process_group_had_remaining_children']
    old_env=yaml.load((HERE/'raw/original_env.yaml').read_text(),Loader=SavedConfigLoader)
    source_rewards=result['cases'][0]['reward_config']
    reward_path=ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py'
    compare_reference(source_rewards,reward_path)
    tree=ast.parse(reward_path.read_text())
    gated_functions={node.name for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))
        and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='get_gait_level_tensor' for n in ast.walk(node))}
    gated=[name for name,cfg in source_rewards.items() if cfg['func'].split(':')[-1] in gated_functions]
    baseline=dict(old_environment_sha256=digest(HERE/'raw/original_env.yaml'),
        old_agent_sha256=digest(HERE/'raw/original_agent.yaml'),
        old_rewards={k:v for k,v in old_env['rewards'].items() if isinstance(v,dict) and v.get('weight')},
        source_active_rewards=source_rewards,gated_source_terms=gated,
        original_source_snapshot='raw/original_training_source.diff')
    old_std=float(old_env['rewards']['track_lin_vel_xy_exp']['params']['std'])
    new_std=float(source_rewards['track_lin_vel_xy_exp']['params']['std'])
    baseline['tracking_kernel']=dict(weight=4,command_mps=.3,old_std=old_std,source_std=new_std,
        old_stationary_fraction=float(np.exp(-.3**2/old_std**2)),
        source_stationary_fraction=float(np.exp(-.3**2/new_std**2)))
    dump('reward_configuration.json',baseline)
    overall=[];by_condition=[];terms=[];joint_rows=[];case_evidence=[]
    for case in result['cases']:
        label=case['label'];arrays=np.load(raw/(label+'.npz'))
        rewards=arrays['rewards'].astype(np.float64)
        state=arrays['state'].astype(np.float64)
        actions=arrays['actions'].astype(np.float64)
        columns=arrays['columns'];names=case['reward_names'];dt=case['control_dt']
        assert rewards.shape==(STEPS,NUM_ENVS,len(names))
        assert state.shape==(STEPS,NUM_ENVS,len(STATE_NAMES))
        assert actions.shape==(STEPS,NUM_ENVS,len(ACTION_NAMES))
        assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
        assert case['identity']==result['cases'][0]['identity']
        assert case['reward_config']==source_rewards and case['actor_parameters_unchanged'] and case['input_actions_unchanged']
        assert case['actor_parity_max_abs']==0 and case['reward_reconstruction_max_abs']<1e-5
        assert np.all(arrays['gate']==0)
        assert all(np.all(rewards[:,:,names.index(name)]==0) for name in gated)
        tracking=rewards[:,:,names.index('track_lin_vel_xy_exp')]
        assert np.max(np.abs(tracking-4*np.exp(-state[:,:,3]/new_std**2)))<1e-6
        # Change only the kernel width on recorded states; this is not a new rollout or the old full reward.
        rescored=rewards.sum(-1)-tracking+4*np.exp(-state[:,:,3]/old_std**2)
        for window,start in [('all_400',0),('after_100',100)]:
            def row(mask):
                s=state[start:][mask];a=actions[start:][mask];r=rewards[start:][mask]
                return dict(case=label,window=window,transitions=len(s),
                    reward_per_control=float(r.sum(-1).mean()*dt),
                    vx_mps=float(s[:,0].mean()),velocity_rms_error_mps=float(np.sqrt(s[:,3].mean())),
                    base_height_above_scan_m=float(s[:,4].mean()),joint_deviation_l2=float(s[:,6].mean()),
                    noise_rms_mean=float(a[:,1].mean()),clipped_joint_fraction=float(a[:,3].mean()),
                    projected_fraction=float(a[:,4].mean()),paused_fraction=float(a[:,5].mean()),
                    narrow_tracking_rescore_per_control=float(rescored[start:][mask].mean()*dt))
            overall.append(row(np.ones_like(columns[start:],dtype=bool)))
            for col in range(15):
                by_condition.append(dict(column=col,**row(columns[start:]==col)))
            for i,name in enumerate(names):
                terms.append(dict(case=label,window=window,term=name,
                    weighted_per_second=float(rewards[start:,:,i].mean()),
                    weighted_per_control=float(rewards[start:,:,i].mean()*dt)))
        snapshots=torch.load(raw/(label+'_observations.pt'),map_location='cpu',weights_only=True)
        for snap in snapshots.values():
            assert torch.equal(snap['mean'],snap['reference_mean'])
        mean=torch.cat([snapshots[step]['mean'] for step in (100,200,400)])
        lo=torch.tensor(LITE3_SAFE_ACTION_LOWER);hi=torch.tensor(LITE3_SAFE_ACTION_UPPER)
        for j,name in enumerate(LITE3_POLICY_JOINT_NAMES):
            joint_rows.append(dict(case=label,joint=name,sampled_observations=len(mean),
                mean_output=float(mean[:,j].mean()),min_output=float(mean[:,j].min()),max_output=float(mean[:,j].max()),
                clipped_fraction=float(((mean[:,j]<lo[j])|(mean[:,j]>hi[j])).float().mean())))
        case_evidence.append(dict(case=label,done_count=int(arrays['done'].sum()),terminal_terms=case['terminal_counts'],
            gate_min=case['gait_level_min'],gate_max=case['gait_level_max'],
            actor_parity_max_abs=case['actor_parity_max_abs'],reward_reconstruction_max_abs=case['reward_reconstruction_max_abs']))
    write_csv('case_metrics.csv',overall)
    write_csv('condition_metrics.csv',by_condition)
    write_csv('reward_components.csv',terms)
    write_csv('sampled_joint_outputs.csv',joint_rows)
    tracked=[r for r in overall if r['window']=='after_100']
    macro=[]
    for label,_,_ in CASES:
        rows=[r for r in by_condition if r['window']=='after_100' and r['case']==label]
        macro.append(dict(case=label,reward_per_control=float(np.mean([r['reward_per_control'] for r in rows])),
            vx_mps=float(np.mean([r['vx_mps'] for r in rows]))))
    # Relevant files are byte-identical to the revision that performed the 90 updates.
    training_revision='67d92d7443491c8a2b05e453f7a7ad34eeb58759'
    paths=[reward_path,ROOT/'source/rl_training/rl_training/source_scan/training_env.py',
        ROOT/'source/rl_training/rl_training/source_scan/frozen_actor.py']
    cfg_dir=reward_path.parent.parent/'config/quadruped/deeprobotics_lite3'
    paths += [cfg_dir/name for name in ('source_training_env_cfg.py','source_terrain_env_cfg.py','rough_env_cfg.py')]
    source_checks=[]
    for path in paths:
        rel=str(path.relative_to(ROOT))
        old_bytes=subprocess.check_output(['git','show',training_revision+':'+rel],cwd=ROOT)
        assert old_bytes==path.read_bytes()
        source_checks.append(dict(path=rel,sha256=digest(path),identical_to_training_revision=True))
    checkpoint_stds={}
    for label,path in [('old',OLD),('new',NEW)]:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        state=saved['actor_state_dict'] if label=='old' else saved
        checkpoint_stds[label]=state['distribution.log_std_param'].exp().tolist()
    assert [digest(p) for p in (OLD,NEW)]==HASHES
    dump('summary.json',dict(primary_window='controls_101_to_400',primary_weighting='each_sample_equal',
        primary_metrics=tracked,condition_equal_means=macro,case_evidence=case_evidence,
        checkpoint_stds=checkpoint_stds,execution=execution,diagnostic_transitions=240000,
        training_transitions=0,ppo_updates=0,gated_source_terms=gated,source_checks=source_checks,
        interpretation_limits=['single fixed geometry bank and seed; not independent terrain replicates',
            'short reward/noise diagnosis, not a completion or selection evaluation',
            'same standard-normal stream; states diverge and noisy cases can reset onto other columns',
            'narrow tracking rescore changes one term on recorded states, not policy performance after retraining',
            'historical training did not record reward components or intermediate actors; causal shares not identified']))
    print(json.dumps(dict(verified=True,primary_metrics=tracked,case_evidence=case_evidence),indent=2))


if __name__=='__main__':
    main()
