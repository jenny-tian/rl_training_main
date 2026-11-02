"""Prepare unapproved cumulative budgets, scale plans and Source-only protocols on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/rl_training'), str(ROOT/'scripts/tools')]
from rl_training.source_scan.cumulative_training import SCHEMA
from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
from rl_training.source_scan.training_budget_ledger import BudgetLedger, canonical_budget_path
from rl_training.source_scan.training_lifecycle import SourceRunSeeds
from rl_training.source_scan.source_selection import proposal
from rl_training.source_scan.runtime_readiness import selected_scale_proposal, numerics_readiness_template
from source_training_entrypoint import _input_hashes, _git_state
from source_live_ppo_supervisor import dependencies


def write_bundle(output, *, ledger, revision, remote, inputs, dependency_hashes, frozen=False):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('preparation output exists')
    before = ledger.reference()
    plans = {str(run): ledger.numerics_plan(run) for run in (1,2,3)}
    output.mkdir(parents=True,exist_ok=False)
    def write(name, payload):
        path = output/name
        path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    protocol = proposal()
    selection_ref = write('source_selection.proposal.json',protocol)
    runtime_ref = write('runtime_readiness.not_run.json',numerics_readiness_template(inputs))
    metrics_ref = write('metrics_qualification.not_run.json',{'schema':'lite3_source_metrics_qualification_v1',
        'status':'not_run','input_sha256':inputs,
        'registered_cells':protocol['registered_cells'],'front_foot_detector_force_N':20.,
        'normalized_episode_schema':'lite3_source_episode_metrics_v1',
        'live_normalized_metric_collector_qualified':False})
    write('selected_scale.proposal.json',selected_scale_proposal())
    write('cumulative_budgets.json',plans)
    write('ledger_snapshot.json',ledger.read())
    authorizations = {}
    for run in (1,2,3):
        budget = plans[str(run)]; node = budget['nodes']['numerics']; seeds = SourceRunSeeds(run)
        plan = {'schema':SCHEMA,'status':'proposed','user_approved':False,
            'action':'source_ppo_bounded_numerics','run':run,'budget_node':'numerics',
            'ppo_seed':seeds.ppo,'geometry_seed':seeds.geometry,'sampler_seed':seeds.sampler,
            'reset_state_id':seeds.reset_state_id,'num_envs':120,'rollout_length':24,'level':'L1',
            'registered_cells':15,'six_families_retained':True,'device':'cuda:0',
            'minimum_free_vram_mib':12288,'retry':False,'resume':False,'automatic_scale_increase':False,
            'development_or_sealed_feedback_allowed':False,'base_revision':revision,
            'remote_revision_verified':remote if frozen else None,'input_sha256':inputs,
            'dependency_sha256':dependency_hashes,'checkpoint_sha256':CHECKPOINT_SHA256,
            'budget_ledger':before,'cumulative_budget':budget,'max_updates':node['remaining_updates'],
            'max_transitions':node['planned_transitions'],'hard_transition_ceiling':node['ceiling'],
            'initialization':budget['initialization'],'diagnostic_weights_reused':False,
            'checkpoint_nodes':[node['remaining_updates']],
            'output_dir':f'/tmp/lite3_source_cumulative_numerics_run{run}_20261102',
            'startup_timeout_s':120,'wall_timeout_s':900,'cleanup_grace_s':15,
            'runtime_qualification':runtime_ref,'source_selection_lock':selection_ref,
            'metrics_qualification':metrics_ref}
        authorizations[str(run)] = write(f'run{run}.authorization.proposal.json',plan)
    if ledger.reference() != before: raise ValueError('production budget changed during preparation')
    import torch
    if torch.cuda.is_initialized(): raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    result = {'schema':'lite3_source_cumulative_preparation_bundle_v1',
        'status':'frozen_proposals_runtime_pending' if frozen else 'working_tree_proposals_runtime_pending',
        'base_revision':revision,'production_budget_before':before,'production_budget_after':ledger.reference(),
        'authorizations':authorizations,'production_inputs':len(inputs),'core_dependencies':len(dependency_hashes),
        'cuda_initialized':False,'real_source_ppo_updates':0,'isaac_sim_launches':0,'gpu_jobs':0,
        'formal_launch_ready':False,'readiness_missing':['shared_scale_startup_ladder','G2_G3_120',
            'G4_collection','120_env_sparse_capture','live_normalized_metric_collection',
            'approved_Source_selection_protocol','separate_numeric_launch_approval']}
    write('preparation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
        raise ValueError('original Source checkpoint differs from registry')
    if args.freeze:
        revision,remote,dirty=_git_state()
        if dirty or revision!=remote: raise ValueError('freeze requires clean, committed and remote-matching worktree')
    else:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();remote=None
    result=write_bundle(args.output_dir,ledger=BudgetLedger(canonical_budget_path(ROOT)),
        revision=revision,remote=remote,inputs=_input_hashes(),dependency_hashes=dependencies(),frozen=args.freeze)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
