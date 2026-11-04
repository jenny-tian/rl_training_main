"""Freeze/run a single approved delivery slot; no implicit retries or extra launches."""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
from rl_training.source_scan.delivery_stage import (LIMITS,reference,read_reference,validate_grant,
    validate_receipt,job_contract,claim_slot,seal_receipt,file_digest)
from rl_training.source_scan.training_budget_ledger import BudgetLedger,canonical_budget_path
from rl_training.source_scan.smoke_artifacts import atomic_json
from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
from source_training_entrypoint import _git_state,_input_hashes


def numeric_plan(root,common,grant_ref,prior):
    from rl_training.source_scan.cumulative_training import NUMERICS_SCHEMA
    from rl_training.source_scan.training_lifecycle import SourceRunSeeds
    from rl_training.source_scan.runtime_readiness import numerics_readiness_template
    from source_training_supervisor import dependencies
    from rl_training.source_scan.cumulative_training import validate_authorization
    validate_receipt(prior,'g4',inputs=common['input_sha256'])
    g4_receipt=read_reference(prior,'g4');g4=read_reference(g4_receipt['acceptance'],'acceptance')
    readiness=numerics_readiness_template(common['input_sha256'])
    readiness.update(status='qualified_for_bounded_numerics',g0_passed=True,g1_passed=True,g2_passed=True,
        g3_passed=True,g4_collection_passed=True,
        measured_collection_transitions_per_second=g4['measured_collection_transitions_per_second'],
        evidence=prior)
    receipt_path=root/'numeric_runtime_readiness.json'
    if receipt_path.exists():raise FileExistsError('numeric readiness already frozen')
    atomic_json(receipt_path,readiness)
    ledger=BudgetLedger(canonical_budget_path(ROOT));budget=ledger.numerics_plan(1);node=budget['nodes']['numerics']
    seeds=SourceRunSeeds(1)
    plan=dict(schema=NUMERICS_SCHEMA,status='approved',user_approved=True,
        action='source_ppo_bounded_numerics',run=1,budget_node='numerics',
        ppo_seed=seeds.ppo,geometry_seed=seeds.geometry,sampler_seed=seeds.sampler,reset_state_id=seeds.reset_state_id,
        num_envs=120,rollout_length=24,level='L1',registered_cells=15,six_families_retained=True,
        device='cuda:0',minimum_free_vram_mib=12288,retry=False,resume=False,automatic_scale_increase=False,
        development_or_sealed_feedback_allowed=False,base_revision=common['base_revision'],
        remote_revision_verified=common['remote_revision_verified'],input_sha256=common['input_sha256'],
        checkpoint_sha256=CHECKPOINT_SHA256,output_dir=common['output_dir'],budget_ledger=ledger.reference(),
        cumulative_budget=budget,max_updates=node['remaining_updates'],max_transitions=node['planned_transitions'],
        hard_transition_ceiling=node['ceiling'],initialization='registered_actor_fresh_critic_optimizer',
        diagnostic_weights_reused=False,checkpoint_nodes=[node['remaining_updates']],
        startup_timeout_s=120,wall_timeout_s=900,cleanup_grace_s=15,
        runtime_qualification=dict(path=str(receipt_path),sha256=file_digest(receipt_path)),
        dependency_sha256=dependencies(),delivery_stage=grant_ref,delivery_job='numerics',
        predecessor_qualification=prior,formal_selection_authorized=False,metrics_scope='diagnostic_only')
    validated=validate_authorization(plan,run=1,budget_node='numerics',output_dir=Path(common['output_dir']),
        checkpoint_sha256=CHECKPOINT_SHA256,revision=common['base_revision'],input_sha256=common['input_sha256'],ledger_path=ledger.path)
    if validated!=plan:raise ValueError('numeric manifest differs from its canonical authorized lock')
    return plan


def freeze(args):
    from source_scale_startup_supervisor import dependencies
    grant_ref=reference(args.approval,'stage_approval');grant=validate_grant(grant_ref)
    root=Path(grant['output_root']);root.mkdir(parents=True,exist_ok=True)
    supplement=getattr(args,'g3_supplement',None)
    if supplement is not None and args.job!='g3':raise ValueError('G3 supplement cannot authorize another job')
    label='g3_fast' if supplement is not None else args.job
    output=root/label;manifest=root/(label+'.approved.json')
    if output.exists() or manifest.exists():raise FileExistsError('delivery attempt/manifest already exists')
    revision,remote,dirty=_git_state()
    if dirty or revision!=remote:raise ValueError('freeze delivery only after clean commit/push verification')
    if file_digest(args.checkpoint)!=CHECKPOINT_SHA256:raise ValueError('original checkpoint changed')
    previous={'g2':'g1','g3':'g2','g4':'g3','numerics':'g4','compare':'numerics'}.get(args.job)
    combined=args.job=='g2' and getattr(args,'combine_startup',False)
    if combined:
        from rl_training.source_scan.scale_reset import PREDECESSOR_120
        prior=dict(PREDECESSOR_120)
    else:
        previous_folder='g3_fast' if previous=='g3' and (root/'g3_fast/receipt.json').exists() else previous
        prior=reference(root/previous_folder/'receipt.json',previous) if previous else None
    common=dict(status='approved',user_approved=True,base_revision=revision,remote_revision_verified=remote,
        input_sha256=_input_hashes(),dependency_sha256=dependencies(),
        budget_ledger=BudgetLedger(canonical_budget_path(ROOT)).reference(),output_dir=str(output),
        checkpoint_sha256=CHECKPOINT_SHA256,delivery_stage=grant_ref,delivery_job=args.job)
    if prior:common['predecessor_qualification']=prior
    if combined:common['startup_covered_by_g2']=True
    if supplement is not None:
        common.update(g3_supplement=reference(supplement,'g3_supplement'),contact_digest_scheme='sha256_blocks_4096_v1')
    if args.job=='compare':common['new_checkpoint']=reference(root/'numerics/training/actor_update_90.pt','actor_checkpoint')
    plan=numeric_plan(root,common,grant_ref,prior) if args.job=='numerics' else common|job_contract(args.job,common)
    with manifest.open('x') as stream:json.dump(plan,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='frozen_within_user_approved_stage',job=args.job,manifest=str(manifest),
        revision=revision,launches=0,training_budget_reserved=False),indent=2))
    return manifest,plan


def run(args):
    manifest,plan=freeze(args)
    claim_slot(plan['delivery_stage'],args.job,manifest)
    out=Path(plan['output_dir']);started=time.monotonic()
    modules={'g1':'source_scale_startup_supervisor','g2':'source_scale_reset_supervisor',
             'g3':'source_scale_motion_supervisor','locate':'source_scale_motion_supervisor',
             'g4':'source_delivery_light','compare':'source_delivery_light','numerics':'source_training_supervisor'}
    ns=NS(manifest=manifest,checkpoint=args.checkpoint.resolve(),output_dir=out,num_envs=120)
    if args.job=='numerics':
        ns=NS(authorization=manifest,checkpoint=args.checkpoint.resolve(),output_dir=out,run=1,budget_node='numerics',device='cuda:0')
        print('[delivery] reserving 259200 training transitions; partial/failed execution retains the full charge',flush=True)
    module=importlib.import_module(modules[args.job]);code=None
    try:
        code=module.supervise(ns) if args.job in ('g4','compare') else module.run(ns)
        if args.job=='numerics' and (out/'training/run_report.json').exists():
            report=json.loads((out/'training/run_report.json').read_text())
            report.update(simulation_control_steps=report['returned_transitions']//120,
                completed_diagnostic_transitions=0,derived_from=reference(out/'training/run_report.json','training_report'))
            atomic_json(out/'result.json',report)
        if (out/'acceptance.json').exists() and (out/'result.json').exists():seal_receipt(out,args.job)
        return code
    finally:
        atomic_json(manifest.parent/(args.job+'.supervisor_receipt.json'),dict(job=args.job,returncode=code,
            supervisor_elapsed_s=time.monotonic()-started,automatic_retry=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--job',choices=tuple(LIMITS),required=True)
    parser.add_argument('--approval',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--combine-startup',action='store_true',help='Reuse the real startup anchor and validate current startup inside the G2 job')
    parser.add_argument('--g3-supplement',type=Path,help='Separate approval of one replacement G3; never reuses the consumed slot')
    args=parser.parse_args();raise SystemExit(run(args) if args.action=='run' else (freeze(args),0)[1])
