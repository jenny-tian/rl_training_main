"""复用固定 100 次更新训练器，只替换已批准的纯转向相位门控。"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import traceback

import reference_low_stairs_zero_preservation_train as native
from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_yaw_phase_off_cfg import FUNCTION

OUTPUT=ROOT/'logs/reference_six/low_stairs_yaw_phase_off_20261115'
DOCS=ROOT/'docs/lite3-low-stairs-yaw-phase-off-20261115'
ZERO_OUTPUT=native.OUTPUT
VARIANT='persistent_clearance_yaw_coverage_zero_preservation_pure_yaw_phase_off'
NATIVE_SHA='80c13c103b2f717e31a535d341b01c4d05e37778b4bc1dafef3cb30600ac431a'
ORIGINAL_SOURCES=native.sources
ORIGINAL_PREFLIGHT=native.preflight


def read(path):
    return json.loads(Path(path).read_text())


def directory(phase):
    if phase not in (0,1):raise ValueError('本阶段仅允许一次零更新集成和一次 100 次更新训练')
    return OUTPUT/('integration' if phase==0 else 'train_001_100')


def sources():
    return ORIGINAL_SOURCES()+[dict(path=str(p),sha256=sha(p)) for p in
        (Path(__file__),Path(__file__).with_name('reference_low_stairs_yaw_phase_off_cfg.py'))]


def expected_configuration():
    value=read(ZERO_OUTPUT/'train_001_100/configuration.json')
    assert value['environment']['rewards']['phase_foot_trajectory_exp']['func']=='reference_low_stairs_rewards:phase_outside_stair_translation'
    value['environment']['rewards']['phase_foot_trajectory_exp']['func']=FUNCTION
    return value


def checked_configuration(value):
    assert value==expected_configuration(),'唯一允许变化是纯转向相位函数'
    save(OUTPUT/'configuration_difference.json',dict(status='passed',only_changed_field='environment.rewards.phase_foot_trajectory_exp.func',
        agent_exact=True,terrain_contract_exact=True,all_other_environment_exact=True,
        old='reference_low_stairs_rewards:phase_outside_stair_translation',new=FUNCTION,auxiliary_coefficient=0.))
    return value


def preflight(phase):
    directory(phase);assert sha(native.__file__)==NATIVE_SHA
    historical=ORIGINAL_PREFLIGHT(0)
    # 零保持历史运行源码和模型保留原身份，新的包装器不改变历史文件。
    report=read(ZERO_OUTPUT/'train_001_100/report.json')
    assert report['status']=='passed' and report['optimizer_updates']==100
    assert sha(report['output_checkpoint'])==report['output_sha256']=='290d94746c467b5cd21ef74c47a908011521c1f145da35ba7b94f1b92d9b7497'
    for row in report['implementation_sources']:assert sha(row['path'])==row['sha256']
    if phase:
        integration=read(directory(0)/'report.json');execution=read(directory(0)/'execution.json')
        assert execution['status']=='completed' and execution['exit_code']==0
        assert integration['status']=='passed' and integration['optimizer_updates']==0
        assert integration['environment_steps']==64 and integration['implementation_sources']==sources()
        audit=integration['phase_gate_audit']
        assert audit['steps']==64 and audit['frames']==4096*64 and audit['pure_yaw_frames']>0
        assert audit['removed_nonzero_frames']>0 and audit['non_yaw_exact'] and audit['reference_initialized']
    return dict(status='passed',historical=historical,native_trainer_sha256=NATIVE_SHA,
                zero_checkpoint_sha256=report['output_sha256'])


def install(args):
    """仅修改本子进程中的组合入口，旧源码、旧输出目录均只读。"""
    from reference_low_stairs_yaw_phase_off_cfg import apply_phase_gate, RuntimeAudit
    original_loader=native.load_reference; original_runner=native.make_resumed_runner
    state={}
    def loader():
        reference,agent=original_loader()
        import reference_low_stairs_guarded_cfg as configuration
        original_configure=configuration.configure
        def configure(reference):
            cfg,audits=original_configure(reference);cfg,audit=apply_phase_gate(cfg)
            return cfg,dict(audits,pure_yaw_phase_off=audit)
        configuration.configure=configure
        return reference,agent
    def runner(base,*a,**kw):
        value=original_runner(base,*a,**kw)
        audit=RuntimeAudit(base);state['audit']=audit;compute=base.reward_manager.compute
        def capture(dt):
            reward=compute(dt);audit.capture(reward,dt);return reward
        base.reward_manager.compute=capture
        return value
    def save_report(path,value):
        if Path(path)==args.output/'report.json':
            value=dict(value,phase_gate_audit=state['audit'].summary(),native_trainer_sha256=NATIVE_SHA)
            if value['status']=='passed':
                assert value['phase_gate_audit']['steps']==value['environment_steps']
                assert value['phase_gate_audit']['removed_nonzero_frames']>0
        save(path,value)
    native.OUTPUT=OUTPUT;native.DOCS=DOCS;native.VARIANT=VARIANT
    native.directory=directory;native.sources=sources;native.preflight=preflight
    native.checked_configuration=checked_configuration;native.load_reference=loader
    native.make_resumed_runner=runner;native.save=save_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',type=int,choices=(0,1),required=True)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args();args.output=directory(args.phase)
    if not args.child:
        preflight(args.phase)
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],
                   gpu=1,output=args.output,timeout=7200,minimum_free_mib=10240)
    try:
        install(args);native.worker(args)
    except Exception:
        traceback.print_exc();save(args.output/'error.json',dict(error=traceback.format_exc()))
        sys.stdout.flush();os._exit(1)


if __name__=='__main__':sys.exit(main())
