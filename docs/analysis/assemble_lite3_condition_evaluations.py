"""Assemble independently registered, disjoint full evaluation conditions."""
import argparse
import json
import os
from pathlib import Path

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--part',action='append',nargs=2,metavar=('PROTOCOL','RUN'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--audit',required=True)
    args=parser.parse_args()
    evidence={}
    def read(path):
        evidence[str(path)]=sha256_file(path)
        return json.loads(Path(path).read_text())
    p=read(args.protocol);phases={};roots={};sources=models=None;total=0;executions=[]
    assert len(args.part)==len(p['conditions'])
    for child_path,run in args.part:
        child=read(child_path);root=Path(run)
        assert set(child)==set(p)
        assert all(child[k]==v for k,v in p.items() if k not in {'conditions','max_controls','scope','registered_at_utc'})
        assert len(child['conditions'])==1 and child['conditions'][0] in p['conditions']
        report=read(root/'report.json');ledger=read(root/'ledger.json')
        assert report['status']==ledger['status']=='complete'
        assert ledger['pending'] is None and ledger['uncertain_transitions']==0
        assert report['ledger']==ledger and ledger['charged']==ledger['confirmed']<=child['max_controls']
        assert report['protocol_sha256']==sha256_file(child_path)
        if sources is None:sources,models=report['sources'],report['models']
        assert report['sources']==sources and report['models']==models
        assert set(report['phases'])=={child['conditions'][0]+'-'+model for model in p['models']}
        assert not set(phases)&set(report['phases'])
        assert sum(v['confirmed_controls'] for v in report['phases'].values())==ledger['confirmed']
        for name,phase in report['phases'].items():
            assert phase['status']=='complete' and read(root/name/'report.json')==phase
            manifest=read(root/name/'trajectory/manifest.json')
            assert manifest['status']=='complete' and manifest['steps']*p['num_envs']==phase['confirmed_controls']
            phases[name]=phase;roots[name]=root/name
        total+=ledger['confirmed']
        executions.append(dict(protocol=child_path,run=run,condition=child['conditions'][0],confirmed_controls=ledger['confirmed']))
    assert set(phases)=={c+'-'+m for c in p['conditions'] for m in p['models']}
    assert total<=p['max_controls']
    for path,digest in sources.items():assert sha256_file(path)==digest
    for name,model in models.items():
        assert {k:model[k] for k in p['models'][name]}==p['models'][name]
        assert sha256_file(model['path'])==model['actual_sha256']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    for name,path in roots.items():(out/name).symlink_to(os.path.relpath(path.resolve(),out.resolve()),target_is_directory=True)
    ledger=dict(schema='lite3_vector_step_ledger_v1',status='complete',num_envs=p['num_envs'],charged=total,confirmed=total,
        pending=None,uncertain_transitions=0,max_transitions=p['max_controls'],metadata=dict(assembled_disjoint_conditions=True))
    execution=dict(processes=len(executions),parts=executions,repeated_or_discarded_controls=0,interrupted_processes=0)
    report=dict(schema='lite3_assembled_evaluation_v2',status='complete',protocol_sha256=sha256_file(args.protocol),
        sources=sources,models=models,phases=phases,ledger=ledger,execution=execution,evidence=evidence,
        scope='All planned conditions ran once in separate processes, same shared model/seed/physics protocol fields. Explicit derived view of complete raw trajectories; independent audit must still verify outcomes and initial hashes.')
    write_json_atomic(out/'report.json',report);write_json_atomic(out/'ledger.json',ledger)
    write_json_atomic(args.audit,dict(status='verified',controls=total,execution=execution,
        evidence=dict(evidence,**{str(out/'report.json'):sha256_file(out/'report.json')})))


if __name__=='__main__':
    main()
