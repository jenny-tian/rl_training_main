"""Audit a training-only angular-velocity counterfactual and early foot contacts."""
import json
from pathlib import Path
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def main():
    base=Path('artifacts/recovery_teacher_20261008');out={};initials={};evidence={}
    bank_meta=json.loads(Path('docs/lite3_recovery_diagnostic_bank_20261008.json').read_text())
    with np.load(bank_meta['source'],allow_pickle=False) as d:original_bank={k:d[k] for k in d.files}
    with np.load(bank_meta['path'],allow_pickle=False) as d:half_bank={k:d[k] for k in d.files}
    for k,arr in original_bank.items():
        expected=arr.copy()
        if k=='root':expected[~original_bank['validation'],10:13]*=.5
        np.testing.assert_array_equal(half_bank[k],expected)
    for label in ('original','half-angular'):
        root=base/f'diagnostic-{label}';report_path=root/'report.json';report=json.loads(report_path.read_text())
        audit_path=Path(f'docs/lite3_recovery_diagnostic_{label}_audit_20261008.json');audit=json.loads(audit_path.read_text())
        assert audit['status']=='verified' and audit['evidence'][str(report_path)]==sha256_file(report_path)
        evidence[str(audit_path)]=sha256_file(audit_path);evidence[str(report_path)]=sha256_file(report_path)
        folder=root/'smoke-trial';initials[label]={k:v for k,v in np.load(folder/'initial.npz',allow_pickle=False).items()}
        arrays={}
        for path in sorted((folder/'trajectory').glob('chunk-*.npz')):
            with np.load(path,allow_pickle=False) as d:
                for k in ('active','foot_contact_force_w','post_root_state','requested_action','emitted_action'):arrays.setdefault(k,[]).append(d[k])
        arrays={k:np.concatenate(v) for k,v in arrays.items()};cases={}
        for e in report['phases']['smoke-trial']['episodes']:
            if e['terrain']=='flat':continue
            i=e['env_id'];first=np.flatnonzero(arrays['active'][:10,i])
            forces=arrays['foot_contact_force_w'][first,i]
            contacts=(np.linalg.norm(forces,axis=-1)>5).sum(-1)
            loaded=(forces[...,2]>5).sum(-1)
            sample=dict(env_id=i,steps=e['steps'],failure=any(e.get('failure_terms',{}).values()),
                failure_terms=e.get('failure_terms',{}),collector_cutoff=e.get('collector_cutoff',False),
                first10_active_contact_counts=contacts.tolist(),first10_active_upward_load_counts=loaded.tolist(),
                first10_action_clip_fraction=float(np.mean(np.abs(arrays['requested_action'][first,i]-arrays['emitted_action'][first,i])>1e-6)))
            cases.setdefault(str(e['case']),[]).append(sample)
        out[label]=dict(cases=cases,rough_trials=sum(map(len,cases.values())),failures=sum(e['failure'] for es in cases.values() for e in es),controls=report['ledger']['confirmed'])
    a=initials['original'];b=initials['half-angular'];mask=a['case']>=0
    for k in a:
        expected=a[k].copy()
        if k=='root':expected[mask,10:13]*=.5
        elif k=='observation':expected[mask,:3]*=.5
        np.testing.assert_allclose(b[k],expected,atol=2e-6,rtol=1e-6)
    assert json.loads((base/'diagnostic-original/smoke-trial/delays.json').read_text())==json.loads((base/'diagnostic-half-angular/smoke-trial/delays.json').read_text())
    write_json_atomic('docs/lite3_recovery_diagnostic_result_20261008.json',dict(status='verified',paired_except_intended_angular_change=True,results=out,evidence=evidence,
        scope='20 TRAIN cases repeated across48 rough slots. 128-control cutoff is not complete-course success. Contacts are last-substep net force >5N; upward-load count uses world Fz>5N. Only active first10 controls used, never post-reset data. No static stability or whole-body COM claim.'))


if __name__=='__main__':main()
