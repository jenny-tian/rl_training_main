"""Audit a frozen full-smoke pair: identical physics and only heading weight4x."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('control-run','control-protocol','run','protocol','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    oldroot,newroot=Path(args.control_run),Path(args.run)
    cp=json.loads(Path(args.control_protocol).read_text());p=json.loads(Path(args.protocol).read_text())
    a=json.loads((oldroot/'report.json').read_text());b=json.loads((newroot/'report.json').read_text())
    assert a['status']==b['status']=='complete'
    assert a['protocol_sha256']==sha256_file(args.control_protocol) and b['protocol_sha256']==sha256_file(args.protocol)
    assert set(cp)==set(p)
    for k,v in cp.items():
        if k not in {'registered_at_utc','scope','route_training'}:assert p[k]==v,k
    assert set(p['route_training'])==set(cp['route_training'])
    for k,v in cp['route_training'].items():
        if k!='heading_weight':assert p['route_training'][k]==v,k
    assert p['route_training']['heading_weight']==4*cp['route_training']['heading_weight']==-4.
    assert a['source_files']==b['source_files']
    assert sha256_file(oldroot/'delays.json')==sha256_file(newroot/'delays.json')
    with np.load(oldroot/'initial.npz',allow_pickle=False) as before,np.load(newroot/'initial.npz',allow_pickle=False) as after:
        assert set(before.files)==set(after.files)
        for k in before.files:np.testing.assert_array_equal(before[k],after[k])
    files=sorted((oldroot/'trajectory').glob('chunk-*.npz'))
    assert [f.name for f in files]==[f.name for f in sorted((newroot/'trajectory').glob('chunk-*.npz'))]
    hashes={};controls=0;maximum=0.
    for file in files:
        new=newroot/'trajectory'/file.name
        hashes[str(file)]=sha256_file(file);hashes[str(new)]=sha256_file(new)
        with np.load(file,allow_pickle=False) as before,np.load(new,allow_pickle=False) as after:
            assert set(before.files)==set(after.files)
            for k in before.files:
                if k not in {'reward','heading_reward'}:np.testing.assert_array_equal(before[k],after[k])
            np.testing.assert_allclose(after['heading_reward'],4*before['heading_reward'],atol=1e-7,rtol=1e-6)
            delta=3*before['heading_reward'];residual=after['reward']-before['reward']-delta
            np.testing.assert_allclose(after['reward']-before['reward'],delta,atol=3e-6,rtol=1e-4)
            maximum=max(maximum,float(abs(residual).max()));controls+=after['done'].size
    assert controls==a['ledger']['confirmed']==b['ledger']['confirmed']==b['ledger']['charged']==p['max_controls']
    assert b['ledger']['uncertain_transitions']==0
    evidence={str(path):sha256_file(path) for path in (oldroot/'report.json',newroot/'report.json',oldroot/'initial.npz',newroot/'initial.npz',oldroot/'delays.json',newroot/'delays.json',Path(args.control_protocol),Path(args.protocol))}
    write_json_atomic(args.output,dict(status='verified',new_controls=controls,all_frozen_physics_fields_bitwise_equal=True,
        maximum_heading_scaling_residual=maximum,trace_sha256=hashes,evidence=evidence,
        scope='Full frozen smoke pair; no optimizer. New25600 controls charged even though physics reproduced. Only reward weight scaling differs.'))


if __name__=='__main__':main()
