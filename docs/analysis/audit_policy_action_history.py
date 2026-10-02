"""Compare recorded next action-history inputs with emitted clipped commands."""
import argparse
import json
from pathlib import Path
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic
from rl_training.lite3_joint_conventions import LITE3_SAFE_ACTION_LOWER,LITE3_SAFE_ACTION_UPPER


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.training);manifest=json.loads((root/'trajectory/manifest.json').read_text())
    report={'training_report_sha256':sha256_file(root/'report.json'),'rows':0,'nonterminal_rows':0,
            'clipped_rows':0,'nonterminal_clipped_history_mismatches':0,'nonterminal_raw_history_mismatches':0,
            'max_emitted_history_error':0.,'chunk_hashes':{}}
    for c in manifest['chunks']:
        path=root/'trajectory'/c['file'];report['chunk_hashes'][str(path)]=sha256_file(path)
        with np.load(path) as data:
            requested=data['requested_action'];history=data['next_policy_observation'][...,-12:];active=~data['done'].astype(bool)
            emitted=np.clip(requested,np.array(LITE3_SAFE_ACTION_LOWER),np.array(LITE3_SAFE_ACTION_UPPER))
            raw_error=np.max(np.abs(history-requested),axis=-1);error=np.max(np.abs(history-emitted),axis=-1)
            report['rows']+=active.size;report['nonterminal_rows']+=int(active.sum())
            report['clipped_rows']+=int((np.max(np.abs(requested-emitted),axis=-1)>1e-6).sum())
            report['nonterminal_clipped_history_mismatches']+=int(((error>1e-6)&active).sum())
            report['nonterminal_raw_history_mismatches']+=int(((raw_error>1e-6)&active).sum())
            if active.any():report['max_emitted_history_error']=max(report['max_emitted_history_error'],float(error[active].max()))
    report['emitted_action_history_compatible']=report['nonterminal_clipped_history_mismatches']==0
    write_json_atomic(args.output,report);print({k:v for k,v in report.items() if k!='chunk_hashes'})


if __name__=='__main__':main()
