"""Verify demo MP4 timing and overlay metadata against the physical trajectory."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.run)
    report=json.loads((root/'report.json').read_text());assert report['status']=='complete'
    checks={}
    for name,phase in report['phases'].items():
        meta=phase['video'];video=Path(meta['path']);env_id=meta['env_id']
        assert sha256_file(video)==meta['sha256']
        samples=json.loads(video.with_suffix('.frames.json').read_text())
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(video)]))['streams'][0]
        assert int(probe['nb_frames'])==meta['frames']==len(samples['frames'])
        numerator,denominator=map(int,probe['avg_frame_rate'].split('/'))
        np.testing.assert_allclose(numerator/denominator,samples['fps'])
        obs=[]
        for file in sorted((root/name/'trajectory').glob('chunk-*.npz')):
            with np.load(file,allow_pickle=False) as data:obs.append(data['policy_observation'][:,env_id].copy())
        obs=np.concatenate(obs);episode=phase['episodes'][env_id]
        for index,sample in enumerate(samples['frames']):
            step=sample['step'];assert sample['frame']==index and step<episode['steps']
            np.testing.assert_allclose(sample['time_s'],step*.02,atol=1e-8)
            np.testing.assert_allclose(sample['lateral_m'],obs[step,9]*1.65,atol=1e-6)
            np.testing.assert_allclose(sample['tilt_deg'],np.degrees(np.arccos(np.clip(-obs[step,5],-1,1))),atol=.002)
        np.testing.assert_allclose(float(probe['duration']),len(samples['frames'])/samples['fps'],atol=.001)
        checks[name]=dict(video=str(video.resolve()),video_sha256=meta['sha256'],frames=meta['frames'],
            fps=samples['fps'],duration_s=float(probe['duration']),width=int(probe['width']),height=int(probe['height']),
            filmed_episode=episode,overlay_matches_trajectory=True,
            scope='New simulation run. Metadata alignment is audited; not original-evaluation footage or an independent training replication.')
    write_json_atomic(args.output,dict(status='verified',videos=checks,report_sha256=sha256_file(root/'report.json')))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
