"""Draw the registered reward geometry, not a simulated robot trajectory."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    p=json.loads(Path(args.plan).read_text())['route_training']
    length=p['heading_lookahead_m'];bound=np.deg2rad(p['heading_maximum_deg'])
    lateral=.6
    target=float(np.clip(np.arctan2(-lateral,length),-bound,bound))
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    ax=axes[0]
    ax.axhline(0,color='#65758B',label='Route centerline')
    ax.plot([0,length],[lateral,0],'--',color='#AAAAAA',label='Unbounded lookahead direction')
    ax.scatter([0,length],[lateral,0],color=['#65758B','#65758B'])
    ax.annotate('Robot position (illustration)',(0,lateral),xytext=(.05,lateral+.1))
    ax.annotate('',(length,lateral),(0,lateral),arrowprops=dict(arrowstyle='->',color='#B27C48',lw=2))
    ax.annotate('',(length,lateral+length*np.tan(target)),(0,lateral),arrowprops=dict(arrowstyle='->',color='#2A8C82',lw=2))
    ax.text(.65,lateral+.035,'Old: 0 deg',color='#B27C48')
    ax.text(.60,.40,f'New: {np.rad2deg(target):.0f} deg',color='#2A8C82')
    ax.set(xlim=(-.1,1.6),ylim=(-.1,.85),xlabel='Forward offset (m)',ylabel='Lateral offset (m)',title='Target turns toward the route, bounded at 20 deg')
    ax.legend(frameon=True,loc='upper right',fontsize=7)
    ax=axes[1];yaw=np.deg2rad(np.linspace(-45,45,361))
    ax.plot(np.rad2deg(yaw),1-np.cos(yaw),color='#B27C48',label='Old target: 0 deg')
    ax.plot(np.rad2deg(yaw),1-np.cos(yaw-target),color='#2A8C82',label='New bounded target')
    ax.set(xlabel='Robot heading (deg)',ylabel='Heading cost (lower is better)',title='Same weight; the minimum moves with lateral error')
    ax.legend(frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Training reward only: no extra steering controller at inference')
    out=Path(args.output)
    for suffix in ('.png','.svg'):fig.savefig(out.with_suffix(suffix),dpi=160)
    svg=out.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
