"""Stream annotated simulator RGB frames to MP4 without advancing physics."""
import json
from pathlib import Path
import subprocess

import numpy as np


class SimulationVideo:
    def __init__(self, path, *, width, height, fps):
        from PIL import ImageFont
        self.path=Path(path);self.width=width;self.height=height;self.fps=fps;self.samples=[]
        self.log=self.path.with_suffix('.ffmpeg.log').open('wb')
        self.process=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
            '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(self.path)],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log)
        font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        self.font=ImageFont.truetype(str(font),18) if font.exists() else ImageFont.load_default()

    def append(self, rgb, *, step, time_s, lateral_m, tilt_deg, terrain, env_id):
        from PIL import Image,ImageDraw
        if rgb.shape!=(self.height,self.width,3) or rgb.dtype!=np.uint8:
            raise ValueError('unexpected simulator RGB format')
        frame=Image.fromarray(rgb);draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,self.width,62),fill=(15,22,30))
        draw.text((12,6),'NEW SIMULATION RUN | PPO recovery teacher',font=self.font,fill='white')
        draw.text((12,34),f'Terrain {terrain} | env {env_id} | t={time_s:.2f}s | y={lateral_m:+.3f}m | tilt={tilt_deg:.1f}deg',font=self.font,fill=(140,220,250))
        if not self.samples:frame.save(self.path.with_suffix('.preview.png'))
        self.process.stdin.write(np.asarray(frame).tobytes())
        self.samples.append(dict(frame=len(self.samples),step=step,time_s=time_s,lateral_m=lateral_m,tilt_deg=tilt_deg))

    def close(self):
        if self.process is None:return
        self.process.stdin.close();code=self.process.wait();self.log.close();self.process=None
        self.path.with_suffix('.frames.json').write_text(json.dumps(dict(fps=self.fps,frames=self.samples),indent=2)+'\n')
        if code:raise RuntimeError(f'ffmpeg exited {code}; see video log')
