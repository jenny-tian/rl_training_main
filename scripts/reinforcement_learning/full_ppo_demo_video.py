"""Method-specific overlay; all frames use the existing audited video writer."""
from rl_training.online_adaptation.simulation_video import SimulationVideo


class ComparisonVideo(SimulationVideo):
    def __init__(self, *args, label, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def append(self, rgb, *, step, time_s, lateral_m, tilt_deg, terrain, env_id):
        import numpy as np
        from PIL import Image, ImageDraw

        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError('unexpected simulator RGB format')
        frame = Image.fromarray(rgb)
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, self.width, 62), fill=(15, 22, 30))
        draw.text((12, 6), 'NEW DEMO RUN | ' + self.label, font=self.font, fill='white')
        draw.text((12, 34),
            f'Terrain {terrain} | env {env_id} | t={time_s:.2f}s | y={lateral_m:+.3f}m | tilt={tilt_deg:.1f}deg',
            font=self.font, fill=(140, 220, 250))
        if not self.samples:
            frame.save(self.path.with_suffix('.preview.png'))
        self.process.stdin.write(np.asarray(frame).tobytes())
        self.samples.append(dict(frame=len(self.samples), step=step, time_s=time_s,
            lateral_m=lateral_m, tilt_deg=tilt_deg))
