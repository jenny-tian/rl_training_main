"""Low-friction, gently undulating grass-like terrain.

The geometry is intentionally similar to continuous rough terrain, while the
task configuration supplies a lower-friction physics material. This isolates
surface interaction from obstacle height and keeps the experiment reproducible.
"""

from __future__ import annotations

from .continuous_rough_track import ContinuousRoughTrackCfg, continuous_rough_track


GrassTrackCfg = ContinuousRoughTrackCfg
grass_track = continuous_rough_track

__all__ = ["GrassTrackCfg", "grass_track"]
