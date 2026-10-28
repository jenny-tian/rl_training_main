"""Configuration for the material-aware Source Terrain importer."""

from __future__ import annotations

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .source_terrain_importer import SourceTerrainImporter


@configclass
class SourceTerrainImporterCfg(TerrainImporterCfg):
    """Use :class:`SourceTerrainImporter` for the six-family Source mix."""

    class_type: type = SourceTerrainImporter
    parallel_reset_sampler_enabled: bool = False
    parallel_reset_sampler_num_envs: int | None = None
    shared_bank_rows: int | None = None


__all__ = ["SourceTerrainImporterCfg"]
