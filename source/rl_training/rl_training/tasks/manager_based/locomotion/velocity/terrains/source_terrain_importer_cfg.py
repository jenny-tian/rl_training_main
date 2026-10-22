"""Configuration for the material-aware Source Terrain importer."""

from __future__ import annotations

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .source_terrain_importer import SourceTerrainImporter


@configclass
class SourceTerrainImporterCfg(TerrainImporterCfg):
    """Use :class:`SourceTerrainImporter` for the six-family Source mix."""

    class_type: type = SourceTerrainImporter


__all__ = ["SourceTerrainImporterCfg"]
