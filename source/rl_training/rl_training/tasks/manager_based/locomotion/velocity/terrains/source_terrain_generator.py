"""Keep tile face ranges when Isaac Lab concatenates the Source terrain map."""

from isaaclab.terrains import TerrainGenerator

from ._mesh_chunking import SOURCE_TILE_FACE_RANGES, validate_tile_face_ranges


class SourceTerrainGenerator(TerrainGenerator):
    """Use the stock geometry, seed order and transforms, recording tile ownership.

    The stock generator still assembles one mesh for the importer. Face ranges
    allow collision chunks to follow physical tiles instead of arbitrary face
    counts; they do not claim to remove the full-map assembly cost.
    """

    def __init__(self, cfg, device="cpu"):
        self._source_face_ranges = []
        self._source_face_count = 0
        super().__init__(cfg, device)
        ranges = validate_tile_face_ranges(self._source_face_ranges, len(self.terrain_mesh.faces))
        self.terrain_mesh.metadata[SOURCE_TILE_FACE_RANGES] = ranges

    def _record_faces(self, name, count):
        start = self._source_face_count
        self._source_face_count += count
        self._source_face_ranges.append((name, start, self._source_face_count))

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        self._record_faces(f"tile_r{row:04d}_c{col:02d}", len(mesh.faces))

    def _add_terrain_border(self):
        # With zero width, the stock border consists of zero-area triangles.
        # It contributes no surface; do not pass it to PhysX cooking.
        if self.cfg.border_width == 0.0:
            return
        super()._add_terrain_border()
        self._record_faces("border", len(self.terrain_meshes[-1].faces))


__all__ = ["SourceTerrainGenerator"]
