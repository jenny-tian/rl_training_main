"""Terrain importer that preserves spatial friction regions as PhysX materials."""

from __future__ import annotations

import logging

import torch
import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.utils import create_prim_from_mesh

from rl_training.source_scan.parallel_terrain_allocation import (
    SOURCE_RESET_BLOCK_COUNTS,
    initialize_reserved_row_environment_origins,
)

from ._mesh_chunking import SOURCE_TILE_FACE_RANGES, balanced_face_chunk_slices, validate_tile_face_ranges
from .source_terrain_isaac import (
    FRICTION_COEFFICIENT_SCALE,
    FRICTION_REGION_MARKER_ALPHA,
)

_LOG = logging.getLogger(__name__)
_VISUAL_FACE_COLOUR = np.asarray([150, 150, 150, 255], dtype=np.uint8)
# A construction budget, not a claimed PhysX limit or runtime qualification.
# Current Source tiles fit below this after bottom simplification. Tile
# ownership is the primary partition; count splitting is only a fallback.
MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES = 100_000


class SourceTerrainImporter(TerrainImporter):
    """Import ordinary terrain faces together and friction regions by material.

    Isaac Lab's stock importer binds one rigid-body material to a generated
    mesh.  Source friction tracks encode their candidate coefficient in face
    colours; this importer separates only those faces into independently bound
    static triangle-mesh colliders. Each material class is then split by the
    generator's tile ranges, preserving triangle coordinates and winding.
    """

    cfg: TerrainImporterCfg

    def __init__(self, cfg: TerrainImporterCfg):
        self._parallel_reset_sampler_enabled = bool(
            getattr(cfg, "parallel_reset_sampler_enabled", False)
        )
        self._shared_bank_rows = getattr(cfg, "shared_bank_rows", None)
        if self._shared_bank_rows is not None and not self._parallel_reset_sampler_enabled:
            raise ValueError("shared Source bank requires the explicit parallel reset sampler")
        if self._parallel_reset_sampler_enabled:
            generator_cfg = getattr(cfg, "terrain_generator", None)
            configured_num_envs = getattr(cfg, "parallel_reset_sampler_num_envs", None)
            if type(configured_num_envs) is not int or configured_num_envs < 1:
                raise ValueError(
                    "parallel Source importer requires an explicit positive environment count"
                )
            if (
                cfg.terrain_type != "generator"
                or not cfg.use_terrain_origins
                or generator_cfg is None
            ):
                raise ValueError(
                    "parallel Source importer requires generated terrain origins"
                )
            if cfg.num_envs != configured_num_envs:
                raise ValueError(
                    "parallel Source importer environment count changed after terrain config construction"
                )
            names = tuple(generator_cfg.sub_terrains)
            if names != tuple(SOURCE_RESET_BLOCK_COUNTS):
                raise ValueError(
                    "parallel Source importer sub-terrain classes do not match the registered layout"
                )
            rows = cfg.num_envs if self._shared_bank_rows is None else self._shared_bank_rows
            if type(rows) is not int or not 1 <= rows <= cfg.num_envs:
                raise ValueError("shared Source bank row count is invalid")
            if generator_cfg.num_rows != rows or generator_cfg.num_cols != len(
                names
            ):
                raise ValueError(
                    "parallel Source importer requires the declared row count and 15 columns"
                )
            expected_proportion = 1.0 / len(names)
            if any(
                abs(float(sub_cfg.proportion) - expected_proportion) > 1e-12
                for sub_cfg in generator_cfg.sub_terrains.values()
            ):
                raise ValueError(
                    "parallel Source importer requires equal physical tile proportions"
                )
        super().__init__(cfg)

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        root_path = f"{self.cfg.prim_path}/{name}"
        if any(
            path == root_path or path.startswith(f"{root_path}/")
            for path in self.terrain_prim_paths
        ):
            raise ValueError(
                f"A terrain with the name '{name}' already exists. "
                f"Existing terrains: {', '.join(self.terrain_names)}."
            )

        tile_ranges = mesh.metadata.get(SOURCE_TILE_FACE_RANGES)
        if tile_ranges is not None:
            validate_tile_face_ranges(tile_ranges, len(mesh.faces))
        elif getattr(self.cfg, "parallel_reset_sampler_enabled", False):
            raise ValueError("parallel Source terrain requires generator tile face ranges")

        face_colours = np.asarray(mesh.visual.face_colors, dtype=np.uint8)
        if face_colours.shape != (len(mesh.faces), 4):
            raise ValueError(
                "Source Terrain mesh face colours do not align with its triangles"
            )
        tagged = face_colours[:, 3] == FRICTION_REGION_MARKER_ALPHA
        if not tagged.any():
            self._import_mesh_chunks(
                root_path,
                mesh,
                self.cfg.physics_material,
                face_indices=None,
            )
            return

        coefficient_q = (
            face_colours[:, 0].astype(np.uint32) << 16
            | face_colours[:, 1].astype(np.uint32) << 8
            | face_colours[:, 2].astype(np.uint32)
        )
        if np.any(coefficient_q[tagged] == 0):
            raise ValueError(
                "Source Terrain friction-region marker contains a zero coefficient"
            )
        coefficient_values = (
            coefficient_q.astype(np.float64) / FRICTION_COEFFICIENT_SCALE
        )

        expected_values = self._configured_friction_candidates()
        present_values = np.unique(np.round(coefficient_values[tagged], 6))
        unknown = [
            value
            for value in present_values
            if not np.any(np.isclose(expected_values, value, atol=5e-7))
        ]
        if unknown:
            raise ValueError(
                "friction-region mesh contains coefficients absent from the Source Terrain configuration: "
                f"{unknown}"
            )

        sim_utils.create_prim(root_path, "Xform")
        ordinary_faces = np.flatnonzero(~tagged)
        if ordinary_faces.size:
            self._import_mesh_chunks(
                f"{root_path}/ordinary",
                mesh,
                self.cfg.physics_material,
                face_indices=ordinary_faces,
                reset_face_colours=True,
            )

        for coefficient in present_values:
            coefficient_faces = np.flatnonzero(
                tagged & np.isclose(coefficient_values, coefficient, atol=5e-7)
            )
            material = self._make_region_material(float(coefficient))
            region_name = f"friction_{int(round(float(coefficient) * 1_000_000)):08d}"
            self._import_mesh_chunks(
                f"{root_path}/{region_name}",
                mesh,
                material,
                face_indices=coefficient_faces,
                reset_face_colours=True,
            )

        _LOG.info(
            "Imported Source Terrain '%s' with %d friction material(s): %s",
            name,
            len(present_values),
            present_values.tolist(),
        )

    def _import_mesh_chunks(
        self,
        root_path: str,
        mesh: trimesh.Trimesh,
        physics_material: sim_utils.RigidBodyMaterialCfg,
        *,
        face_indices: np.ndarray | None,
        reset_face_colours: bool = False,
    ) -> None:
        """Import a material class by whole tiles, then enforce a count budget."""

        face_count = len(mesh.faces) if face_indices is None else len(face_indices)
        if face_count == 0:
            raise ValueError("Source Terrain colliders must contain at least one triangle")
        tile_ranges = mesh.metadata.get(SOURCE_TILE_FACE_RANGES)
        chunks = []
        if tile_ranges is not None:
            for tile_name, start, stop in validate_tile_face_ranges(tile_ranges, len(mesh.faces)):
                if face_indices is None:
                    selected = np.arange(start, stop, dtype=np.int64)
                else:
                    # Material selections come from flatnonzero and stay sorted.
                    lower, upper = np.searchsorted(face_indices, (start, stop))
                    selected = face_indices[lower:upper]
                for index, span in enumerate(balanced_face_chunk_slices(
                    len(selected), MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES
                )):
                    suffix = f"/part_{index:03d}" if len(selected) > MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES else ""
                    chunks.append((f"{tile_name}{suffix}", selected[span]))
        else:
            for index, span in enumerate(balanced_face_chunk_slices(
                face_count, MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES
            )):
                selected = (
                    np.arange(span.start, span.stop, dtype=np.int64)
                    if face_indices is None else face_indices[span]
                )
                chunks.append((f"part_{index:03d}", selected))

        if len(chunks) == 1 and tile_ranges is None and face_indices is None and not reset_face_colours:
            self._import_mesh_part(root_path, mesh, physics_material)
            return

        use_children = tile_ranges is not None or len(chunks) > 1
        if use_children:
            sim_utils.create_prim(root_path, "Xform")

        for chunk_name, selected_faces in chunks:
            chunk_path = f"{root_path}/{chunk_name}" if use_children else root_path
            if "/part_" in chunk_name:
                parent_path = chunk_path.rsplit("/", 1)[0]
                if chunk_name.endswith("/part_000"):
                    sim_utils.create_prim(parent_path, "Xform")
            chunk_mesh = self._extract_faces(mesh, selected_faces)
            if reset_face_colours:
                chunk_mesh.visual.face_colors = np.tile(
                    _VISUAL_FACE_COLOUR, (len(chunk_mesh.faces), 1)
                )
            self._import_mesh_part(chunk_path, chunk_mesh, physics_material)

        if use_children:
            _LOG.info(
                "Imported terrain collider '%s' as %d triangle-mesh parts "
                "(%d total triangles; budget %d per part; tile layout: %s)",
                root_path,
                len(chunks),
                face_count,
                MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES,
                tile_ranges is not None,
            )

    def _configured_friction_candidates(self) -> np.ndarray:
        generator_cfg = self.cfg.terrain_generator
        if generator_cfg is None:
            raise ValueError(
                "friction-region meshes require a Source Terrain generator configuration"
            )
        friction_cfgs = [
            sub_cfg
            for sub_cfg in generator_cfg.sub_terrains.values()
            if getattr(sub_cfg, "family", None) == "friction_variation"
        ]
        if not friction_cfgs:
            raise ValueError(
                "friction-region faces were found but no friction Source Terrain is configured"
            )
        palettes = {
            (
                tuple(float(value) for value in cfg.friction_range),
                int(cfg.friction_region_count),
            )
            for cfg in friction_cfgs
        }
        if len(palettes) != 1:
            raise ValueError(
                "all friction-variation sub-terrains in one generated mesh must share a coefficient palette"
            )
        friction_range, region_count = palettes.pop()
        return np.linspace(
            friction_range[0], friction_range[1], region_count, dtype=np.float64
        )

    def _make_region_material(
        self, coefficient: float
    ) -> sim_utils.RigidBodyMaterialCfg:
        base = self.cfg.physics_material
        if not isinstance(base, sim_utils.RigidBodyMaterialCfg):
            raise TypeError(
                "Source Terrain friction variation requires a rigid-body terrain material"
            )
        return sim_utils.RigidBodyMaterialCfg(
            static_friction=coefficient,
            dynamic_friction=coefficient,
            restitution=base.restitution,
            friction_combine_mode=base.friction_combine_mode,
            restitution_combine_mode=base.restitution_combine_mode,
            compliant_contact_stiffness=base.compliant_contact_stiffness,
            compliant_contact_damping=base.compliant_contact_damping,
        )

    @staticmethod
    def _extract_faces(
        mesh: trimesh.Trimesh, face_indices: np.ndarray
    ) -> trimesh.Trimesh:
        if len(face_indices) == 0:
            raise RuntimeError(
                "failed to extract non-empty Source Terrain collider faces"
            )
        # Remap only vertices referenced by this chunk. Generic submesh
        # extraction can allocate full-map vertex masks for every small tile.
        selected_faces = mesh.faces[face_indices]
        vertex_ids, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
        return trimesh.Trimesh(
            vertices=mesh.vertices[vertex_ids],
            faces=inverse.reshape(-1, 3),
            face_colors=np.asarray(mesh.visual.face_colors)[face_indices],
            process=False,
        )

    def _import_mesh_part(
        self,
        prim_path: str,
        mesh: trimesh.Trimesh,
        physics_material: sim_utils.RigidBodyMaterialCfg,
    ) -> None:
        if prim_path in self.terrain_prim_paths:
            raise ValueError(f"A terrain collider already exists at '{prim_path}'")
        create_prim_from_mesh(
            prim_path,
            mesh,
            visual_material=self.cfg.visual_material,
            physics_material=physics_material,
        )
        self.terrain_prim_paths.append(prim_path)

    def _compute_env_origins_curriculum(
        self, num_envs: int, origins: np.ndarray | "torch.Tensor"
    ):
        """Reserve one terrain row per environment before scene startup/reset."""

        if not self._parallel_reset_sampler_enabled:
            return super()._compute_env_origins_curriculum(num_envs, origins)
        if num_envs != self.cfg.parallel_reset_sampler_num_envs:
            raise ValueError(
                "parallel Source terrain origin count differs from the configured environment count"
            )
        terrain_origins = origins
        if isinstance(origins, np.ndarray):
            terrain_origins = torch.from_numpy(origins).to(
                self.device, dtype=torch.float
            )
        self.terrain_origins = terrain_origins.to(self.device, dtype=torch.float)
        bank_rows = self._shared_bank_rows
        row_count = num_envs if bank_rows is None else bank_rows
        self.max_terrain_level = row_count
        self.terrain_levels = self.terrain_origins.new_tensor(
            [i % row_count for i in range(num_envs)], dtype=torch.long
        )
        self.terrain_types = self.terrain_levels.new_zeros((num_envs,))
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types]
        if bank_rows is None:
            initialize_reserved_row_environment_origins(self, num_envs)
        else:
            from rl_training.source_scan.parallel_terrain_allocation import initialize_source_environment_origins

            initialize_source_environment_origins(self, num_envs, bank_rows=bank_rows)
        return self.env_origins


__all__ = ["SourceTerrainImporter"]
