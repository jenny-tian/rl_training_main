"""Terrain importer that preserves spatial friction regions as PhysX materials."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.utils import create_prim_from_mesh

from .source_terrain_isaac import (
    FRICTION_COEFFICIENT_SCALE,
    FRICTION_REGION_MARKER_ALPHA,
)


_LOG = logging.getLogger(__name__)
_VISUAL_FACE_COLOUR = np.asarray([150, 150, 150, 255], dtype=np.uint8)


class SourceTerrainImporter(TerrainImporter):
    """Import ordinary terrain faces together and friction regions by material.

    Isaac Lab's stock importer binds one rigid-body material to a generated
    mesh.  Source friction tracks encode their candidate coefficient in face
    colours; this importer separates only those faces into independently bound
    static triangle-mesh colliders.  All other terrain families remain in one
    ordinary collider with the configured terrain material.
    """

    cfg: TerrainImporterCfg

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        root_path = f"{self.cfg.prim_path}/{name}"
        if any(path == root_path or path.startswith(f"{root_path}/") for path in self.terrain_prim_paths):
            raise ValueError(
                f"A terrain with the name '{name}' already exists. "
                f"Existing terrains: {', '.join(self.terrain_names)}."
            )

        face_colours = np.asarray(mesh.visual.face_colors, dtype=np.uint8)
        if face_colours.shape != (len(mesh.faces), 4):
            raise ValueError("Source Terrain mesh face colours do not align with its triangles")
        tagged = face_colours[:, 3] == FRICTION_REGION_MARKER_ALPHA
        if not tagged.any():
            self._import_mesh_part(root_path, mesh, self.cfg.physics_material)
            return

        coefficient_q = (
            face_colours[:, 0].astype(np.uint32) << 16
            | face_colours[:, 1].astype(np.uint32) << 8
            | face_colours[:, 2].astype(np.uint32)
        )
        if np.any(coefficient_q[tagged] == 0):
            raise ValueError("Source Terrain friction-region marker contains a zero coefficient")
        coefficient_values = coefficient_q.astype(np.float64) / FRICTION_COEFFICIENT_SCALE

        expected_values = self._configured_friction_candidates()
        present_values = np.unique(np.round(coefficient_values[tagged], 6))
        unknown = [value for value in present_values if not np.any(np.isclose(expected_values, value, atol=5e-7))]
        if unknown:
            raise ValueError(
                "friction-region mesh contains coefficients absent from the Source Terrain configuration: "
                f"{unknown}"
            )

        sim_utils.create_prim(root_path, "Xform")
        ordinary_faces = np.flatnonzero(~tagged)
        if ordinary_faces.size:
            ordinary_mesh = self._extract_faces(mesh, ordinary_faces)
            ordinary_mesh.visual.face_colors = np.tile(_VISUAL_FACE_COLOUR, (len(ordinary_mesh.faces), 1))
            self._import_mesh_part(f"{root_path}/ordinary", ordinary_mesh, self.cfg.physics_material)

        for coefficient in present_values:
            coefficient_faces = np.flatnonzero(tagged & np.isclose(coefficient_values, coefficient, atol=5e-7))
            region_mesh = self._extract_faces(mesh, coefficient_faces)
            # The marker is an internal transport tag, not a terrain texture.
            region_mesh.visual.face_colors = np.tile(_VISUAL_FACE_COLOUR, (len(region_mesh.faces), 1))
            material = self._make_region_material(float(coefficient))
            region_name = f"friction_{int(round(float(coefficient) * 1_000_000)):08d}"
            self._import_mesh_part(f"{root_path}/{region_name}", region_mesh, material)

        _LOG.info(
            "Imported Source Terrain '%s' with %d friction-material collider(s): %s",
            name,
            len(present_values),
            present_values.tolist(),
        )

    def _configured_friction_candidates(self) -> np.ndarray:
        generator_cfg = self.cfg.terrain_generator
        if generator_cfg is None:
            raise ValueError("friction-region meshes require a Source Terrain generator configuration")
        friction_cfgs = [
            sub_cfg
            for sub_cfg in generator_cfg.sub_terrains.values()
            if getattr(sub_cfg, "family", None) == "friction_variation"
        ]
        if not friction_cfgs:
            raise ValueError("friction-region faces were found but no friction Source Terrain is configured")
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
        return np.linspace(friction_range[0], friction_range[1], region_count, dtype=np.float64)

    def _make_region_material(self, coefficient: float) -> sim_utils.RigidBodyMaterialCfg:
        base = self.cfg.physics_material
        if not isinstance(base, sim_utils.RigidBodyMaterialCfg):
            raise TypeError("Source Terrain friction variation requires a rigid-body terrain material")
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
    def _extract_faces(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> trimesh.Trimesh:
        extracted = mesh.submesh([face_indices], append=True, repair=False)
        if not isinstance(extracted, trimesh.Trimesh) or len(extracted.faces) == 0:
            raise RuntimeError("failed to extract non-empty Source Terrain collider faces")
        extracted.remove_unreferenced_vertices()
        return extracted

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


__all__ = ["SourceTerrainImporter"]
