"""Six terrain families on the reference's native curriculum and allocation."""
from __future__ import annotations

import copy
import numpy as np

from isaaclab.terrains import HfPyramidSlopedTerrainCfg, MeshPlaneTerrainCfg
from isaaclab.terrains.height_field.hf_terrains import pyramid_sloped_terrain, random_uniform_terrain
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from lite3_terrain_reuse.source_terrain_importer import SourceTerrainImporter
from lite3_terrain_reuse.source_terrain_importer_cfg import SourceTerrainImporterCfg
from lite3_terrain_reuse.source_terrain_isaac import SourceTerrainTrackCfg, source_terrain_track


@height_field_to_mesh
def slope_with_roughness(difficulty, cfg):
    """Compose the two native height fields before their single mesh conversion."""
    slope = pyramid_sloped_terrain.__wrapped__(difficulty, cfg).astype(np.int32)
    rough = random_uniform_terrain.__wrapped__(difficulty, cfg).astype(np.int32)
    # Keep the reference slope's centre platform level, including reset jitter.
    x = (np.arange(slope.shape[0]) - slope.shape[0] // 2) * cfg.horizontal_scale
    y = (np.arange(slope.shape[1]) - slope.shape[1] // 2) * cfg.horizontal_scale
    platform = (np.abs(x[:, None]) <= cfg.platform_width / 2) & (np.abs(y[None, :]) <= cfg.platform_width / 2)
    rough[platform] = 0
    combined = slope + rough
    if combined.min() < -32768 or combined.max() > 32767:
        raise ValueError("combined terrain exceeds int16 height range")
    return combined.astype(np.int16)


@configclass
class RoughSlopeCfg(HfPyramidSlopedTerrainCfg):
    function = slope_with_roughness
    noise_range: tuple[float, float] = (.01, .06)
    noise_step: float = .01
    downsampled_scale: float | None = None


def centered_friction(difficulty, cfg):
    native = copy.deepcopy(cfg)
    if native.seed is None:
        native.seed = int(np.random.get_state()[1][0])
    meshes, _ = source_terrain_track(difficulty, native)
    return meshes, np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.])


@configclass
class CenteredFrictionCfg(SourceTerrainTrackCfg):
    function = centered_friction
    family: str = "friction_variation"
    friction_range: tuple[float, float] = (.5, 1.)
    friction_region_count: int = 4


class ReferenceFrictionImporter(SourceTerrainImporter):
    """Reuse material binding while retaining native origin allocation/curriculum.

    Native RayCaster reads the first USD mesh. Add an invisible, non-colliding
    copy of the full geometry first, then import the material-split colliders.
    Sensor configuration and sensor initialization remain unchanged.
    """
    def import_mesh(self, name, mesh):
        import isaaclab.sim as sim_utils
        from pxr import UsdGeom, UsdPhysics
        scan_path = f"{self.cfg.prim_path}/height_scan_geometry"
        prim = sim_utils.create_prim(scan_path, "Mesh", attributes={
            "points": mesh.vertices, "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.full(len(mesh.faces), 3), "subdivisionScheme": "none"})
        UsdGeom.Imageable(prim).MakeInvisible()
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError("height-only mesh must not introduce duplicate contacts")
        super().import_mesh(name, mesh)
        first = sim_utils.get_first_matching_child_prim(self.cfg.prim_path, lambda p: p.GetTypeName() == "Mesh")
        if str(first.GetPath()) != scan_path:
            raise RuntimeError("native scanner would not select the full terrain")
        self.reference_scan_path = scan_path
        self.reference_ray_mesh_vertices = len(mesh.vertices)
        self.reference_ray_mesh_triangles = len(mesh.faces)


@configclass
class ReferenceFrictionImporterCfg(SourceTerrainImporterCfg):
    class_type: type = ReferenceFrictionImporter


def replace_terrain(reference_cfg):
    cfg = copy.deepcopy(reference_cfg)
    old = reference_cfg.scene.terrain
    terrain = ReferenceFrictionImporterCfg()
    for name in old.__dataclass_fields__:
        if name != "class_type":
            setattr(terrain, name, copy.deepcopy(getattr(old, name)))
    source = copy.deepcopy(old.terrain_generator.sub_terrains)
    rough = source["random_rough"]
    rough.proportion = .20
    # From the centre platform, inverted terrain is uphill and a raised
    # pyramid is downhill. Names describe the robot's outward movement.
    uphill, downhill = source["hf_pyramid_slope_inv"], source["hf_pyramid_slope"]
    stairs_up, stairs_down = source["pyramid_stairs_inv"], source["pyramid_stairs"]
    for sub in (uphill, downhill, stairs_up, stairs_down):
        sub.proportion = .10
    rough_slopes = []
    for sub in (uphill, downhill):
        combined = RoughSlopeCfg()
        for name in sub.__dataclass_fields__:
            if name != "function":
                setattr(combined, name, copy.deepcopy(getattr(sub, name)))
        combined.proportion = .05
        rough_slopes.append(combined)
    terrain.terrain_generator.sub_terrains = {
        "flat": MeshPlaneTerrainCfg(proportion=.20),
        "rough": rough,
        "slope_up": uphill,
        "slope_down": downhill,
        "stairs_up": stairs_up,
        "stairs_down": stairs_down,
        "friction_lateral": CenteredFrictionCfg(proportion=.05, friction_layout="left_right_bands"),
        "friction_longitudinal": CenteredFrictionCfg(proportion=.05, friction_layout="front_back_bands"),
        "slope_rough_up": rough_slopes[0],
        "slope_rough_down": rough_slopes[1],
    }
    cfg.scene.terrain = terrain
    return cfg


FAMILIES = {"flat": "flat", "rough": "rough", "slope_up": "slope", "slope_down": "slope",
            "stairs_up": "stairs", "stairs_down": "stairs", "friction_lateral": "friction",
            "friction_longitudinal": "friction", "slope_rough_up": "slope_rough",
            "slope_rough_down": "slope_rough"}


def column_names(cfg):
    sub = cfg.scene.terrain.terrain_generator.sub_terrains
    names = list(sub)
    weights = np.array([v.proportion for v in sub.values()])
    cumulative = np.cumsum(weights / weights.sum())
    count = cfg.scene.terrain.terrain_generator.num_cols
    return [names[np.flatnonzero(i / count + .001 < cumulative)[0]] for i in range(count)]
