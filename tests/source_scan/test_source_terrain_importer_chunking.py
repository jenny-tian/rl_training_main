"""Large ordinary terrain meshes are split without launching a simulator."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import trimesh


_ROOT = Path(__file__).resolve().parents[2]
_TERRAIN_DIR = (
    _ROOT
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains"
)
_TEST_PACKAGE = "source_terrain_importer_chunking_test_package"
_PACKAGE_NAMES = (
    "isaaclab",
    "isaaclab.sim",
    "isaaclab.terrains",
    "isaaclab.terrains.utils",
    "rl_training",
    "rl_training.source_scan",
    "rl_training.source_scan.parallel_terrain_allocation",
)
_MISSING = object()


def _load_importer_with_simulator_stubs():
    """Load the importer module without importing Isaac Sim or starting Kit."""

    saved_modules = {name: sys.modules.get(name, _MISSING) for name in _PACKAGE_NAMES}
    try:
        isaaclab = ModuleType("isaaclab")
        isaaclab.__path__ = []
        sim_utils = ModuleType("isaaclab.sim")
        sim_utils.RigidBodyMaterialCfg = type("RigidBodyMaterialCfg", (), {})
        sim_utils.create_prim = lambda *_args, **_kwargs: None
        isaaclab.sim = sim_utils

        terrains = ModuleType("isaaclab.terrains")
        terrains.__path__ = []
        terrains.TerrainImporter = type("TerrainImporter", (), {})
        terrains.TerrainImporterCfg = type("TerrainImporterCfg", (), {})
        terrain_utils = ModuleType("isaaclab.terrains.utils")
        terrain_utils.create_prim_from_mesh = lambda *_args, **_kwargs: None

        rl_training = ModuleType("rl_training")
        rl_training.__path__ = []
        source_scan = ModuleType("rl_training.source_scan")
        source_scan.__path__ = []
        allocation = ModuleType("rl_training.source_scan.parallel_terrain_allocation")
        allocation.SOURCE_RESET_BLOCK_COUNTS = {}
        allocation.initialize_reserved_row_environment_origins = lambda *_args: None

        sys.modules.update(
            {
                "isaaclab": isaaclab,
                "isaaclab.sim": sim_utils,
                "isaaclab.terrains": terrains,
                "isaaclab.terrains.utils": terrain_utils,
                "rl_training": rl_training,
                "rl_training.source_scan": source_scan,
                "rl_training.source_scan.parallel_terrain_allocation": allocation,
            }
        )

        package = ModuleType(_TEST_PACKAGE)
        package.__path__ = [str(_TERRAIN_DIR)]
        sys.modules[_TEST_PACKAGE] = package
        source_terrain_isaac = ModuleType(f"{_TEST_PACKAGE}.source_terrain_isaac")
        source_terrain_isaac.FRICTION_COEFFICIENT_SCALE = 1_000_000
        source_terrain_isaac.FRICTION_REGION_MARKER_ALPHA = 254
        sys.modules[source_terrain_isaac.__name__] = source_terrain_isaac

        module_name = f"{_TEST_PACKAGE}.source_terrain_importer"
        spec = importlib.util.spec_from_file_location(
            module_name, _TERRAIN_DIR / "source_terrain_importer.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in saved_modules.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


importer_module = _load_importer_with_simulator_stubs()


def _mesh_with_disjoint_triangles(face_count: int) -> trimesh.Trimesh:
    vertices = []
    faces = []
    for index in range(face_count):
        x = float(index * 2)
        first = len(vertices)
        vertices.extend(((x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x, 1.0, 0.0)))
        faces.append((first, first + 1, first + 2))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.visual.face_colors = np.tile(
        np.asarray([150, 150, 150, 255], dtype=np.uint8), (face_count, 1)
    )
    return mesh


def _empty_importer():
    importer = object.__new__(importer_module.SourceTerrainImporter)
    importer.cfg = SimpleNamespace(
        prim_path="/World/ground",
        physics_material=object(),
        visual_material=object(),
    )
    importer.terrain_prim_paths = []
    return importer


def test_large_ordinary_mesh_is_imported_as_bounded_child_colliders(monkeypatch):
    monkeypatch.setattr(
        importer_module, "MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES", 2
    )
    parent_prims = []
    imported_meshes = []
    monkeypatch.setattr(
        importer_module.sim_utils,
        "create_prim",
        lambda path, prim_type: parent_prims.append((path, prim_type)),
    )
    monkeypatch.setattr(
        importer_module,
        "create_prim_from_mesh",
        lambda path, mesh, **kwargs: imported_meshes.append(
            (path, len(mesh.faces), kwargs["physics_material"])
        ),
    )

    importer = _empty_importer()
    importer.import_mesh("terrain", _mesh_with_disjoint_triangles(3))

    assert parent_prims == [("/World/ground/terrain", "Xform")]
    assert [item[:2] for item in imported_meshes] == [
        ("/World/ground/terrain/part_000", 2),
        ("/World/ground/terrain/part_001", 1),
    ]
    assert [path for path in importer.terrain_prim_paths] == [
        "/World/ground/terrain/part_000",
        "/World/ground/terrain/part_001",
    ]
    assert sum(face_count for _, face_count, _ in imported_meshes) == 3


def test_under_limit_ordinary_mesh_keeps_its_existing_prim_path(monkeypatch):
    monkeypatch.setattr(
        importer_module, "MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES", 3
    )
    parent_prims = []
    imported_meshes = []
    monkeypatch.setattr(
        importer_module.sim_utils,
        "create_prim",
        lambda path, prim_type: parent_prims.append((path, prim_type)),
    )
    monkeypatch.setattr(
        importer_module,
        "create_prim_from_mesh",
        lambda path, mesh, **kwargs: imported_meshes.append((path, len(mesh.faces))),
    )

    importer = _empty_importer()
    importer.import_mesh("terrain", _mesh_with_disjoint_triangles(3))

    assert parent_prims == []
    assert imported_meshes == [("/World/ground/terrain", 3)]
    assert importer.terrain_prim_paths == ["/World/ground/terrain"]


def test_chunking_preserves_triangle_coordinates_winding_and_material(monkeypatch):
    monkeypatch.setattr(importer_module, "MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES", 2)
    imported_meshes = []
    monkeypatch.setattr(
        importer_module,
        "create_prim_from_mesh",
        lambda path, mesh, **kwargs: imported_meshes.append((mesh, kwargs)),
    )
    importer = _empty_importer()
    mesh = _mesh_with_disjoint_triangles(5)
    original_triangles = mesh.triangles.copy()

    importer.import_mesh("terrain", mesh)

    np.testing.assert_array_equal(
        np.concatenate([part.triangles for part, _ in imported_meshes]),
        original_triangles,
    )
    assert all(len(part.faces) <= 2 for part, _ in imported_meshes)
    assert all(
        material["physics_material"] is importer.cfg.physics_material
        for _, material in imported_meshes
    )
    np.testing.assert_array_equal(mesh.triangles, original_triangles)


def test_tagged_mesh_splits_only_ordinary_faces_and_keeps_friction_collider(monkeypatch):
    monkeypatch.setattr(
        importer_module, "MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES", 2
    )
    parent_prims = []
    imported_meshes = []
    monkeypatch.setattr(
        importer_module.sim_utils,
        "create_prim",
        lambda path, prim_type: parent_prims.append((path, prim_type)),
    )
    monkeypatch.setattr(
        importer_module,
        "create_prim_from_mesh",
        lambda path, mesh, **kwargs: imported_meshes.append(
            (path, len(mesh.faces), kwargs["physics_material"])
        ),
    )

    importer = _empty_importer()
    importer.cfg.terrain_generator = SimpleNamespace(
        sub_terrains={
            "friction": SimpleNamespace(
                family="friction_variation",
                friction_range=(0.7, 1.0),
                friction_region_count=4,
            )
        }
    )
    friction_material = object()
    importer._make_region_material = lambda _coefficient: friction_material
    mesh = _mesh_with_disjoint_triangles(4)
    face_colours = np.asarray(mesh.visual.face_colors, dtype=np.uint8).copy()
    quantized = 700_000
    face_colours[-1] = (
        (quantized >> 16) & 0xFF,
        (quantized >> 8) & 0xFF,
        quantized & 0xFF,
        importer_module.FRICTION_REGION_MARKER_ALPHA,
    )
    mesh.visual.face_colors = face_colours

    importer.import_mesh("terrain", mesh)

    assert parent_prims == [
        ("/World/ground/terrain", "Xform"),
        ("/World/ground/terrain/ordinary", "Xform"),
    ]
    assert [item[:2] for item in imported_meshes] == [
        ("/World/ground/terrain/ordinary/part_000", 2),
        ("/World/ground/terrain/ordinary/part_001", 1),
        ("/World/ground/terrain/friction_00700000", 1),
    ]
    assert imported_meshes[-1][2] is friction_material
    assert sum(face_count for _, face_count, _ in imported_meshes) == 4


def _two_tiles():
    first = trimesh.creation.box(extents=(12.0, 4.0, 1.0))
    second = first.copy()
    second.apply_translation((12.0, 0.0, 0.0))
    mesh = trimesh.util.concatenate((first, second))
    mesh.metadata[importer_module.SOURCE_TILE_FACE_RANGES] = (
        ("tile_r0000_c00", 0, 12), ("tile_r0001_c00", 12, 24)
    )
    return mesh


def test_tile_partition_keeps_closed_neighbors_even_below_count_budget(monkeypatch):
    parts = []
    monkeypatch.setattr(importer_module, "create_prim_from_mesh", lambda path, mesh, **kw: parts.append((path, mesh)))
    importer = _empty_importer()
    mesh = _two_tiles()
    original = mesh.triangles.copy()
    importer.import_mesh("terrain", mesh)
    assert [path for path, _ in parts] == [
        "/World/ground/terrain/tile_r0000_c00", "/World/ground/terrain/tile_r0001_c00"
    ]
    assert all(part.is_watertight and part.is_winding_consistent for _, part in parts)
    assert all(np.allclose(part.extents, (12.0, 4.0, 1.0)) for _, part in parts)
    np.testing.assert_array_equal(np.concatenate([part.triangles for _, part in parts]), original)
    np.testing.assert_array_equal(mesh.triangles, original)


def test_tile_material_partition_preserves_friction_and_face_coverage(monkeypatch):
    parts = []
    monkeypatch.setattr(importer_module, "create_prim_from_mesh", lambda path, mesh, **kw: parts.append((path, mesh, kw)))
    importer = _empty_importer()
    importer._configured_friction_candidates = lambda: np.array([0.7])
    friction_material = object()
    importer._make_region_material = lambda _value: friction_material
    mesh = _two_tiles()
    colour = [700000 >> 16, (700000 >> 8) & 255, 700000 & 255, 254]
    colours = np.tile([150, 150, 150, 255], (24, 1)).astype(np.uint8)
    colours[12:] = colour
    mesh.visual.face_colors = colours
    importer.import_mesh("terrain", mesh)
    assert len(parts) == 2
    assert parts[0][0].endswith("/ordinary/tile_r0000_c00")
    assert parts[1][0].endswith("/friction_00700000/tile_r0001_c00")
    assert parts[1][2]["physics_material"] is friction_material
    np.testing.assert_array_equal(np.concatenate([part.triangles for _, part, _ in parts]), mesh.triangles)
    assert all((part.visual.face_colors[:, 3] == 255).all() for _, part, _ in parts)


def test_large_tile_fallback_does_not_mix_neighboring_tiles(monkeypatch):
    monkeypatch.setattr(importer_module, "MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES", 5)
    parts, parents = [], []
    monkeypatch.setattr(importer_module.sim_utils, "create_prim", lambda path, kind: parents.append(path))
    monkeypatch.setattr(importer_module, "create_prim_from_mesh", lambda path, mesh, **kw: parts.append((path, mesh)))
    importer = _empty_importer()
    mesh = _two_tiles()
    importer.import_mesh("terrain", mesh)
    assert len(parts) == 6
    assert len(parents) == len(set(parents)) == 3
    assert all(len(part.faces) <= 5 for _, part in parts)
    assert all(f"tile_r{index // 3:04d}_c00/part_{index % 3:03d}" in path
               for index, (path, _) in enumerate(parts))
    np.testing.assert_array_equal(np.concatenate([part.triangles for _, part in parts]), mesh.triangles)


def test_missing_or_invalid_tile_ranges_fail_before_creating_prims(monkeypatch):
    created = []
    monkeypatch.setattr(importer_module.sim_utils, "create_prim", lambda *args: created.append(args))
    importer = _empty_importer()
    importer.cfg.parallel_reset_sampler_enabled = True
    with pytest.raises(ValueError, match="requires generator tile"):
        importer.import_mesh("terrain", _mesh_with_disjoint_triangles(3))
    mesh = _two_tiles()
    mesh.metadata[importer_module.SOURCE_TILE_FACE_RANGES] = [("tile_r0000_c00", 0, 23)]
    with pytest.raises(ValueError, match="do not cover"):
        importer.import_mesh("terrain", mesh)
    assert created == []
