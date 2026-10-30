"""CPU checks of the approved instance and the existing material/coverage path."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
import trimesh

from scripts.tools.source_terrain_mesh_audit import load_offline_pipeline
from scripts.reinforcement_learning.online_adaptation import run_source_qualification_scan as scan

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def inspect_material_path():
    case = json.loads((HERE / "preparation.json").read_text())["cases"][0]
    factory, _, namespace, _ = load_offline_pipeline(ROOT, Path("/home/tjx/IsaacLab"))
    cfg = factory("L1")[case["subterrain"]]
    cfg.seed = case["geometry_seed"]
    cfg.size = tuple(case["cell"]["resolved_parameters"]["geometry"]["size"])
    adapter = cfg.function.__globals__
    geometry = adapter["generate_source_terrain_geometry"](case["geometry_difficulty"], cfg)
    regions = [asdict(region) for region in geometry.friction_regions]
    assert [r["bounds_xy_m"] for r in regions] == [(float(i * 3), float((i + 1) * 3), 0., 4.) for i in range(4)]
    family_index = adapter["SOURCE_TERRAIN_FAMILIES"].index("friction_variation")
    seed = int(np.random.SeedSequence([10001, 0, family_index, int(case["geometry_difficulty"] * 2**32)]).generate_state(1)[0])
    expected = np.linspace(0.7, 1.0, 4)
    np.random.default_rng(seed).shuffle(expected)
    np.testing.assert_array_equal([r["friction_coefficient_candidate"] for r in regions], expected)
    meshes, origin = cfg.function(case["geometry_difficulty"], cfg)
    for mesh, region in zip(meshes, regions, strict=True):
        colours = mesh.visual.face_colors.astype(np.int64)
        assert np.all(colours[:, 3] == 254)
        decoded = ((colours[:, 0] << 16) | (colours[:, 1] << 8) | colours[:, 2]) / 1e6
        np.testing.assert_allclose(decoded, region["friction_coefficient_candidate"], atol=1e-12)
        assert mesh.is_watertight and mesh.area_faces.min() > 0
        assert mesh.bounds[1, 2] == 0
    assert geometry.metadata["effective_friction_confirmed"] is False
    importer = object.__new__(namespace["SourceTerrainImporter"])
    class Material(NS):
        pass
    namespace["sim_utils"].RigidBodyMaterialCfg = Material
    base = Material(static_friction=1., dynamic_friction=1., restitution=0., friction_combine_mode="multiply",
                    restitution_combine_mode="multiply", compliant_contact_stiffness=0., compliant_contact_damping=0.)
    importer.cfg = NS(prim_path="/World/ground", physics_material=base, visual_material=None,
                      parallel_reset_sampler_enabled=False, terrain_generator=NS(sub_terrains={"case": cfg}))
    importer.terrain_prim_paths = []
    captured = []
    def capture(path, mesh, **kwargs):
        material = kwargs["physics_material"]
        assert material.static_friction == material.dynamic_friction
        assert material.static_friction in [0.7, 0.8, 0.9, 1.]
        for key in ("friction_combine_mode", "restitution_combine_mode", "restitution",
                    "compliant_contact_stiffness", "compliant_contact_damping"):
            assert getattr(material, key) == getattr(base, key)
        captured.append(dict(path=path, bounds=mesh.bounds.tolist(), faces=len(mesh.faces), material=vars(material)))
    namespace["create_prim_from_mesh"] = capture
    importer.import_mesh("terrain", trimesh.util.concatenate(meshes))
    assert len(captured) == 4 and sum(p["faces"] for p in captured) == sum(len(m.faces) for m in meshes)
    np.testing.assert_allclose(importer._configured_friction_candidates(), [0.7, 0.8, 0.9, 1.])
    return dict(instance_seed=seed, regions=regions, geometry_metadata=geometry.metadata,
                origin_local=origin.tolist(), importer_material_calls=captured,
                scope="Actual geometry/face encoding/importer methods on CPU; material config and USD creation stubbed. No effective contact friction readback.")


def test_exact_geometry_marker_and_importer_materials():
    inspect_material_path()


def test_per_foot_region_coverage_and_slip():
    report = inspect_material_path()
    regions = report["regions"]
    geometry = dict(size=[12., 4.], flat_length_m=3.)
    origins = torch.tensor([[-4.5, 0., 0.]])
    sample = dict(foot_positions_w_m=[[x, 0., 0.] for x in [-5.5, -2.5, 0.5, 3.5]],
                  foot_contact=[True, True, True, False], foot_horizontal_speed_mps=[0.1, 0.35, 0.36, 0.9],
                  foot_body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"])
    observations = {}
    assert scan._update_friction_observations(observations, sample, origins, geometry, {"friction_regions": regions})
    result = scan._finalize_friction_observations(observations)
    assert result["effective_contact_friction_measured"] is False
    assert set(result["regions"]) == {"0", "1", "2"}
    for index, name in enumerate(sample["foot_body_names"][:3]):
        entry = result["regions"][str(index)]
        assert entry["contact_samples"] == 1
        assert entry["slipping_contact_samples"] == int(index == 2)
        assert set(entry["by_foot"]) == {name}
    assert scan._friction_region_for_foot([-4.5, 0., 0.], origins, geometry, regions)["region_id"] == 0
    assert scan._friction_region_for_foot([6.1, 0., 0.], origins, geometry, regions) is None
    assert not scan._update_friction_observations({}, {}, origins, geometry, {"friction_regions": regions})
    assert scan._finalize_friction_observations({}) is None


def test_family_report_does_not_claim_measured_contact_friction():
    case = json.loads((HERE / "preparation.json").read_text())["cases"][0]
    result = scan._family_metrics(case["cell"], [0.])
    assert result["actual_contact_friction"] is None
    assert result["material_parameters"]["effective_contact_friction_readback"] is False
    assert result["material_parameters"]["effective_contact_friction_status"] == "unavailable_without_contact_material_readback"
