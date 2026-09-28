import math

import numpy as np
import pytest

from rl_training.online_adaptation.spawn_geometry import collision_bounds, certify_spawn_range


def shape(kind="Sphere", center=(0, 0, 1), **kwargs):
    matrix = np.eye(4)
    matrix[:3, 3] = center
    return {"path":"test", "type":kind, "radius":.1, "world_from_shape":matrix, **kwargs}


@pytest.mark.parametrize("kind", ["Cube", "Sphere", "Capsule", "Cylinder"])
def test_support_bounds_contain_rotated_scaled_surface_samples(kind):
    rng = np.random.default_rng(92)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    item = shape(kind, size=2., height=1.4, axis="Z")
    item["world_from_shape"][:3, :3] = q @ np.diag([.4,.7,1.3])
    points = rng.normal(size=(3000,3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    if kind == "Cube":
        points = np.sign(points)
    elif kind == "Sphere":
        points *= .1
    elif kind == "Capsule":
        points *= .1
        points[:,2] += np.sign(points[:,2]) * .7
    else:
        points[:,:2] /= np.linalg.norm(points[:,:2],axis=1,keepdims=True)
        points[:,:2] *= .1
        points[:,2] = np.sign(points[:,2]) * .7
    matrix = item["world_from_shape"]
    world = points @ matrix[:3,:3].T + matrix[:3,3]
    lower,upper = collision_bounds(item)
    assert (world >= lower - 1e-12).all() and (world <= upper + 1e-12).all()
    assert np.max(np.abs(world.min(axis=0)-lower)) < .025
    assert np.max(np.abs(world.max(axis=0)-upper)) < .025


def test_disk_covers_continuous_yaw_and_translation_and_rejects_edge_or_penetration():
    body = shape("Cube",center=(.2,0,.3),size=.1)
    report = certify_spawn_range([body], root_position=[0,0,.4], platform_bounds=[[-1,1],[-1,1]])
    assert report["valid"]
    for angle in np.linspace(-math.pi,math.pi,1001):
        x,y = .25*math.cos(angle)-.05*math.sin(angle), .25*math.sin(angle)+.05*math.cos(angle)
        assert max(abs(x)+.02,abs(y)+.02) <= report["horizontal_enclosing_radius_m"]+.02+1e-12
    assert not certify_spawn_range([body], root_position=[0,0,.4], platform_bounds=[[-.1,.1],[-1,1]])["valid"]
    body["world_from_shape"][2,3] = .01
    assert not certify_spawn_range([body], root_position=[0,0,.4], platform_bounds=[[-1,1],[-1,1]])["valid"]


@pytest.mark.parametrize("items", [[], [shape("Mesh")], [shape(radius=float("nan"))]])
def test_incomplete_geometry_cannot_be_certified(items):
    with pytest.raises(ValueError):
        certify_spawn_range(items, root_position=[0,0,1], platform_bounds=[[-1,1],[-1,1]])


def test_certificate_reproduces_proof_and_rejects_changed_asset(tmp_path):
    import json
    from rl_training.online_adaptation.spawn_geometry import validate_spawn_certificate
    from rl_training.online_adaptation.provenance import sha256_file

    asset = tmp_path / "asset.usd"
    asset.write_bytes(b"fixture model")
    item = shape()
    item["world_from_shape"] = item["world_from_shape"].tolist()
    inputs = {"root_position":[0,0,1], "platform_bounds":[[-1,1],[-1,1]]}
    certificate = {"schema":"lite3_spawn_certificate_v1", "status":"passed", "certificate_path":"certificate.json",
                   "file_sha256":{"asset.usd":sha256_file(asset)}, "nominal_shapes":[item], "proof_inputs":inputs,
                   "nominal_proof":certify_spawn_range([item], **inputs)}
    (tmp_path/"certificate.json").write_text(json.dumps(certificate))
    assert validate_spawn_certificate(tmp_path,certificate) == sha256_file(tmp_path/"certificate.json")
    changed = dict(certificate, nominal_proof={**certificate["nominal_proof"], "valid":False})
    with pytest.raises(ValueError,match="proof"):
        validate_spawn_certificate(tmp_path,changed)
    asset.write_bytes(b"changed geometry")
    with pytest.raises(ValueError,match="stale"):
        validate_spawn_certificate(tmp_path,certificate)


def test_committed_lite3_certificate_matches_deployed_inputs():
    import json
    from pathlib import Path
    from rl_training.online_adaptation.spawn_geometry import validate_spawn_certificate

    root = Path(__file__).resolve().parents[2]
    certificate = json.loads((root/"docs/lite3_spawn_geometry_certificate_v1.json").read_text())
    assert len(validate_spawn_certificate(root,certificate)) == 64
    assert certificate["nominal_proof"]["collision_count"] == 22
    assert len(certificate["corner_checks"]) == 9
