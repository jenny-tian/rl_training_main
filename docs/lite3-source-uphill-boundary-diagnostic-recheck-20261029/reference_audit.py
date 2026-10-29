"""Static USD/API inspection only: no application or physics initialization."""

import ast
import hashlib
import json
from pathlib import Path

from pxr import Usd, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    dependency = Path('/home/tjx/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py')
    mapping = Path('/home/tjx/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/__editable___isaaclab_0_54_3_finder.py')
    source = dependency.read_text()
    names = {'root_pos_w', 'root_lin_vel_w', 'root_lin_vel_b', 'body_pos_w',
             'body_lin_vel_w', 'root_link_vel_w', 'body_link_vel_w'}
    methods = {node.name: ast.get_source_segment(source, node) for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.FunctionDef) and node.name in names}
    assert len(methods) == len(names)
    assert 'return self.root_link_pos_w' in methods['root_pos_w']
    assert 'return self.root_com_lin_vel_w' in methods['root_lin_vel_w']
    assert 'return self.body_link_pos_w' in methods['body_pos_w']
    assert 'return self.body_com_lin_vel_w' in methods['body_lin_vel_w']
    assets = [ROOT / 'deep_robotics_model/Lite3/Lite3_usd/Lite3.usd',
              ROOT / 'deep_robotics_model/Lite3/Lite3_usd/configuration/Lite3_physics.usd',
              ROOT / 'deep_robotics_model/Lite3/Lite3_usd/configuration/Lite3_base.usd']
    stage = Usd.Stage.Open(str(assets[0]))
    bodies = {}
    for prim in stage.Traverse():
        if prim.GetName() in ('TORSO', 'FL_THIGH') and prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            bodies[prim.GetName()] = {'prim_path': str(prim.GetPath()),
                                     'center_of_mass_in_link_m': list(api.GetCenterOfMassAttr().Get()),
                                     'mass_kg': api.GetMassAttr().Get()}
    assert bodies['TORSO']['center_of_mass_in_link_m'] == [0., 0., 0.]
    assert any(v != 0 for v in bodies['FL_THIGH']['center_of_mass_in_link_m'])
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for asset in assets:
        assert sha(asset) == manifest['input_sha256'][str(asset.relative_to(ROOT))]
    cfg = ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py'
    assert 'self.events.randomize_com_positions = None' in cfg.read_text()
    assert sha(cfg) == manifest['input_sha256'][str(cfg.relative_to(ROOT))]
    return {
        'schema': 'lite3_link_com_reference_point_static_audit_v1',
        'simulation_started': False,
        'source_dependency': {'path': str(dependency), 'sha256': sha(dependency),
                              'included_in_original_run_hash_inventory': str(dependency) in manifest['input_sha256']},
        'editable_package_mapping': {'path': str(mapping), 'sha256': sha(mapping),
                                     'isaaclab_root': '/home/tjx/IsaacLab/source/isaaclab/isaaclab'},
        'api_method_source': methods,
        'usd_sha256': {str(p.relative_to(ROOT)): sha(p) for p in assets},
        'physics_variant': stage.GetDefaultPrim().GetVariantSet('Physics').GetVariantSelection(),
        'static_body_properties': bodies,
        'source_com_randomization_disabled': True,
        'guard_input_definition': 'body_pos_w is the link-frame origin; body_lin_vel_w is the CoM linear velocity.',
        'point_consistent_formula': 'v_link_w = v_com_w - omega_w cross (R_link_w * r_com_in_link)',
        'conclusion': 'The recorded predicate replay is complete, but its position/velocity reference points are mixed. Correct the inputs offline before another GPU proposal.',
        'limitations': [
            'ArticulationData and editable-finder hashes were first captured after this run; do not retroactively add them to its frozen manifest.',
            'USD values and disabled COM randomization are static evidence; actual runtime COM offsets were not read back.',
            'Full body angular velocities and orientations were not captured, so the point-consistent final thigh prediction cannot be exactly reconstructed from this run.',
            'TORSO has zero authored COM offset. The nonzero thigh offset does not itself explain the torso lateral integration residual.',
            'No false-positive or true physical-edge-crossing claim follows from the recorded mixed-point prediction.',
        ],
    }


if __name__ == '__main__':
    with (OUT / 'reference_point_audit.json').open('x') as stream:
        json.dump(audit(), stream, indent=2, allow_nan=False)
        stream.write('\n')
