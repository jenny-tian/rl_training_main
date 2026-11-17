"""Candidate indexed-delay repair; original failed source remains hash-locked."""
from __future__ import annotations

import copy
import json
import sys

import numpy as np
import torch

from reference_lite3_loader import ROOT, load_reference, namespace, resolve_lite3_asset, source_manifest
from reference_stairs_eval import MODELS, load_actor, save
from reference_online_targets import MODEL_PATHS, HASHES, read
from reference_online_c1b import OUTPUT, protocol


class Environment:
    def __init__(self, args, specs):
        from isaaclab.envs import ManagerBasedRLEnv
        reference, _ = load_reference()
        from reference_six_terrain_cfg import replace_terrain, column_names
        from reference_low_stairs_cfg import apply_terrain
        cfg = replace_terrain(reference)
        training = read(MODEL_PATHS['B'].parent / 'configuration.json')['environment']
        for key in ('actions','observations','terminations'):
            assert json.loads(json.dumps(cfg.to_dict()[key], default=str)) == training[key], key
        package = ROOT / 'source/rl_training/rl_training'
        sys.modules['rl_training'].__path__.append(str(package))
        namespace('rl_training.online_adaptation', package / 'online_adaptation')
        namespace('rl_training.ea_lpl', package / 'ea_lpl')
        from reference_online_interface import CurrentIsaacAdapter, wrap_manager_compute
        self.args, self.specs = args, specs
        self.n = len(specs)
        self.policy_age = np.zeros(self.n, int)
        self.policy_active = np.zeros(self.n, bool)
        self.deferred_events = 0
        if args.kind == 'retention':
            cfg, _ = apply_terrain(cfg)
            gen = cfg.scene.terrain.terrain_generator
            gen.seed = 42
            for name in ('stairs_up','stairs_down'):
                gen.sub_terrains[name].step_height_range = (.05,.05)
            names = column_names(cfg)
            assert all(names[s['column']] == s['condition'] for s in specs)
        else:
            gen = cfg.scene.terrain.terrain_generator
            sub = copy.deepcopy(gen.sub_terrains['stairs_up'])
            sub.step_height_range = (.06,.06); sub.proportion = 1.
            gen.num_rows = gen.num_cols = 1
            gen.sub_terrains = {'stairs_06': sub}
            gen.seed, gen.difficulty_range = 20261116, (.5,.5)
        gen.curriculum, gen.use_cache, gen.size = True, False, (8.,8.)
        cfg.seed = 20261117
        cfg.scene.num_envs = self.n
        cfg.sim.device = 'cuda:0'
        cfg.episode_length_s = 20.
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.curriculum.terrain_levels = cfg.curriculum.gait_level = None
        cfg.commands.base_velocity.debug_vis = False
        cfg.observations.policy.enable_corruption = False
        for name in ('randomize_rigid_body_mass','randomize_rigid_body_inertia','randomize_com_positions','randomize_actuator_gains'):
            setattr(cfg.events, name, None)
        cfg.events.randomize_rigid_body_material.params.update(static_friction_range=(.9,.9),
            dynamic_friction_range=(.9,.9), restitution_range=(0.,0.), make_consistent=True)
        cfg.events.randomize_reset_base.params.update(
            pose_range={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')},
            velocity_range={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')})
        resolve_lite3_asset(cfg)
        save(args.output / 'config.json', dict(configuration=json.loads(json.dumps(cfg.to_dict(),default=str)),
            specs=specs, protocol=protocol(), sources=source_manifest(), checkpoint_sha256=HASHES['B']))
        self.base = ManagerBasedRLEnv(cfg=cfg)
        self.obs, _ = self.base.reset()
        self.device = self.base.device
        self.robot, self.terrain = self.base.scene['robot'], self.base.scene.terrain
        self.sensor = self.base.scene['contact_forces']
        levels = torch.tensor([s['row'] for s in specs], device=self.device)
        columns = torch.tensor([s['column'] if args.kind == 'retention' else 0 for s in specs], device=self.device)
        self.terrain.terrain_levels.copy_(levels); self.terrain.terrain_types.copy_(columns)
        self.terrain.env_origins.copy_(self.terrain.terrain_origins[levels,columns])
        assert cfg.scene.filter_collisions and self.obs['policy'].shape == (self.n,45)
        assert self.obs['critic'].shape == (self.n,393)
        self.adapter = CurrentIsaacAdapter(self.base)
        self.native_reset = self.base._reset_idx
        def defer(ids):
            self.deferred_events += len(ids)
        self.base._reset_idx = defer
        self.command = self.base.command_manager.get_term('base_velocity')
        wrap_manager_compute(self.base.command_manager, lambda dt: self.set_command(next_step=True))
        self.foot_ids = [self.robot.body_names.index(f'{leg}_FOOT') for leg in ('FL','FR','HL','HR')]
        self.contact_ids = [self.sensor.body_names.index(f'{leg}_FOOT') for leg in ('FL','FR','HL','HR')]
        self.torso_id = self.sensor.body_names.index('TORSO')
        MODELS['B'] = MODEL_PATHS['B']
        self.actor = load_actor('B').to(self.device).eval()
        self.actor.requires_grad_(False)
        self.geometry = self.audit_geometry(cfg.scene.terrain.prim_path)
        save(args.output / 'geometry.json', self.geometry)

    def set_command(self, next_step=False):
        age = self.policy_age + int(next_step)
        speeds = .4 * np.clip((age*.02-.5)/.5, 0., 1.) * self.policy_active
        self.command.is_heading_env[:] = self.command.is_standing_env[:] = False
        self.command.vel_command_b[:] = 0
        self.command.vel_command_b[:,0] = torch.tensor(speeds, device=self.device)

    def refresh(self):
        self.set_command()
        self.obs = self.base.observation_manager.compute()

    def reset(self, ids, episodes):
        from isaaclab.utils.math import quat_from_euler_xyz
        from rl_training.ea_lpl.reset_table import initial_state_offsets
        from rl_training.ea_lpl.c1b_delay import set_indexed_time_lag
        index = torch.tensor(ids, device=self.device, dtype=torch.long)
        self.native_reset(index)
        offsets, keys = [], []
        namespace_id = protocol()['namespaces'][self.args.kind]
        for i in ids:
            spec = self.specs[i]
            stair = spec['condition'].startswith('stairs')
            ranges = dict(x=(.2,.3) if stair else (-.05,.05), y=(-.1,.1), yaw=(-.03,.03))
            offsets.append(initial_state_offsets(20261117,namespace_id,[spec['slot']],[int(episodes[i])],ranges)[0])
            keys.append([20261117,namespace_id,spec['slot'],int(episodes[i])])
        delta = torch.tensor(np.asarray(offsets), device=self.device, dtype=torch.float32)
        roots = self.robot.data.default_root_state[index].clone()
        roots[:,:3] += self.terrain.env_origins[index]
        roots[:,:2] += delta[:,:2]
        roots[:,3:7] = quat_from_euler_xyz(delta[:,2]*0, delta[:,2]*0, delta[:,2])
        roots[:,7:] = 0
        self.robot.write_root_state_to_sim(roots, env_ids=index)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[index],
            torch.zeros_like(self.robot.data.default_joint_vel[index]), env_ids=index)
        delays = []
        for actuator_index, actuator in enumerate(self.robot.actuators.values()):
            lag = torch.tensor([int(np.random.default_rng([*key,actuator_index]).integers(0,2)) for key in keys], device=self.device)
            delays.append(lag.cpu().tolist())
            for name in ('positions_delay_buffer','velocities_delay_buffer','efforts_delay_buffer'):
                set_indexed_time_lag(getattr(actuator,name), lag, index)
                getattr(actuator,name).reset(batch_ids=index)
        self.base.reset_buf[index] = False
        self.base.reset_terminated[index] = False
        self.base.reset_time_outs[index] = False
        self.base.sim.forward(); self.base.scene.update(self.base.step_dt)
        self.policy_age[ids] = 0; self.policy_active[ids] = False
        self.refresh()
        return dict(ids=ids, keys=keys, offsets=np.asarray(offsets).tolist(), delays=delays)

    def prepared_state(self, ids):
        data = self.robot.data
        return dict(root=data.root_state_w[ids].cpu().numpy().copy(),
            joint_pos=data.joint_pos[ids].cpu().numpy().copy(), joint_vel=data.joint_vel[ids].cpu().numpy().copy(),
            observation=self.obs['policy'][ids].cpu().numpy().copy())

    def frame(self):
        data, forces = self.robot.data, self.sensor.data.net_forces_w
        root = (data.root_pos_w-self.terrain.env_origins).cpu().numpy().copy()
        feet = (data.body_pos_w[:,self.foot_ids]-self.terrain.env_origins[:,None,:]).cpu().numpy().copy()
        ground = self.base.scene['height_scanner_base'].data.ray_hits_w[:,:,2].amax(-1)
        return dict(root=root, feet=feet, gravity=data.projected_gravity_b.cpu().numpy().copy(),
            yaw_rate=data.root_ang_vel_b[:,2].cpu().numpy().copy(),
            mean_joint_speed=data.joint_vel.abs().mean(-1).cpu().numpy().copy(),
            clearance=(data.root_pos_w[:,2]-ground).cpu().numpy().copy(),
            supports=(forces[:,self.contact_ids,2]>1.).sum(-1).cpu().numpy().copy(),
            torso_contact=(forces[:,self.torso_id].norm(dim=-1)>1.).cpu().numpy().copy())

    def audit_geometry(self, path):
        from isaaclab.sensors import RayCaster
        from isaaclab.utils.warp import raycast_mesh
        mesh = RayCaster.meshes[path]
        result = {}
        for i, spec in enumerate(self.specs):
            key = f'{spec["condition"]}:{spec["row"]}'
            if key in result: continue
            origin = self.terrain.env_origins[i]
            xs = torch.linspace(0.,3.8,3401,device=self.device)
            starts = origin+torch.stack((xs,torch.zeros_like(xs),torch.full_like(xs,20.)),-1)
            directions = torch.zeros_like(starts); directions[:,2] = -1
            hits = raycast_mesh(starts,directions,mesh)[0]
            assert torch.isfinite(hits).all()
            z, x = (hits[:,2]-origin[2]).cpu().numpy(), xs.cpu().numpy()
            fronts, tops = [], []
            if spec['condition'].startswith('stairs'):
                sign = -1 if spec['condition'] == 'stairs_down' else 1
                h = .05 if self.args.kind == 'retention' else .06
                edges = np.flatnonzero(np.diff(z)*sign > h*.5)+1
                fronts = ((x[edges]+x[edges-1])/2).tolist(); tops = z[edges].tolist()
                assert len(fronts) >= 5 and np.allclose(np.diff([0.]+tops),sign*h,atol=.002)
            result[key] = dict(fronts=fronts,tops=tops,goal_x=fronts[-1]+.3 if fronts else None,
                               centerline_min_z=float(z.min()),centerline_max_z=float(z.max()))
        return result
