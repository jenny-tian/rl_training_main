"""Candidate native-reset API repair; no simulation has validated this version."""
import numpy as np
import torch

from reference_online_c1b import protocol
from reference_online_c1b_env_repaired import Environment as DelayRepairedEnvironment


class Environment(DelayRepairedEnvironment):
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
        # Native _reset_idx resets managers and episode lengths. The three
        # reset_* outputs are created by step(), not reset(); leave them owned
        # by the termination manager. The collector copies terminal flags
        # before any explicit reset, and compute() recomputes them next step.
        # Root/joint state writes above go directly to PhysX. Retain the
        # registered forward/update sequence; an extra scene-wide write would
        # advance actuator delay buffers for untouched environments too.
        self.base.sim.forward()
        self.base.scene.update(self.base.step_dt)
        self.policy_age[ids] = 0; self.policy_active[ids] = False
        self.refresh()
        return dict(ids=ids, keys=keys, offsets=np.asarray(offsets).tolist(), delays=delays)
