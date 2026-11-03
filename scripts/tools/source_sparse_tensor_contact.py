"""Native PhysX tensor adapter with compact evidence; no simulator import."""
from pathlib import Path
import hashlib
import os

import numpy as np

from rl_training.source_scan.sparse_contact import ContactEncoder, compact_topology, owner
from source_shared_contact_report import ContactCapture
from source_shared_tensor_contact import _copy, tensor_runtime_passed

CAPACITIES = {16:131072,32:262144,60:524288,120:1048576}


class SparseTensorContactCapture(ContactCapture):
    uses_contact_events = False

    def __init__(self, *, physics_view, ground_colliders, num_envs, physics_dt, fault_path,
                 require_ground_positive=True, diagnostic_scope='G1_startup', **kwargs):
        super().__init__(decode_path=None,event_types={},**kwargs)
        if num_envs not in CAPACITIES or physics_dt != .005:
            raise ValueError('sparse capture supports the registered 16/32/60/120 scale ladder at 200Hz')
        self.physics_view,self.physics_dt = physics_view,physics_dt
        self.ground_colliders,self.num_envs = sorted(ground_colliders),num_envs
        self.capacity = CAPACITIES[num_envs]
        self.fault_path = Path(fault_path)
        self.require_ground_positive = require_ground_positive
        if diagnostic_scope not in ('G1_startup','G2_reset_recovery'):
            raise ValueError('unregistered sparse diagnostic scope')
        if diagnostic_scope=='G2_reset_recovery' and (num_envs!=120 or self.control_steps!=4
                or self.decimation!=4 or require_ground_positive is not False):
            raise ValueError('G2 sparse capture requires exactly 120 environments and four recovery controls')
        self.diagnostic_scope = diagnostic_scope
        self.view = self.encoder = self.topology = None
        self.samples = self.peak_used = 0
        self.fault_written = False

    def _topology(self):
        topology = compact_topology(list(self.view.sensor_paths),self.view.filter_paths,self.ground_colliders,
            num_envs=self.num_envs,capacity=self.capacity)
        if (self.view.sensor_count!=topology['sensor_count'] or self.view.filter_count!=topology['filter_count']
                or self.view.max_contact_data_count!=self.capacity):
            raise ValueError('native compact contact dimensions changed')
        return topology

    def runtime(self, interface, get_runtime):
        super().runtime(interface,get_runtime)
        if not tensor_runtime_passed(self.runtime_state,self.stage_id):
            raise ValueError('native sparse tensor backend changed or is not cuda')
        if self.view is None:
            bodies = sorted(self.bodies);targets = bodies+self.ground_colliders
            self.view = self.physics_view.create_rigid_contact_view(bodies,
                filter_patterns=[list(targets) for _ in bodies],max_contact_data_count=self.capacity)
            self.topology = self._topology();self.encoder = ContactEncoder(self.topology)
            self.record({'phase':'sparse_contact_topology',**self.topology})
        elif (self.view.sensor_count!=self.topology['sensor_count'] or self.view.filter_count!=self.topology['filter_count']
                or self.view.max_contact_data_count!=self.capacity):
            raise ValueError('native sparse contact dimensions/capacity changed')

    def poll(self, env, interface, get_runtime):
        step = env._sim_step_counter
        self.record({'phase':'contact_poll_attempt','sim_step':step})
        self.runtime(interface,get_runtime)
        if type(step) is not int or step!=self.start_step+len(self.poll_steps)+1 or env.physics_dt!=self.physics_dt:
            raise ValueError('nonconsecutive sparse contact poll or changed physics timestep')
        net=matrix=data=None
        try:
            net = _copy(self.view.get_net_contact_forces(self.physics_dt))
            matrix = _copy(self.view.get_contact_force_matrix(self.physics_dt))
            data = tuple(_copy(v) for v in self.view.get_contact_data(self.physics_dt))
            frame = self.encoder.encode(net,matrix,data,step=step)
            self.record({'phase':'sparse_contact_sample',**frame})
            # Count ground positive controls using detailed force magnitudes.
            detail = dict(zip(frame['detail_indices'],frame['normal_forces_N']))
            t = self.topology
            for (i,j),count,start in zip(frame['pair_indices'],frame['pair_counts'],frame['pair_starts']):
                target=t['target_paths'][t['filter_orders'][t['sensor_filter_order'][i]][j]]
                if target in self.ground_colliders:
                    self.ground['poll'][owner(t['sensor_paths'][i])] += sum(abs(detail[k])>1e-4 for k in range(start,start+count))
            self.peak_used=max(self.peak_used,frame['used_contact_count'])
            self.samples += 1;self.poll_steps.append(step)
        except Exception as error:
            if not self.fault_written:
                # Keep all native buffers, including undefined unused slots, on failure only.
                snapshot={}
                if net is not None:snapshot['net']=net
                if matrix is not None:snapshot['matrix']=matrix
                if data is not None:snapshot.update(dict(zip(('forces','points','normals','distances','counts','starts'),data)))
                with self.fault_path.open('xb') as stream:
                    np.savez_compressed(stream,**snapshot)
                    stream.flush();os.fsync(stream.fileno())
                self.fault_written=True
                self.record({'phase':'sparse_contact_fault','sim_step':step,'error':str(error),
                    'file':self.fault_path.name,'available_buffers':list(snapshot),
                    'all_native_buffers_read':len(snapshot)==8,
                    'sha256':hashlib.sha256(self.fault_path.read_bytes()).hexdigest()})
            raise

    def check_control(self, env, completed):
        super().check_control(env,completed)
        # Paths are static within each control interval; re-resolve after resets.
        if self._topology()!=self.topology: raise ValueError('resolved sparse contact topology changed')

    def finish(self):
        if self.require_ground_positive:
            super().finish()
        elif self.diagnostic_scope=='G2_reset_recovery':
            if (self.control_steps!=4 or self.samples!=16 or self.control_poll_counts!=[4]*4
                    or self.errors or self.cross_contacts):
                raise ValueError('reset-recovery sparse capture incomplete')
        elif (self.control_steps!=1 or self.samples!=4 or len(self.control_poll_counts)!=1
              or self.errors or self.cross_contacts):
            raise ValueError('startup-only sparse capture incomplete')

    def summary(self):
        result=super().summary()
        result.update(schema='lite3_sparse_tensor_capture_v1',topology=self.topology,samples=self.samples,
            peak_used=self.peak_used,require_ground_positive=self.require_ground_positive,
            ground_positive_qualified=self.positive_controls_validated,fault_written=self.fault_written,
            physics_dt=self.physics_dt)
        if self.diagnostic_scope=='G2_reset_recovery':result['diagnostic_scope']=self.diagnostic_scope
        return result
