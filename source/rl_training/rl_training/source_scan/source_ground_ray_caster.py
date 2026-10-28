"""Source's static ground ray caster with the documented (N, B, 1) hit IDs.

The installed MultiMeshRayCaster allocates rank-3 IDs but its Warp helper
returns rank-2 IDs. Keep the public cache contract without modifying Isaac Lab
or its module globals. This adapter only supports one shared, static ground
mesh; other ray-caster configurations retain the upstream implementation.
"""

from collections.abc import Sequence

import torch

from isaaclab.sensors.ray_caster import MultiMeshRayCaster
from isaaclab.utils.warp import raycast_dynamic_meshes


class SourceGroundRayCaster(MultiMeshRayCaster):
    """Use upstream mesh/ray initialization and normalize hit IDs at the write."""

    def _initialize_warp_meshes(self):
        targets = self._raycast_targets_cfg
        if (
            not self.cfg.update_mesh_ids
            or len(targets) != 1
            or targets[0].prim_expr != "/World/ground"
            or not targets[0].merge_prim_meshes
            or targets[0].track_mesh_transforms
        ):
            raise ValueError("SourceGroundRayCaster requires hit IDs and one merged static /World/ground target")
        super()._initialize_warp_meshes()
        ids = self._mesh_ids_wp.numpy()
        if (
            self._num_meshes_per_env != {"/World/ground": 1}
            or ids.shape != (self._num_envs, 1)
            or not (ids == self.meshes["/World/ground"].id).all()
            or any(view is not None for view in self._mesh_views)
        ):
            raise RuntimeError("Source ground mesh must be static and identical for every environment")

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        self._update_ray_infos(env_ids)
        hits, _, _, _, mesh_ids = raycast_dynamic_meshes(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            # Every row holds the same single ground ID (checked at init).
            # Thus local row indices also work for reordered/partial batches.
            mesh_ids_wp=self._mesh_ids_wp,
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_mesh_id=True,
        )
        expected = self._ray_starts_w[env_ids].shape
        if not isinstance(hits, torch.Tensor) or hits.shape != expected:
            raise RuntimeError("Source ray hits do not match the selected environment/ray dimensions")
        if not isinstance(mesh_ids, torch.Tensor) or mesh_ids.dtype != torch.int16:
            raise RuntimeError("Source ray mesh IDs must be an int16 tensor")
        if mesh_ids.shape == expected[:2]:
            mesh_ids = mesh_ids.unsqueeze(-1)
        if mesh_ids.shape != (*expected[:2], 1):
            raise RuntimeError("Source ray mesh IDs do not match the selected environment/ray dimensions")
        if self._data.ray_mesh_ids.shape != (self._num_envs, self.num_rays, 1):
            raise RuntimeError("Source ray mesh ID cache must retain its (N, B, 1) contract")
        # Normalize before either cache write, so shape errors cannot leave a
        # new hit-position cache paired with stale hit IDs. Preserve -1 misses.
        self._data.ray_hits_w[env_ids] = hits
        self._data.ray_mesh_ids[env_ids] = mesh_ids
