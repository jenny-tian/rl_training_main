"""Additional B1.1 runtime probes, called by the shared block checker."""
import torch


class ReferenceRoughProbe:
    def __init__(self, env, checks, report):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from isaaclab.utils.warp import raycast_dynamic_meshes

        self.env, self.checks, self.report = env, checks, report
        terrain, robot = env.scene.terrain, env.scene["robot"]
        self.mask = (terrain.terrain_types == 6) | (terrain.terrain_types == 15)
        self.foot_ids, names = robot.find_bodies(".*_FOOT")
        self.contact_ids = env.scene["contact_forces"].find_bodies(".*_FOOT")[0]
        # Read the effective USD collision spheres; never assume the visual
        # mesh or a guessed foot radius describes the collider.
        root = env.scene.stage.GetPrimAtPath(env.scene.env_prim_paths[0] + "/Robot")
        spheres = {}
        cache = UsdGeom.XformCache()
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            body = prim
            while body and body.GetName() not in names and body != root:
                body = body.GetParent()
            if not body or body.GetName() not in names:
                continue
            if prim.GetTypeName() != "Sphere":
                raise ValueError(f"unsupported foot collider for clearance proof: {prim.GetPath()}")
            transform, _ = cache.ComputeRelativeTransform(prim, body)
            offset = transform.Transform(Gf.Vec3d(0, 0, 0))
            if offset.GetLength() > 1e-6:
                raise ValueError("foot sphere offset needs an explicit transformed-centre probe")
            scales = [transform.TransformDir(Gf.Vec3d(*axis)).GetLength()
                      for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
            radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * max(scales)
            spheres[body.GetName()] = dict(path=str(prim.GetPath()), radius_m=radius)
        if set(spheres) != set(names):
            raise ValueError(f"foot collision discovery incomplete: {spheres}")
        self.radii = torch.tensor([spheres[name]["radius_m"] for name in names], device=env.device)
        self.report["foot_colliders"] = spheres
        self.min_gap = float("inf")
        self.max_gap = -float("inf")
        self.max_foot_offset = 0.
        self.min_origin_clearance = float("inf")
        self.max_origin_clearance = -float("inf")
        self.contact_seen = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        self.forces_finite = True
        self.reset_count = 0

        sensor = env.scene["height_scanner_base"]
        coordinates = torch.tensor([-3.9, -3.7, -3., -2., -1., 0., 1., 2., 3., 3.7, 3.9], device=env.device)
        xy = torch.cartesian_prod(coordinates, coordinates)
        starts = terrain.env_origins[:, None, :].expand(-1, len(xy), -1).clone()
        starts[:, :, :2] += xy
        starts[:, :, 2] = 2.
        directions = torch.zeros_like(starts)
        directions[:, :, 2] = -1.
        hits, *_ = raycast_dynamic_meshes(starts, directions, sensor._mesh_ids_wp,
            mesh_positions_w=sensor._mesh_positions_w, mesh_orientations_w=sensor._mesh_orientations_w)
        z = hits[self.mask, :, 2]
        self.report["rough_imported_ray_probe"] = dict(rays=int(z.numel()),
            height_range_m=[float(z.min()), float(z.max())],
            min_tile_peak_valley_m=float((z.amax(-1) - z.amin(-1)).min()))
        checks.add("reference_rough_imported_surface", bool(torch.isfinite(z).all())
            and bool((z >= -2e-4).all()) and bool((z <= .0602).all())
            and bool(((z.amax(-1) - z.amin(-1)) >= .04).all()), str(self.report["rough_imported_ray_probe"]))
        checks.add("reference_rough_origins", bool((terrain.terrain_origins[:, (6, 15), 2] - .06).abs().max() < 1e-6),
                   "all 20 rough origins are central maximum +0.06m")

    def observe_reset(self):
        robot, terrain = self.env.scene["robot"], self.env.scene.terrain
        feet = robot.data.body_link_pos_w[self.mask][:, self.foot_ids]
        origin = terrain.env_origins[self.mask]
        # The entire terrain is <= origin z. A positive gap above that
        # ceiling proves the complete foot sphere cannot intersect it.
        gap = feet[:, :, 2] - self.radii - origin[:, None, 2]
        self.min_gap = min(self.min_gap, float(gap.min()))
        self.max_gap = max(self.max_gap, float(gap.max()))
        self.max_foot_offset = max(self.max_foot_offset, float((feet[:, :, :2] - origin[:, None, :2]).abs().max()))
        clearance = robot.data.root_link_pos_w[self.mask, 2] - origin[:, 2]
        self.min_origin_clearance = min(self.min_origin_clearance, float(clearance.min()))
        self.max_origin_clearance = max(self.max_origin_clearance, float(clearance.max()))
        self.reset_count += 1

    def finish_resets(self):
        data = dict(resets=self.reset_count, conservative_foot_bottom_gap_m=[self.min_gap, self.max_gap],
                    max_foot_xy_offset_m=self.max_foot_offset,
                    root_above_support_origin_m=[self.min_origin_clearance, self.max_origin_clearance])
        self.report["rough_reset_probe"] = data
        self.checks.add("rough_reset_no_foot_overlap", self.min_gap >= -2e-4 and self.max_foot_offset < .9
                        and abs(self.min_origin_clearance - .375) < 2e-4
                        and abs(self.max_origin_clearance - .375) < 2e-4, str(data))

    def observe_step(self):
        forces = self.env.scene["contact_forces"].data.net_forces_w[:, self.contact_ids].norm(dim=-1)
        self.forces_finite &= bool(torch.isfinite(forces).all())
        self.contact_seen |= forces > 1.

    def finish_rollout(self):
        contact = self.contact_seen[self.mask]
        data = dict(rough_environments=int(self.mask.sum()), feet_with_contact=int(contact.sum()),
                    expected_feet=int(contact.numel()), all_forces_finite=self.forces_finite)
        self.report["rough_collision_probe"] = data
        self.checks.add("rough_physical_foot_contacts", self.forces_finite and bool(contact.all()), str(data))
        snapshot = self.env.source_block_state.stats.snapshot()
        self.checks.add("rough_statistics_label", snapshot["physical_difficulty"] == "L2_except_reference_uniform_rough"
            and snapshot["physical_difficulty_by_column"][6] == "reference_rough"
            and snapshot["physical_difficulty_by_column"][15] == "reference_rough"
            and "blocks/reference_uniform_rough_a/reference_rough_transitions" in snapshot["metrics"],
            "reference rough interactions are not labelled as original L2 or multi-scale rough")
