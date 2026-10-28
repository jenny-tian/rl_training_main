"""Opt-in shared Source geometry and explicit collision/scan contracts.

Static contracts do not prove PhysX isolation. The two-environment runtime
diagnostic must pass separately before expanding this experimental layout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from .parallel_terrain_allocation import ReservedRowTerrainAllocator, TerrainAssignment


class SharedBankTerrainAllocator:
    """Keep the existing episode schedule but map robots to a bounded tile bank.

    Environment i uses geometry variant i % bank_rows and the scheduled column.
    Multiple environments may occupy exactly the same tile; this is deliberate
    and must NEVER be applied to a scene that permits collisions between them.
    Geometry diversity is bank_rows variants per class, not one per robot.
    """

    def __init__(
        self, num_envs: int, block_counts: Mapping[str, int], sampler_seed: int, *, bank_rows: int
    ) -> None:
        if type(bank_rows) is not int or bank_rows < 1:
            raise ValueError("bank_rows must be a positive integer")
        self._schedule = ReservedRowTerrainAllocator(num_envs, block_counts, sampler_seed)
        self._bank_rows = bank_rows

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self._bank_rows, len(self._schedule.column_labels)

    @property
    def sampler_seed(self) -> int:
        return self._schedule.sampler_seed

    @property
    def column_labels(self) -> tuple[str, ...]:
        return self._schedule.column_labels

    @property
    def block_size(self) -> int:
        return self._schedule.block_size

    @property
    def assignments(self) -> tuple[TerrainAssignment | None, ...]:
        return tuple(None if assignment is None else self._map(assignment)
                     for assignment in self._schedule.assignments)

    def _map(self, assignment: TerrainAssignment) -> TerrainAssignment:
        return replace(assignment, row=assignment.env_id % self._bank_rows)

    def reset(self, env_ids: Sequence[int]) -> tuple[TerrainAssignment, ...]:
        return tuple(self._map(assignment) for assignment in self._schedule.reset(env_ids))


def validate_shared_scene_cfg(scene_cfg) -> None:
    """Require Isaac Lab's authored USD groups and terrain-only ray targets."""
    if (
        getattr(scene_cfg, "replicate_physics", None) is not False
        or getattr(scene_cfg, "filter_collisions", None) is not True
        or getattr(scene_cfg, "clone_in_fabric", None) is not False
    ):
        raise RuntimeError("shared Source requires explicit USD collision filtering (no replication/Fabric clones)")
    terrain = getattr(scene_cfg, "terrain", None)
    robot = getattr(scene_cfg, "robot", None)
    if getattr(terrain, "prim_path", None) != "/World/ground" or getattr(terrain, "collision_group", None) != -1:
        raise RuntimeError("shared Source terrain must be the global /World/ground collider")
    if getattr(robot, "collision_group", None) != 0 or getattr(robot, "prim_path", None) not in (
        "{ENV_REGEX_NS}/Robot", "/World/envs/env_.*/Robot",
    ):
        raise RuntimeError("shared Source robots must belong to their own environment collision groups")
    for name in ("height_scanner", "height_scanner_base"):
        scanner = getattr(scene_cfg, name, None)
        targets = getattr(scanner, "mesh_prim_paths", ())
        if (
            len(targets) != 1
            or getattr(targets[0], "prim_expr", None) != "/World/ground"
            or getattr(targets[0], "merge_prim_meshes", None) is not True
            or getattr(targets[0], "track_mesh_transforms", None) is not False
            or getattr(scanner, "update_mesh_ids", None) is not True
        ):
            raise RuntimeError(f"shared Source {name} must report hits against the static ground only")


def audit_shared_collision_groups(scene) -> dict:
    """Read the actual USD filter graph before allocating overlapping origins.

    Matches the installed Isaac Sim Cloner.filter_collisions implementation:
    the inverted relationships permit self and ground contacts only. No Kit or
    USD import is needed here; the scene provides its already-created stage.
    """
    validate_shared_scene_cfg(scene.cfg)
    paths = list(scene.env_prim_paths)
    if paths != [f"/World/envs/env_{i}" for i in range(scene.cfg.num_envs)]:
        raise RuntimeError("shared Source environment prim paths are not canonical")
    if set(scene._global_prim_paths) != {"/World/ground"}:
        raise RuntimeError("shared Source global colliders contain missing or unexpected assets")
    stage = scene.stage
    physics = stage.GetPrimAtPath(scene.physics_scene_path)
    if physics.GetAttribute("physxScene:invertCollisionGroupFilter").Get() is not True:
        raise RuntimeError("shared Source collision group filter is not inverted")
    root = "/World/collisions"
    global_group = root + "/global_group"
    groups = [root + f"/group{i}" for i in range(len(paths))]
    scope = stage.GetPrimAtPath(root)
    if not scope or {str(p.GetPath()) for p in scope.GetChildren()} != {*groups, global_group}:
        raise RuntimeError("shared Source collision group inventory is incomplete or unexpected")
    records = {}
    for group_path, includes, allowed in [
        (global_group, ["/World/ground"], [global_group, *groups]),
        *((group, [path], [group, global_group]) for group, path in zip(groups, paths)),
    ]:
        prim = stage.GetPrimAtPath(group_path)
        actual_includes = [str(p) for p in prim.GetRelationship("collection:colliders:includes").GetTargets()]
        actual_allowed = [str(p) for p in prim.GetRelationship("physics:filteredGroups").GetTargets()]
        if (
            prim.GetTypeName() != "PhysicsCollisionGroup"
            or set(actual_includes) != set(includes)
            or set(actual_allowed) != set(allowed)
            or prim.GetRelationship("collection:colliders:excludes").GetTargets()
            or prim.GetAttribute("collection:colliders:expansionRule").Get() != "expandPrims"
            or prim.GetAttribute("collection:colliders:includeRoot").Get() is True
        ):
            raise RuntimeError(f"shared Source collision group contract violated: {group_path}")
        records[group_path] = {"includes": actual_includes, "allowed_groups": actual_allowed}
    return {"inverted": True, "groups": records}


__all__ = ["SharedBankTerrainAllocator", "validate_shared_scene_cfg", "audit_shared_collision_groups"]
