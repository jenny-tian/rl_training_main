"""Small, dependency-free helpers for bounding terrain mesh face counts."""

from __future__ import annotations

from collections.abc import Iterator
import re


SOURCE_TILE_FACE_RANGES = "source_tile_face_ranges_v1"


def validate_tile_face_ranges(ranges, face_count: int) -> tuple[tuple[str, int, int], ...]:
    """Require ordered, disjoint, nonempty tile ranges covering the whole mesh."""

    if type(face_count) is not int or face_count < 1:
        raise ValueError("tile mesh face_count must be a positive integer")
    result = []
    names = set()
    expected_start = 0
    for name, start, stop in ranges:
        if not isinstance(name, str) or re.fullmatch(r"tile_r[0-9]+_c[0-9]+|border", name) is None:
            raise ValueError("invalid Source tile mesh name")
        if name in names:
            raise ValueError("duplicate Source tile mesh name")
        if type(start) is not int or type(stop) is not int or start != expected_start or stop <= start:
            raise ValueError("Source tile face ranges must be contiguous and nonempty")
        names.add(name)
        result.append((name, start, stop))
        expected_start = stop
    if expected_start != face_count:
        raise ValueError("Source tile face ranges do not cover the mesh")
    return tuple(result)


def balanced_face_chunk_slices(face_count: int, max_face_count: int) -> Iterator[slice]:
    """Partition ``face_count`` faces into balanced nonempty slices.

    The number of chunks is the minimum needed to respect ``max_face_count``.
    Balancing prevents the first chunk from sitting exactly at the limit when
    the remaining faces would make the final chunk much smaller.
    """

    if type(face_count) is not int or face_count < 0:
        raise ValueError("face_count must be a nonnegative integer")
    if type(max_face_count) is not int or max_face_count < 1:
        raise ValueError("max_face_count must be a positive integer")
    if face_count == 0:
        return

    chunk_count = (face_count + max_face_count - 1) // max_face_count
    chunk_size, remainder = divmod(face_count, chunk_count)
    start = 0
    for index in range(chunk_count):
        size = chunk_size + int(index < remainder)
        stop = start + size
        yield slice(start, stop)
        start = stop


__all__ = ["SOURCE_TILE_FACE_RANGES", "balanced_face_chunk_slices", "validate_tile_face_ranges"]
