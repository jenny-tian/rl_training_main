"""Small, dependency-free helpers for bounding terrain mesh face counts."""

from __future__ import annotations

from collections.abc import Iterator


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


__all__ = ["balanced_face_chunk_slices"]
