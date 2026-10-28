"""Face-count chunking must be exact and must not emit empty collider parts."""

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/_mesh_chunking.py"
)
SPEC = importlib.util.spec_from_file_location(
    "source_mesh_chunking_under_test", MODULE_PATH
)
mesh_chunking_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mesh_chunking_module
SPEC.loader.exec_module(mesh_chunking_module)

balanced_face_chunk_slices = mesh_chunking_module.balanced_face_chunk_slices


@pytest.mark.parametrize(
    "face_count,max_face_count,expected_sizes",
    [
        (0, 5, []),
        (1, 5, [1]),
        (5, 5, [5]),
        (6, 5, [3, 3]),
        (11, 5, [4, 4, 3]),
        (10_000_001, 10_000_000, [5_000_001, 5_000_000]),
    ],
)
def test_balanced_chunks_cover_every_face_once(face_count, max_face_count, expected_sizes):
    chunks = tuple(balanced_face_chunk_slices(face_count, max_face_count))
    assert [chunk.stop - chunk.start for chunk in chunks] == expected_sizes
    assert all(chunk.start < chunk.stop for chunk in chunks)
    assert all(size <= max_face_count for size in expected_sizes)
    if face_count == 0:
        assert chunks == ()
    else:
        assert chunks[0].start == 0
        assert chunks[-1].stop == face_count
        assert all(left.stop == right.start for left, right in zip(chunks, chunks[1:]))
    assert sum(expected_sizes) == face_count


@pytest.mark.parametrize(
    "face_count,max_face_count",
    [(-1, 5), (True, 5), (1.0, 5), (1, 0), (1, -1), (1, True)],
)
def test_chunk_planner_rejects_invalid_counts(face_count, max_face_count):
    with pytest.raises(ValueError):
        tuple(balanced_face_chunk_slices(face_count, max_face_count))
