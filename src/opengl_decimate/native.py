"""Whether the compiled reducer is here, and running a reduction through it.

The package installs and works without a compiler. Where the accelerator built,
a reduction on the ``heap`` schedule, and the corner handover every recorded
reduction works out, run through it instead of through NumPy and Python -- the
same algorithm, the same order, the same result, at a size the NumPy loop
cannot reach.

    >>> isinstance(ACCELERATED, bool)
    True

Set ``OPENGL_DECIMATE_NO_ACCEL=1`` to keep to the NumPy path, which is what
lets the suite run both and hold them to the same answers.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np

from opengl_decimate.topology import Topology, VertexClass
from opengl_decimate.types import DecimateError, FloatArray, IndexArray

__all__ = ['ACCELERATED', 'check_index_width', 'ends_given_up', 'reduce_mesh']


def _load() -> Any:
    if os.environ.get('OPENGL_DECIMATE_NO_ACCEL'):
        return None
    try:
        # Imported by name rather than from the package, because it is built
        # only where a compiler was and a checker cannot see it at all.
        return importlib.import_module('opengl_decimate._reduce_native')
    except ImportError:  # pragma: no cover - depends on the build
        return None


_NATIVE = _load()

#: True where the compiled reducer is available and not switched off.
ACCELERATED = _NATIVE is not None


#: The largest index the compiled loop holds: a C ``int``.
_INT_MAX = 2**31 - 1


def check_index_width(points: int, faces: int) -> None:
    """Refuse a mesh whose indices do not fit the compiled loop's 32-bit ints.

    A point is named by its index, and a face corner by ``3 * face + slot``, so
    the limits are about two billion points and about 715 million faces. NumPy
    would wrap a larger index silently on the way in.
    """
    if points > _INT_MAX:
        raise DecimateError(
            'the compiled reducer holds point indices in 32-bit ints; %d points is too many'
            % (points,)
        )
    if 3 * faces > _INT_MAX:
        raise DecimateError(
            'the compiled reducer holds face corners in 32-bit ints; %d faces is too many'
            % (faces,)
        )


def reduce_mesh(
    mesh: Topology,
    quadrics: FloatArray,
    weights: FloatArray,
    kinds: np.ndarray,
    target_faces: int,
    placement: str,
    max_normal_flip: float,
    min_triangle_quality: float,
    error_limit: float | None,
) -> tuple[IndexArray, IndexArray, FloatArray, FloatArray, IndexArray]:
    """Run the whole contraction loop in C, mutating ``mesh`` as it goes.

    Returns the log the NumPy path also produces: which point died into which,
    where the survivor landed, what it cost, and the step each face was removed
    at. The mesh's faces, positions and live count are left as the reduction
    finished them.
    """
    check_index_width(mesh.vertex_count, len(mesh.faces))
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    alive = np.ascontiguousarray(mesh.alive).view(np.uint8)
    reducer = _NATIVE.Reducer(
        mesh.positions,
        faces,
        alive,
        quadrics,
        weights,
        np.ascontiguousarray(kinds, dtype=np.int8),
        np.ascontiguousarray(mesh.copies, dtype=np.int32),
        np.ascontiguousarray(mesh.charts, dtype=np.int32),
        0 if placement == 'optimal' else 1,
        float(np.cos(np.radians(max_normal_flip))),
        float(min_triangle_quality),
        -1.0 if error_limit is None else float(error_limit),
    )
    edges = np.ascontiguousarray(mesh.edges(), dtype=np.int32).reshape(-1, 2)
    dying, surviving, placements, deviation, removed_at = reducer.run(edges, int(target_faces))

    mesh.faces[:] = faces
    mesh.alive[:] = alive.view(bool)
    mesh.set_live_count(int(np.count_nonzero(mesh.alive)))
    # The compiled path rewrites the faces wholesale, so any face sets built
    # earlier describe a surface that is no longer there.
    mesh.forget_adjacency()
    return dying, surviving, placements, deviation, removed_at


def ends_given_up(
    points: FloatArray,
    placement: FloatArray,
    dying: IndexArray,
    surviving: IndexArray,
    charts: IndexArray,
    point_count: int,
) -> tuple[IndexArray, IndexArray]:
    """:func:`opengl_decimate.corners.ends_given_up_in_python`, in C."""
    return _NATIVE.ends_given_up(
        np.ascontiguousarray(points, dtype='d').reshape(-1, 3),
        np.ascontiguousarray(placement, dtype='d').reshape(-1, 3),
        np.ascontiguousarray(dying, dtype=np.int64),
        np.ascontiguousarray(surviving, dtype=np.int64),
        np.ascontiguousarray(charts, dtype=np.int64),
        int(point_count),
    )


def vertex_class_values() -> tuple[int, int, int]:
    """The classification values the compiled reducer is written against.

    It compares against the numbers rather than the names, so a change to the
    enum that did not reach the ``.pyx`` would silently mean something else.
    """
    return (
        int(VertexClass.MANIFOLD),
        int(VertexClass.BORDER),
        int(VertexClass.LOCKED),
    )
