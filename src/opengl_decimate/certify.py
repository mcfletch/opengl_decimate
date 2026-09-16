"""Measuring how far a reduced surface actually moved.

A quadric's value is an estimate: it accumulates squared distance to *planes*,
and planes extend past the triangles that produced them, so the number drifts
optimistic as a reduction goes on. What a level of detail has to promise is a
bound on the surface, which is a different quantity and has to be measured.

:func:`surface_deviation` measures it both ways round. One direction alone
cannot see a hole -- every point of a mesh with a piece missing is still on the
original -- so both surfaces are sampled and each set of samples is asked how
far it is from the other.

    >>> import numpy as np
    >>> square = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype='f4')
    >>> faces = np.array([0, 1, 2, 1, 3, 2], dtype='u4')
    >>> float(round(surface_deviation(square, faces, square, faces, samples=64).max, 6))
    0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opengl_decimate.spatial import CellGrid, shell, shell_size
from opengl_decimate.types import FloatArray, IndexArray

__all__ = ['Deviation', 'distance_to_mesh', 'sample_surface', 'surface_deviation']

#: Most points compared against triangles at once. The distance is exact
#: whatever this is; it only decides how much memory the measurement asks for.
BLOCK = 256

#: Ceiling on one block of the comparison. Every point is tested against every
#: triangle, so the work is the product of the two counts -- and so is the
#: memory, unless the product is what gets divided up rather than one side of
#: it. A 1.5M-triangle reference against 256 points at a time is nine gigabytes
#: of temporaries for a single block, which is not a slow measurement but a dead
#: process. Blocking both ways bounds it here regardless of either count.
BLOCK_BYTES = 64 * 1024 * 1024

#: Doubles held per point-triangle pair while the closest point is found: the
#: seven vectors and the dozen or so scalars in :func:`_closest_on_triangles`.
_PER_PAIR = 33

#: Below this many point-triangle pairs, comparing them all costs less than
#: building a grid to avoid comparing them.
_SCAN_PAIRS = 1 << 22

#: Cells a triangle may be registered in, averaged over the mesh. The cell side
#: is doubled until the grid is this economical, which is what stops one large
#: triangle in an otherwise fine mesh from filling it.
_CELLS_PER_TRIANGLE = 16

#: Most ``(point, cell)`` queries asked of the grid at once, which sets how many
#: of a ring's cells are looked up in one call.
_QUERY_ROWS = 1 << 20

_TINY = 1e-30


@dataclass(frozen=True)
class Deviation:
    """How far two surfaces are apart, in model units.

    ``max`` is the sampled Hausdorff distance -- the worst place either surface
    is from the other -- and is the number a level of detail's error bound has to
    cover. ``rms`` is the root-mean-square over the samples, which is what the
    eye is closer to seeing, since one bad corner moves ``max`` and almost
    nothing else.
    """

    max: float
    rms: float


def distance_to_mesh(points: FloatArray, positions: FloatArray, indices: IndexArray) -> FloatArray:
    """Distance from each point to the nearest triangle of a mesh.

    Exact, not an approximation to the vertices: the closest point of a triangle
    may be inside it, along an edge, or at a corner, and all three are found. A
    mesh with no triangles is infinitely far from everywhere, which is what lets
    a caller measure against an empty surface without a special case.

    The triangles go into a uniform grid and each point is measured against the
    handful in the cells around it, widening until the answer is known to be
    inside what has been searched. Testing every point against every triangle
    instead is the product of the two counts, and for the meshes this package
    exists for that is not a slow measurement but one nobody waits for.
    """
    points = np.asarray(points, dtype='d')
    faces = np.asarray(indices).reshape(-1, 3)
    if not len(faces) or not len(points):
        return np.full(len(points), np.inf)

    corners = np.asarray(positions, dtype='d')[faces]
    if len(points) * len(corners) <= _SCAN_PAIRS:
        return np.sqrt(_nearest_among(points, corners))
    return _nearest_by_grid(points, corners)


def _nearest_among(points: FloatArray, corners: FloatArray) -> FloatArray:
    """Squared distance from every point to the nearest of every triangle.

    Blocked both ways. Every point is tested against every triangle, so the work
    is the product of the two counts -- and so is the memory, unless the product
    is what gets divided up rather than one side of it. A 1.5M-triangle mesh
    against 256 points at a time is nine gigabytes of temporaries for a single
    block, which is not a slow measurement but a dead process.
    """
    points_at_once = min(BLOCK, len(points))
    pairs = max(1, BLOCK_BYTES // (_PER_PAIR * 8))
    triangles_at_once = max(1, min(len(corners), pairs // points_at_once))

    best = np.full(len(points), np.inf)
    for start in range(0, len(points), points_at_once):
        block = points[start : start + points_at_once]
        nearest = np.full(len(block), np.inf)
        for first in range(0, len(corners), triangles_at_once):
            some = corners[first : first + triangles_at_once]
            offsets = _closest_on_triangles(block, some) - block[:, None, :]
            np.minimum(
                nearest,
                np.min(np.einsum('ijk,ijk->ij', offsets, offsets), axis=1),
                out=nearest,
            )
        best[start : start + points_at_once] = nearest
    return best


def _grid_side(low: FloatArray, high: FloatArray) -> float:
    """A cell side that registers each triangle in a handful of cells.

    Starting from the mean triangle's largest extent, so an ordinary triangle
    spans about two cells on each axis, and doubled until the total number of
    registrations is a small multiple of the triangle count -- which is what
    keeps one huge triangle in a fine mesh from filling the grid.
    """
    side = float(np.mean(np.max(high - low, axis=1)))
    whole = float(np.max(np.max(high, axis=0) - np.min(low, axis=0)))
    if not side > 0.0:
        side = whole if whole > 0.0 else 1.0
    while side < whole:
        spans = np.floor(high / side) - np.floor(low / side) + 1.0
        if float(np.sum(np.prod(spans, axis=1))) <= _CELLS_PER_TRIANGLE * len(low):
            break
        side *= 2.0
    return side


def _register(low: IndexArray, high: IndexArray) -> tuple[IndexArray, IndexArray]:
    """Every cell each box covers: the cell coordinates, and whose box it is."""
    extents = high - low + 1
    total = np.prod(extents, axis=1)
    owner = np.repeat(np.arange(len(low), dtype=np.int64), total)
    ramp = np.arange(int(total.sum()), dtype=np.int64) - np.repeat(np.cumsum(total) - total, total)
    plane = np.repeat(extents[:, 1] * extents[:, 2], total)
    depth = np.repeat(extents[:, 2], total)
    inside = np.stack([ramp // plane, (ramp % plane) // depth, ramp % depth], axis=1)
    return low[owner] + inside, owner


def scan_is_cheaper(ring_cells: int, per_cell: float, triangles: int) -> bool:
    """Whether measuring the waiting points against every triangle costs less.

    Both sides are paid per waiting point, so the points cancel and what is left
    is a comparison of triangles tested: one more ring asks for the triangles in
    its cells, and a scan asks for the whole reference.

    The currency matters. Comparing a *count of points* or a *count of cells*
    against a count of triangles -- which is the mistake this replaces -- reads
    as a sensible rule and is out by whatever the two units differ by. On a mesh
    of a million triangles at eight to a cell, one more ring of three hundred
    cells is two thousand tests a point and the scan is a million: a factor of
    four hundred, spent on the handful of points that had almost arrived.

    A grid holding nothing makes every ring free, which would widen the search
    for ever; there is nothing out there to find, so it scans instead.
    """
    return not per_cell or ring_cells * per_cell >= triangles


def _nearest_by_grid(points: FloatArray, corners: FloatArray) -> FloatArray:
    """Distance from each point to the nearest triangle, through a uniform grid.

    A triangle is registered in every cell its bounding box covers. So every
    triangle within ``radius * side`` of a point has a point of its own inside
    the box of cells out to ``radius``, and is therefore registered in one of
    them: once a point's best is no further than that, searching wider cannot
    improve it and the answer is exact. Points not yet settled go round again
    one ring wider.

    The widening stops where it stops paying -- see :func:`scan_is_cheaper` --
    and whatever is still waiting is measured against every triangle. Both
    routes are exact, so the rule decides only what the answer costs.
    """
    low, high = np.min(corners, axis=1), np.max(corners, axis=1)
    side = _grid_side(low, high)
    cells, owner = _register(
        np.floor(low / side).astype(np.int64), np.floor(high / side).astype(np.int64)
    )
    grid = CellGrid(cells)

    at = np.floor(points / side).astype(np.int64)
    best = np.full(len(points), np.inf)
    pending = np.arange(len(points), dtype=np.int64)
    # What one cell of the ring is worth asking for, averaged over the grid.
    per_cell = len(cells) / grid.occupied if grid.occupied else 0.0
    radius = 0
    while len(pending) and not scan_is_cheaper(shell_size(radius), per_cell, len(corners)):
        offsets = np.asarray(shell(radius), dtype=np.int64)
        here = at[pending]
        # Whole rings at a time rather than a cell at a time: a ring past the
        # first few holds hundreds of cells, and asking for each on its own
        # spends more on NumPy's per-call overhead than on the search.
        at_once = max(1, _QUERY_ROWS // len(pending))
        for first in range(0, len(offsets), at_once):
            batch = offsets[first : first + at_once]
            for query, entry in grid.members((here[:, None, :] + batch).reshape(-1, 3)):
                asking = query // len(batch)
                rows = pending[asking]
                found = _paired_distances(points[rows], corners[owner[entry]])
                # The rows come back grouped by point, so taking each point's
                # best is one `reduceat` -- `np.minimum.at` over them would be
                # scattered, which NumPy does one element at a time.
                opens = np.empty(len(asking), dtype=bool)
                opens[0] = True
                opens[1:] = asking[1:] != asking[:-1]
                starts = np.flatnonzero(opens)
                each = pending[asking[starts]]
                best[each] = np.minimum(best[each], np.minimum.reduceat(found, starts))
        pending = pending[best[pending] > (radius * side) ** 2]
        radius += 1
    if len(pending):
        best[pending] = np.minimum(best[pending], _nearest_among(points[pending], corners))
    return np.sqrt(best)


def _paired_distances(points: FloatArray, corners: FloatArray) -> FloatArray:
    """Squared distance from each point to the triangle in the same row."""
    offsets = _closest(points, corners[:, 0], corners[:, 1], corners[:, 2]) - points
    return np.einsum('ij,ij->i', offsets, offsets)


def _closest_on_triangles(points: FloatArray, corners: FloatArray) -> FloatArray:
    """The closest point of every triangle to every point: ``(p, t, 3)``."""
    return _closest(
        points[:, None, :], corners[None, :, 0], corners[None, :, 1], corners[None, :, 2]
    )


def _closest(points: FloatArray, a: FloatArray, b: FloatArray, c: FloatArray) -> FloatArray:
    """The closest point of triangle ``(a, b, c)`` to ``points``.

    The triangle's plane is divided into the seven regions a nearest point can
    fall in -- the face, three edges, three corners -- and each is selected by
    the sign of a pair of barycentric-like quantities. Every region is evaluated
    for every pair and the right one is chosen, which is what makes it one pass
    over arrays rather than a branch per triangle.

    Everything broadcasts over leading axes, so the same pass serves a point
    against every triangle and a point against the one triangle paired with it.
    """
    along, across = b - a, c - a
    offset = points - a

    edge_here = np.einsum('...j,...j->...', along, offset)
    edge_there = np.einsum('...j,...j->...', across, offset)
    from_b = points - b
    b_along = np.einsum('...j,...j->...', along, from_b)
    b_across = np.einsum('...j,...j->...', across, from_b)
    from_c = points - c
    c_along = np.einsum('...j,...j->...', along, from_c)
    c_across = np.einsum('...j,...j->...', across, from_c)

    face_c = edge_here * b_across - b_along * edge_there
    face_b = c_along * edge_there - edge_here * c_across
    face_a = b_along * c_across - c_along * b_across

    total = face_a + face_b + face_c
    scale = np.where(np.abs(total) > _TINY, total, 1.0)
    inside = a + along * (face_b / scale)[..., None] + across * (face_c / scale)[..., None]

    on_ab = a + along * _fraction(edge_here, edge_here - b_along)[..., None]
    on_ac = a + across * _fraction(edge_there, edge_there - c_across)[..., None]
    on_bc = (
        b
        + (c - b)
        * _fraction(b_across - b_along, (b_across - b_along) + (c_along - c_across))[..., None]
    )

    at_a = (edge_here <= 0) & (edge_there <= 0)
    at_b = (b_along >= 0) & (b_across <= b_along)
    at_c = (c_across >= 0) & (c_along <= c_across)
    near_ab = (face_c <= 0) & (edge_here >= 0) & (b_along <= 0)
    near_ac = (face_b <= 0) & (edge_there >= 0) & (c_across <= 0)
    near_bc = (face_a <= 0) & ((b_across - b_along) >= 0) & ((c_along - c_across) >= 0)

    closest = inside
    closest = np.where(near_bc[..., None], on_bc, closest)
    closest = np.where(near_ac[..., None], on_ac, closest)
    closest = np.where(near_ab[..., None], on_ab, closest)
    closest = np.where(at_c[..., None], np.broadcast_to(c, closest.shape), closest)
    closest = np.where(at_b[..., None], np.broadcast_to(b, closest.shape), closest)
    return np.where(at_a[..., None], np.broadcast_to(a, closest.shape), closest)


def _fraction(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """``numerator / denominator`` clamped to ``[0, 1]``, safe where it is zero."""
    safe = np.where(np.abs(denominator) > _TINY, denominator, 1.0)
    return np.clip(numerator / safe, 0.0, 1.0)


def sample_surface(
    positions: FloatArray, indices: IndexArray, count: int, seed: int = 0
) -> FloatArray:
    """``count`` points spread over a mesh in proportion to triangle area.

    Area-proportional rather than per-triangle, so a decimated mesh -- where one
    triangle may cover what fifty used to -- is measured over its surface rather
    than over its triangle list.
    """
    faces = np.asarray(indices).reshape(-1, 3)
    corners = np.asarray(positions, dtype='d')[faces]
    if not len(faces):
        return np.zeros((0, 3), dtype='d')

    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    total = float(areas.sum())
    weights = areas / total if total > _TINY else np.full(len(areas), 1.0 / len(areas))

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=count, p=weights)
    # Reflecting the unit square into the triangle keeps the sample uniform
    # without rejecting any draw.
    first, second = rng.random(count), rng.random(count)
    outside = first + second > 1.0
    first[outside], second[outside] = 1.0 - first[outside], 1.0 - second[outside]
    picked = corners[chosen]
    return (
        picked[:, 0]
        + (picked[:, 1] - picked[:, 0]) * first[:, None]
        + (picked[:, 2] - picked[:, 0]) * second[:, None]
    )


def surface_deviation(
    reference_positions: FloatArray,
    reference_indices: IndexArray,
    candidate_positions: FloatArray,
    candidate_indices: IndexArray,
    samples: int = 4000,
    seed: int = 0,
) -> Deviation:
    """How far the two surfaces are from each other, measured both ways.

    ``samples`` points are taken from each surface. Raising it tightens the
    estimate of the worst case, which is the half that needs the samples; the
    root-mean-square settles quickly.
    """
    forward = sample_surface(reference_positions, reference_indices, samples, seed)
    backward = sample_surface(candidate_positions, candidate_indices, samples, seed + 1)
    distances = np.concatenate(
        [
            distance_to_mesh(forward, candidate_positions, candidate_indices),
            distance_to_mesh(backward, reference_positions, reference_indices),
        ]
    )
    finite = distances[np.isfinite(distances)]
    if not len(finite):
        return Deviation(max=float('inf'), rms=float('inf'))
    return Deviation(max=float(np.max(finite)), rms=float(np.sqrt(np.mean(finite**2))))
