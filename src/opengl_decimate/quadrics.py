"""Error quadrics: what a point costs, and where the cheapest point is.

A quadric is a symmetric 4x4 that answers one question -- given a point, what is
the summed squared distance to a set of planes? -- in a form with two properties
that make decimation possible. Quadrics **add**, so the cost of merging two
vertices is the sum of their quadrics; and the point that minimises one is found
by a single 3x3 solve rather than by search.

The 4x4 is stored as its ten distinct coefficients, in the order

    a00 a01 a02 a03 a11 a12 a13 a22 a23 a33

so a whole mesh's quadrics are one ``(vertices, 10)`` array and every operation
here is a whole-array operation.

Two accumulations are offered, and they share the storage and the solve:

``position_noise`` / ``normal_noise`` at zero
    The classical plane quadric -- exactly the squared distance to each plane.

either above zero
    The **probabilistic** quadric: each plane is read as a sample from a
    Gaussian, and the value is the *expected* squared distance. For a plane with
    mean normal ``n``, mean point ``p``, and isotropic variances
    ``s_n^2`` and ``s_p^2``, expanding ``E[((x - p).n)^2]`` over both
    distributions gives

        A = n n' + s_n^2 I
        b = (n.p) n + s_n^2 p
        c = (n.p)^2 + s_n^2 |p|^2 + s_p^2 + 3 s_n^2 s_p^2

    for ``E(x) = x'Ax - 2b'x + c``.

    Note which variance does what. ``s_n`` reaches ``A``, and is what makes it
    positive *definite* rather than rank one -- so a single plane has a minimum
    and a noisy neighbourhood is well conditioned. ``s_p`` reaches ``c`` alone:
    it comes from averaging a fixed quadratic form over a Gaussian cloud of
    sample points, which adds ``s_p^2 tr(A)`` and nothing else. So it raises
    every cost by the same amount and leaves the choice of contraction exactly
    where it was; with ``s_n`` at zero this is the classical metric with an
    offset.

    >>> import numpy as np
    >>> one_plane = plane_quadric(np.array([[0.0, 1.0, 0.0]]),
    ...                           np.array([[0.0, 2.0, 0.0]]))
    >>> bool(minimize(one_plane)[1][0])          # rank one: no single minimum
    False
    >>> noisy = plane_quadric(np.array([[0.0, 1.0, 0.0]]),
    ...                       np.array([[0.0, 2.0, 0.0]]), normal_noise=0.1)
    >>> bool(minimize(noisy)[1][0])
    True
"""

from __future__ import annotations

import numpy as np

from opengl_decimate.types import FloatArray, IndexArray

__all__ = [
    'QUADRIC_SIZE',
    'plane_quadric',
    'triangle_quadrics',
    'scatter',
    'accumulate',
    'evaluate',
    'minimize',
]

#: Distinct coefficients in a symmetric 4x4.
QUADRIC_SIZE = 10

#: Below this length a normal describes no plane. Squared lengths are compared,
#: so a normal shorter than this contributes nothing at all.
_TINY = 1e-300


def plane_quadric(
    normals: FloatArray,
    points: FloatArray,
    weights: FloatArray | None = None,
    position_noise: float = 0.0,
    normal_noise: float = 0.0,
) -> FloatArray:
    """Quadrics for planes through ``points`` with the given ``normals``.

    ``normals`` and ``points`` are ``(n, 3)``; the normals need not be unit
    length, since a plane is the same plane however it is described. A normal of
    zero length describes nothing and yields an all-zero quadric, which is what
    keeps a degenerate triangle from poisoning the vertices around it.

    ``weights`` multiplies each plane's contribution, and is where triangle area
    enters. ``position_noise`` and ``normal_noise`` are the standard deviations
    of the probabilistic form, in model units and in normal-vector units; both
    at zero gives the classical quadric.
    """
    normals = np.asarray(normals, dtype='d')
    points = np.asarray(points, dtype='d')
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > _TINY
    unit = np.zeros_like(normals)
    np.divide(normals, lengths[:, None], out=unit, where=usable[:, None])

    scale = np.ones(len(normals), dtype='d') if weights is None else np.asarray(weights, dtype='d')
    scale = np.where(usable, scale, 0.0)

    var_n = float(normal_noise) ** 2
    var_p = float(position_noise) ** 2

    offsets = np.einsum('ij,ij->i', unit, points)
    matrix = np.einsum('ij,ik->ijk', unit, unit)
    if var_n:
        matrix = matrix + var_n * np.eye(3)
    vector = offsets[:, None] * unit
    if var_n:
        vector = vector + var_n * points
    constant = offsets**2
    if var_n:
        constant = constant + var_n * np.einsum('ij,ij->i', points, points)
    if var_p:
        constant = constant + var_p + 3.0 * var_n * var_p

    out = np.empty((len(normals), QUADRIC_SIZE), dtype='d')
    out[:, 0] = matrix[:, 0, 0]
    out[:, 1] = matrix[:, 0, 1]
    out[:, 2] = matrix[:, 0, 2]
    out[:, 3] = -vector[:, 0]
    out[:, 4] = matrix[:, 1, 1]
    out[:, 5] = matrix[:, 1, 2]
    out[:, 6] = -vector[:, 1]
    out[:, 7] = matrix[:, 2, 2]
    out[:, 8] = -vector[:, 2]
    out[:, 9] = constant
    out *= scale[:, None]
    return out


def triangle_quadrics(
    positions: FloatArray,
    faces: IndexArray,
    position_noise: float = 0.0,
    normal_noise: float = 0.0,
) -> FloatArray:
    """One quadric per triangle, weighted by the triangle's area.

    Area weighting is what stops a dense patch of tiny triangles from outvoting
    the large ones around it: the quadric then measures deviation of *surface*
    rather than a count of planes.
    """
    positions = np.asarray(positions, dtype='d')
    faces = np.asarray(faces)
    corners = positions[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return plane_quadric(
        cross,
        corners[:, 0],
        weights=areas,
        position_noise=position_noise,
        normal_noise=normal_noise,
    )


def scatter(values: FloatArray, targets: IndexArray, count: int) -> FloatArray:
    """Sum row ``i`` of ``values`` onto vertex ``targets[i]``.

    ``count`` is how many vertices the result covers, which is not always the
    largest index mentioned -- a mesh can carry vertices no triangle uses.
    """
    values = np.asarray(values, dtype='d')
    targets = np.asarray(targets).reshape(-1)
    out = np.zeros((count, QUADRIC_SIZE), dtype='d')
    for coefficient in range(QUADRIC_SIZE):
        out[:, coefficient] = np.bincount(targets, weights=values[:, coefficient], minlength=count)
    return out


def accumulate(per_face: FloatArray, faces: IndexArray, count: int) -> FloatArray:
    """Sum each face's quadric onto the three vertices of that face."""
    per_face = np.asarray(per_face, dtype='d')
    return scatter(np.repeat(per_face, 3, axis=0), np.asarray(faces).reshape(-1), count)


def evaluate(quadrics: FloatArray, points: FloatArray) -> FloatArray:
    """The cost each quadric assigns to the matching point.

    ``quadrics`` is ``(n, 10)`` and ``points`` is ``(n, 3)``; the result is
    ``(n,)``. The value is a squared distance, so it is never negative beyond
    rounding.
    """
    quadrics = np.asarray(quadrics, dtype='d')
    points = np.asarray(points, dtype='d')
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (
        quadrics[:, 0] * x * x
        + quadrics[:, 4] * y * y
        + quadrics[:, 7] * z * z
        + 2.0 * (quadrics[:, 1] * x * y + quadrics[:, 2] * x * z + quadrics[:, 5] * y * z)
        + 2.0 * (quadrics[:, 3] * x + quadrics[:, 6] * y + quadrics[:, 8] * z)
        + quadrics[:, 9]
    )


def minimize(quadrics: FloatArray, tolerance: float = 1e-10) -> tuple[FloatArray, np.ndarray]:
    """The point minimising each quadric, and whether that point is determined.

    Returns ``(points, determined)``. Where ``determined`` is false the quadric
    has no single minimum -- one plane leaves a plane of equally good points, two
    leave a line -- and the point returned there is the origin, which a caller
    must replace with a fallback of its own. Being told is the useful part: a
    flat neighbourhood is the common case, not an error.

    ``tolerance`` is the smallest reciprocal condition number accepted. The
    matrix's largest entry times its largest cofactor, over the determinant, is
    the condition number in the largest-entry norm, so the test means the same
    thing for a quadric of any size. It separates a plane of equally good
    points, whose condition is infinite, from a noisy plane, whose condition is
    about one over the squared normal noise: the probabilistic metric's default
    of 1e-3 conditions the solve to 1e6, which it accepts.

    A determined minimum can still be far from any edge it is asked about,
    where the matrix is nearly singular: the point then slides along the
    valley to wherever rounding put it. The reducer drops a minimum further
    from its edge's midpoint than the edge is long.
    """
    quadrics = np.asarray(quadrics, dtype='d')
    a00, a01, a02 = quadrics[:, 0], quadrics[:, 1], quadrics[:, 2]
    a11, a12, a22 = quadrics[:, 4], quadrics[:, 5], quadrics[:, 7]
    # Minimising x'Ax - 2b'x + c means solving Ax = b, and the packed form holds
    # -b in the fourth column.
    b = -np.stack([quadrics[:, 3], quadrics[:, 6], quadrics[:, 8]], axis=1)

    # Cofactors of the symmetric 3x3. Symmetry makes the cofactor matrix its own
    # transpose, so the adjugate is the cofactors as they stand.
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c11 = a00 * a22 - a02 * a02
    c12 = a01 * a02 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    determinant = a00 * c00 + a01 * c01 + a02 * c02

    magnitude = np.max(np.abs(np.stack([a00, a01, a02, a11, a12, a22], axis=1)), axis=1)
    cofactor = np.max(np.abs(np.stack([c00, c01, c02, c11, c12, c22], axis=1)), axis=1)
    determined = np.abs(determinant) > tolerance * magnitude * cofactor

    points = np.zeros((len(quadrics), 3), dtype='d')
    safe = np.where(determined, determinant, 1.0)
    points[:, 0] = (c00 * b[:, 0] + c01 * b[:, 1] + c02 * b[:, 2]) / safe
    points[:, 1] = (c01 * b[:, 0] + c11 * b[:, 1] + c12 * b[:, 2]) / safe
    points[:, 2] = (c02 * b[:, 0] + c12 * b[:, 1] + c22 * b[:, 2]) / safe
    points[~determined] = 0.0
    return points, determined
