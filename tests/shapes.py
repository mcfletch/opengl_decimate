"""Meshes with known properties, for tests that need a real surface.

Each returns ``(positions, indices)`` with positions ``(n, 3)`` float32 and
indices a flat ``uint32`` triple-per-triangle array -- the form the package
takes. The closed ones are watertight and consistently wound; the open ones have
a border on purpose; the broken ones are broken on purpose.
"""

from __future__ import annotations

import numpy as np


def _mesh(positions, faces):
    return (
        np.asarray(positions, dtype='f4'),
        np.asarray(faces, dtype=np.uint32).reshape(-1),
    )


def tetrahedron():
    """Four vertices, four faces, closed and outward-wound."""
    positions = [
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    ]
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    return _mesh(positions, faces)


def octahedron():
    """Six vertices, eight faces, closed."""
    positions = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    faces = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ]
    return _mesh(positions, faces)


_ICOSAHEDRON_FACES = [
    (0, 11, 5),
    (0, 5, 1),
    (0, 1, 7),
    (0, 7, 10),
    (0, 10, 11),
    (1, 5, 9),
    (5, 11, 4),
    (11, 10, 2),
    (10, 7, 6),
    (7, 1, 8),
    (3, 9, 4),
    (3, 4, 2),
    (3, 2, 6),
    (3, 6, 8),
    (3, 8, 9),
    (4, 9, 5),
    (2, 4, 11),
    (6, 2, 10),
    (8, 6, 7),
    (9, 8, 1),
]


def _icosahedron():
    phi = (1.0 + 5.0**0.5) / 2.0
    raw = [
        (-1, phi, 0),
        (1, phi, 0),
        (-1, -phi, 0),
        (1, -phi, 0),
        (0, -1, phi),
        (0, 1, phi),
        (0, -1, -phi),
        (0, 1, -phi),
        (phi, 0, -1),
        (phi, 0, 1),
        (-phi, 0, -1),
        (-phi, 0, 1),
    ]
    points = np.asarray(raw, dtype='d')
    points /= np.linalg.norm(points, axis=1)[:, None]
    return points, np.asarray(_ICOSAHEDRON_FACES, dtype=np.int64)


def icosphere(subdivisions: int = 2):
    """A closed, near-uniform triangulation of the unit sphere.

    Twenty faces at ``subdivisions=0``, four times as many at each step, so the
    surface it approximates is known exactly and the deviation of a decimated
    copy can be compared against the sphere itself.
    """
    points, faces = _icosahedron()
    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}
        grown = list(points)
        out = []
        for a, b, c in faces:

            def middle(i, j, _grown=grown, _midpoints=midpoints):
                key = (min(i, j), max(i, j))
                found = _midpoints.get(key)
                if found is None:
                    point = (_grown[i] + _grown[j]) / 2.0
                    point = point / np.linalg.norm(point)
                    found = len(_grown)
                    _grown.append(point)
                    _midpoints[key] = found
                return found

            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        points = np.asarray(grown, dtype='d')
        faces = np.asarray(out, dtype=np.int64)
    return _mesh(points, faces)


def grid(side: int = 5, bump: float = 0.0):
    """An open ``side`` x ``side`` patch in the XZ plane, with a border.

    ``bump`` raises the middle into a dome, so the surface is not planar and the
    quadrics have something to say about it.
    """
    axis = np.linspace(-1.0, 1.0, side)
    gx, gz = np.meshgrid(axis, axis, indexing='ij')
    gy = bump * np.cos(gx * np.pi / 2.0) * np.cos(gz * np.pi / 2.0)
    positions = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    # Built with array arithmetic rather than a loop over cells: the performance
    # tests want a patch of hundreds of thousands of triangles, and generating
    # one has to cost less than what it is being used to measure.
    i, j = np.meshgrid(np.arange(side - 1), np.arange(side - 1), indexing='ij')
    corner = (i * side + j).reshape(-1)
    faces = np.concatenate(
        [
            np.stack([corner, corner + 1, corner + side], axis=1),
            np.stack([corner + 1, corner + side + 1, corner + side], axis=1),
        ]
    )
    return _mesh(positions, faces)


def fan(valence: int = 16):
    """An open fan of ``valence - 1`` triangles around one apex.

    The apex is point 0 and the rim is ``1..valence``, so its valence is exactly
    the argument. A fan-triangulated n-gon, a lathe pole and a CAD hub all
    present this: one point carrying arbitrarily many faces, in the *input*,
    where no reduction had a chance to grow it.
    """
    angle = np.linspace(0.0, 1.5 * np.pi, valence)
    rim = np.stack([np.cos(angle), np.zeros(valence), np.sin(angle)], axis=1)
    positions = np.concatenate([[[0.0, 0.0, 0.0]], rim])
    faces = np.stack(
        [
            np.zeros(valence - 1, dtype=np.int64),
            np.arange(1, valence, dtype=np.int64),
            np.arange(2, valence + 1, dtype=np.int64),
        ],
        axis=1,
    )
    return _mesh(positions, faces)


def tube(around: int = 48, along: int = 40, noise: float = 0.0, seed: int = 0):
    """An open cylinder of radius one along Y, from -2 to 2.

    Every plane on a cylinder contains its axis, so the summed quadric of an
    edge is singular along it. ``noise`` jitters the points by that standard
    deviation, which leaves the solve determined but nearly singular -- the
    shape a scanned pipe or trunk presents.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    height = np.linspace(-2.0, 2.0, along)
    a, h = np.meshgrid(angle, height, indexing='ij')
    positions = np.stack([np.cos(a), h, np.sin(a)], axis=-1).reshape(-1, 3)
    positions += np.random.default_rng(seed).normal(scale=noise, size=positions.shape)
    i, j = np.meshgrid(np.arange(around), np.arange(along - 1), indexing='ij')
    here = (i * along + j).reshape(-1)
    beside = (((i + 1) % around) * along + j).reshape(-1)
    faces = np.concatenate(
        [
            np.stack([here, here + 1, beside], axis=1),
            np.stack([beside, here + 1, beside + 1], axis=1),
        ]
    )
    return _mesh(positions, faces)


def cone(valence: int = 64):
    """A closed cone on a flat base: a rim of ``valence`` points and two poles.

    The apex and the centre of the base each carry ``valence`` faces, which is
    what a lathed solid and a fan-triangulated cap present to a reducer. The
    base is flat, so its spokes cost nothing and are contracted early.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, valence, endpoint=False)
    rim = np.stack([np.cos(angle), np.sin(angle), np.zeros(valence)], axis=1)
    positions = np.concatenate([rim, [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    here = np.arange(valence)
    beside = (here + 1) % valence
    faces = np.concatenate(
        [
            np.stack([here, beside, np.full(valence, valence)], axis=1),
            np.stack([beside, here, np.full(valence, valence + 1)], axis=1),
        ]
    )
    return _mesh(positions, faces)


def three_charts(side: int = 9, first: int = 3, apart: float = 10.0):
    """A domed patch unwrapped as three charts, the middle one a column of quads.

    Returns ``(attributes, indices)`` with ``TEXCOORD_0``. Each chart is the
    patch's own ``(x, z)`` mapped to the unit square and moved ``apart`` along
    ``u`` per chart, so the chart a coordinate belongs to is ``u // apart``.
    Every point on a chart boundary is one vertex per chart it is drawn in.
    """
    positions, indices = grid(side, bump=0.3)
    column = np.round((positions[:, 0] + 1.0) * 0.5 * (side - 1)).astype(int)
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    chart = np.where(
        column[faces].max(axis=1) <= first,
        0,
        np.where(column[faces].min(axis=1) >= first + 1, 2, 1),
    )
    drawn = np.unique(np.stack([faces.reshape(-1), np.repeat(chart, 3)], axis=1), axis=0)
    vertex = {(int(point), int(where)): index for index, (point, where) in enumerate(drawn)}
    uv = (positions[drawn[:, 0]][:, [0, 2]] + 1.0) * 0.5
    uv[:, 0] += apart * drawn[:, 1]
    remapped = [
        vertex[(int(point), int(where))]
        for row, where in zip(faces, chart, strict=True)
        for point in row
    ]
    return (
        {'POSITION': positions[drawn[:, 0]], 'TEXCOORD_0': uv.astype('f4')},
        np.asarray(remapped, dtype=np.uint32),
    )


def nearly_coincident(count: int = 200, spread: float = 1e-7):
    """A grid whose points are each duplicated a hair away, as a scan's are.

    Returns ``(positions, indices)`` where every triangle's corners were jittered
    independently, so an exact weld finds nothing and a tolerance weld has real
    work to do.
    """
    positions, indices = grid(count)
    rng = np.random.default_rng(0)
    jittered = positions.astype('d') + rng.normal(scale=spread, size=positions.shape)
    return _mesh(jittered, np.asarray(indices, dtype=np.int64).reshape(-1, 3))


def bowtie():
    """Two triangles meeting at one vertex and nowhere else."""
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, -1.0, 0.0),
    ]
    return _mesh(positions, [(0, 1, 2), (0, 3, 4)])


def nonmanifold_edge():
    """Three triangles sharing one edge -- a surface no orientation covers."""
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    return _mesh(positions, [(0, 1, 2), (0, 1, 3), (0, 1, 4)])


def cube_with_hard_normals():
    """A closed cube whose normals are per-face, so every edge is a seam.

    Eight positions carried by twenty-four vertices -- the arrangement a glTF
    export of a hard-edged model has, and the one attribute handling has to cope
    with.
    """
    corners = np.asarray(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ],
        dtype='d',
    )
    quads = [
        ((0, 3, 2, 1), (0, 0, -1)),
        ((4, 5, 6, 7), (0, 0, 1)),
        ((0, 1, 5, 4), (0, -1, 0)),
        ((2, 3, 7, 6), (0, 1, 0)),
        ((1, 2, 6, 5), (1, 0, 0)),
        ((0, 4, 7, 3), (-1, 0, 0)),
    ]
    positions, normals, uvs, faces = [], [], [], []
    for quad, normal in quads:
        base = len(positions)
        for offset, index in enumerate(quad):
            positions.append(corners[index])
            normals.append(normal)
            uvs.append(((offset in (1, 2)) * 1.0, (offset in (2, 3)) * 1.0))
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    pos, idx = _mesh(positions, faces)
    return (
        {
            'POSITION': pos,
            'NORMAL': np.asarray(normals, dtype='f4'),
            'TEXCOORD_0': np.asarray(uvs, dtype='f4'),
        },
        idx,
    )
