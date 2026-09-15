"""The surface under the arrays: what is joined to what, and what may move.

A mesh arrives as vertices and triangles, but a vertex is not the same thing as
a *point*: a hard-edged export carries the same corner of a cube as three
vertices with three different normals. Decimation is a question about the
surface, so :func:`build` welds vertices that share a position and works in
those positions, keeping each corner's original vertex alongside so the
attributes it carried can be handed back at the end.

:meth:`Topology.classify` then says what each position is allowed to do:

``MANIFOLD``
    A closed fan of faces around it. Free to collapse onto any neighbour.
``BORDER``
    On the edge of the surface. May collapse only along the border, which is
    what keeps a patch's outline where the author put it.
``LOCKED``
    Never moves. A non-manifold edge, a bowtie, an isolated point, or a vertex
    the caller named -- a cluster group's outer boundary arrives this way.

The classification is computed once and stays true for the whole reduction: the
collapse rules in :mod:`opengl_decimate.collapse` admit only contractions that
preserve the link of the edge, and those leave every border a border and every
manifold fan a fan.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from opengl_decimate.types import DecimateError, FloatArray, IndexArray

__all__ = ['VertexClass', 'Topology', 'build', 'weld_positions']


class VertexClass(IntEnum):
    """What a position is allowed to do during a reduction."""

    MANIFOLD = 0
    BORDER = 1
    LOCKED = 2


class _DisjointSet:
    """Union-find over a fixed range, for grouping points a tolerance welds.

    No path compression, because the trees never get deep enough to want it:
    :func:`weld_positions` unions in ascending order and always attaches the
    larger root to the smaller, so every group ends up a star rooted at its
    lowest member and a lookup is one hop.
    """

    def __init__(self, count: int) -> None:
        self._parent = np.arange(count, dtype=np.int64)

    def find(self, item: int) -> int:
        parent = self._parent
        while parent[item] != item:
            item = int(parent[item])
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


def weld_positions(positions: FloatArray, tolerance: float = 0.0) -> tuple[FloatArray, IndexArray]:
    """Merge vertices that occupy the same point.

    Returns ``(points, vertex_point)``: the distinct points, and which point each
    input vertex belongs to. Points are numbered by **first appearance**, so a
    mesh with no coincident vertices welds to itself and a caller's vertex
    indices still mean what they meant. At ``tolerance`` zero the comparison is
    exact, which is one vectorised sort. A positive tolerance merges anything
    within that distance -- which is what a scan needs, since the same corner
    reconstructed twice differs in the last several bits -- by grouping into
    cells of that size and joining across the twenty-seven cells a point can
    have a partner in.

    >>> import numpy as np
    >>> points, belongs = weld_positions(np.array([[1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]]))
    >>> belongs.tolist()
    [0, 1, 0]
    """
    positions = np.asarray(positions, dtype='d')
    if tolerance <= 0.0:
        _, first, inverse = np.unique(positions, axis=0, return_index=True, return_inverse=True)
        return _by_first_appearance(positions, inverse.reshape(-1), len(first))

    cells = np.floor(positions / tolerance).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(map(tuple, cells)):
        buckets.setdefault(cell, []).append(index)

    groups = _DisjointSet(len(positions))
    offsets = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
    limit = tolerance * tolerance
    for index, cell in enumerate(map(tuple, cells)):
        for offset in offsets:
            neighbours = buckets.get(
                (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
            )
            if not neighbours:
                continue
            for other in neighbours:
                if other > index:
                    delta = positions[index] - positions[other]
                    if float(delta @ delta) <= limit:
                        groups.union(index, other)

    roots = np.asarray([groups.find(i) for i in range(len(positions))], dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    return _by_first_appearance(positions, inverse.reshape(-1), len(unique_roots), average=True)


def _by_first_appearance(
    positions: FloatArray, inverse: IndexArray, count: int, average: bool = False
) -> tuple[FloatArray, IndexArray]:
    """Renumber welded groups so group ``k`` is the ``k``-th to appear.

    ``np.unique`` numbers its groups in sorted order, which would permute a
    caller's vertices for no reason a caller can see. ``average`` places a group
    at the mean of its members, which is what a tolerance weld wants; an exact
    weld has nothing to average.
    """
    inverse = np.asarray(inverse, dtype=np.int64)
    first_member = np.full(count, len(positions), dtype=np.int64)
    np.minimum.at(first_member, inverse, np.arange(len(positions), dtype=np.int64))
    order = np.argsort(first_member)
    relabel = np.empty(count, dtype=np.int64)
    relabel[order] = np.arange(count, dtype=np.int64)
    renumbered = relabel[inverse]

    if not average:
        return positions[first_member[order]], renumbered
    points = np.zeros((count, 3), dtype='d')
    members = np.bincount(renumbered, minlength=count).astype('d')
    for axis in range(3):
        points[:, axis] = (
            np.bincount(renumbered, weights=positions[:, axis], minlength=count) / members
        )
    return points, renumbered


class Topology:
    """A welded surface, its adjacency, and the faces still alive on it.

    ``positions`` are the welded points and are moved in place by a collapse;
    ``faces`` holds position indices and ``corners`` the vertex each of those
    came from, so an attribute follows its corner rather than its point.
    """

    def __init__(
        self,
        positions: FloatArray,
        faces: IndexArray,
        corners: IndexArray,
        vertex_point: IndexArray,
    ) -> None:
        self.positions = np.ascontiguousarray(positions, dtype='d')
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.corners = np.ascontiguousarray(corners, dtype=np.int64)
        #: Which welded point each *input* vertex went to.
        self.vertex_point = np.ascontiguousarray(vertex_point, dtype=np.int64)
        #: False once a collapse has removed the face.
        self.alive = np.ones(len(self.faces), dtype=bool)
        # Kept as a running total rather than counted on demand: a reduction
        # asks how many faces are left once per contraction, and scanning a
        # million flags to answer costs more than the contraction does.
        self._alive_count = len(self.faces)
        self.vertex_faces: list[set[int]] = [set() for _ in range(len(self.positions))]
        for index, face in enumerate(self.faces):
            for point in face:
                self.vertex_faces[point].add(index)

    @property
    def vertex_count(self) -> int:
        """How many welded points the surface has."""
        return len(self.positions)

    @property
    def face_count(self) -> int:
        """How many triangles are still alive."""
        return self._alive_count

    def kill_face(self, face: int) -> None:
        """Remove a face from the live surface and from its points' adjacency."""
        self.alive[face] = False
        self._alive_count -= 1
        for point in self.faces[face]:
            self.vertex_faces[point].discard(face)

    def neighbours(self, point: int) -> set[int]:
        """The points joined to ``point`` by a live triangle."""
        found: set[int] = set()
        for face in self.vertex_faces[point]:
            found.update(int(v) for v in self.faces[face])
        found.discard(point)
        return found

    def edge_faces(self, left: int, right: int) -> set[int]:
        """The live faces using both ends of an edge."""
        return self.vertex_faces[left] & self.vertex_faces[right]

    def is_boundary_edge(self, left: int, right: int) -> bool:
        """True where exactly one face uses the edge, so the surface stops there."""
        return len(self.edge_faces(left, right)) == 1

    def live_faces(self) -> IndexArray:
        """The face array with the dead rows removed."""
        return self.faces[self.alive]

    def edges(self) -> IndexArray:
        """Every distinct undirected edge of the live surface, as ``(e, 2)``."""
        faces = self.live_faces()
        pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        pairs = np.sort(pairs, axis=1)
        return _unique_pairs(pairs, self.vertex_count)[0]

    def classify(
        self,
        lock_boundary: bool = False,
        locked: IndexArray | None = None,
    ) -> np.ndarray:
        """What each point may do, as an array of :class:`VertexClass`.

        ``lock_boundary`` holds the surface's border where it is, which is what a
        cluster group's simplification needs so its neighbours still meet it.
        ``locked`` names further points to hold.
        """
        kinds = np.full(self.vertex_count, VertexClass.MANIFOLD, dtype=np.int8)
        faces = self.live_faces()
        if len(faces):
            pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
            pairs = np.sort(pairs, axis=1)
            edges, counts = _unique_pairs(pairs, self.vertex_count)

            over = edges[counts > 2].reshape(-1)
            kinds[over] = VertexClass.LOCKED

            border_ends = edges[counts == 1].reshape(-1)
            border_count = np.bincount(border_ends, minlength=self.vertex_count)
            # A point on a well-formed border has exactly two border edges. More
            # than two is a pinch: two sheets meeting at a point, which no single
            # placement can serve.
            kinds[border_count == 2] = np.where(
                kinds[border_count == 2] == VertexClass.LOCKED,
                VertexClass.LOCKED,
                VertexClass.BORDER,
            )
            kinds[border_count > 2] = VertexClass.LOCKED

            kinds[_multi_fan_points(faces)] = VertexClass.LOCKED

        used = np.zeros(self.vertex_count, dtype=bool)
        used[faces.reshape(-1)] = True
        kinds[~used] = VertexClass.LOCKED

        if lock_boundary:
            kinds[kinds == VertexClass.BORDER] = VertexClass.LOCKED
        if locked is not None:
            kinds[np.asarray(locked, dtype=np.int64)] = VertexClass.LOCKED
        return kinds


def _pair_keys(pairs: IndexArray, span: int) -> IndexArray:
    """Two index columns as one integer each, so a sort is a sort of ``int64``.

    ``np.unique(..., axis=0)`` views every row as a void and compares it byte by
    byte, which on the millions of edges a scan carries costs several times what
    an ordinary integer sort does. ``span`` has to exceed anything in the second
    column for the encoding to be reversible.
    """
    return np.asarray(pairs[:, 0], dtype=np.int64) * span + pairs[:, 1]


def _unique_pairs(pairs: IndexArray, span: int) -> tuple[IndexArray, np.ndarray]:
    """The distinct rows of a two-column index array, and how often each occurs."""
    keys, counts = np.unique(_pair_keys(pairs, span), return_counts=True)
    return np.stack([keys // span, keys % span], axis=1), counts


def _paired_half_edges(faces: IndexArray) -> tuple[IndexArray, IndexArray]:
    """For every edge with exactly two faces, the corners each face puts on it.

    A corner is identified by ``face * 3 + slot``. Returns two arrays naming
    corners that sit at the same point on the same edge, so joining each pair
    walks a fan from one face to the next.
    """
    count = len(faces)
    slot = np.arange(3, dtype=np.int64)
    following = np.asarray([1, 2, 0], dtype=np.int64)
    here = faces.reshape(-1)
    there = faces[:, following].reshape(-1)
    corner_here = (np.arange(count, dtype=np.int64)[:, None] * 3 + slot).reshape(-1)
    corner_there = (np.arange(count, dtype=np.int64)[:, None] * 3 + following).reshape(-1)

    # Orient every half-edge the same way, so the two that share an edge name
    # their corners in the same order and can be paired off directly.
    backwards = here > there
    low = np.where(backwards, there, here)
    high = np.where(backwards, here, there)
    corner_low = np.where(backwards, corner_there, corner_here)
    corner_high = np.where(backwards, corner_here, corner_there)

    order = np.lexsort((high, low))
    opens = np.empty(len(order), dtype=bool)
    opens[0] = True
    opens[1:] = (low[order][1:] != low[order][:-1]) | (high[order][1:] != high[order][:-1])
    sizes = np.bincount(np.cumsum(opens) - 1)
    starts = np.flatnonzero(opens)[sizes == 2]

    first, second = order[starts], order[starts + 1]
    return (
        np.concatenate([corner_low[first], corner_high[first]]),
        np.concatenate([corner_low[second], corner_high[second]]),
    )


def _connected(links_from: IndexArray, links_to: IndexArray, count: int) -> IndexArray:
    """Label every item by its connected component, as whole-array rounds.

    Hooking each end of a link onto the lower of the two labels and then
    pointer-jumping until nothing moves. A fan is a handful of faces, so the
    components here are tiny and this settles in a few rounds whatever the size
    of the mesh.
    """
    label = np.arange(count, dtype=np.int64)
    while True:
        lowest = np.minimum(label[links_from], label[links_to])
        np.minimum.at(label, links_from, lowest)
        np.minimum.at(label, links_to, lowest)
        while True:
            jumped = label[label]
            if np.array_equal(jumped, label):
                break
            label = jumped
        if np.array_equal(label[links_from], label[links_to]):
            return label


def _multi_fan_points(faces: IndexArray) -> IndexArray:
    """Points whose faces form more than one fan -- a bowtie.

    Corners are joined across every interior edge; a point whose corners end up
    in more than one group is being shared by sheets that only touch. Counting
    faces and edges cannot answer this -- two closed fans around one point have
    as many of each as one does -- so the fans are actually walked.
    """
    label = _connected(*_paired_half_edges(faces), count=faces.size)
    fans = np.unique(_pair_keys(np.stack([faces.reshape(-1), label], axis=1), faces.size))
    return np.flatnonzero(np.bincount(fans // faces.size) > 1)


def build(positions: FloatArray, indices: IndexArray, tolerance: float = 0.0) -> Topology:
    """A :class:`Topology` from the arrays a caller holds.

    Vertices sharing a position are welded, and a triangle left with a repeated
    corner by that welding is dropped: it covers no area, so it describes no
    surface and its plane is undefined.
    """
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise DecimateError('positions must be (n, 3), got %r' % (positions.shape,))
    flat = np.asarray(indices).reshape(-1)
    if len(flat) % 3:
        raise DecimateError('indices must be a multiple of three, got %d' % (len(flat),))
    corners = flat.reshape(-1, 3).astype(np.int64)
    if len(flat) and (int(corners.min()) < 0 or int(corners.max()) >= len(positions)):
        raise DecimateError(
            'index out of range: %d vertices, indices up to %d'
            % (len(positions), int(corners.max()))
        )

    points, vertex_point = weld_positions(positions, tolerance)
    faces = vertex_point[corners]
    usable = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    return Topology(points, faces[usable], corners[usable], vertex_point)
