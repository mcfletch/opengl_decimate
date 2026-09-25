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

import math
from enum import IntEnum

import numpy as np

from opengl_decimate.corners import copies_per_point
from opengl_decimate.spatial import NEIGHBOURHOOD, CellGrid
from opengl_decimate.types import POSITION, AttributeMap, DecimateError, FloatArray, IndexArray

__all__ = [
    'VertexClass',
    'Topology',
    'build',
    'positions_of',
    'atlas_charts',
    'components',
    'smoothing_groups',
    'weld_positions',
    'local_origin',
]


class VertexClass(IntEnum):
    """What a position is allowed to do during a reduction."""

    MANIFOLD = 0
    BORDER = 1
    LOCKED = 2


#: The cell offsets a partner can be in, taken once per unordered pair of cells:
#: the lexicographically positive half of the twenty-seven. A point's own cell is
#: handled separately, with a ``left < right`` filter.
_HALF_NEIGHBOURHOOD = tuple(offset for offset in NEIGHBOURHOOD if offset > (0, 0, 0))


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

    Welding is transitive, so a run of points each near the next becomes one
    even where its ends are further apart than the tolerance -- which is what a
    seam reconstructed as a smear of points needs.

    **Pick a tolerance smaller than the typical spacing between points.** The
    work is the number of pairs that share a cell, so a tolerance that sweeps a
    whole cluster into one cell costs the square of that cluster's size.

    >>> import numpy as np
    >>> points, belongs = weld_positions(np.array([[1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]]))
    >>> belongs.tolist()
    [0, 1, 0]
    """
    positions = np.asarray(positions, dtype='d')
    if tolerance <= 0.0:
        _, first, inverse = np.unique(positions, axis=0, return_index=True, return_inverse=True)
        return _by_first_appearance(positions, inverse.reshape(-1), len(first))

    left, right = _close_pairs(positions, tolerance)
    roots = _connected(left, right, len(positions))
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    return _by_first_appearance(positions, inverse.reshape(-1), len(unique_roots), average=True)


def _cells_of(positions: FloatArray, tolerance: float) -> IndexArray:
    """Which cell of side ``tolerance`` each point falls in.

    A tolerance so fine against these coordinates that the cell number will not
    fit in an ``int64`` is refused rather than wrapped: it is also far below the
    spacing ``float64`` can represent there, so nothing it could have merged is
    distinguishable in the first place.
    """
    scaled = np.floor(positions / tolerance)
    if not np.all(np.abs(scaled) < 2.0**62):
        raise DecimateError(
            'weld_tolerance %r is too fine for coordinates up to %r'
            % (tolerance, float(np.abs(positions).max()))
        )
    return scaled.astype(np.int64)


def _close_pairs(positions: FloatArray, tolerance: float) -> tuple[IndexArray, IndexArray]:
    """Index pairs naming points within ``tolerance`` of each other.

    Points are bucketed into cells of side ``tolerance``, so a partner can only
    be in one of the twenty-seven cells around a point's own. Each of those is
    one lookup into the occupied cells and one gather of that cell's members, so
    every comparison is whole-array: the work is still the number of candidate
    pairs, but the constant is NumPy's rather than Python's.
    """
    empty = np.zeros(0, dtype=np.int64)
    if not len(positions):
        return empty, empty

    cells = _cells_of(positions, tolerance)
    grid = CellGrid(cells)
    limit = tolerance * tolerance
    found: list[tuple[IndexArray, IndexArray]] = []
    for offset in ((0, 0, 0),) + _HALF_NEIGHBOURHOOD:
        for left, right in grid.members(cells + np.asarray(offset, dtype=np.int64)):
            if offset == (0, 0, 0):
                # A point's own cell offers every pair twice and every point
                # against itself; the other offsets name each cell pair once.
                keep = left < right
                left, right = left[keep], right[keep]
            delta = positions[left] - positions[right]
            close = np.einsum('ij,ij->i', delta, delta) <= limit
            if np.any(close):
                found.append((left[close], right[close]))

    if not found:
        return empty, empty
    return (
        np.concatenate([pair[0] for pair in found]),
        np.concatenate([pair[1] for pair in found]),
    )


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


def local_origin(points: FloatArray) -> FloatArray:
    """Where to put the origin so a quadric keeps its digits.

    A plane quadric's constant term is ``(n.p)**2``, taken from the point's
    coordinates as they stand. A model at Earth-centred coordinates -- ``p``
    around 6.4e6, which is what 3D Tiles and any ECEF-referenced world uses --
    gives a constant around 4e13, while the squared distance the quadric exists
    to report is around 1e-6. That is nineteen decimal digits of range inside a
    type that carries sixteen, and the cancellation takes the answer with it:
    at that scale the metric's own noise floor is 0.147 model units, which is
    larger than most level-of-detail budgets before a triangle has been removed.

    Working about the model's own bounding-box centre costs one subtraction per
    point and puts the whole computation back where ``float64`` has the digits.

    The shift is taken on an axis only where the model sits at least twice its
    own half-extent from the origin. That is where absolute coordinates cost
    something, and it is also exactly where subtracting the centre and adding it
    back are both exact operations -- Sterbenz's lemma gives ``p - o`` exactly
    for ``o/2 <= p <= 2o``, and the sum of an exact difference and ``o`` is
    ``p``, which is representable. So a model near the origin is left where it
    is rather than moved by a rounding on account of a problem it does not have.

    >>> import numpy as np
    >>> local_origin(np.array([[-1.0, 0, 0], [1.0, 2.0, 0]])).tolist()
    [0.0, 0.0, 0.0]
    >>> local_origin(np.array([[6.4e6, 0, 0], [6.4e6 + 1.0, 0, 0]])).tolist()
    [6400000.5, 0.0, 0.0]
    """
    points = np.asarray(points, dtype='d')
    if not len(points):
        return np.zeros(3, dtype='d')
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    centre = 0.5 * (low + high)
    half_extent = 0.5 * (high - low)
    return np.where(np.abs(centre) >= 2.0 * half_extent, centre, 0.0)


class Topology:
    """A welded surface, its adjacency, and the faces still alive on it.

    ``positions`` are the welded points and are moved in place by a collapse;
    ``faces`` holds position indices and ``corners`` the vertex each of those
    came from, so an attribute follows its corner rather than its point.

    Positions are held **relative to** ``origin``, which :func:`build` takes
    from the model's own bounding box -- see :func:`local_origin` for why, and
    for the rule that leaves it at zero for a model near the origin. Adding
    ``origin`` back is what
    :meth:`~opengl_decimate.sequence.CollapseSequence.state` does before handing
    positions to a caller.
    """

    def __init__(
        self,
        positions: FloatArray,
        faces: IndexArray,
        corners: IndexArray,
        vertex_point: IndexArray,
        origin: FloatArray | None = None,
        input_faces: int | None = None,
        dropped_faces: int = 0,
        copies: IndexArray | None = None,
        charts: IndexArray | None = None,
    ) -> None:
        self.positions = np.ascontiguousarray(positions, dtype='d')
        #: What ``positions`` are measured from; zero for a model near it.
        self.origin = np.zeros(3, dtype='d') if origin is None else np.asarray(origin, dtype='d')
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.corners = np.ascontiguousarray(corners, dtype=np.int64)
        #: Which welded point each *input* vertex went to.
        self.vertex_point = np.ascontiguousarray(vertex_point, dtype=np.int64)
        #: How many different sets of carried values are drawn at each point --
        #: see :func:`~opengl_decimate.corners.copies_per_point`. One
        #: everywhere for a mesh of positions alone, which is what a caller
        #: naming no attributes gets.
        self.copies = (
            np.ones(len(self.positions), dtype=np.int32)
            if copies is None
            else np.ascontiguousarray(copies, dtype=np.int32)
        )
        #: Which piece of the texture atlas each face is laid out in -- see
        #: :func:`atlas_charts`. One chart for a mesh whose seams were not
        #: asked about, so no edge is between two.
        self.charts = (
            np.zeros(len(self.faces), dtype=np.int64)
            if charts is None
            else np.ascontiguousarray(charts, dtype=np.int64)
        )
        #: Triangles the caller handed in, before welding dropped any. A ratio
        #: is a share of these, which is what a caller counted; ``face_count``
        #: is what is left to work on.
        self.input_faces = len(self.faces) if input_faces is None else int(input_faces)
        #: Input triangles removed for belonging to a component too small to be
        #: worth the triangles -- see ``drop_below`` on :func:`build`.
        self.dropped_faces = int(dropped_faces)
        #: False once a collapse has removed the face.
        self.alive = np.ones(len(self.faces), dtype=bool)
        # Kept as a running total rather than counted on demand: a reduction
        # asks how many faces are left once per contraction, and scanning a
        # million flags to answer costs more than the contraction does.
        self._alive_count = len(self.faces)
        self._vertex_faces: list[set[int]] | None = None

    @property
    def adjacency_built(self) -> bool:
        """Whether the per-point face sets have been built yet."""
        return self._vertex_faces is not None

    @property
    def vertex_faces(self) -> list[set[int]]:
        """The live faces on each point, built the first time it is asked for.

        A set per point is the largest thing this object holds -- far larger
        than the arrays it describes, since a scan has hundreds of thousands of
        points and a Python set costs hundreds of bytes empty. Welding,
        classifying and any schedule that works in whole arrays need none of it,
        so it is not built until something walks the surface point by point.
        """
        if self._vertex_faces is None:
            built: list[set[int]] = [set() for _ in range(len(self.positions))]
            for index in np.flatnonzero(self.alive):
                for point in self.faces[index]:
                    built[point].add(int(index))
            self._vertex_faces = built
        return self._vertex_faces

    def forget_adjacency(self) -> None:
        """Drop the face sets, so whole-array work does not carry them.

        A schedule that rewrites the faces in bulk leaves them wrong, and
        rebuilding on the next ask is both cheaper and safer than maintaining
        them through work that does not use them.
        """
        self._vertex_faces = None

    @property
    def vertex_count(self) -> int:
        """How many welded points the surface has."""
        return len(self.positions)

    @property
    def face_count(self) -> int:
        """How many triangles are still alive."""
        return self._alive_count

    def set_live_count(self, count: int) -> None:
        """Tell the mesh how many faces are alive, after a bulk rewrite.

        The compiled reducer flips the flags itself, so the running total it
        left behind has to be handed back rather than inferred.
        """
        self._alive_count = int(count)

    def kill_face(self, face: int) -> None:
        """Remove a face from the live surface and from its points' adjacency.

        The adjacency is only updated where it exists: asking for it here would
        build half a gigabyte of sets on behalf of a caller that never wanted
        them.
        """
        self.alive[face] = False
        self._alive_count -= 1
        if self._vertex_faces is not None:
            for point in self.faces[face]:
                self._vertex_faces[point].discard(face)

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

    def along_a_seam(self, left: int, right: int) -> bool:
        """True where the edge has a face on each side and they are in different charts."""
        on_edge = self.edge_faces(left, right)
        if len(on_edge) != 2:
            return False
        first, second = on_edge
        return bool(self.charts[first] != self.charts[second])

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
        ``locked`` names further points to hold -- by *welded point*, which is
        where the count to check it against lives, and which is why the check is
        here rather than with the rest of the option validation.
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
            held = np.asarray(locked, dtype=np.int64)
            if held.size and int(held.max()) >= self.vertex_count:
                raise DecimateError(
                    'locked index %d is past the end: the mesh has %d welded points'
                    % (int(held.max()), self.vertex_count)
                )
            kinds[held] = VertexClass.LOCKED
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

    Each round points the higher of a link's two **roots** at the lower, then
    pointer-jumps until every item names a root again. Hooking roots rather than
    the items themselves is what makes this settle in a number of rounds that
    grows with the logarithm of the longest chain rather than with its length:
    an item's label already reaches its root, so moving the root moves
    everything under it at once, while moving the item moves only the item and
    the news travels one link per round.

    The difference is the whole cost on anything but a fan. A photogrammetry
    scan's surface is a handful of components over millions of corners, and a
    texture atlas is a hundred or so; those are the chains that are long.
    """
    label = np.arange(count, dtype=np.int64)
    while True:
        here, there = label[links_from], label[links_to]
        higher, lower = np.maximum(here, there), np.minimum(here, there)
        apart = higher != lower
        if not np.any(apart):
            return label
        np.minimum.at(label, higher[apart], lower[apart])
        while True:
            jumped = label[label]
            if np.array_equal(jumped, label):
                break
            label = jumped


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


def smoothing_groups(
    positions: FloatArray,
    faces: IndexArray,
    corners: IndexArray,
    normals: FloatArray | None = None,
    crease_angle: float = 0.0,
) -> IndexArray:
    """Which normal each face corner belongs to, as a label per corner.

    A normal is shared by the corners around a point *up to the edges the model
    is hard across*. Two things say where those are, and either is enough:

    ``normals``
        What the input drew. A hard-edged export carries one position as several
        vertices with several normals, and that is the author saying which edges
        are edges. Corners whose carried normals differ are never accumulated
        together, so the edge stays.
    ``crease_angle``
        The fold itself, in degrees, for a mesh that arrived with no normals or
        with smooth ones over a geometric edge. Two faces meeting at more than
        this are not accumulated together.

    Corners are joined across the interior edges that pass both tests and then
    labelled by connected component, which walks each point's fan and stops
    wherever the fan is hard. Labels index the flat corner array, so a corner's
    label is its place in ``faces.reshape(-1)``.
    """
    links_from, links_to = _paired_half_edges(faces)
    if len(links_from):
        keep = np.ones(len(links_from), dtype=bool)
        if normals is not None:
            carried = np.asarray(normals, dtype='d')[np.asarray(corners).reshape(-1)]
            keep &= np.all(
                np.isclose(carried[links_from], carried[links_to], rtol=0.0, atol=1e-6), axis=1
            )
        if crease_angle > 0.0:
            plane = _face_normals(np.asarray(positions, dtype='d'), faces)
            # A corner names its face: the flat index divided by three.
            meeting = np.einsum('ij,ij->i', plane[links_from // 3], plane[links_to // 3])
            keep &= meeting >= math.cos(math.radians(crease_angle))
        links_from, links_to = links_from[keep], links_to[keep]
    return _connected(links_from, links_to, count=faces.size)


def atlas_charts(faces: IndexArray, corners: IndexArray, texcoords: FloatArray) -> IndexArray:
    """Which piece of the texture atlas each face is laid out in.

    A chart is a connected region of the surface that was unwrapped as one, and
    a texture coordinate is only meaningful inside one of them: across a chart
    boundary the two sides are at unrelated places in the image. So the faces
    are joined across every interior edge whose two sides agree on where they
    are in the texture, and labelled by connected component.

    What that measures is how far a model can be reduced before its texture
    stops being able to describe it. An output triangle covering more surface
    than a chart holds has no coordinate that fits it, whatever a reducer does
    with the ones it inherited.
    """
    if not len(faces):
        return np.zeros(0, dtype=np.int64)
    own = np.arange(len(faces), dtype=np.int64) * 3
    # A face is never cut: its own three corners are always one chart.
    inner_from = np.concatenate([own, own + 1])
    inner_to = np.concatenate([own + 1, own + 2])
    here, there = _paired_half_edges(faces)
    at = np.asarray(texcoords, dtype='d').reshape(len(np.asarray(texcoords)), -1)
    at = at[np.asarray(corners, dtype=np.int64).reshape(-1)]
    shared = np.all(at[here] == at[there], axis=1)
    labels = _connected(
        np.concatenate([inner_from, here[shared]]),
        np.concatenate([inner_to, there[shared]]),
        count=faces.size,
    )
    return labels[own]


def _face_normals(positions: FloatArray, faces: IndexArray) -> FloatArray:
    """Unit normal of each face; zero where it has no area to take one from."""
    corners = positions[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(normal, axis=1)
    usable = length > 0.0
    normal[usable] /= length[usable][:, None]
    normal[~usable] = 0.0
    return normal


def components(faces: IndexArray, count: int) -> IndexArray:
    """Which connected surface each point belongs to, as a label per point.

    Two points are in the same component where a chain of triangles joins them.
    A point no triangle uses keeps a label of its own, which costs nothing and
    means the answer needs no separate account of what is used.

    This is what a reduction's floor is usually about. Every component reduces
    independently and each has a floor of its own -- a closed shell cannot go
    below four triangles -- so a scan that arrived with the subject and two
    hundred crumbs spends four triangles on each crumb however coarse a target
    it is given. :func:`build` takes ``drop_below`` for that reason.

    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [3, 4, 5]])
    >>> components(faces, 6).tolist()
    [0, 0, 0, 3, 3, 3]
    """
    if not len(faces):
        return np.arange(count, dtype=np.int64)
    faces = np.asarray(faces).reshape(-1, 3)
    # All three edges of every face at once. Hooking on one edge at a time would
    # join only the pairs that edge happens to touch.
    here = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    there = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return _connected(here, there, count)


def _too_small(points: FloatArray, faces: IndexArray, share: float) -> np.ndarray:
    """Which faces belong to a component smaller than ``share`` of the model.

    Size is the component's own bounding-box diagonal against the whole model's,
    so the answer means the same thing whatever units the model is in. It is
    also the measure that maps onto the screen: a component at one per cent of
    the model's diagonal covers about one pixel where the model covers a hundred.

    The largest component is never among them. A share above one would otherwise
    take the whole model, and a caller who asked for too much wants the subject
    back rather than an empty mesh.
    """
    label = components(faces, len(points))
    low = np.full((len(points), 3), np.inf)
    high = np.full((len(points), 3), -np.inf)
    np.minimum.at(low, label, points)
    np.maximum.at(high, label, points)
    with np.errstate(invalid='ignore'):
        reach = np.linalg.norm(high - low, axis=1)
    reach[~np.isfinite(reach)] = 0.0
    whole = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
    if not whole > 0.0:
        return np.zeros(len(faces), dtype=bool)
    per_face = label[faces[:, 0]]
    biggest = int(np.argmax(reach))
    return (reach[per_face] < share * whole) & (per_face != biggest)


def positions_of(attributes: AttributeMap) -> FloatArray:
    """The positions, once the attribute set has been shown to be a mesh.

    ``POSITION`` has to be there, and every attribute has to be one row per
    vertex, with as many rows as ``POSITION``.
    """
    if POSITION not in attributes:
        raise DecimateError("attributes must include 'POSITION'")
    positions = np.asarray(attributes[POSITION])
    for name, value in attributes.items():
        rows = np.asarray(value)
        if not rows.ndim:
            raise DecimateError('%s is a single value, not one row per vertex' % (name,))
        if len(rows) != len(positions):
            raise DecimateError(
                '%s has %d rows, POSITION has %d' % (name, len(rows), len(positions))
            )
    return positions


def build(
    positions: FloatArray,
    indices: IndexArray,
    tolerance: float = 0.0,
    drop_below: float = 0.0,
    carried: dict[str, np.ndarray] | None = None,
) -> Topology:
    """A :class:`Topology` from the arrays a caller holds.

    Vertices sharing a position are welded, and a triangle left with a repeated
    corner by that welding is dropped: it covers no area, so it describes no
    surface and its plane is undefined.

    The welded points are held relative to :func:`local_origin`, so a model far
    from the origin is measured where ``float64`` still has the digits to
    measure it in.

    ``drop_below`` removes whole connected components smaller than that share of
    the model's bounding-box diagonal, before any reduction sees them. A scan
    arrives with the subject and whatever else was in the room, and every crumb
    is a closed shell with a floor of four triangles -- so a target of a few
    hundred triangles is spent on crumbs unless they go. ``0.01`` is about one
    pixel where the whole model covers a hundred. The largest component is never
    dropped, whatever share is asked for.

    ``carried`` is what the mesh's vertices hold besides their positions. Its
    texture coordinates are read to count how many different ones each point is
    drawn with, and to find the chart of the atlas each face is in, which is
    where the surface's seams are; without it every point counts as drawn once,
    every face is in one chart, and a reduction is free to collapse across them.
    """
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise DecimateError('positions must be (n, 3), got %r' % (positions.shape,))
    # A NaN or an infinity here is accepted by every operation downstream and
    # comes out the far end in the result, with a reported error of zero, so it
    # has to be refused where it arrives.
    finite = np.isfinite(positions).all(axis=1)
    if not finite.all():
        raise DecimateError(
            'positions must all be finite: row %d is %r'
            % (int(np.argmin(finite)), positions[int(np.argmin(finite))].tolist())
        )
    flat = np.asarray(indices).reshape(-1)
    if len(flat) and flat.dtype.kind not in 'iu':
        raise DecimateError('indices must be integers, got %s' % (flat.dtype,))
    if len(flat) % 3:
        raise DecimateError('indices must be a multiple of three, got %d' % (len(flat),))
    corners = flat.reshape(-1, 3).astype(np.int64)
    if len(flat) and (int(corners.min()) < 0 or int(corners.max()) >= len(positions)):
        raise DecimateError(
            'index out of range: %d vertices, indices up to %d'
            % (len(positions), int(corners.max()))
        )

    points, vertex_point = weld_positions(positions, tolerance)
    origin = local_origin(points)
    faces = vertex_point[corners]
    usable = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    faces, corners = faces[usable], corners[usable]
    dropped = 0
    if drop_below > 0.0 and len(faces):
        small = _too_small(points, faces, drop_below)
        if np.any(small):
            dropped = int(np.count_nonzero(small))
            faces, corners = faces[~small], corners[~small]
    copies = charts = None
    if carried is not None:
        copies = copies_per_point(vertex_point, corners, carried, len(points))
        mapped = [
            np.asarray(value, dtype='d').reshape(len(positions), -1)
            for name, value in sorted(carried.items())
            if name.startswith('TEXCOORD')
        ]
        if mapped:
            charts = atlas_charts(faces, corners, np.concatenate(mapped, axis=1))
    return Topology(
        points - origin,
        faces,
        corners,
        vertex_point,
        origin,
        input_faces=len(flat) // 3,
        dropped_faces=dropped,
        copies=copies,
        charts=charts,
    )
