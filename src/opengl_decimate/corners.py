"""Which vertex a corner reads its attributes from, once its point has moved.

A contraction merges two points into one. The triangles joined to both are
handed to the point that remains, and their corners have to be handed over with
them: a corner drawn at the merged point but still reading the texture
coordinate of a vertex the collapse left behind samples the texture from where
that vertex used to be, and a corner still reading its normal shades as though
the surface had not moved. Both grow with the reduction, because the mistake
compounds -- a triangle can end up drawn across half a model while sampling a
patch the size it started as.

So every contraction rewrites corners as well as faces, and the rule is the one
the rest of the package follows: a value is *carried*, never invented. The
corner takes up a vertex that was already on the surface, and its attributes are
the ones measured there.

Two questions decide which vertex. The first is which end of the edge the
merged point is at. The placement is somewhere on or near the edge, and the two ends were measured in different places, so the
copies to keep are the ones measured nearest to where the surface now is. Which
end that is has little to do with which *index* survived -- the placement is
chosen from the edge before the legality tests say which end may die -- so it is
asked of the geometry. The losing end's copies are let go, and that end's index
never sources a corner again, which keeps the answer a single hop per vertex for
the whole reduction.

The second is which copy, where the winning end has several. A point on a
texture seam is one position drawn twice, once per chart, and the copies differ only in what
they carry. The copy nearest in attribute space is the copy on the same side of
the seam, so choosing it keeps each chart's triangles reading from their own
chart and a seam stays a seam rather than becoming a smear across the texture.

The moves are recorded rather than applied, in the same shape as the
contractions themselves, so a prefix of the reduction is replayed by
pointer-jumping them exactly as
:meth:`~opengl_decimate.sequence.CollapseSequence.state` pointer-jumps points.
"""

from __future__ import annotations

import numpy as np

from opengl_decimate.types import POSITION, FloatArray, IndexArray

__all__ = ['corner_moves', 'copies_per_point', 'follow']

#: Values in one block of the candidate match -- corners times copies times the
#: width of a row -- so a model whose points carry many copies cannot ask for an
#: arbitrarily large intermediate. Eight bytes each, so this is about 4 MB.
_BLOCK = 1 << 19


def corner_moves(
    points: FloatArray,
    vertex_point: IndexArray,
    corners: IndexArray,
    attributes: dict[str, np.ndarray],
    dying: IndexArray,
    surviving: IndexArray,
    placement: FloatArray,
) -> tuple[IndexArray, IndexArray, IndexArray]:
    """Where each contraction sends the corners of the end it gives up.

    ``points`` and ``placement`` are in the sequence's own coordinates, which is
    all this needs: every comparison here is between two of them.

    Returns ``(moved, moved_to, step_start)``: the vertices whose corners change
    hands, the vertex each hands over to, and where each contraction's moves
    begin -- so the moves belonging to the first ``n`` contractions are the first
    ``step_start[n]`` entries, which is what makes a prefix replayable. Every
    vertex appears in ``moved`` at most once.
    """
    vertex_point = np.asarray(vertex_point, dtype=np.int64)
    dying = np.asarray(dying, dtype=np.int64)
    surviving = np.asarray(surviving, dtype=np.int64)
    point_count = int(vertex_point.max()) + 1 if len(vertex_point) else 0

    used = np.zeros(len(vertex_point), dtype=bool)
    used[np.asarray(corners, dtype=np.int64).reshape(-1)] = True
    order, start = _copies_by_point(vertex_point, used, point_count)
    counts = np.diff(start)

    loser, winner = _ends_given_up(
        np.asarray(points, dtype='d'),
        np.asarray(placement, dtype='d'),
        dying,
        surviving,
        copies_per_point(vertex_point, corners, attributes, point_count),
        point_count,
    )

    per_step = counts[loser] if len(loser) else np.zeros(0, dtype=np.int64)
    step_start = np.concatenate([[0], np.cumsum(per_step)]).astype(np.int64)
    total = int(step_start[-1])
    if not total:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), step_start

    # The copies of every end given up, concatenated: a flat run per step, read
    # straight out of `order` rather than gathered a step at a time.
    within = np.arange(total, dtype=np.int64) - np.repeat(step_start[:-1], per_step)
    moved = order[np.repeat(start[loser], per_step) + within]

    # The winning end's first copy is the answer wherever it is its only one,
    # which is every point off a seam.
    kept = np.repeat(winner, per_step)
    moved_to = order[start[kept]]

    key = _attribute_key(attributes, len(vertex_point))
    split = np.flatnonzero(counts[kept] > 1)
    if key is not None and len(split):
        # Each corner is compared against a row padded out to the widest point
        # in its block, so the block is sized in values rather than in rows: a
        # model that draws one position a hundred times would otherwise ask for
        # a hundred times the intermediate.
        widest = max(1, int(counts[kept[split]].max()) * key.shape[1])
        block = max(1, _BLOCK // widest)
        for first in range(0, len(split), block):
            rows = split[first : first + block]
            moved_to[rows] = _nearest_copy(key, order, start, kept[rows], moved[rows])

    return moved, moved_to, step_start


def copies_per_point(
    vertex_point: IndexArray,
    corners: IndexArray,
    attributes: dict[str, np.ndarray],
    point_count: int,
) -> IndexArray:
    """How many different texture coordinates are drawn at each point.

    One almost everywhere; two where a seam runs, because a chart boundary is
    one position the model draws twice, once at the end of one chart and once at
    the start of the next. Vertices carrying the same coordinate count once
    however many times an exporter wrote them out, and vertices no triangle uses
    do not count at all.

    Texture coordinates and not the rest of what a corner carries, because they
    are the values that **jump**. A normal splits at a crease and the two sides
    differ by an angle; taking the nearer of them is a small error in a
    direction. A chart boundary puts the two sides at unrelated ends of the
    texture, and taking either draws a band of the whole image across the
    triangles on the other side. So this is what
    :func:`~opengl_decimate.collapse.is_legal` holds: a point drawn at several
    coordinates dies only into another drawn at the same number, which shortens
    a seam along its own length and never lets it wander off into a chart. A
    crease is left to the metric, which is what it is for.
    """
    counted = np.zeros(point_count, dtype=np.int64)
    used = np.unique(np.asarray(corners, dtype=np.int64).reshape(-1))
    if not len(used):
        return counted
    mapped = {name: value for name, value in attributes.items() if name.startswith('TEXCOORD')}
    key = _attribute_key(mapped, len(np.asarray(vertex_point)))
    point = np.asarray(vertex_point, dtype=np.int64)[used]
    if key is None:
        counted[np.unique(point)] = 1
        return counted
    rows = np.concatenate([point[:, None].astype('d'), key[used]], axis=1)
    distinct = np.unique(rows, axis=0)[:, 0].astype(np.int64)
    counted[distinct] = np.bincount(distinct, minlength=point_count)[distinct]
    return counted


def follow(moved: IndexArray, moved_to: IndexArray, vertex_count: int, upto: int) -> IndexArray:
    """Which vertex each vertex's corners have become after ``upto`` moves.

    The moves chain -- a corner handed to a copy whose own end is given up later
    is handed on again -- so the map is closed by pointer-jumping, as the points'
    own map is. No vertex is written twice, so the order they are applied in does
    not come into it.
    """
    roots = np.arange(vertex_count, dtype=np.int64)
    roots[moved[:upto]] = moved_to[:upto]
    while True:
        jumped = roots[roots]
        if np.array_equal(jumped, roots):
            return roots
        roots = jumped


def _copies_by_point(
    vertex_point: IndexArray, used: np.ndarray, point_count: int
) -> tuple[IndexArray, IndexArray]:
    """The copies at each point, as a flat array and a start index per point.

    Only vertices some triangle uses are listed. A model can carry vertices no
    face mentions -- a duplicate an exporter left behind -- and handing a corner
    to one would give it attributes nothing was ever drawn with.
    """
    order = np.argsort(np.where(used, vertex_point, point_count), kind='stable')
    order = order[: int(used.sum())].astype(np.int64)
    counts = np.bincount(vertex_point[used], minlength=point_count)
    return order, np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def _ends_given_up(
    points: FloatArray,
    placement: FloatArray,
    dying: IndexArray,
    surviving: IndexArray,
    charts: IndexArray,
    point_count: int,
) -> tuple[IndexArray, IndexArray]:
    """:func:`ends_given_up_in_python`, through the compiled reducer where it built."""
    # Imported here: `native` imports `topology`, which imports this module.
    from opengl_decimate import native  # noqa: PLC0415 cycle native -> topology -> corners

    if native.ACCELERATED:
        return native.ends_given_up(points, placement, dying, surviving, charts, point_count)
    return ends_given_up_in_python(points, placement, dying, surviving, charts, point_count)


def ends_given_up_in_python(
    points: FloatArray,
    placement: FloatArray,
    dying: IndexArray,
    surviving: IndexArray,
    charts: IndexArray,
    point_count: int,
) -> tuple[IndexArray, IndexArray]:
    """For each contraction, which end's copies are let go and which are kept.

    A point's copies are the copies of whichever end has won every merge it has
    been through, which is what ``sourced`` tracks -- so the comparison is
    against the place those copies were *measured*, not against where the point
    that owns them has since been moved to.

    The end drawn at **more** texture coordinates wins outright, because it is
    the only one with a coordinate to give each side of the seam it is on.
    Letting distance decide there would hand a whole chart's triangles the
    coordinate of the chart next door. Distance decides between ends drawn at
    the same number, which is every edge that is not on a seam.

    The loop is the reduction's own order: a contraction's answer depends on what
    the contractions before it decided, which is the same reason the reduction
    itself is sequential. ``_reduce_native.ends_given_up`` is the same loop in
    C, and the two are held to the same answers.
    """
    sourced = list(range(point_count))
    drawn = charts.tolist()
    coordinate = points.reshape(-1).tolist()
    target = placement.reshape(-1).tolist()
    steps = len(dying)
    loser = [0] * steps
    winner = [0] * steps
    for step, (dies, lives) in enumerate(zip(dying.tolist(), surviving.tolist(), strict=True)):
        from_dying, from_surviving = sourced[dies], sourced[lives]
        richer = drawn[from_dying] - drawn[from_surviving]
        if richer:
            nearer = richer > 0
        else:
            at = step * 3
            x, y, z = target[at], target[at + 1], target[at + 2]
            near, far = from_dying * 3, from_surviving * 3
            # A tie is the midpoint of an edge whose ends were measured equally
            # far away, where either answer is as good; keeping the surviving
            # end's copies is the one that moves fewer corners.
            nearer = (
                (x - coordinate[near]) ** 2
                + (y - coordinate[near + 1]) ** 2
                + (z - coordinate[near + 2]) ** 2
            ) < (
                (x - coordinate[far]) ** 2
                + (y - coordinate[far + 1]) ** 2
                + (z - coordinate[far + 2]) ** 2
            )
        keep, give_up = (from_dying, from_surviving) if nearer else (from_surviving, from_dying)
        sourced[lives] = keep
        winner[step] = keep
        loser[step] = give_up
    return np.asarray(loser, dtype=np.int64), np.asarray(winner, dtype=np.int64)


def _attribute_key(attributes: dict[str, np.ndarray], vertex_count: int) -> np.ndarray | None:
    """Every vertex's carried values as one row, each column scaled to its range.

    Scaling is what makes a texture coordinate and a normal comparable: without
    it whichever attribute happens to use the larger numbers would decide which
    side of a seam a corner is on. ``None`` where the mesh carries nothing but
    positions, which leaves no choice to make.
    """
    columns = []
    for name in sorted(attributes):
        if name == POSITION:
            continue
        values = np.asarray(attributes[name], dtype='d').reshape(vertex_count, -1)
        span = np.ptp(values, axis=0)
        columns.append(np.where(span > 0.0, values / np.where(span > 0.0, span, 1.0), 0.0))
    return np.concatenate(columns, axis=1) if columns else None


def _nearest_copy(
    key: np.ndarray,
    order: IndexArray,
    start: IndexArray,
    kept: IndexArray,
    moved: IndexArray,
) -> IndexArray:
    """Of the copies at each kept point, the one carrying the nearest values.

    The candidate lists are ragged, so they are padded out to the longest by
    repeating each point's last copy. A repeat can only tie with itself, so it
    never wins a comparison it should have lost.
    """
    counts = start[kept + 1] - start[kept]
    reach = np.minimum(np.arange(int(counts.max()), dtype=np.int64)[None, :], (counts - 1)[:, None])
    candidates = order[start[kept][:, None] + reach]
    apart = key[candidates] - key[moved][:, None, :]
    return candidates[np.arange(len(moved)), np.argmin(np.einsum('ijk,ijk->ij', apart, apart), 1)]
