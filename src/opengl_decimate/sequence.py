"""The whole reduction, recorded once, and any point along it reached by replay.

A decimation is an ordered list of contractions. Recording that list rather than
only its outcome is what lets an editor's target slider answer immediately: the
mesh at any triangle count is a *prefix* of the list, and applying a prefix is
four array operations rather than a re-run.

- Which point a point ended up as: ``parent`` filled from the prefix, then
  pointer-jumped until it stops changing.
- Where each surviving point sits: the placement of the last contraction in the
  prefix that it survived, gathered with one scatter.
- Which faces are left: each face records the step that removed it, so the live
  set is a comparison.
- Which vertex each corner now reads its attributes from: the corner moves
  recorded alongside the contractions, pointer-jumped the same way.

So a target costs a *replay*, not a reduction: a few milliseconds against the
tens the reduction itself took, and the same few whether the sequence being
replayed over is a thousand contractions long or fifty thousand. What it does
scale with is the mesh it hands back, since that has to be assembled -- asking
for twenty thousand triangles costs a good deal more than asking for twenty.
Fifty asks cost fifty replays, which is still a fraction of one reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from opengl_decimate import topology
from opengl_decimate.corners import corner_moves, follow
from opengl_decimate.types import POSITION, FloatArray, IndexArray

__all__ = ['SimplifyResult', 'CollapseSequence']


@dataclass
class SimplifyResult:
    """A decimated mesh, in the form it was handed over in.

    ``attributes`` and ``indices`` are what a renderer wants. ``vertex_map``
    says, for every vertex of the *input*, which output vertex it became or was
    merged into -- the correspondence a geomorph needs to lerp one level toward
    the next -- and is ``-1`` for an input vertex whose surface is gone.
    """

    attributes: dict[str, np.ndarray]
    indices: IndexArray
    error: float
    vertex_map: IndexArray
    collapses: int
    measured_error: float | None = None
    #: Triangles the caller handed in, which is what ``target_ratio`` is a
    #: share of.
    input_triangles: int = 0
    #: Input triangles welding dropped before the reduction began, because
    #: their corners landed on the same welded point and they covered no area.
    #: Zero on a clean export; on scanned or badly-exported data it is the gap
    #: between what the caller counted and what there was to reduce.
    welded_away: int = 0
    #: Input triangles ``drop_components_below`` removed, for belonging to a
    #: piece of the model too small to be worth spending triangles on.
    dropped_away: int = 0

    @property
    def triangle_count(self) -> int:
        """How many triangles came back."""
        return len(self.indices) // 3


@dataclass
class CollapseSequence:
    """Every contraction of a reduction, in the order they were applied.

    Held against the *welded* mesh: ``points`` are the distinct positions of the
    input, ``faces`` index them, and ``corners`` records which input vertex each
    corner came from so its attributes can be handed back. A contraction hands
    the dying point's corners to the surviving point, and those handovers are
    recorded too -- see :mod:`opengl_decimate.corners` -- so a corner's
    attributes belong to the position it is drawn at however far the reduction
    has gone.

    ``points`` and ``placement`` are measured from ``origin``, which
    :func:`~opengl_decimate.topology.local_origin` sets to the model's own
    bounding-box centre where the model is far enough from the world origin for
    absolute coordinates to cost the metric its digits, and to zero otherwise.
    :meth:`state` adds it back.
    """

    points: FloatArray
    faces: IndexArray
    corners: IndexArray
    vertex_point: IndexArray
    attributes: dict[str, np.ndarray]
    dying: IndexArray
    surviving: IndexArray
    placement: FloatArray
    deviation: FloatArray
    removed_at: IndexArray
    recompute_normals: bool = False
    #: Where a recomputed normal keeps an edge hard, in degrees, on top of what
    #: the input's own normals already say. See ``SimplifyOptions``.
    crease_angle: float = 0.0
    origin: FloatArray = field(default_factory=lambda: np.zeros(3, dtype='d'))
    #: Triangles the caller handed in, before welding dropped any. A ratio is a
    #: share of these; ``faces`` is what welding left to reduce.
    input_faces: int | None = None
    #: Triangles removed for belonging to a component too small to be worth them.
    dropped_faces: int = 0
    #: Filled in from the fields above: how many faces the first ``k``
    #: contractions removed, and the corner handovers of
    #: :func:`~opengl_decimate.corners.corner_moves`.
    _removed_by_step: IndexArray = field(init=False, repr=False)
    _moved: IndexArray = field(init=False, repr=False)
    _moved_to: IndexArray = field(init=False, repr=False)
    _moved_by_step: IndexArray = field(init=False, repr=False)

    def __len__(self) -> int:
        """How many contractions were recorded."""
        return len(self.dying)

    def __post_init__(self) -> None:
        if self.input_faces is None:
            self.input_faces = len(self.faces)
        removals = self.removed_at[self.removed_at >= 0]
        per_step = np.bincount(removals, minlength=len(self.dying) + 1)
        self._removed_by_step = np.concatenate([[0], np.cumsum(per_step)])
        self._moved, self._moved_to, self._moved_by_step = corner_moves(
            self.points,
            self.vertex_point,
            self.corners,
            self.attributes,
            self.dying,
            self.surviving,
            self.placement,
        )

    def triangles_after(self, steps: int) -> int:
        """How many triangles are left once ``steps`` contractions are applied."""
        steps = int(np.clip(steps, 0, len(self)))
        return int(len(self.faces) - self._removed_by_step[steps])

    def steps_for(
        self,
        target_count: int | None = None,
        target_ratio: float | None = None,
        target_error: float | None = None,
    ) -> int:
        """The fewest contractions that satisfy every target given.

        With no target the whole sequence is used. Triangle counts fall as the
        sequence is replayed and the deviation of a prefix is the largest
        deviation in it, so both are monotone and each target is one search.

        ``target_ratio`` is a share of ``input_faces`` -- the triangles the
        caller handed in -- rather than of what welding left behind.
        """
        steps = len(self)
        if target_ratio is not None:
            target_count = min(
                target_count if target_count is not None else len(self.faces),
                int(target_ratio * (self.input_faces or len(self.faces))),
            )
        if target_count is not None:
            remaining = len(self.faces) - self._removed_by_step
            steps = min(steps, int(np.searchsorted(-remaining, -target_count)))
        if target_error is not None and len(self):
            reached = np.maximum.accumulate(self.deviation)
            steps = min(steps, int(np.searchsorted(reached, target_error, side='right')))
        return int(np.clip(steps, 0, len(self)))

    def state(self, steps: int) -> tuple[FloatArray, IndexArray, IndexArray, IndexArray]:
        """The mesh after ``steps`` contractions: positions, roots, faces, corners.

        ``roots`` maps every original point to the point it has become, which is
        the correspondence the caller needs to follow a vertex forward.
        Positions come back in the caller's own coordinates, with ``origin``
        added back on.

        The corners come back remapped the same way the faces do, so a corner
        names the vertex whose attributes belong at the position the corner is
        now drawn at rather than the vertex it started as.
        """
        steps = int(np.clip(steps, 0, len(self)))
        roots = np.arange(len(self.points), dtype=np.int64)
        roots[self.dying[:steps]] = self.surviving[:steps]
        while True:
            jumped = roots[roots]
            if np.array_equal(jumped, roots):
                break
            roots = jumped

        positions = self.points.copy()
        if steps:
            last = np.full(len(self.points), -1, dtype=np.int64)
            np.maximum.at(last, self.surviving[:steps], np.arange(steps, dtype=np.int64))
            moved = last >= 0
            positions[moved] = self.placement[last[moved]]

        alive = (self.removed_at < 0) | (self.removed_at >= steps)
        handed = follow(
            self._moved, self._moved_to, len(self.vertex_point), int(self._moved_by_step[steps])
        )
        return (
            positions + self.origin,
            roots,
            roots[self.faces[alive]],
            handed[self.corners[alive]],
        )

    def at(
        self,
        target_count: int | None = None,
        target_ratio: float | None = None,
        target_error: float | None = None,
    ) -> SimplifyResult:
        """The mesh at the first point in the sequence that meets the targets."""
        steps = self.steps_for(target_count, target_ratio, target_error)
        positions, roots, faces, corners = self.state(steps)
        error = float(np.max(self.deviation[:steps])) if steps else 0.0
        return _emit(
            positions=positions,
            roots=roots,
            faces=faces,
            corners=corners,
            attributes=self.attributes,
            vertex_point=self.vertex_point,
            error=error,
            collapses=steps,
            recompute_normals=self.recompute_normals,
            crease_angle=self.crease_angle,
            input_triangles=self.input_faces or len(self.faces),
            welded_away=(
                (self.input_faces or len(self.faces)) - len(self.faces) - self.dropped_faces
            ),
            dropped_away=self.dropped_faces,
        )


def _position_dtype(attributes: dict[str, np.ndarray]) -> np.dtype:
    """What ``POSITION`` comes back as: whatever it arrived as, if that is float.

    A model at Earth-centred coordinates hands in ``float64`` for a reason -- a
    ``float32`` ulp at 6.4e6 is half a metre -- and handing it back narrowed
    would undo the accuracy the reduction was careful about. An integer or
    otherwise unusual ``POSITION`` has no meaningful width to keep, so it comes
    back ``float32``, which is what a vertex buffer wants.
    """
    dtype = np.asarray(attributes[POSITION]).dtype
    return dtype if dtype.kind == 'f' else np.dtype('f4')


def _first_appearance(
    inverse: IndexArray, representative: IndexArray
) -> tuple[IndexArray, IndexArray]:
    """Renumber unique-group labels so group ``k`` is the ``k``-th to appear.

    ``np.unique`` numbers in sorted order, which would make the output vertex
    order depend on coordinate values rather than on the mesh.
    """
    order = np.argsort(representative)
    relabel = np.empty(len(representative), dtype=np.int64)
    relabel[order] = np.arange(len(representative), dtype=np.int64)
    return relabel[inverse], representative[order]


def _emit(
    positions: FloatArray,
    roots: IndexArray,
    faces: IndexArray,
    corners: IndexArray,
    attributes: dict[str, np.ndarray],
    vertex_point: IndexArray,
    error: float,
    collapses: int,
    recompute_normals: bool,
    crease_angle: float,
    input_triangles: int,
    welded_away: int,
    dropped_away: int,
) -> SimplifyResult:
    """Turn the live surface back into vertex arrays and an index array.

    An output vertex is a distinct combination of *point* and *attributes*: two
    corners at the same point carrying the same normal and the same texture
    coordinate are one vertex, and two carrying different ones are two, which is
    what keeps a seam a seam.
    """
    flat_points = faces.reshape(-1)
    flat_corners = corners.reshape(-1)
    place_dtype = _position_dtype(attributes)
    if not len(flat_points):
        empty = {name: value[:0].copy() for name, value in attributes.items()}
        empty[POSITION] = np.zeros((0, 3), dtype=place_dtype)
        if recompute_normals:
            empty['NORMAL'] = np.zeros((0, 3), dtype='f4')
        return SimplifyResult(
            attributes=empty,
            indices=np.zeros((0,), dtype=np.uint32),
            error=error,
            vertex_map=np.full(len(vertex_point), -1, dtype=np.int64),
            collapses=collapses,
            input_triangles=input_triangles,
            welded_away=welded_away,
            dropped_away=dropped_away,
        )

    # Which corners share a normal, where one is to be computed. This is part
    # of what makes a vertex distinct: an edge the model is hard across needs a
    # vertex per side to *be* hard, and on a mesh that arrived carrying no
    # normals there is nothing else to tell the two sides apart.
    group = None
    if recompute_normals:
        group = topology.smoothing_groups(
            positions, faces, corners, attributes.get('NORMAL'), crease_angle
        )

    columns = [flat_points.astype('d')[:, None]]
    carried = [name for name in sorted(attributes) if name != POSITION]
    for name in carried:
        values = np.asarray(attributes[name])[flat_corners]
        columns.append(values.reshape(len(flat_corners), -1).astype('d'))
    if group is not None:
        columns.append(group.astype('d')[:, None])
    key = np.concatenate(columns, axis=1)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    out_index, representative = _first_appearance(inverse.reshape(-1), first)

    out_point = flat_points[representative]
    out_source = flat_corners[representative]
    out_attributes: dict[str, np.ndarray] = {
        POSITION: np.ascontiguousarray(positions[out_point], dtype=place_dtype)
    }
    for name in carried:
        # Gathered from one surviving corner, never blended between two. Every
        # value in the output is therefore a value that was in the input, which
        # is what an attribute whose values are identifiers rather than
        # quantities needs: the mean of joint 3 and joint 9 is joint 6, which is
        # some unrelated bone.
        out_attributes[name] = np.ascontiguousarray(np.asarray(attributes[name])[out_source])

    indices = np.ascontiguousarray(out_index.reshape(-1, 3), dtype=np.uint32).reshape(-1)
    if group is not None:
        out_attributes['NORMAL'] = _surface_normals(positions, faces, group, representative)

    # An input vertex maps to the output vertex that carries its own attributes
    # where one survived, and otherwise to any output vertex at the point it
    # became -- which is still the right place to morph toward.
    preferred = np.full(len(vertex_point), -1, dtype=np.int64)
    preferred[out_source] = np.arange(len(out_source), dtype=np.int64)
    by_point = np.full(len(roots), -1, dtype=np.int64)
    by_point[out_point] = np.arange(len(out_point), dtype=np.int64)
    vertex_map = np.where(preferred >= 0, preferred, by_point[roots[vertex_point]])

    return SimplifyResult(
        attributes=out_attributes,
        indices=indices,
        error=error,
        vertex_map=vertex_map,
        collapses=collapses,
        input_triangles=input_triangles,
        welded_away=welded_away,
        dropped_away=dropped_away,
    )


def _surface_normals(
    positions: FloatArray,
    faces: IndexArray,
    group: IndexArray,
    representative: IndexArray,
) -> FloatArray:
    """Area-weighted vertex normals of the surface as it now stands.

    Accumulated per **smoothing group** -- the corners around a point that the
    model is smooth across -- and then gathered to the output vertices.

    Per *point* would be too coarse: a hard edge is one position the model draws
    twice, once per side, and averaging over it turns a cube into a ball. Per
    *output vertex* would be too fine: an output vertex is split by attributes,
    so a texture seam carrying no geometric meaning at all would come back as a
    crease in the shading. The group is the middle of those two, and
    :func:`~opengl_decimate.topology.smoothing_groups` finds it from the normals
    the input carried and from the folds in the surface itself.
    """
    at = np.asarray(positions, dtype='d')[faces]
    face_normal = np.cross(at[:, 1] - at[:, 0], at[:, 2] - at[:, 0])
    summed = np.zeros((faces.size, 3), dtype='d')
    np.add.at(summed, group, np.repeat(face_normal, 3, axis=0))
    lengths = np.linalg.norm(summed, axis=1)
    usable = lengths > 0.0
    summed[usable] /= lengths[usable][:, None]
    return np.ascontiguousarray(summed[group[representative]], dtype='f4')
