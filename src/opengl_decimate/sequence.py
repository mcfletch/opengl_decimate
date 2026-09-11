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
- The corners' attributes never move at all, so nothing has to be recomputed.

So :meth:`CollapseSequence.at` costs the same whether it is asked for ninety per
cent of the triangles or two per cent, and asking for fifty different counts in
a row costs fifty times that and no more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

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

    @property
    def triangle_count(self) -> int:
        """How many triangles came back."""
        return len(self.indices) // 3


@dataclass
class CollapseSequence:
    """Every contraction of a reduction, in the order they were applied.

    Held against the *welded* mesh: ``points`` are the distinct positions of the
    input, ``faces`` index them, and ``corners`` records which input vertex each
    corner came from so its attributes can be handed back.
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
    _removed_by_step: Any = field(default=None, repr=False)

    def __len__(self) -> int:
        """How many contractions were recorded."""
        return len(self.dying)

    def __post_init__(self) -> None:
        removals = self.removed_at[self.removed_at >= 0]
        per_step = np.bincount(removals, minlength=len(self.dying) + 1)
        self._removed_by_step = np.concatenate([[0], np.cumsum(per_step)])

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
        """
        steps = len(self)
        if target_ratio is not None:
            target_count = min(
                target_count if target_count is not None else len(self.faces),
                int(target_ratio * len(self.faces)),
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
        return positions, roots, roots[self.faces[alive]], self.corners[alive]

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
        )


def _first_appearance(inverse: IndexArray, representative: IndexArray) -> tuple[Any, Any]:
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
) -> SimplifyResult:
    """Turn the live surface back into vertex arrays and an index array.

    An output vertex is a distinct combination of *point* and *attributes*: two
    corners at the same point carrying the same normal and the same texture
    coordinate are one vertex, and two carrying different ones are two, which is
    what keeps a seam a seam.
    """
    flat_points = faces.reshape(-1)
    flat_corners = corners.reshape(-1)
    if not len(flat_points):
        empty = {name: value[:0].copy() for name, value in attributes.items()}
        empty[POSITION] = np.zeros((0, 3), dtype='f4')
        return SimplifyResult(
            attributes=empty,
            indices=np.zeros((0,), dtype=np.uint32),
            error=error,
            vertex_map=np.full(len(vertex_point), -1, dtype=np.int64),
            collapses=collapses,
        )

    columns = [flat_points.astype('d')[:, None]]
    carried = [name for name in sorted(attributes) if name != POSITION]
    for name in carried:
        values = np.asarray(attributes[name])[flat_corners]
        columns.append(values.reshape(len(flat_corners), -1).astype('d'))
    key = np.concatenate(columns, axis=1)
    _, representative, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    out_index, representative = _first_appearance(inverse.reshape(-1), representative)

    out_point = flat_points[representative]
    out_source = flat_corners[representative]
    out_attributes: dict[str, np.ndarray] = {
        POSITION: np.ascontiguousarray(positions[out_point], dtype='f4')
    }
    for name in carried:
        out_attributes[name] = np.ascontiguousarray(np.asarray(attributes[name])[out_source])

    indices = np.ascontiguousarray(out_index.reshape(-1, 3), dtype=np.uint32).reshape(-1)
    if recompute_normals:
        out_attributes['NORMAL'] = _surface_normals(out_attributes[POSITION], indices)

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
    )


def _surface_normals(positions: FloatArray, indices: IndexArray) -> FloatArray:
    """Area-weighted vertex normals of the surface as it now stands."""
    faces = np.asarray(indices).reshape(-1, 3)
    corners = np.asarray(positions, dtype='d')[faces]
    face_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    summed = np.zeros((len(positions), 3), dtype='d')
    np.add.at(summed, faces.reshape(-1), np.repeat(face_normal, 3, axis=0))
    lengths = np.linalg.norm(summed, axis=1)
    usable = lengths > 0.0
    summed[usable] /= lengths[usable][:, None]
    return np.ascontiguousarray(summed, dtype='f4')
