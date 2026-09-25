"""Driving a reduction: what to contract next, and when to stop.

:func:`simplify` is the whole job in one call. :func:`collapse_sequence` runs the
same reduction but records every step and stops at nothing, so a caller can then
ask for any triangle count without decimating again.

Two schedules choose the next contraction, and the difference between them is
the difference between the two halves of the problem:

``heap``
    The cheapest candidate anywhere on the mesh, kept in a lazily-updated
    priority queue. This is the best quality available, and it is sequential by
    construction -- every contraction changes the price of its neighbours, so
    the next choice depends on the last.

``multiple-choice``
    The cheapest of a few candidates drawn at random. There is no global
    ordering to maintain and no queue to keep, each step costs the same as any
    other, and the quality loss is small. Giving up the global ordering is what
    makes a reduction divisible, so this is the schedule a parallel or GPU
    implementation is built on.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from opengl_decimate import certify, collapse, native, quadrics, topology
from opengl_decimate.options import SimplifyOptions
from opengl_decimate.sequence import CollapseSequence, SimplifyResult
from opengl_decimate.topology import Topology, VertexClass
from opengl_decimate.types import POSITION, AttributeMap, DecimateError, FloatArray, IndexArray

__all__ = ['simplify', 'collapse_sequence']

_TINY = 1e-30

#: What :func:`collapse_sequence` uses where the caller named no options. The
#: target is there because :class:`SimplifyOptions` requires one and is
#: ignored because the whole reduction is recorded either way.
_EXHAUSTIVE = SimplifyOptions(target_ratio=1.0)


def simplify(
    attributes: AttributeMap, indices: IndexArray, options: SimplifyOptions
) -> SimplifyResult:
    """Reduce a mesh to the target ``options`` names.

    ``attributes`` maps glTF semantics to arrays and must include ``POSITION``;
    ``indices`` is a flat triangle list. Everything else the mesh carries is
    handed back for the vertices that survive.

    ``result.error`` is the reduction's own estimate, accumulated from the
    quadrics. ``options.certify`` additionally measures the deviation of the
    result from the input and puts it in ``result.measured_error``, which is the
    number a level of detail's error bound should be built on. Pieces
    ``drop_components_below`` removed are not part of the input it is measured
    against.
    """
    sequence = _reduce(attributes, indices, options, exhaust=False)
    result = sequence.at()
    if options.certify:
        # Measured against the surface the reduction started from: the input's
        # own vertices, less the triangles welding left without area and the
        # pieces drop_components_below was asked to remove.
        result.measured_error = certify.surface_deviation(
            np.asarray(attributes[POSITION]),
            sequence.corners,
            result.attributes[POSITION],
            result.indices,
            samples=options.certify_samples,
            seed=options.seed,
        ).max
    return result


def collapse_sequence(
    attributes: AttributeMap, indices: IndexArray, options: SimplifyOptions | None = None
) -> CollapseSequence:
    """Record the whole reduction, ignoring the targets in ``options``.

    The targets are what :meth:`~opengl_decimate.sequence.CollapseSequence.at`
    is asked for afterwards, one at a time and as often as wanted. This is the
    call an editor makes once when a model is loaded.

    ``options`` says how to reduce -- the metric, the noise, the locks, the
    placement, the weld, the schedule -- and every target in it is ignored, so
    it may be left out entirely where the defaults will do. What stops the
    reduction here is the surface running out, not a budget.
    """
    return _reduce(attributes, indices, options or _EXHAUSTIVE, exhaust=True)


def _reduce(
    attributes: AttributeMap, indices: IndexArray, options: SimplifyOptions, exhaust: bool
) -> CollapseSequence:
    """Build the mesh, run a schedule over it, and return what it did."""
    mesh = topology.build(
        topology.positions_of(attributes),
        indices,
        options.weld_tolerance,
        options.drop_components_below,
        carried=dict(attributes) if options.lock_seams else None,
    )
    engine = _Engine(mesh, options, exhaust=exhaust)
    if mesh.face_count:
        if options.schedule == 'heap' and native.ACCELERATED:
            engine.run_native(engine.face_limit)
        else:
            runner: Callable[[int], None] = (
                engine.run_heap if options.schedule == 'heap' else engine.run_multiple_choice
            )
            runner(engine.face_limit)
    return engine.record(dict(attributes))


class _Engine:
    """One reduction in progress: the surface, its quadrics, and the log."""

    def __init__(self, mesh: Topology, options: SimplifyOptions, exhaust: bool = False) -> None:
        self.mesh = mesh
        self.options = options
        #: A recording run stops at nothing, so every target in ``options`` is
        #: set aside here rather than half of them here and half in the loops.
        self.exhaust = exhaust
        # A contraction moves positions and rewrites faces in place, so the
        # sequence has to be handed the surface as it was: replaying a prefix
        # starts from the input, not from wherever the reduction finished.
        self.origin_positions = mesh.positions.copy()
        self.origin_faces = mesh.faces.copy()
        self.origin_corners = mesh.corners.copy()
        self.kinds = mesh.classify(
            lock_boundary=options.lock_boundary, locked=_locked_points(mesh, options.locked)
        )

        diagonal = _diagonal(mesh.positions)
        position_noise, normal_noise = options.noise(diagonal)
        faces = mesh.live_faces()
        per_face = quadrics.triangle_quadrics(mesh.positions, faces, position_noise, normal_noise)
        self.quadrics = quadrics.accumulate(per_face, faces, mesh.vertex_count)
        self.quadrics += self._border_constraints(position_noise, normal_noise)

        areas = _face_areas(mesh.positions, faces)
        self.weights = np.bincount(
            faces.reshape(-1), weights=np.repeat(areas, 3), minlength=mesh.vertex_count
        )

        self.face_limit = 0 if exhaust else self._face_limit()
        self.error_limit = None if exhaust else options.target_error
        self._dying: list[int] = []
        self._surviving: list[int] = []
        self._placement: list[Any] = []
        self._deviation: list[float] = []
        self.removed_at = np.full(len(mesh.faces), -1, dtype=np.int64)
        #: Set by the compiled path, which returns its log as arrays rather
        #: than appending to the lists above.
        self._compiled: tuple[Any, ...] | None = None

    def run_native(self, limit: int) -> None:
        """The same reduction, through the compiled loop."""
        self._compiled = native.reduce_mesh(
            self.mesh,
            self.quadrics,
            self.weights,
            self.kinds,
            limit,
            self.options.placement,
            self.options.max_normal_flip,
            self.options.min_triangle_quality,
            self.error_limit,
        )
        self.removed_at = np.asarray(self._compiled[4], dtype=np.int64)

    def _face_limit(self) -> int:
        """The triangle count to stop at, from whichever targets were given.

        With only an error budget there is no count to stop at, so the schedule
        runs until the budget stops it.

        A ratio is a share of the triangles the *caller* handed in, not of what
        is left after welding merged coincident vertices and dropped the
        triangles that left with a repeated corner. On badly-exported or scanned
        data those are not a rounding, and a caller asking for half of their
        three hundred triangles means a hundred and fifty of them.
        """
        options = self.options
        if options.target_count is None and options.target_ratio is None:
            return 0
        limit = self.mesh.face_count
        if options.target_count is not None:
            limit = min(limit, options.target_count)
        if options.target_ratio is not None:
            limit = min(limit, int(options.target_ratio * self.mesh.input_faces))
        return limit

    def _border_constraints(self, position_noise: float, normal_noise: float) -> FloatArray:
        """Quadrics holding an open surface's edge in the plane through it.

        A border edge has one face, and a plane standing perpendicular to that
        face along the edge is the wall the border is free to slide along but not
        to leave. Without it the outline of a patch shrinks inward as the patch
        is reduced.
        """
        mesh = self.mesh
        empty = np.zeros((mesh.vertex_count, quadrics.QUADRIC_SIZE), dtype='d')
        if not self.options.boundary_weight or not mesh.face_count:
            return empty
        faces = mesh.live_faces()
        owners = np.tile(np.arange(len(faces), dtype=np.int64), 3)
        pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        pairs = np.sort(pairs, axis=1)
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs, owners = pairs[order], owners[order]
        unique, start, counts = np.unique(pairs, axis=0, return_index=True, return_counts=True)
        alone = counts == 1
        if not np.any(alone):
            return empty

        edges = unique[alone]
        corners = mesh.positions[faces[owners[start[alone]]]]
        face_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        along = mesh.positions[edges[:, 1]] - mesh.positions[edges[:, 0]]
        wall = np.cross(along, face_normal)
        weight = self.options.boundary_weight * np.einsum('ij,ij->i', along, along)
        constraint = quadrics.plane_quadric(
            wall,
            mesh.positions[edges[:, 0]],
            weights=weight,
            position_noise=position_noise,
            normal_noise=normal_noise,
        )
        return quadrics.scatter(
            np.concatenate([constraint, constraint], axis=0),
            np.concatenate([edges[:, 0], edges[:, 1]]),
            mesh.vertex_count,
        )

    def candidates(self, pairs: IndexArray) -> tuple[FloatArray, FloatArray]:
        """Deviation and placement for each candidate pair, vectorised.

        A pair whose ends are both locked, or whose only admissible placements
        are ruled out, comes back with an infinite deviation and is passed over.
        """
        left, right = pairs[:, 0], pairs[:, 1]
        summed = self.quadrics[left] + self.quadrics[right]
        here, there = self.mesh.positions[left], self.mesh.positions[right]
        middle = 0.5 * (here + there)
        if self.options.placement == 'optimal':
            best, solvable = quadrics.minimize(summed)
        else:
            best, solvable = middle, np.zeros(len(pairs), dtype=bool)

        places = np.stack([here, there, middle, best], axis=1)
        allowed = np.ones((len(pairs), 4), dtype=bool)
        allowed[:, 2] = self.options.placement == 'optimal'
        allowed[:, 3] = solvable & _near_the_edge(best, middle, here, there)
        held_here = self.kinds[left] == VertexClass.LOCKED
        held_there = self.kinds[right] == VertexClass.LOCKED
        allowed[held_here] = (True, False, False, False)
        allowed[held_there] = (False, True, False, False)
        allowed[held_here & held_there] = False

        costs = quadrics.evaluate(np.repeat(summed, 4, axis=0), places.reshape(-1, 3)).reshape(
            -1, 4
        )
        costs = np.where(allowed, costs, np.inf)
        chosen = np.argmin(costs, axis=1)
        rows = np.arange(len(pairs))
        weight = np.maximum(self.weights[left] + self.weights[right], _TINY)
        deviation = np.sqrt(np.maximum(costs[rows, chosen], 0.0) / weight)
        return deviation, places[rows, chosen]

    def try_contract(self, left: int, right: int, placement: FloatArray, deviation: float) -> bool:
        """Apply one contraction if every rule allows it; say whether it happened."""
        mesh, kinds = self.mesh, self.kinds
        if collapse.is_legal(mesh, kinds, left, right):
            dying, surviving = left, right
        elif collapse.is_legal(mesh, kinds, right, left):
            dying, surviving = right, left
        else:
            return False
        if collapse.would_distort(
            mesh,
            dying,
            surviving,
            placement,
            self.options.max_normal_flip,
            self.options.min_triangle_quality,
        ):
            return False

        step = len(self._dying)
        for face in collapse.contract(mesh, dying, surviving, placement):
            self.removed_at[face] = step
        self.quadrics[surviving] += self.quadrics[dying]
        self.weights[surviving] += self.weights[dying]
        self._dying.append(dying)
        self._surviving.append(surviving)
        self._placement.append(np.array(placement, dtype='d'))
        self._deviation.append(float(deviation))
        return True

    def run_heap(self, limit: int) -> None:
        """Contract the cheapest candidate on the mesh, over and over."""
        counter = itertools.count()
        stamps: dict[tuple[int, int], int] = {}
        queue: list[tuple[float, int, int, int, Any]] = []
        self._offer(queue, stamps, counter, self.mesh.edges())

        while queue and self.mesh.face_count > limit:
            deviation, stamp, left, right, placement = heapq.heappop(queue)
            if stamps.get((left, right)) != stamp:
                continue
            del stamps[(left, right)]
            if self.error_limit is not None and deviation > self.error_limit:
                break
            # A pair that is no longer an edge -- its faces taken by an earlier
            # contraction -- needs no check of its own here: `is_legal` refuses
            # a pair with no face on it.
            departing = self.mesh.neighbours(left) | self.mesh.neighbours(right)
            if not self.try_contract(left, right, placement, deviation):
                continue
            surviving = self._surviving[-1]
            for point in departing:
                stamps.pop(_pair(point, self._dying[-1]), None)
            self._offer(
                queue,
                stamps,
                counter,
                np.asarray(
                    [_pair(surviving, point) for point in sorted(self.mesh.neighbours(surviving))],
                    dtype=np.int64,
                ).reshape(-1, 2),
            )

    def _offer(
        self,
        queue: list[tuple[float, int, int, int, Any]],
        stamps: dict[tuple[int, int], int],
        counter: Any,
        pairs: IndexArray,
    ) -> None:
        """Price a batch of pairs in one pass and put them on the queue."""
        deviation, placement = self.candidates(pairs)
        for index in np.flatnonzero(np.isfinite(deviation)):
            key = (int(pairs[index, 0]), int(pairs[index, 1]))
            stamp = next(counter)
            stamps[key] = stamp
            heapq.heappush(
                queue, (float(deviation[index]), stamp, key[0], key[1], placement[index])
            )

    def run_multiple_choice(self, limit: int) -> None:
        """Contract the cheapest of a few candidates drawn at random.

        The pool is the set of pairs worth considering. A pair leaves it when it
        is drawn and not contracted: no longer an edge, priced out, over the
        error budget, or refused by the rules. Nothing about such a pair changes
        until one of its ends survives a contraction -- its price is its two
        ends' quadrics and positions -- and the survivor's pairs are put back
        when that happens, as the heap re-queues them. So every pair drawn is
        either contracted or set aside, the pool runs dry exactly when nothing
        is left to contract, and the work is bounded by the pairs that ever
        entered it.

        The error budget filters candidates rather than ending the reduction.
        These are a few edges drawn at random out of thousands, so the cheapest
        of them is over budget long before the cheapest on the mesh is, and
        ending there would stop wherever the draw happened to fall.
        """
        rng = np.random.default_rng(self.options.seed)
        pool = [_pair(int(a), int(b)) for a, b in self.mesh.edges()]
        known = set(pool)

        while pool and self.mesh.face_count > limit:
            slots = np.unique(rng.integers(len(pool), size=min(self.options.candidates, len(pool))))
            drawn = [pool[slot] for slot in slots]
            live = [rank for rank, pair in enumerate(drawn) if self.mesh.edge_faces(*pair)]
            spent = [int(slots[rank]) for rank in range(len(drawn)) if rank not in live]
            pairs = np.asarray([drawn[rank] for rank in live], dtype=np.int64).reshape(-1, 2)
            deviation, placement = self.candidates(pairs)
            contracted = False
            for rank in np.argsort(deviation, kind='stable'):
                affordable = np.isfinite(deviation[rank]) and (
                    self.error_limit is None or deviation[rank] <= self.error_limit
                )
                left, right = int(pairs[rank, 0]), int(pairs[rank, 1])
                spent.append(int(slots[live[rank]]))
                if affordable and self.try_contract(
                    left, right, placement[rank], float(deviation[rank])
                ):
                    contracted = True
                    break
            for slot in sorted(spent, reverse=True):
                known.discard(pool[slot])
                pool[slot] = pool[-1]
                pool.pop()
            if not contracted:
                continue
            surviving = self._surviving[-1]
            for point in self.mesh.neighbours(surviving):
                key = _pair(surviving, int(point))
                if key not in known:
                    known.add(key)
                    pool.append(key)

    def record(self, attributes: dict[str, np.ndarray]) -> CollapseSequence:
        """Everything the reduction did, as the replayable sequence."""
        if self._compiled is not None:
            dying, surviving, placement, deviation, removed_at = self._compiled
            return CollapseSequence(
                points=self.origin_positions,
                faces=self.origin_faces,
                corners=self.origin_corners,
                vertex_point=self.mesh.vertex_point,
                attributes=attributes,
                dying=dying,
                surviving=surviving,
                placement=placement,
                deviation=deviation,
                removed_at=removed_at,
                recompute_normals=self.options.recompute_normals,
                crease_angle=self.options.crease_angle,
                origin=self.mesh.origin,
                input_faces=self.mesh.input_faces,
                dropped_faces=self.mesh.dropped_faces,
            )
        count = len(self._dying)
        return CollapseSequence(
            points=self.origin_positions,
            faces=self.origin_faces,
            corners=self.origin_corners,
            vertex_point=self.mesh.vertex_point,
            attributes=attributes,
            dying=np.asarray(self._dying, dtype=np.int64),
            surviving=np.asarray(self._surviving, dtype=np.int64),
            placement=(
                np.asarray(self._placement, dtype='d') if count else np.zeros((0, 3), dtype='d')
            ),
            deviation=np.asarray(self._deviation, dtype='d'),
            removed_at=self.removed_at,
            recompute_normals=self.options.recompute_normals,
            crease_angle=self.options.crease_angle,
            origin=self.mesh.origin,
            input_faces=self.mesh.input_faces,
            dropped_faces=self.mesh.dropped_faces,
        )


def _near_the_edge(
    points: FloatArray, middle: FloatArray, here: FloatArray, there: FloatArray
) -> np.ndarray:
    """Whether each point is no further from its edge's midpoint than the edge is long.

    A minimum further than that comes from a nearly singular solve, and sits
    wherever rounding put it along the direction the quadric cannot see. It is
    written out coordinate by coordinate, as the compiled reducer computes it,
    so the two reducers draw the line in the same place.
    """
    away = points - middle
    edge = there - here
    return (
        away[:, 0] * away[:, 0] + away[:, 1] * away[:, 1] + away[:, 2] * away[:, 2]
        <= edge[:, 0] * edge[:, 0] + edge[:, 1] * edge[:, 1] + edge[:, 2] * edge[:, 2]
    )


def _locked_points(mesh: Topology, locked: Sequence[int] | None) -> IndexArray | None:
    """The welded points holding the caller's ``locked`` vertices.

    The caller names vertices of the arrays they handed in. Welding merges
    every vertex that shares a position -- each side of a texture seam, each
    face of a hard edge -- and numbers the points by first appearance, so
    after the first merged vertex the two numberings differ.
    """
    if locked is None:
        return None
    held = np.asarray(locked, dtype=np.int64).reshape(-1)
    if held.size and int(held.max()) >= len(mesh.vertex_point):
        raise DecimateError(
            'locked index %d is past the end: the mesh has %d vertices'
            % (int(held.max()), len(mesh.vertex_point))
        )
    return mesh.vertex_point[held]


def _pair(left: int, right: int) -> tuple[int, int]:
    """An undirected edge as an ordered key."""
    return (left, right) if left < right else (right, left)


def _diagonal(positions: FloatArray) -> float:
    """The model's bounding-box diagonal, which every tolerance is relative to."""
    if not len(positions):
        return 1.0
    span = np.max(positions, axis=0) - np.min(positions, axis=0)
    return float(np.linalg.norm(span)) or 1.0


def _face_areas(positions: FloatArray, faces: IndexArray) -> FloatArray:
    """Area of each triangle, which is what a vertex's quadric is weighted by."""
    corners = positions[faces]
    return 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
