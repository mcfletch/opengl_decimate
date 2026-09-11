"""Contracting an edge, and the questions asked before one is allowed.

A contraction merges the two ends of an edge into one point. Left unchecked it
is also how a decimator produces a surface that is not a surface: an edge with
three triangles on it, two triangles occupying the same three corners, or a fan
turned inside out so the model renders with its back faces forward. Four
questions stand between a candidate and the mesh, and all four have to be asked
every time -- the cases they catch are not rare, they are what a dense mesh is
made of.

:func:`is_legal`
    Topology and classification: may this vertex move at all, is there an edge
    here, would the result keep the surface a surface, and would it leave two
    triangles on the same corners.
:func:`would_distort`
    Geometry: does the proposed placement turn a neighbouring face over, or
    thin it to a sliver.
:func:`contract`
    The operation itself, once both have passed.
"""

from __future__ import annotations

import math

import numpy as np

from opengl_decimate.topology import Topology, VertexClass
from opengl_decimate.types import FloatArray

__all__ = ['link_condition', 'is_legal', 'would_distort', 'triangle_quality', 'contract']

#: Below this a triangle has no meaningful normal.
_TINY = 1e-30


def link_condition(mesh: Topology, dying: int, surviving: int) -> bool:
    """True where contracting the edge leaves the surface a surface.

    The link of an edge is what surrounds it. Contracting is safe exactly when
    the points joined to both ends are precisely the points opposite the edge --
    a shared neighbour anywhere else becomes, after the merge, an edge carrying
    three triangles.
    """
    shared = mesh.neighbours(dying) & mesh.neighbours(surviving)
    opposite: set[int] = set()
    for face in mesh.edge_faces(dying, surviving):
        opposite.update(int(point) for point in mesh.faces[face] if point not in (dying, surviving))
    return shared == opposite


def is_legal(mesh: Topology, kinds: np.ndarray, dying: int, surviving: int) -> bool:
    """Whether ``dying`` may be merged into ``surviving``.

    ``kinds`` is what :meth:`~opengl_decimate.topology.Topology.classify`
    returned. A locked point never dies; a border point dies only along its
    border, which is what holds a patch's outline in place.
    """
    if dying == surviving:
        return False
    if kinds[dying] == VertexClass.LOCKED:
        return False
    on_edge = mesh.edge_faces(dying, surviving)
    if not on_edge:
        return False
    if kinds[dying] == VertexClass.BORDER and len(on_edge) != 1:
        return False
    if not link_condition(mesh, dying, surviving):
        return False
    return not _would_duplicate_a_face(mesh, dying, surviving)


def _would_duplicate_a_face(mesh: Topology, dying: int, surviving: int) -> bool:
    """True where two surviving triangles would end up on the same three points.

    The link condition alone lets a small closed shape fold onto itself -- a
    tetrahedron contracts to two triangles back to back, which is a surface by
    every local test and encloses nothing.
    """
    on_edge = mesh.edge_faces(dying, surviving)
    seen = set()
    for face in (mesh.vertex_faces[dying] | mesh.vertex_faces[surviving]) - on_edge:
        points = tuple(
            sorted(surviving if point == dying else int(point) for point in mesh.faces[face])
        )
        if points in seen:
            return True
        seen.add(points)
    return False


def triangle_quality(corners: FloatArray) -> FloatArray:
    """How near each triangle is to equilateral: 1 for equilateral, 0 for a line.

    ``corners`` is ``(n, 3, 3)``. The measure is four root three times the area
    over the summed squared edge lengths, which is scale free -- a sliver scores
    the same whether it is a millimetre long or a kilometre.
    """
    corners = np.asarray(corners, dtype='d')
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 1]
    third = corners[:, 0] - corners[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(first, corners[:, 2] - corners[:, 0]), axis=1)
    lengths = (
        np.einsum('ij,ij->i', first, first)
        + np.einsum('ij,ij->i', second, second)
        + np.einsum('ij,ij->i', third, third)
    )
    return np.where(lengths > _TINY, 4.0 * math.sqrt(3.0) * area / np.maximum(lengths, _TINY), 0.0)


def _surviving_corners(
    mesh: Topology, dying: int, surviving: int, placement: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """The faces that outlive the contraction, before and after it.

    Returns two ``(n, 3, 3)`` arrays of corner positions. The faces on the edge
    itself are excluded: they are removed by the contraction, so how they would
    have looked is not a question.
    """
    affected = sorted(
        (mesh.vertex_faces[dying] | mesh.vertex_faces[surviving])
        - mesh.edge_faces(dying, surviving)
    )
    faces = mesh.faces[affected]
    before = mesh.positions[faces]
    moved = mesh.positions.copy()
    moved[surviving] = placement
    after = moved[np.where(faces == dying, surviving, faces)]
    return before, after


def would_distort(
    mesh: Topology,
    dying: int,
    surviving: int,
    placement: FloatArray,
    max_normal_flip: float = 90.0,
    min_quality: float = 0.0,
) -> bool:
    """True where the placement damages a face that has to outlive the collapse.

    ``max_normal_flip`` is in degrees: a face whose normal turns further than
    this has been folded over rather than moved, and the surface would render
    inside out there. ``min_quality`` refuses a contraction that leaves a
    triangle thinner than :func:`triangle_quality` allows, which trades
    reduction for triangle shape and is off at zero.
    """
    before, after = _surviving_corners(mesh, dying, surviving, placement)
    if not len(before):
        return False

    old = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
    new = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
    old_length = np.linalg.norm(old, axis=1)
    new_length = np.linalg.norm(new, axis=1)
    # A face that has collapsed to a line has no normal to compare, and a face
    # that had none to begin with cannot be said to have turned.
    vanished = new_length <= _TINY
    if np.any(vanished & (old_length > _TINY)):
        return True
    comparable = (old_length > _TINY) & ~vanished
    if np.any(comparable):
        cosines = np.einsum('ij,ij->i', old[comparable], new[comparable]) / (
            old_length[comparable] * new_length[comparable]
        )
        if np.any(cosines < math.cos(math.radians(max_normal_flip))):
            return True

    return bool(min_quality > 0.0 and np.any(triangle_quality(after) < min_quality))


def contract(mesh: Topology, dying: int, surviving: int, placement: FloatArray) -> list[int]:
    """Merge ``dying`` into ``surviving`` at ``placement``; return the faces removed.

    The faces on the edge lose their area and go; every other face of ``dying``
    is handed to ``surviving``. Nothing is reallocated -- a removed face is
    marked dead in place -- so face indices mean the same thing for the whole
    reduction and a collapse sequence can be replayed against them.
    """
    removed = sorted(mesh.edge_faces(dying, surviving))
    for face in removed:
        mesh.alive[face] = False
        for point in mesh.faces[face]:
            mesh.vertex_faces[point].discard(face)

    for face in sorted(mesh.vertex_faces[dying]):
        mesh.faces[face][mesh.faces[face] == dying] = surviving
        mesh.vertex_faces[surviving].add(face)
    mesh.vertex_faces[dying] = set()
    mesh.positions[surviving] = placement
    return removed
