"""What the surface is: welding, adjacency, and what each vertex is allowed to do.

Classification decides most of the output quality, so these ask it about shapes
whose answers are settled by inspection -- a closed tetrahedron has no border, a
grid's rim is all border, a bowtie vertex is not on a surface at all.
"""

import numpy as np
import pytest
import shapes

from opengl_decimate import topology
from opengl_decimate.types import DecimateError


class TestBuild:
    def test_a_closed_shape_keeps_every_face(self):
        mesh = topology.build(*shapes.tetrahedron())
        assert mesh.vertex_count == 4
        assert mesh.face_count == 4

    def test_coincident_positions_are_welded(self):
        """A hard-edged cube carries twenty-four vertices over eight positions."""
        attributes, indices = shapes.cube_with_hard_normals()
        mesh = topology.build(attributes['POSITION'], indices)
        assert mesh.vertex_count == 8
        assert mesh.face_count == 12

    def test_the_attribute_vertex_of_each_corner_is_kept(self):
        """Welding positions must not lose which vertex a corner came from."""
        attributes, indices = shapes.cube_with_hard_normals()
        mesh = topology.build(attributes['POSITION'], indices)
        assert sorted(np.unique(mesh.corners).tolist()) == list(range(24))

    def test_a_tolerance_welds_what_is_merely_close(self):
        positions = np.asarray(
            [(0.0, 0.0, 0.0), (1e-7, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], dtype='f4'
        )
        indices = np.asarray([0, 2, 3, 1, 2, 3], dtype=np.uint32)
        assert topology.build(positions, indices).vertex_count == 4
        assert topology.build(positions, indices, tolerance=1e-5).vertex_count == 3

    def test_a_face_with_a_repeated_position_is_dropped(self):
        """Welding turns a sliver into a line, and a line is not a triangle."""
        positions = np.asarray(
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], dtype='f4'
        )
        indices = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        mesh = topology.build(positions, indices)
        assert mesh.face_count == 1

    def test_indices_must_be_whole_triangles(self):
        positions = np.zeros((3, 3), dtype='f4')
        with pytest.raises(DecimateError, match='multiple of three'):
            topology.build(positions, np.asarray([0, 1], dtype=np.uint32))

    def test_an_index_past_the_end_is_refused(self):
        positions = np.zeros((3, 3), dtype='f4')
        with pytest.raises(DecimateError, match='out of range'):
            topology.build(positions, np.asarray([0, 1, 7], dtype=np.uint32))

    def test_positions_must_be_three_dimensional(self):
        with pytest.raises(DecimateError, match=r'\(n, 3\)'):
            topology.build(np.zeros((3, 2), dtype='f4'), np.asarray([0, 1, 2], dtype=np.uint32))


class TestAdjacency:
    def test_neighbours_of_a_tetrahedron_vertex_are_the_other_three(self):
        mesh = topology.build(*shapes.tetrahedron())
        assert mesh.neighbours(0) == {1, 2, 3}

    def test_every_face_is_registered_on_each_of_its_vertices(self):
        mesh = topology.build(*shapes.icosphere(1))
        for face_index, face in enumerate(mesh.faces):
            for vertex in face:
                assert face_index in mesh.vertex_faces[vertex]

    def test_edge_count_satisfies_euler(self):
        """V - E + F = 2 for any closed surface of genus zero."""
        mesh = topology.build(*shapes.octahedron())
        edges = mesh.edges()
        assert mesh.vertex_count - len(edges) + mesh.face_count == 2

    def test_a_closed_shape_has_no_boundary_edge(self):
        mesh = topology.build(*shapes.icosphere(1))
        assert not any(mesh.is_boundary_edge(a, b) for a, b in mesh.edges())

    def test_a_patch_has_a_boundary(self):
        mesh = topology.build(*shapes.grid(4))
        boundary = [(a, b) for a, b in mesh.edges() if mesh.is_boundary_edge(a, b)]
        # A 4x4 grid's rim is twelve edges.
        assert len(boundary) == 12


class TestClassification:
    def test_a_closed_surface_is_all_manifold(self):
        mesh = topology.build(*shapes.icosphere(1))
        assert np.all(mesh.classify() == topology.VertexClass.MANIFOLD)

    def test_a_patch_is_border_at_the_rim_and_manifold_inside(self):
        mesh = topology.build(*shapes.grid(5))
        kinds = mesh.classify()
        rim = [i for i in range(mesh.vertex_count) if np.any(np.abs(mesh.positions[i]) >= 1.0)]
        inside = [i for i in range(mesh.vertex_count) if i not in rim]
        assert np.all(kinds[rim] == topology.VertexClass.BORDER)
        assert np.all(kinds[inside] == topology.VertexClass.MANIFOLD)

    def test_a_bowtie_vertex_is_locked(self):
        """Its faces form two fans, so no single placement serves both."""
        mesh = topology.build(*shapes.bowtie())
        kinds = mesh.classify()
        assert kinds[0] == topology.VertexClass.LOCKED
        assert np.all(kinds[1:] == topology.VertexClass.BORDER)

    def test_the_ends_of_a_non_manifold_edge_are_locked(self):
        mesh = topology.build(*shapes.nonmanifold_edge())
        kinds = mesh.classify()
        assert kinds[0] == topology.VertexClass.LOCKED
        assert kinds[1] == topology.VertexClass.LOCKED

    def test_locking_the_boundary_makes_a_rim_immovable(self):
        mesh = topology.build(*shapes.grid(4))
        kinds = mesh.classify(lock_boundary=True)
        assert not np.any(kinds == topology.VertexClass.BORDER)
        assert np.count_nonzero(kinds == topology.VertexClass.LOCKED) == 12

    def test_named_vertices_are_locked(self):
        """A cluster group's outer border arrives as a list of vertices to hold."""
        mesh = topology.build(*shapes.icosphere(1))
        kinds = mesh.classify(locked=np.asarray([3, 7], dtype=np.int64))
        assert kinds[3] == topology.VertexClass.LOCKED
        assert kinds[7] == topology.VertexClass.LOCKED
        assert np.count_nonzero(kinds == topology.VertexClass.LOCKED) == 2


def _two_closed_fans():
    """Two closed tetrahedral shells sharing exactly one point.

    The counting rules cannot tell this from a single closed fan -- both have as
    many faces around the point as edges -- so it is the case that decides
    whether fan connectivity is really being computed.
    """
    lower = np.asarray(
        [(0.0, 0.0, 0.0), (1.0, -1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, 0.0)], dtype='f4'
    )
    upper = lower * np.asarray([1.0, -1.0, 1.0], dtype='f4')
    positions = np.concatenate([lower, upper[1:]])
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2), (0, 5, 4), (0, 4, 6), (0, 6, 5), (4, 5, 6)]
    return positions, np.asarray(faces, dtype=np.uint32).reshape(-1)


class TestFanConnectivity:
    def test_a_point_shared_by_two_closed_shells_is_locked(self):
        mesh = topology.build(*_two_closed_fans())
        kinds = mesh.classify()
        assert kinds[0] == topology.VertexClass.LOCKED
        assert np.all(kinds[1:] == topology.VertexClass.MANIFOLD)

    def test_classification_keeps_up_with_a_large_mesh(self):
        """A scan is millions of triangles; classification is one pass over it.

        The bound is loose enough not to measure the machine, and far below what
        a per-corner Python pass costs at this size.
        """
        import time

        mesh = topology.build(*shapes.grid(300))
        assert mesh.face_count > 175_000
        start = time.perf_counter()
        mesh.classify()
        assert time.perf_counter() - start < 0.5


class TestToleranceWeldChains:
    def test_a_run_of_points_each_near_the_next_becomes_one(self):
        """Welding is transitive: a chain collapses even though its ends are
        further apart than the tolerance. That is what a scan needs, where a
        seam is reconstructed as a smear of points rather than as a pair."""
        step = 4e-6
        positions = np.asarray(
            [(i * step, 0.0, 0.0) for i in range(5)] + [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            dtype='f4',
        )
        indices = np.asarray([0, 5, 6, 4, 5, 6, 2, 5, 6], dtype=np.uint32)
        loose = topology.build(positions, indices, tolerance=1e-7)
        assert loose.vertex_count == 7
        welded = topology.build(positions, indices, tolerance=step * 1.5)
        # The five stepping points are one; the far two are themselves.
        assert welded.vertex_count == 3

    def test_the_welded_point_sits_at_the_middle_of_its_run(self):
        step = 4e-6
        positions = np.asarray(
            [(i * step, 0.0, 0.0) for i in range(5)] + [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            dtype='f4',
        )
        indices = np.asarray([0, 5, 6, 4, 5, 6], dtype=np.uint32)
        welded = topology.build(positions, indices, tolerance=step * 1.5)
        assert welded.positions[0][0] == pytest.approx(2.0 * step, rel=1e-3)
