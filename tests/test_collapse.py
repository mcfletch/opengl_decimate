"""The one operation, and the four questions asked before it is allowed.

An unchecked contraction is how a decimator produces a mesh that renders inside
out, has an edge on three triangles, or has two triangles occupying the same
place. Each test here is one of those outcomes, set up so it would happen.
"""

import numpy as np
import pytest
import shapes

from opengl_decimate import collapse, topology


def _tent():
    """A tetrahedron with one face removed: three faces round an open triangle."""
    positions = np.asarray(
        [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.9), (-0.5, 0.0, -0.9)], dtype='f4'
    )
    return positions, np.asarray([0, 1, 2, 0, 2, 3, 0, 3, 1], dtype=np.uint32)


class TestLinkCondition:
    def test_an_ordinary_interior_edge_passes(self):
        mesh = topology.build(*shapes.icosphere(1))
        edge = mesh.edges()[0]
        assert collapse.link_condition(mesh, int(edge[0]), int(edge[1]))

    def test_an_edge_whose_ends_share_a_third_neighbour_is_refused(self):
        """Contracting it would leave an edge with three triangles on it."""
        mesh = topology.build(*_tent())
        assert mesh.is_boundary_edge(1, 2)
        assert not collapse.link_condition(mesh, 1, 2)

    def test_the_condition_is_symmetric(self):
        mesh = topology.build(*_tent())
        assert collapse.link_condition(mesh, 1, 2) == collapse.link_condition(mesh, 2, 1)


class TestLegality:
    def test_a_locked_vertex_never_dies(self):
        mesh = topology.build(*shapes.icosphere(1))
        kinds = mesh.classify(locked=np.asarray([0], dtype=np.int64))
        neighbour = next(iter(mesh.neighbours(0)))
        assert not collapse.is_legal(mesh, kinds, 0, neighbour)
        assert collapse.is_legal(mesh, kinds, neighbour, 0)

    def test_a_border_vertex_moves_only_along_its_border(self):
        mesh = topology.build(*shapes.grid(4))
        kinds = mesh.classify()
        # A rim vertex that is not a corner: it has neighbours both along the
        # border and into the middle of the patch.
        on_rim = next(
            v
            for v in range(mesh.vertex_count)
            if kinds[v] == topology.VertexClass.BORDER
            and any(not mesh.is_boundary_edge(v, n) for n in mesh.neighbours(v))
        )
        along = next(n for n in mesh.neighbours(on_rim) if mesh.is_boundary_edge(on_rim, n))
        inward = next(n for n in mesh.neighbours(on_rim) if not mesh.is_boundary_edge(on_rim, n))
        assert collapse.is_legal(mesh, kinds, on_rim, along)
        assert not collapse.is_legal(mesh, kinds, on_rim, inward)

    def test_a_contraction_that_would_duplicate_a_face_is_refused(self):
        """Every edge of a tetrahedron: collapsing one folds it onto itself."""
        mesh = topology.build(*shapes.tetrahedron())
        kinds = mesh.classify()
        assert not any(collapse.is_legal(mesh, kinds, int(a), int(b)) for a, b in mesh.edges())

    def test_vertices_that_do_not_share_an_edge_are_refused(self):
        mesh = topology.build(*shapes.grid(4))
        kinds = mesh.classify()
        assert not collapse.is_legal(mesh, kinds, 0, 15)


class TestDistortion:
    def test_placing_the_survivor_across_the_patch_flips_faces(self):
        mesh = topology.build(*shapes.grid(5, bump=0.3))
        far_side = mesh.positions[12] + np.asarray([0.0, -4.0, 0.0])
        assert collapse.would_distort(mesh, 11, 12, far_side)

    def test_a_placement_on_the_edge_does_not(self):
        mesh = topology.build(*shapes.grid(5, bump=0.3))
        midpoint = 0.5 * (mesh.positions[11] + mesh.positions[12])
        assert not collapse.would_distort(mesh, 11, 12, midpoint)

    def test_a_quality_floor_refuses_a_sliver(self):
        """The same contraction, allowed without a floor and refused with one."""
        mesh = topology.build(*shapes.grid(5))
        placement = mesh.positions[12] + np.asarray([0.0, 0.0, 0.24])
        assert not collapse.would_distort(mesh, 11, 12, placement)
        assert collapse.would_distort(mesh, 11, 12, placement, min_quality=0.9)


class TestTriangleQuality:
    def test_an_equilateral_triangle_scores_one(self):
        corners = np.asarray(
            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 3.0**0.5 / 2.0)]], dtype='d'
        )
        assert collapse.triangle_quality(corners)[0] == pytest.approx(1.0)

    def test_a_collinear_triangle_scores_zero(self):
        corners = np.asarray([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]], dtype='d')
        assert collapse.triangle_quality(corners)[0] == pytest.approx(0.0)

    def test_a_triangle_with_no_extent_scores_zero(self):
        corners = np.zeros((1, 3, 3), dtype='d')
        assert collapse.triangle_quality(corners)[0] == pytest.approx(0.0)


class TestContract:
    def test_an_interior_edge_loses_two_faces(self):
        mesh = topology.build(*shapes.icosphere(1))
        before = mesh.face_count
        edge = next(e for e in mesh.edges() if not mesh.is_boundary_edge(*e))
        collapse.contract(mesh, int(edge[0]), int(edge[1]), mesh.positions[edge[1]])
        assert mesh.face_count == before - 2

    def test_a_border_edge_loses_one_face(self):
        mesh = topology.build(*shapes.grid(4))
        before = mesh.face_count
        edge = next(e for e in mesh.edges() if mesh.is_boundary_edge(*e))
        collapse.contract(mesh, int(edge[0]), int(edge[1]), mesh.positions[edge[1]])
        assert mesh.face_count == before - 1

    def test_the_survivor_takes_the_placement_and_the_faces(self):
        mesh = topology.build(*shapes.icosphere(1))
        dying, surviving = 1, int(next(iter(mesh.neighbours(1))))
        expected = set(mesh.vertex_faces[dying]) | set(mesh.vertex_faces[surviving])
        expected -= mesh.edge_faces(dying, surviving)
        placement = np.asarray([0.25, 0.5, 0.75])
        collapse.contract(mesh, dying, surviving, placement)
        assert mesh.vertex_faces[dying] == set()
        assert mesh.vertex_faces[surviving] == expected
        assert mesh.positions[surviving] == pytest.approx(placement)

    def test_the_dying_vertex_is_gone_from_every_face(self):
        mesh = topology.build(*shapes.icosphere(1))
        dying, surviving = 1, int(next(iter(mesh.neighbours(1))))
        collapse.contract(mesh, dying, surviving, mesh.positions[surviving])
        assert dying not in mesh.live_faces()

    def test_adjacency_stays_consistent(self):
        """Every live face must still be registered on each vertex it uses."""
        mesh = topology.build(*shapes.icosphere(2))
        kinds = mesh.classify()
        rng = np.random.default_rng(3)
        for _ in range(40):
            edges = mesh.edges()
            a, b = edges[rng.integers(len(edges))]
            if collapse.is_legal(mesh, kinds, int(a), int(b)):
                collapse.contract(mesh, int(a), int(b), mesh.positions[b])
        for index, face in enumerate(mesh.faces):
            if mesh.alive[index]:
                for point in face:
                    assert index in mesh.vertex_faces[point]
            else:
                for point in face:
                    assert index not in mesh.vertex_faces[point]


class TestDegenerateAsks:
    def test_a_vertex_cannot_be_merged_into_itself(self):
        mesh = topology.build(*shapes.icosphere(1))
        kinds = mesh.classify()
        assert not collapse.is_legal(mesh, kinds, 3, 3)

    def test_a_surface_with_no_triangles_has_no_edges(self):
        mesh = topology.build(np.zeros((0, 3), dtype='f4'), np.zeros((0,), dtype=np.uint32))
        assert mesh.edges().shape == (0, 2)


class TestCostOfOneContraction:
    """A contraction touches one neighbourhood, and must read only that.

    A scan is millions of triangles. Anything in the inner loop that reads,
    copies or scans the whole mesh makes each contraction cost as much as the
    mesh is big, which turns the reduction into quadratic work and never
    finishes. Asserted by watching what is allocated rather than by timing,
    so it is a fact about the code rather than about the machine.
    """

    @staticmethod
    def _big():
        mesh = topology.build(*shapes.grid(400))  # 160,000 points
        middle = 400 * 200 + 200
        return mesh, middle, int(next(iter(mesh.neighbours(middle))))

    def test_testing_the_placement_allocates_a_neighbourhood_not_a_mesh(self):
        import tracemalloc

        mesh, dying, surviving = self._big()
        placement = mesh.positions[surviving]
        tracemalloc.start()
        try:
            collapse.would_distort(mesh, dying, surviving, placement)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # The mesh's positions alone are 160,000 * 3 * 8 bytes.
        assert peak < 200_000, 'allocated %.1f MB to test one contraction' % (peak / 1e6)

    def test_applying_it_allocates_a_neighbourhood_not_a_mesh(self):
        import tracemalloc

        mesh, dying, surviving = self._big()
        placement = mesh.positions[surviving]
        tracemalloc.start()
        try:
            collapse.contract(mesh, dying, surviving, placement)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 200_000, 'allocated %.1f MB to apply one contraction' % (peak / 1e6)

    def test_counting_the_live_faces_does_not_scan_them(self):
        """It is asked once per contraction, so it cannot be a pass over the mesh."""
        import tracemalloc

        mesh, dying, surviving = self._big()
        before = mesh.face_count
        collapse.contract(mesh, dying, surviving, mesh.positions[surviving])
        tracemalloc.start()
        try:
            counted = mesh.face_count
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert counted == before - 2
        assert peak < 10_000


class TestNothingToDistort:
    def test_a_contraction_with_no_surviving_faces_distorts_nothing(self):
        """An isolated triangle: both faces on the edge go, and nothing is left
        whose shape could have been damaged."""
        positions = np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], dtype='f4')
        mesh = topology.build(positions, np.asarray([0, 1, 2], dtype=np.uint32))
        assert not collapse.would_distort(mesh, 0, 1, mesh.positions[1])
