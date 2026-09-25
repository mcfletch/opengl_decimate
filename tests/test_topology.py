"""What the surface is: welding, adjacency, and what each vertex is allowed to do.

Classification decides most of the output quality, so these ask it about shapes
whose answers are settled by inspection -- a closed tetrahedron has no border, a
grid's rim is all border, a bowtie vertex is not on a surface at all.
"""

import tracemalloc

import numpy as np
import pytest
import shapes

from opengl_decimate import SimplifyOptions, collapse_sequence, topology
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


class TestTheLocalOrigin:
    """Where the quadrics are accumulated from, and why it is not always zero."""

    def test_a_model_around_the_origin_is_left_where_it_is(self):
        mesh = topology.build(*shapes.icosphere(1))
        assert mesh.origin.tolist() == [0.0, 0.0, 0.0]

    def test_a_model_far_from_the_origin_is_measured_from_its_own_centre(self):
        positions, indices = shapes.icosphere(1)
        far = positions.astype('d') + np.asarray([6.4e6, 0.0, 0.0])
        mesh = topology.build(far, indices)
        # Shifted on X, where the model is far from the origin relative to its
        # size, and left alone on Y and Z, where it is not.
        assert mesh.origin[0] == pytest.approx(6.4e6, abs=1.0)
        assert mesh.origin[1] == 0.0
        assert mesh.origin[2] == 0.0

    @pytest.mark.parametrize('shift', [1e3, 1e6, 6.4e6, -6.4e6, 1e12])
    def test_the_shift_and_the_shift_back_are_both_exact(self, shift):
        """The reason for the rule: no model pays a rounding for the fix.

        The shift is taken only where the model sits at least twice its own
        half-extent from the origin, which is the condition under which
        Sterbenz's lemma makes the subtraction exact -- and an exact difference
        added back to what it was taken from is the original, exactly.
        """
        points = np.linspace(-1.0, 1.0, 97)[:, None] * np.asarray([1.0, 0.5, 0.25])
        points = points + np.asarray([shift, 0.0, 0.0])
        origin = topology.local_origin(points)
        assert np.array_equal((points - origin) + origin, points)

    def test_state_hands_positions_back_in_the_callers_coordinates(self):
        positions, indices = shapes.icosphere(1)
        far = positions.astype('d') + np.asarray([6.4e6, 0.0, 0.0])
        sequence = collapse_sequence({'POSITION': far}, indices, SimplifyOptions(target_ratio=0.5))
        held, _roots, _faces, _corners = sequence.state(0)
        assert held == pytest.approx(far[: len(held)], abs=1e-9)


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

    def test_a_point_past_the_end_is_refused(self):
        mesh = topology.build(*shapes.octahedron())
        with pytest.raises(DecimateError, match='6 welded points'):
            mesh.classify(locked=np.asarray([6], dtype=np.int64))

    def test_a_mesh_with_no_faces_has_no_components_to_join_and_no_charts(self):
        assert topology.components(np.zeros((0, 3), dtype=np.int64), 3).tolist() == [0, 1, 2]
        empty = np.zeros((0, 3), dtype=np.int64)
        assert len(topology.atlas_charts(empty, empty, np.zeros((0, 2)))) == 0


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

    def test_classifying_a_large_mesh_never_walks_it_point_by_point(self):
        """A scan is millions of triangles, and a set per point is what stops it.

        Classification is whole-array work over the face list, so the structure
        a point-by-point walk would need -- a Python set per point, hundreds of
        bytes each before it holds anything -- is never built. Asserted by
        asking whether it was built rather than by timing the pass, so it is a
        fact about the code rather than about the machine.
        """
        mesh = topology.build(*shapes.grid(300))
        assert mesh.face_count > 175_000
        kinds = mesh.classify()
        assert not mesh.adjacency_built
        # And it is the right answer at that size, not merely a fast one: the
        # rim of the patch is border, and everything inside it is manifold.
        on_rim = np.any(np.abs(mesh.positions[:, [0, 2]]) >= 1.0, axis=1)
        assert np.all(kinds[on_rim] == topology.VertexClass.BORDER)
        assert np.all(kinds[~on_rim] == topology.VertexClass.MANIFOLD)


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

    def test_a_cluster_that_lands_in_one_cell_is_still_welded_whole(self):
        """The case a bad tolerance produces, and the one that used to hang.

        Every point here is within the tolerance of every other, so the cell
        holding them offers the square of its size in candidate pairs. The
        answer is one point; getting there is the part that has to stay
        whole-array and has to bound the memory it asks for.
        """
        rng = np.random.default_rng(0)
        points = rng.normal(scale=1e-9, size=(4000, 3))
        welded, belongs = topology.weld_positions(points, tolerance=1e-6)
        assert len(welded) == 1
        assert np.all(belongs == 0)

    def test_a_tolerance_weld_of_nothing_is_nothing(self):
        welded, belongs = topology.weld_positions(np.zeros((0, 3)), tolerance=1e-6)
        assert welded.shape == (0, 3)
        assert belongs.shape == (0,)

    def test_a_tolerance_finer_than_the_coordinates_can_carry_is_refused(self):
        """Below the float64 spacing there, nothing it could merge is distinct."""
        points = np.asarray([(0.0, 0.0, 0.0), (1e6, 5e5, 3e5)])
        with pytest.raises(DecimateError, match='weld_tolerance'):
            topology.weld_positions(points, tolerance=1e-18)

    def test_a_fine_tolerance_on_a_large_model_still_welds(self):
        """The grid is then too big to pack a cell into one integer.

        Cell coordinates are compared as rows instead of as a packed key, which
        is slower and is the only thing that is correct at that span.
        """
        points = np.asarray([(0.0, 0.0, 0.0), (5e-13, 0.0, 0.0), (1e6, 5e5, 3e5)])
        welded, belongs = topology.weld_positions(points, tolerance=1e-12)
        assert len(welded) == 2
        assert belongs.tolist() == [0, 0, 1]

    def test_the_welded_point_sits_at_the_middle_of_its_run(self):
        step = 4e-6
        positions = np.asarray(
            [(i * step, 0.0, 0.0) for i in range(5)] + [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            dtype='f4',
        )
        indices = np.asarray([0, 5, 6, 4, 5, 6], dtype=np.uint32)
        welded = topology.build(positions, indices, tolerance=step * 1.5)
        assert welded.positions[0][0] == pytest.approx(2.0 * step, rel=1e-3)


class TestAdjacencyIsNotBuiltUntilItIsWanted:
    """771k Python sets cost half a gigabyte before a reduction has started.

    The face-per-vertex adjacency is what the heap schedule walks, and it is
    the single largest thing a `Topology` holds -- far larger than the arrays
    it describes. Building it for a caller who only wants to classify a mesh,
    or who reduces it with a schedule that works in whole arrays, is the
    difference between a scan fitting in memory and not.
    """

    def test_building_a_mesh_does_not_build_the_adjacency(self):
        positions, indices = shapes.grid(300)  # 90,000 points
        tracemalloc.start()
        try:
            mesh = topology.build(positions, indices)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        arrays = mesh.faces.nbytes + mesh.positions.nbytes + mesh.corners.nbytes
        assert peak < 4 * arrays, (
            'building a %d-face mesh peaked at %.0f MB against %.0f MB of arrays'
            % (mesh.face_count, peak / 1e6, arrays / 1e6)
        )

    def test_the_adjacency_is_still_there_when_asked_for(self):
        mesh = topology.build(*shapes.icosphere(1))
        for index, face in enumerate(mesh.faces):
            for point in face:
                assert index in mesh.vertex_faces[point]

    def test_classifying_a_mesh_needs_no_adjacency(self):
        """Classification is whole-array work and must not drag the sets in."""
        mesh = topology.build(*shapes.grid(120))
        mesh.classify()
        assert mesh.adjacency_built is False


class TestDroppingWhatIsTooSmallToSee:
    """A speck that will never cover a pixel is not worth a triangle.

    A scan arrives with the subject and whatever else was in the room: crumbs
    of geometry the reconstruction could not attach to anything. Each is its own
    closed shell with a floor of its own, so a reduction asked for a few hundred
    triangles spends some of them there instead of on the subject.
    """

    def _speckled(self):
        """A grid, plus three tetrahedra a thousandth of its size."""
        positions, indices = shapes.grid(9)
        points = [positions.astype('d')]
        faces = [np.asarray(indices, dtype=np.int64).reshape(-1, 3)]
        offset = len(positions)
        for step, where in enumerate(((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0))):
            corners, speck = shapes.tetrahedron()
            points.append(corners.astype('d') * 0.001 + np.asarray(where))
            faces.append(np.asarray(speck, dtype=np.int64).reshape(-1, 3) + offset)
            offset += len(corners)
            del step
        return (
            np.concatenate(points).astype('f4'),
            np.concatenate(faces).astype(np.uint32).reshape(-1),
        )

    def test_the_specks_are_their_own_components(self):
        positions, indices = self._speckled()
        mesh = topology.build(positions, indices)
        labels = topology.components(mesh.live_faces(), mesh.vertex_count)
        used = np.zeros(mesh.vertex_count, dtype=bool)
        used[mesh.live_faces().reshape(-1)] = True
        assert len(np.unique(labels[used])) == 4

    def test_a_component_is_labelled_across_all_three_edges_of_a_face(self):
        """Two triangles sharing only the edge no walk of one edge would take."""
        positions = np.asarray([(0.0, 0, 0), (1.0, 0, 0), (0.0, 1.0, 0), (1.0, 1.0, 0)], dtype='f4')
        mesh = topology.build(positions, np.asarray([0, 1, 2, 2, 1, 3], dtype=np.uint32))
        labels = topology.components(mesh.live_faces(), mesh.vertex_count)
        assert len(set(labels[mesh.live_faces().reshape(-1)].tolist())) == 1

    def test_dropping_removes_the_specks_and_keeps_the_subject(self):
        positions, indices = self._speckled()
        whole = topology.build(positions, indices)
        pruned = topology.build(positions, indices, drop_below=0.01)
        assert pruned.face_count == whole.face_count - 12
        assert pruned.dropped_faces == 12

    def test_what_is_kept_is_measured_against_the_whole_model(self):
        """A share of the model's own size, so it means the same at any scale."""
        positions, indices = self._speckled()
        big = topology.build(positions * 1000.0, indices, drop_below=0.01)
        small = topology.build(positions * 0.001, indices, drop_below=0.01)
        assert big.face_count == small.face_count

    def test_dropping_nothing_is_the_default(self):
        positions, indices = self._speckled()
        assert topology.build(positions, indices).dropped_faces == 0

    def test_a_share_large_enough_to_take_everything_leaves_the_largest(self):
        """The model cannot be smaller than itself, so one component survives."""
        positions, indices = self._speckled()
        mesh = topology.build(positions, indices, drop_below=2.0)
        assert mesh.face_count == 128
        assert mesh.dropped_faces == 12

    def test_an_empty_mesh_drops_nothing(self):
        mesh = topology.build(
            np.zeros((0, 3), dtype='f4'), np.zeros((0,), dtype=np.uint32), drop_below=0.1
        )
        assert mesh.face_count == 0
