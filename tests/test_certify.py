"""Measuring how far a reduced surface actually moved.

The quadric's own number is an estimate accumulated from planes. This module
measures instead: sample both surfaces, and ask each sample how far it is from
the other. The tests pin the primitive it rests on -- point to triangle -- to
distances that can be worked out on paper, then check the whole measure against
shapes whose deviation is known.
"""

import numpy as np
import pytest
import shapes

from opengl_decimate import SimplifyOptions, certify, simplify

#: The unit right triangle in the z = 0 plane, used for the distances below.
_TRIANGLE = (
    np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], dtype='d'),
    np.asarray([0, 1, 2], dtype=np.uint32),
)


def _the_slow_way(point, triangle):
    """Point-to-triangle distance, worked out one case at a time.

    The projection onto the triangle's plane where that lands inside it, and
    otherwise the nearest point of each of the three edges, each a segment
    clamp. Slow, obvious, and independent of the seven-region selector it is
    used to check.
    """
    a, b, c = triangle
    normal = np.cross(b - a, c - a)
    area = float(np.linalg.norm(normal))
    unit = normal / area
    flat = point - unit * float(np.dot(point - a, unit))
    inside = all(
        float(np.dot(np.cross(end - start, flat - start), unit)) / area >= -1e-12
        for start, end in ((a, b), (b, c), (c, a))
    )
    if inside:
        return float(np.linalg.norm(point - flat))
    nearest = float('inf')
    for start, end in ((a, b), (b, c), (c, a)):
        along = end - start
        share = np.clip(float(np.dot(point - start, along)) / float(np.dot(along, along)), 0.0, 1.0)
        nearest = min(nearest, float(np.linalg.norm(point - (start + along * share))))
    return nearest


class TestDistanceToMesh:
    @pytest.mark.parametrize(
        ('point', 'expected'),
        [
            ((0.25, 0.25, 2.0), 2.0),  # over the face
            ((0.25, 0.25, 0.0), 0.0),  # on the face
            ((5.0, 0.0, 0.0), 4.0),  # past corner B
            ((-1.0, -1.0, 0.0), 2.0**0.5),  # past corner A, the right angle
            ((-1.0, 5.0, 0.0), 17.0**0.5),  # past corner C
            ((1.0, 1.0, 0.0), 0.5**0.5),  # past edge BC, the long one
            ((0.5, -1.0, 0.0), 1.0),  # past edge AB
            ((-1.0, 0.5, 0.0), 1.0),  # past edge AC
        ],
    )
    def test_the_closest_point_is_found_in_every_region(self, point, expected):
        """All seven: the face, the three edges, and the three corners."""
        got = certify.distance_to_mesh(np.asarray([point], dtype='d'), *_TRIANGLE)
        assert got[0] == pytest.approx(expected)

    def test_it_agrees_with_the_distance_worked_out_the_obvious_way(self):
        """An oracle that shares no code with the thing it is checking.

        The seven-region selector is one expression evaluated for every pair at
        once, which is what makes it fast and what makes an error in it uniform
        rather than obvious. :func:`_the_slow_way` is the same question asked
        the plodding way -- the projection if it lands inside, and otherwise
        three segment clamps -- and has nothing in common with it but the
        answer.
        """
        positions, indices = shapes.icosphere(1)
        corners = positions.astype('d')[indices.reshape(-1, 3)]
        rng = np.random.default_rng(11)
        probes = rng.normal(size=(60, 3)) * 1.3
        got = certify.distance_to_mesh(probes, positions, indices)
        wanted = [min(_the_slow_way(point, triangle) for triangle in corners) for point in probes]
        assert got == pytest.approx(wanted)

    def test_every_vertex_of_a_mesh_is_on_it(self):
        positions, indices = shapes.icosphere(2)
        got = certify.distance_to_mesh(positions.astype('d'), positions, indices)
        assert np.max(got) < 1e-6

    def test_a_mesh_with_no_triangles_is_infinitely_far_away(self):
        got = certify.distance_to_mesh(
            np.zeros((2, 3), dtype='d'),
            np.zeros((0, 3), dtype='f4'),
            np.zeros((0,), dtype=np.uint32),
        )
        assert np.all(np.isinf(got))


class TestSampling:
    def test_samples_land_on_the_surface(self):
        positions, indices = shapes.icosphere(2)
        points = certify.sample_surface(positions, indices, 500, seed=1)
        assert points.shape == (500, 3)
        assert np.max(certify.distance_to_mesh(points, positions, indices)) < 1e-6

    def test_samples_are_spread_by_area(self):
        """A triangle nine times the area must attract about nine times as many."""
        positions = np.asarray(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -3.0),
                (3.0, 0.0, -3.0),
                (0.0, 3.0, -3.0),
            ],
            dtype='f4',
        )
        indices = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        points = certify.sample_surface(positions, indices, 4000, seed=2)
        on_large = np.count_nonzero(points[:, 2] < -1.0)
        assert on_large / 4000 == pytest.approx(0.9, abs=0.02)

    def test_the_same_seed_samples_the_same_points(self):
        positions, indices = shapes.icosphere(1)
        first = certify.sample_surface(positions, indices, 100, seed=3)
        second = certify.sample_surface(positions, indices, 100, seed=3)
        assert np.array_equal(first, second)


class TestSurfaceDeviation:
    def test_a_mesh_does_not_deviate_from_itself(self):
        positions, indices = shapes.icosphere(2)
        found = certify.surface_deviation(positions, indices, positions, indices, samples=500)
        assert found.max < 1e-6
        assert found.rms < 1e-6

    def test_a_shrunken_copy_deviates_by_how_much_it_shrank(self):
        positions, indices = shapes.icosphere(3)
        smaller = (positions.astype('d') * 0.97).astype('f4')
        found = certify.surface_deviation(positions, indices, smaller, indices, samples=2000)
        assert found.max == pytest.approx(0.03, abs=0.005)

    def test_it_notices_a_surface_that_is_missing_a_piece(self):
        """One-sided sampling would miss it; both directions are measured."""
        positions, indices = shapes.icosphere(2)
        gapped = indices.reshape(-1, 3)[:-40].reshape(-1)
        found = certify.surface_deviation(positions, indices, positions, gapped, samples=2000)
        assert found.max > 0.1

    def test_deviation_grows_as_a_reduction_is_pushed_further(self):
        positions, indices = shapes.icosphere(3)
        previous = 0.0
        for ratio in (0.5, 0.25, 0.1):
            result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=ratio))
            found = certify.surface_deviation(
                positions, indices, result.attributes['POSITION'], result.indices, samples=2000
            )
            assert found.max > previous
            previous = found.max

    def test_a_quarter_of_a_sphere_s_triangles_still_makes_a_sphere(self):
        """The gate the reduction has to hold: a measured bound, not an estimate."""
        positions, indices = shapes.icosphere(4)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.25))
        found = certify.surface_deviation(
            positions, indices, result.attributes['POSITION'], result.indices, samples=4000
        )
        assert found.max < 0.01
        assert found.rms < 0.003


class TestCertifyingAReduction:
    def test_a_reduction_can_measure_itself(self):
        """The estimate and the measurement are different numbers, both wanted."""
        positions, indices = shapes.icosphere(3)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.25, certify=True),
        )
        assert result.measured_error is not None
        assert result.measured_error > 0.0
        assert result.error > 0.0

    def test_it_is_not_measured_unless_asked(self):
        positions, indices = shapes.icosphere(2)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.5))
        assert result.measured_error is None

    def test_the_measured_error_is_the_deviation_of_what_came_back(self):
        positions, indices = shapes.icosphere(3)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.25, certify=True, certify_samples=2000),
        )
        direct = certify.surface_deviation(
            positions, indices, result.attributes['POSITION'], result.indices, samples=2000
        )
        assert result.measured_error == pytest.approx(direct.max)


class TestEmptySurfaces:
    def test_a_mesh_with_no_triangles_cannot_be_sampled(self):
        points = certify.sample_surface(
            np.zeros((0, 3), dtype='f4'), np.zeros((0,), dtype=np.uint32), 10
        )
        assert points.shape == (0, 3)

    def test_two_empty_surfaces_deviate_infinitely(self):
        """Nothing to sample and nothing to measure against: not a zero answer."""
        empty = (np.zeros((0, 3), dtype='f4'), np.zeros((0,), dtype=np.uint32))
        found = certify.surface_deviation(*empty, *empty, samples=8)
        assert found.max == float('inf')
        assert found.rms == float('inf')


class TestMeasuringABigSurface:
    """The measurement has to survive the meshes it exists to measure.

    Testing every sample against every triangle costs memory proportional to
    the product, and on a scan that product is billions: a 1.5M-triangle
    reference and a block of 256 samples is nine gigabytes of temporaries for
    one block. The answer is not to allocate it.
    """

    def test_measuring_against_a_dense_mesh_stays_within_a_budget(self):
        import tracemalloc

        positions, indices = shapes.icosphere(5)  # 20,480 triangles
        points = certify.sample_surface(positions, indices, 2000, seed=1)
        tracemalloc.start()
        try:
            certify.distance_to_mesh(points, positions, indices)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 200e6, 'peaked at %.1f GB measuring a 20k-triangle mesh' % (peak / 1e9)

    def test_it_still_gets_the_same_answer(self):
        """Whatever it does to stay small, the distances must not move."""
        positions, indices = shapes.icosphere(3)
        rng = np.random.default_rng(5)
        probes = rng.normal(size=(200, 3)) * 1.4
        got = certify.distance_to_mesh(probes, positions, indices)
        corners = positions.astype('d')[indices.reshape(-1, 3)]
        wanted = [min(_the_slow_way(point, triangle) for triangle in corners) for point in probes]
        assert got == pytest.approx(wanted)


class TestTheGridGivesTheSameAnswerAsTheScan:
    """Searching nearby cells rather than every triangle is exact, not close.

    A triangle is registered in every cell its bounding box covers, so once a
    point's nearest is no further away than the ring searched so far, no
    triangle outside that ring can beat it. These check the claim where it
    matters: on the surface, off it, far from it, and on a mesh with one huge
    triangle among small ones, which is what decides the cell size.
    """

    @pytest.mark.parametrize('spread', [0.0, 0.05, 1.4, 8.0], ids=['on', 'near', 'off', 'far'])
    def test_the_two_agree_wherever_the_points_are(self, spread):
        positions, indices = shapes.icosphere(4)
        corners = positions.astype('d')[np.asarray(indices).reshape(-1, 3)]
        rng = np.random.default_rng(3)
        probes = certify.sample_surface(positions, indices, 1500, seed=2)
        probes = probes + rng.normal(scale=spread, size=probes.shape)
        by_grid = certify._nearest_by_grid(probes, corners)
        by_scan = np.sqrt(certify._nearest_among(probes, corners))
        assert np.array_equal(by_grid, by_scan)

    def test_one_huge_triangle_among_small_ones_does_not_break_it(self):
        """The cell side is chosen from the mesh, so an outlier has to survive it."""
        positions, indices = shapes.icosphere(3)
        positions = np.concatenate(
            [positions, np.asarray([(40.0, 0, 0), (0, 40.0, 0), (0, 0, 40.0)], dtype='f4')]
        )
        base = len(positions) - 3
        indices = np.concatenate([indices, np.asarray([base, base + 1, base + 2], dtype=np.uint32)])
        corners = positions.astype('d')[np.asarray(indices).reshape(-1, 3)]
        rng = np.random.default_rng(7)
        probes = rng.normal(size=(1200, 3)) * 3.0
        assert np.array_equal(
            certify._nearest_by_grid(probes, corners),
            np.sqrt(certify._nearest_among(probes, corners)),
        )

    @pytest.mark.parametrize(
        ('corners', 'side'),
        [
            # Triangles with no extent at all say nothing about a cell size, so
            # the mesh's own extent stands in -- and where that is nothing
            # either, any positive side will do, since everything is in one cell.
            pytest.param(np.zeros((4, 3, 3)), 1.0, id='a mesh at one point'),
            pytest.param(
                np.stack([np.full((3, 3), 0.0), np.full((3, 3), 6.0)]), 6.0, id='two points apart'
            ),
        ],
    )
    def test_a_cell_size_is_chosen_even_for_a_mesh_with_no_area(self, corners, side):
        low, high = np.min(corners, axis=1), np.max(corners, axis=1)
        assert certify._grid_side(low, high) == side

    def test_a_mesh_of_one_triangle_is_still_a_grid(self):
        probes = np.asarray([(0.25, 0.25, 2.0), (5.0, 0.0, 0.0), (-1.0, 0.5, 0.0)])
        corners = _TRIANGLE[0][_TRIANGLE[1].reshape(-1, 3)]
        assert certify._nearest_by_grid(probes, corners) == pytest.approx([2.0, 4.0, 1.0])


class TestWhenToStopSearchingAndJustScan:
    """The grid search gives up at the point where giving up is cheaper.

    Widening the search by one ring costs every waiting point the triangles in
    that ring; scanning costs every waiting point the *whole* reference. So the
    comparison is between those two, and it has to be made in triangles tested
    rather than in points or cells, which are not the same currency. On a
    million-triangle reference the difference is three orders of magnitude, and
    getting it wrong means the last handful of points cost more than every other
    point put together.
    """

    def test_a_ring_that_is_cheaper_than_the_whole_mesh_is_searched(self):
        # 300 cells of 8 triangles is 2,400 tests a point; the mesh is a
        # million. Widening wins by a wide margin.
        assert not certify.scan_is_cheaper(ring_cells=300, per_cell=8.0, triangles=1_000_000)

    def test_a_ring_that_costs_more_than_the_whole_mesh_is_not(self):
        assert certify.scan_is_cheaper(ring_cells=300, per_cell=8.0, triangles=2_000)

    def test_a_small_reference_is_scanned_almost_at_once(self):
        """Which is what makes the fallback right for the meshes it suits."""
        assert certify.scan_is_cheaper(ring_cells=26, per_cell=4.0, triangles=64)

    def test_an_empty_grid_stops_rather_than_widening_for_ever(self):
        """Free rings would otherwise be searched without end."""
        assert certify.scan_is_cheaper(ring_cells=26, per_cell=0.0, triangles=10)

    def test_the_far_point_of_a_dense_mesh_is_still_measured_exactly(self):
        """The rule may only change what it costs, never what it answers."""
        rng = np.random.default_rng(5)
        positions, indices = shapes.icosphere(4)
        # Points well outside the sphere, so the search has to widen for them,
        # and points on it, which settle in the first ring.
        outside = rng.normal(size=(64, 3))
        outside = 3.0 * outside / np.linalg.norm(outside, axis=1)[:, None]
        on = certify.sample_surface(positions, indices, 64, seed=1)
        points = np.concatenate([outside, on])

        through_the_grid = certify.distance_to_mesh(points, positions, indices)
        corners = np.asarray(positions, dtype='d')[np.asarray(indices).reshape(-1, 3)]
        by_hand = np.sqrt(certify._nearest_among(np.asarray(points, dtype='d'), corners))
        assert np.allclose(through_the_grid, by_hand, atol=1e-9)
