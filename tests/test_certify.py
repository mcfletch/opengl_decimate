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


class TestDistanceToMesh:
    @pytest.mark.parametrize(
        ('point', 'expected'),
        [
            ((0.25, 0.25, 2.0), 2.0),  # over the face
            ((0.25, 0.25, 0.0), 0.0),  # on the face
            ((5.0, 0.0, 0.0), 4.0),  # past a corner
            ((-1.0, -1.0, 0.0), 2.0**0.5),  # past the right-angled corner
            ((1.0, 1.0, 0.0), 0.5**0.5),  # past the long edge
            ((0.5, -1.0, 0.0), 1.0),  # past a short edge
        ],
    )
    def test_the_closest_point_is_found_in_every_region(self, point, expected):
        got = certify.distance_to_mesh(np.asarray([point], dtype='d'), *_TRIANGLE)
        assert got[0] == pytest.approx(expected)

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
