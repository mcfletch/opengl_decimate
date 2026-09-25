"""The quadric algebra: does a quadric measure the distance it claims to?

Every one of these asks a question with an answer known independently of the
implementation -- a point's distance to a plane, the intersection of three
planes, the limit of the probabilistic form as its noise goes to zero.
"""

import numpy as np
import pytest

from opengl_decimate import quadrics
from opengl_decimate.options import DEFAULT_NORMAL_NOISE


def _plane(normal, point):
    """One plane's quadric, as the (1, 10) batch the module works in."""
    return quadrics.plane_quadric(np.asarray([normal], dtype='d'), np.asarray([point], dtype='d'))


class TestPlaneQuadric:
    def test_storage_is_ten_coefficients(self):
        q = _plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
        assert q.shape == (1, quadrics.QUADRIC_SIZE)
        assert quadrics.QUADRIC_SIZE == 10

    def test_a_point_on_the_plane_costs_nothing(self):
        q = _plane((0.0, 1.0, 0.0), (0.0, 3.0, 0.0))
        on = np.asarray([[5.0, 3.0, -2.0]], dtype='d')
        assert quadrics.evaluate(q, on)[0] == pytest.approx(0.0, abs=1e-12)

    def test_cost_is_the_squared_distance_to_the_plane(self):
        q = _plane((0.0, 1.0, 0.0), (0.0, 3.0, 0.0))
        off = np.asarray([[5.0, 7.0, -2.0]], dtype='d')
        assert quadrics.evaluate(q, off)[0] == pytest.approx(16.0)

    def test_an_unnormalised_normal_is_normalised(self):
        """A plane is the same plane however long the vector describing it is."""
        q = quadrics.plane_quadric(
            np.asarray([[0.0, 4.0, 0.0]], dtype='d'), np.asarray([[0.0, 3.0, 0.0]], dtype='d')
        )
        off = np.asarray([[0.0, 7.0, 0.0]], dtype='d')
        assert quadrics.evaluate(q, off)[0] == pytest.approx(16.0)

    def test_weight_scales_the_cost(self):
        q = quadrics.plane_quadric(
            np.asarray([[0.0, 1.0, 0.0]], dtype='d'),
            np.asarray([[0.0, 0.0, 0.0]], dtype='d'),
            weights=np.asarray([2.5], dtype='d'),
        )
        off = np.asarray([[0.0, 2.0, 0.0]], dtype='d')
        assert quadrics.evaluate(q, off)[0] == pytest.approx(10.0)

    def test_a_degenerate_normal_contributes_nothing(self):
        """A zero-length normal describes no plane, so it must cost nothing."""
        q = quadrics.plane_quadric(
            np.asarray([[0.0, 0.0, 0.0]], dtype='d'), np.asarray([[1.0, 2.0, 3.0]], dtype='d')
        )
        anywhere = np.asarray([[9.0, -4.0, 2.0]], dtype='d')
        assert quadrics.evaluate(q, anywhere)[0] == pytest.approx(0.0, abs=1e-12)


class TestAccumulation:
    def test_quadrics_add(self):
        """The whole point of the representation: cost of a sum is a sum of costs."""
        a = _plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
        b = _plane((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        point = np.asarray([[3.0, 4.0, 0.0]], dtype='d')
        assert quadrics.evaluate(a + b, point)[0] == pytest.approx(9.0 + 16.0)

    def test_accumulate_sums_by_vertex(self):
        """Three faces, two vertices: each vertex gets the faces that touch it."""
        per_face = np.asarray(
            [
                _plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0))[0],
                _plane((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))[0],
            ],
            dtype='d',
        )
        faces = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
        summed = quadrics.accumulate(per_face, faces, 2)
        point = np.asarray([[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]], dtype='d')
        # Vertex 0 has the y-plane three times over, vertex 1 the x-plane.
        assert quadrics.evaluate(summed, point) == pytest.approx([3 * 16.0, 3 * 9.0])


class TestTriangleQuadrics:
    def test_area_weighted_by_default(self):
        """A triangle four times the area weighs four times as much."""
        small = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='d')
        large = small * 2.0
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        qs = quadrics.triangle_quadrics(small, faces)
        ql = quadrics.triangle_quadrics(large, faces)
        off = np.asarray([[0.0, 0.0, 1.0]], dtype='d')
        assert quadrics.evaluate(ql, off)[0] == pytest.approx(4.0 * quadrics.evaluate(qs, off)[0])

    def test_a_zero_area_triangle_contributes_nothing(self):
        """Collinear corners describe no plane; a scan is full of them."""
        flat = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype='d')
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        q = quadrics.triangle_quadrics(flat, faces)
        assert np.all(np.isfinite(q))
        off = np.asarray([[0.0, 5.0, 5.0]], dtype='d')
        assert quadrics.evaluate(q, off)[0] == pytest.approx(0.0, abs=1e-12)


class TestMinimize:
    def test_three_orthogonal_planes_meet_at_a_point(self):
        q = (
            _plane((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
            + _plane((0.0, 1.0, 0.0), (0.0, 3.0, 0.0))
            + _plane((0.0, 0.0, 1.0), (0.0, 0.0, 5.0))
        )
        point, ok = quadrics.minimize(q)
        assert ok[0]
        assert point[0] == pytest.approx([2.0, 3.0, 5.0])

    def test_a_single_plane_is_singular(self):
        """One plane constrains one direction; the minimiser must say so."""
        point, ok = quadrics.minimize(_plane((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
        assert not ok[0]
        assert np.all(np.isfinite(point))

    def test_two_planes_are_singular(self):
        """Two planes leave a line of minima, which is still not a point."""
        q = _plane((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)) + _plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
        _, ok = quadrics.minimize(q)
        assert not ok[0]

    def test_the_minimum_is_the_lowest_cost_around(self):
        q = (
            _plane((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
            + _plane((0.0, 1.0, 0.0), (0.0, 3.0, 0.0))
            + _plane((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        )
        point, ok = quadrics.minimize(q)
        assert ok[0]
        best = quadrics.evaluate(q, point)[0]
        nearby = point + np.random.default_rng(0).normal(scale=0.1, size=(64, 3))
        assert np.all(quadrics.evaluate(np.repeat(q, 64, axis=0), nearby) >= best - 1e-12)


class TestProbabilistic:
    def test_zero_noise_is_the_classical_quadric(self):
        normals = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype='d')
        points = np.asarray([[0.0, 2.0, 0.0], [4.0, 0.0, 0.0]], dtype='d')
        classical = quadrics.plane_quadric(normals, points)
        probabilistic = quadrics.plane_quadric(
            normals, points, position_noise=0.0, normal_noise=0.0
        )
        assert probabilistic == pytest.approx(classical)

    def test_noise_makes_a_single_plane_solvable(self):
        """The robustness claim: noise on the normal makes the system full rank."""
        q = quadrics.plane_quadric(
            np.asarray([[0.0, 1.0, 0.0]], dtype='d'),
            np.asarray([[0.0, 1.0, 0.0]], dtype='d'),
            normal_noise=0.1,
        )
        point, ok = quadrics.minimize(q)
        assert ok[0]
        assert point[0] == pytest.approx([0.0, 1.0, 0.0])

    def test_the_default_noise_solves_a_flat_fan(self):
        """What the probabilistic metric is for: a flat neighbourhood has a minimum.

        Six coplanar triangles around a point, at the noise the options default
        to. The minimum is within the fan, at its own plane.
        """
        angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
        rim = np.stack([np.cos(angles), np.zeros(6), np.sin(angles)], axis=1)
        q = quadrics.plane_quadric(
            np.tile([0.0, 1.0, 0.0], (6, 1)), rim, normal_noise=DEFAULT_NORMAL_NOISE
        ).sum(axis=0, keepdims=True)
        point, ok = quadrics.minimize(q)
        assert ok[0]
        assert point[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    def test_cost_stays_non_negative(self):
        """It is an expected squared error, so it cannot be below zero."""
        rng = np.random.default_rng(7)
        normals = rng.normal(size=(32, 3))
        points = rng.normal(size=(32, 3))
        q = quadrics.plane_quadric(normals, points, position_noise=0.05, normal_noise=0.05)
        probes = rng.normal(size=(32, 3)) * 4.0
        assert np.all(quadrics.evaluate(q, probes) > -1e-9)

    def test_noise_raises_the_cost_of_the_plane_s_own_point(self):
        """Uncertainty means even the sample point is no longer free."""
        normals = np.asarray([[0.0, 1.0, 0.0]], dtype='d')
        points = np.asarray([[0.0, 1.0, 0.0]], dtype='d')
        noisy = quadrics.plane_quadric(normals, points, position_noise=0.1, normal_noise=0.1)
        assert quadrics.evaluate(noisy, points)[0] > 1e-6

    def test_matches_a_direct_expectation(self):
        """Against a Monte-Carlo estimate of the same expected squared error."""
        rng = np.random.default_rng(11)
        mean_normal = np.asarray([0.0, 1.0, 0.0], dtype='d')
        mean_point = np.asarray([0.0, 1.0, 0.0], dtype='d')
        sigma_n, sigma_p = 0.08, 0.05
        probe = np.asarray([0.4, 1.7, -0.3], dtype='d')

        draws = 400000
        n = mean_normal + rng.normal(scale=sigma_n, size=(draws, 3))
        p = mean_point + rng.normal(scale=sigma_p, size=(draws, 3))
        sampled = np.mean(np.sum((probe - p) * n, axis=1) ** 2)

        q = quadrics.plane_quadric(
            mean_normal[None, :],
            mean_point[None, :],
            position_noise=sigma_p,
            normal_noise=sigma_n,
        )
        assert quadrics.evaluate(q, probe[None, :])[0] == pytest.approx(sampled, rel=0.02)
