"""The NumPy path rounds its dot products the same on every platform.

``np.einsum`` sums products with a fused multiply-add where the processor has
one, so the last bit of a row's dot product differs between x86-64 and arm64.
A reduction orders its contractions by cost, and a symmetric mesh is full of
ties that bit settles, so the same mesh and options would reduce to a different
mesh on each. The package adds its products in a fixed order instead.
"""

import numpy as np
import pytest
import shapes

from opengl_decimate import SimplifyOptions, _vectors, certify, native, simplify


def _refuse(*args, **named):
    raise AssertionError(
        'np.einsum%r rounds differently from one platform to the next' % (args[:1],)
    )


@pytest.fixture
def no_einsum(monkeypatch):
    """Fail any call to ``np.einsum``, and run the NumPy reduction."""
    monkeypatch.setattr(np, 'einsum', _refuse)
    monkeypatch.setattr(native, 'ACCELERATED', False)


@pytest.mark.usefixtures('no_einsum')
class TestNothingIsSummedByEinsum:
    @pytest.mark.parametrize('schedule', ['heap', 'multiple-choice'])
    def test_a_closed_surface(self, schedule):
        positions, indices = shapes.icosphere(2)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.5, schedule=schedule, min_triangle_quality=0.1),
        )
        assert result.triangle_count < len(indices) // 3

    def test_a_border_under_the_probabilistic_metric(self):
        positions, indices = shapes.grid(10, bump=0.3)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(
                target_ratio=0.5, metric='probabilistic', position_noise=1e-3, normal_noise=1e-3
            ),
        )
        assert result.triangle_count < len(indices) // 3

    def test_welding_within_a_tolerance(self):
        positions, indices = shapes.nearly_coincident()
        result = simplify(
            {'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.5, weld_tolerance=1e-5)
        )
        assert result.triangle_count <= len(indices) // 3

    def test_attributes_across_a_crease(self):
        attributes, indices = shapes.cube_with_hard_normals()
        result = simplify(attributes, indices, SimplifyOptions(target_ratio=0.6))
        assert result.triangle_count < len(indices) // 3

    def test_recomputed_normals(self):
        positions, indices = shapes.icosphere(2)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.5, recompute_normals=True),
        )
        assert 'NORMAL' in result.attributes

    def test_measuring_the_result(self):
        positions, indices = shapes.icosphere(2)
        result = simplify(
            {'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.5, certify=True)
        )
        assert result.measured_error is not None
        points = certify.sample_surface(positions, indices, 200, seed=1)
        assert len(certify.distance_to_mesh(points, positions, indices)) == 200


class TestDot:
    def test_it_adds_the_three_products_in_order(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(2, 500, 3))
        expected = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]
        assert np.array_equal(_vectors.dot(a, b), expected)

    def test_it_takes_the_last_axis_of_any_shape(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(2, 7, 5, 3))
        assert _vectors.dot(a, b).shape == (7, 5)
        assert np.array_equal(_vectors.dot(a, b)[3], _vectors.dot(a[3], b[3]))

    def test_a_length_is_the_root_of_a_vector_with_itself(self):
        assert _vectors.length(np.array([3.0, 4.0, 12.0])) == 13.0
