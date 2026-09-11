"""Decimating a whole mesh: does it hit the target, and is what comes out a mesh?

The shapes are chosen so the right answer is known without reference to the
implementation: a flat patch is exactly representable by two triangles, a sphere
of radius one stays a sphere of radius one, a closed surface stays closed.
"""

import numpy as np
import pytest
import shapes

from opengl_decimate import SimplifyOptions, simplify
from opengl_decimate.types import DecimateError


def _triangles(result):
    return len(result.indices) // 3


def _faces(result):
    return result.indices.reshape(-1, 3)


def _closed(positions, indices):
    """True where every edge carries exactly two triangles."""
    faces = np.asarray(indices).reshape(-1, 3)
    pairs = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0), axis=1
    )
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    del positions
    return bool(np.all(counts == 2))


def _signed_volume(positions, indices):
    corners = np.asarray(positions, dtype='d')[np.asarray(indices).reshape(-1, 3)]
    return float(
        np.einsum('ij,ij->i', corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6.0
    )


class TestTargets:
    def test_a_ratio_reaches_the_share_of_triangles_asked_for(self):
        positions, indices = shapes.icosphere(3)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.25))
        assert _triangles(result) <= 0.25 * (len(indices) // 3)
        assert _triangles(result) >= 0.2 * (len(indices) // 3)

    def test_a_count_is_not_exceeded(self):
        positions, indices = shapes.icosphere(3)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_count=200))
        assert _triangles(result) <= 200

    def test_an_error_budget_stops_the_reduction(self):
        """A tight budget keeps more triangles than a loose one."""
        positions, indices = shapes.icosphere(3)
        tight = simplify({'POSITION': positions}, indices, SimplifyOptions(target_error=0.002))
        loose = simplify({'POSITION': positions}, indices, SimplifyOptions(target_error=0.05))
        assert _triangles(tight) > _triangles(loose)

    def test_a_target_is_required(self):
        with pytest.raises(DecimateError, match='target'):
            SimplifyOptions()

    def test_a_ratio_outside_the_unit_interval_is_refused(self):
        with pytest.raises(DecimateError, match='target_ratio'):
            SimplifyOptions(target_ratio=1.5)

    def test_an_unknown_schedule_is_refused(self):
        with pytest.raises(DecimateError, match='schedule'):
            SimplifyOptions(target_ratio=0.5, schedule='telepathy')

    def test_an_unknown_metric_is_refused(self):
        with pytest.raises(DecimateError, match='metric'):
            SimplifyOptions(target_ratio=0.5, metric='vibes')


class TestInput:
    def test_position_is_required(self):
        with pytest.raises(DecimateError, match='POSITION'):
            simplify(
                {'NORMAL': np.zeros((3, 3), dtype='f4')},
                np.asarray([0, 1, 2]),
                SimplifyOptions(target_ratio=0.5),
            )

    def test_an_attribute_of_the_wrong_length_is_refused(self):
        positions, indices = shapes.octahedron()
        with pytest.raises(DecimateError, match='NORMAL'):
            simplify(
                {'POSITION': positions, 'NORMAL': np.zeros((3, 3), dtype='f4')},
                indices,
                SimplifyOptions(target_ratio=0.5),
            )

    def test_a_mesh_with_no_triangles_comes_back_empty(self):
        result = simplify(
            {'POSITION': np.zeros((0, 3), dtype='f4')},
            np.zeros((0,), dtype=np.uint32),
            SimplifyOptions(target_ratio=0.5),
        )
        assert _triangles(result) == 0


class TestGeometry:
    def test_a_flat_patch_reduces_to_almost_nothing_for_almost_nothing(self):
        """A plane is exactly representable, so the error must stay at zero."""
        positions, indices = shapes.grid(9)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_count=2))
        assert _triangles(result) <= 8
        assert result.error < 1e-6

    def test_a_sphere_stays_on_the_sphere(self):
        positions, indices = shapes.icosphere(3)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.25))
        radii = np.linalg.norm(result.attributes['POSITION'].astype('d'), axis=1)
        assert np.max(np.abs(radii - 1.0)) < 0.02

    def test_a_closed_surface_stays_closed(self):
        positions, indices = shapes.icosphere(3)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.2))
        assert _closed(result.attributes['POSITION'], result.indices)

    def test_the_surface_keeps_its_orientation(self):
        """Every face of a reduced sphere still faces outward."""
        positions, indices = shapes.icosphere(3)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.2))
        assert _signed_volume(result.attributes['POSITION'], result.indices) > 0

    def test_volume_is_broadly_kept(self):
        positions, indices = shapes.icosphere(3)
        before = _signed_volume(positions, indices)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.2))
        after = _signed_volume(result.attributes['POSITION'], result.indices)
        assert after == pytest.approx(before, rel=0.05)

    def test_no_triangle_comes_back_degenerate(self):
        positions, indices = shapes.icosphere(3)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.2))
        faces = _faces(result)
        assert np.all(faces[:, 0] != faces[:, 1])
        assert np.all(faces[:, 1] != faces[:, 2])
        assert np.all(faces[:, 0] != faces[:, 2])

    def test_every_vertex_returned_is_used(self):
        positions, indices = shapes.icosphere(3)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.2))
        used = np.unique(result.indices)
        assert len(used) == len(result.attributes['POSITION'])

    def test_a_border_is_kept_where_it_was(self):
        """The rim of a patch must still trace the same square."""
        positions, indices = shapes.grid(9, bump=0.4)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.2))
        out = result.attributes['POSITION'].astype('d')
        extent = np.max(np.abs(out[:, [0, 2]]))
        assert extent == pytest.approx(1.0)

    def test_locking_the_boundary_keeps_every_rim_vertex(self):
        positions, indices = shapes.grid(6)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_count=2, lock_boundary=True),
        )
        out = result.attributes['POSITION'].astype('d')
        on_rim = np.any(np.abs(positions.astype('d')[:, [0, 2]]) >= 1.0, axis=1)
        for wanted in positions.astype('d')[on_rim]:
            assert np.min(np.linalg.norm(out - wanted, axis=1)) < 1e-9


class TestAttributes:
    def test_attributes_come_back_for_every_output_vertex(self):
        attributes, indices = shapes.cube_with_hard_normals()
        result = simplify(attributes, indices, SimplifyOptions(target_ratio=0.6))
        count = len(result.attributes['POSITION'])
        assert result.attributes['NORMAL'].shape == (count, 3)
        assert result.attributes['TEXCOORD_0'].shape == (count, 2)

    def test_carried_attributes_keep_their_dtype(self):
        attributes, indices = shapes.cube_with_hard_normals()
        joints = np.zeros((len(attributes['POSITION']), 4), dtype=np.uint16)
        joints[:, 0] = np.arange(len(joints), dtype=np.uint16) % 7
        result = simplify(
            dict(attributes, JOINTS_0=joints), indices, SimplifyOptions(target_ratio=0.6)
        )
        assert result.attributes['JOINTS_0'].dtype == np.uint16

    def test_a_carried_attribute_is_always_one_of_the_input_values(self):
        """Nothing is averaged, so a joint index is still a joint that exists."""
        attributes, indices = shapes.cube_with_hard_normals()
        result = simplify(attributes, indices, SimplifyOptions(target_ratio=0.6))
        original = {tuple(row) for row in attributes['NORMAL'].tolist()}
        for row in result.attributes['NORMAL'].tolist():
            assert tuple(row) in original

    def test_normals_can_be_recomputed_for_the_surface_that_is_left(self):
        positions, indices = shapes.icosphere(3)
        stale = np.tile(np.asarray([0.0, 1.0, 0.0], dtype='f4'), (len(positions), 1))
        result = simplify(
            {'POSITION': positions, 'NORMAL': stale},
            indices,
            SimplifyOptions(target_ratio=0.3, recompute_normals=True),
        )
        out = result.attributes
        outward = np.einsum('ij,ij->i', out['NORMAL'].astype('d'), out['POSITION'].astype('d'))
        assert np.all(outward > 0.5)

    def test_the_vertex_map_points_every_input_vertex_at_an_output_vertex(self):
        positions, indices = shapes.icosphere(2)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.3))
        mapped = result.vertex_map
        assert mapped.shape == (len(positions),)
        assert np.all(mapped >= 0)
        assert np.all(mapped < len(result.attributes['POSITION']))

    def test_the_vertex_map_is_the_correspondence_a_geomorph_needs(self):
        """Each input vertex maps somewhere near where it started."""
        positions, indices = shapes.icosphere(3)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.25))
        target = result.attributes['POSITION'].astype('d')[result.vertex_map]
        assert np.max(np.linalg.norm(target - positions.astype('d'), axis=1)) < 0.35


class TestSchedules:
    def test_the_multiple_choice_schedule_reaches_the_target(self):
        positions, indices = shapes.icosphere(3)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.25, schedule='multiple-choice'),
        )
        assert _triangles(result) <= 0.25 * (len(indices) // 3)

    def test_the_heap_is_the_better_of_the_two(self):
        """Sampling trades a little quality for dropping the global ordering."""
        positions, indices = shapes.icosphere(3)
        heap = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.15))
        sampled = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.15, schedule='multiple-choice'),
        )
        assert heap.error <= sampled.error
        assert sampled.error < 4.0 * max(heap.error, 1e-9)

    def test_the_same_seed_gives_the_same_mesh(self):
        positions, indices = shapes.icosphere(3)
        options = SimplifyOptions(target_ratio=0.25, schedule='multiple-choice', seed=5)
        first = simplify({'POSITION': positions}, indices, options)
        second = simplify({'POSITION': positions}, indices, options)
        assert np.array_equal(first.indices, second.indices)
        assert np.array_equal(first.attributes['POSITION'], second.attributes['POSITION'])

    def test_a_different_seed_gives_a_different_mesh(self):
        positions, indices = shapes.icosphere(3)
        first = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.25, schedule='multiple-choice', seed=1),
        )
        second = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.25, schedule='multiple-choice', seed=2),
        )
        assert not np.array_equal(first.attributes['POSITION'], second.attributes['POSITION'])


class TestPlacement:
    def test_endpoint_placement_keeps_every_position_it_was_given(self):
        """Nothing moves, so a decimated vertex is still an original vertex."""
        positions, indices = shapes.icosphere(3)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.3, placement='endpoint'),
        )
        original = {tuple(row) for row in positions.astype('f4').tolist()}
        for row in result.attributes['POSITION'].tolist():
            assert tuple(row) in original

    def test_optimal_placement_is_the_more_accurate_of_the_two(self):
        positions, indices = shapes.icosphere(3)
        optimal = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.2))
        endpoint = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.2, placement='endpoint'),
        )
        assert optimal.error < endpoint.error

    def test_an_unknown_placement_is_refused(self):
        with pytest.raises(DecimateError, match='placement'):
            SimplifyOptions(target_ratio=0.5, placement='wherever')


class TestProbabilisticMetric:
    def test_it_reaches_the_target_too(self):
        positions, indices = shapes.icosphere(3)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.25, metric='probabilistic'),
        )
        assert _triangles(result) <= 0.25 * (len(indices) // 3)

    def test_it_survives_a_noisy_surface(self):
        """A scan is a sphere plus noise; the reduction must still be a sphere."""
        positions, indices = shapes.icosphere(3)
        rng = np.random.default_rng(4)
        noisy = positions.astype('d') * (1.0 + rng.normal(scale=0.01, size=(len(positions), 1)))
        result = simplify(
            {'POSITION': noisy.astype('f4')},
            indices,
            SimplifyOptions(target_ratio=0.25, metric='probabilistic', position_noise=0.01),
        )
        radii = np.linalg.norm(result.attributes['POSITION'].astype('d'), axis=1)
        assert np.max(np.abs(radii - 1.0)) < 0.06


class TestRefusedOptions:
    def test_a_negative_count_is_refused(self):
        with pytest.raises(DecimateError, match='target_count'):
            SimplifyOptions(target_count=-1)

    def test_a_negative_error_budget_is_refused(self):
        with pytest.raises(DecimateError, match='target_error'):
            SimplifyOptions(target_error=-0.5)

    def test_asking_for_no_candidates_is_refused(self):
        with pytest.raises(DecimateError, match='candidates'):
            SimplifyOptions(target_ratio=0.5, candidates=0)


class TestSamplingCorners:
    def test_the_sampling_schedule_honours_an_error_budget(self):
        positions, indices = shapes.icosphere(3)
        tight = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_error=0.002, schedule='multiple-choice'),
        )
        loose = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_error=0.05, schedule='multiple-choice'),
        )
        assert _triangles(tight) > _triangles(loose)

    def test_the_sampling_schedule_stops_when_nothing_is_left_to_do(self):
        """Every remaining pair has both ends locked, so no draw can succeed."""
        positions, indices = shapes.grid(5)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(
                target_count=2, lock_boundary=True, schedule='multiple-choice', candidates=4
            ),
        )
        assert _triangles(result) >= 14
