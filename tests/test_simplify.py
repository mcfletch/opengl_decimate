"""Decimating a whole mesh: does it hit the target, and is what comes out a mesh?

The shapes are chosen so the right answer is known without reference to the
implementation: a flat patch is exactly representable by two triangles, a sphere
of radius one stays a sphere of radius one, a closed surface stays closed.
"""

import numpy as np
import pytest
import shapes

from opengl_decimate import SimplifyOptions, quadrics, reduction, simplify, topology
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


def _three_hundred_with_a_hundred_that_weld_away():
    """A 300-triangle mesh, 100 of whose triangles cover no area.

    Each of the hundred repeats a corner, so welding drops it: it describes no
    surface and has no plane for a quadric to use.
    """
    positions, faces = shapes.grid(11)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    slivers = np.stack(
        [np.arange(100), np.arange(100), np.arange(1, 101)],
        axis=1,
    )
    both = np.concatenate([faces, slivers]).astype(np.uint32).reshape(-1)
    return positions, both


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

    def test_a_ratio_is_a_share_of_the_input_triangles_not_of_what_welding_left(self):
        """Welding drops triangles before the reduction sees them.

        A caller asking for half of their three hundred triangles means a
        hundred and fifty of them, whatever welding merged first -- and on
        scanned or badly-exported data what welding merges is not a rounding.
        """
        positions, indices = _three_hundred_with_a_hundred_that_weld_away()
        assert len(indices) // 3 == 300
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.5))
        assert result.input_triangles == 300
        assert result.welded_away == 100
        # A contraction removes up to two faces, so the count lands on the limit
        # or one below it, never above.
        assert 149 <= _triangles(result) <= 150

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

    def test_a_texture_seam_does_not_become_a_shading_seam(self):
        """Recomputed normals are accumulated per smoothing group, not per output vertex.

        An output vertex is split by *attributes*, so a texture seam splits one.
        Accumulating there would give each side of the split only the faces on
        its own side -- turning a seam with no geometric meaning at all into a
        crease in the shading.
        """
        positions, indices = shapes.icosphere(4)
        seam = np.stack(
            [(positions[:, 0] > 0).astype('f4'), np.zeros(len(positions), dtype='f4')], axis=1
        )
        options = SimplifyOptions(target_ratio=0.3, recompute_normals=True)

        def worst_angle(attributes):
            result = simplify(attributes, indices, options)
            out = result.attributes['POSITION'].astype('d')
            truth = out / np.linalg.norm(out, axis=1)[:, None]
            cosine = np.clip(np.einsum('ij,ij->i', result.attributes['NORMAL'], truth), -1.0, 1.0)
            return float(np.degrees(np.arccos(cosine)).max())

        plain = worst_angle({'POSITION': positions})
        seamed = worst_angle({'POSITION': positions, 'TEXCOORD_0': seam})
        assert seamed == pytest.approx(plain, abs=1e-6)

    def test_recomputed_normals_are_added_where_the_input_had_none(self):
        positions, indices = shapes.icosphere(2)
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.5, recompute_normals=True),
        )
        normals = result.attributes['NORMAL']
        assert normals.shape == result.attributes['POSITION'].shape
        assert np.linalg.norm(normals, axis=1) == pytest.approx(1.0, abs=1e-5)

    def test_an_empty_result_still_carries_the_normals_it_was_asked_for(self):
        result = simplify(
            {'POSITION': np.zeros((0, 3), dtype='f4')},
            np.zeros((0,), dtype=np.uint32),
            SimplifyOptions(target_ratio=0.5, recompute_normals=True),
        )
        assert result.attributes['NORMAL'].shape == (0, 3)
        assert result.attributes['NORMAL'].dtype == np.float32

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

    def test_a_nearly_singular_solve_does_not_place_a_point_far_from_its_edge(self):
        """On a noisy cylinder the minimum along the axis is wherever rounding put it.

        The solve is determined, and the point it finds can be hundreds of
        edge lengths away. Such a point is not a placement: the edge is priced
        at its ends and its midpoint instead.
        """
        positions, indices = shapes.tube(noise=1e-5)
        mesh = topology.build(positions, indices)
        edges = mesh.edges()
        engine = reduction._Engine(mesh, SimplifyOptions(target_ratio=0.5))
        _, placement = engine.candidates(edges)
        here, there = mesh.positions[edges[:, 0]], mesh.positions[edges[:, 1]]
        away = np.linalg.norm(placement - 0.5 * (here + there), axis=1)
        assert np.all(away <= np.linalg.norm(there - here, axis=1))

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


class TestAModelFarFromTheOrigin:
    """A georeferenced world hands in coordinates around 6.4e6.

    A plane quadric's constant term is the squared plane offset, so in absolute
    coordinates at that scale it carries nineteen decimal digits of range in a
    type that holds sixteen. Accumulating about the model's own bounding-box
    centre puts it back inside the type, and the shift is exact, so the answer
    does not merely improve -- it stops depending on where the model is.
    """

    @pytest.mark.parametrize('shift', [0.0, 1e2, 1e4, 1e6, 6.4e6])
    def test_the_reduction_does_not_depend_on_where_the_model_is(self, shift):
        positions, indices = shapes.icosphere(4)
        here = positions.astype('d')
        there = here + np.asarray([shift, 0.0, 0.0])
        first = simplify({'POSITION': here}, indices, SimplifyOptions(target_ratio=0.1))
        second = simplify({'POSITION': there}, indices, SimplifyOptions(target_ratio=0.1))
        assert _triangles(second) == _triangles(first)
        assert second.error == pytest.approx(first.error, rel=1e-12)
        assert np.array_equal(second.indices, first.indices)
        moved = second.attributes['POSITION'] - np.asarray([shift, 0.0, 0.0])
        assert moved == pytest.approx(first.attributes['POSITION'], abs=1e-9)

    def test_the_quadrics_still_score_their_own_points_at_zero(self):
        """The metric's own noise floor, which is what the digits buy.

        A vertex quadric evaluated at the vertex it was accumulated from is the
        summed distance from that point to planes that all pass through it, so
        it is zero. How far from zero it comes back is the floor below which no
        error the metric reports means anything.
        """
        positions, indices = shapes.icosphere(3)
        far = positions.astype('d') + np.asarray([6.4e6, 0.0, 0.0])
        mesh = topology.build(far, indices)
        faces = mesh.live_faces()
        accumulated = quadrics.accumulate(
            quadrics.triangle_quadrics(mesh.positions, faces), faces, mesh.vertex_count
        )
        worst = float(np.max(np.abs(quadrics.evaluate(accumulated, mesh.positions))))
        assert worst < 1e-15

    def test_a_float64_mesh_comes_back_float64(self):
        """A float32 ulp at 6.4e6 is half a metre, so narrowing would undo it."""
        positions, indices = shapes.icosphere(2)
        far = positions.astype('d') + np.asarray([6.4e6, 0.0, 0.0])
        result = simplify({'POSITION': far}, indices, SimplifyOptions(target_ratio=0.5))
        assert result.attributes['POSITION'].dtype == np.dtype('f8')

    def test_a_float32_mesh_still_comes_back_float32(self):
        positions, indices = shapes.icosphere(2)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.5))
        assert result.attributes['POSITION'].dtype == np.dtype('f4')


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
        """A scan is a sphere plus noise; the reduction must still be a sphere.

        Named through ``normal_noise``, because that is the term that makes the
        metric behave differently at all -- see the tests below.
        """
        positions, indices = shapes.icosphere(3)
        rng = np.random.default_rng(4)
        noisy = positions.astype('d') * (1.0 + rng.normal(scale=0.01, size=(len(positions), 1)))
        result = simplify(
            {'POSITION': noisy.astype('f4')},
            indices,
            SimplifyOptions(target_ratio=0.25, metric='probabilistic', normal_noise=0.01),
        )
        radii = np.linalg.norm(result.attributes['POSITION'].astype('d'), axis=1)
        assert np.max(np.abs(radii - 1.0)) < 0.06

    def test_naming_one_noise_does_not_zero_the_other(self):
        """Both default on their own, so a caller gets what they asked for plus
        what they did not ask about, rather than plus nothing."""
        assert SimplifyOptions(target_ratio=0.5, metric='probabilistic').noise(1.0) == (
            0.001,
            0.001,
        )
        assert SimplifyOptions(target_ratio=0.5, metric='probabilistic', position_noise=0.01).noise(
            1.0
        ) == (0.01, 0.001)
        assert SimplifyOptions(target_ratio=0.5, metric='probabilistic', normal_noise=0.01).noise(
            1.0
        ) == (0.001, 0.01)
        # An explicit zero is a zero, which `None` as the default is what makes
        # expressible at all.
        assert SimplifyOptions(target_ratio=0.5, metric='probabilistic', normal_noise=0.0).noise(
            1.0
        ) == (0.001, 0.0)

    def test_the_normal_noise_is_what_makes_the_metric_different(self):
        """`position_noise` adds a constant to every cost, so it cannot reorder.

        It raises the deviation the reduction reports and changes nothing about
        which contraction is chosen -- so with the normal noise at zero, the
        probabilistic metric *is* the classical one, bit for bit. Worth pinning,
        because a caller reading only the `position_noise` row would otherwise
        believe they had switched metric.
        """
        positions, indices = shapes.icosphere(3)
        rng = np.random.default_rng(4)
        noisy = (
            positions.astype('d') * (1.0 + rng.normal(scale=0.01, size=(len(positions), 1)))
        ).astype('f4')
        classical = simplify({'POSITION': noisy}, indices, SimplifyOptions(target_ratio=0.25))
        flat = simplify(
            {'POSITION': noisy},
            indices,
            SimplifyOptions(
                target_ratio=0.25, metric='probabilistic', position_noise=0.01, normal_noise=0.0
            ),
        )
        assert np.array_equal(flat.indices, classical.indices)
        assert np.array_equal(flat.attributes['POSITION'], classical.attributes['POSITION'])

        conditioned = simplify(
            {'POSITION': noisy},
            indices,
            SimplifyOptions(target_ratio=0.25, metric='probabilistic', normal_noise=0.01),
        )
        assert not np.array_equal(conditioned.indices, classical.indices)

    def test_only_the_normal_noise_conditions_the_solve(self):
        """One plane leaves a plane of equally good points until `A` is full rank."""
        plane = (np.asarray([[0.0, 1.0, 0.0]]), np.asarray([[0.0, 2.0, 0.0]]))
        assert not quadrics.minimize(quadrics.plane_quadric(*plane))[1][0]
        assert not quadrics.minimize(quadrics.plane_quadric(*plane, position_noise=0.01))[1][0]
        assert quadrics.minimize(quadrics.plane_quadric(*plane, normal_noise=0.01))[1][0]


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

    @pytest.mark.parametrize(
        ('named', 'says'),
        [
            pytest.param({'position_noise': -1.0}, 'position_noise', id='position_noise'),
            pytest.param({'normal_noise': -1.0}, 'normal_noise', id='normal_noise'),
            pytest.param({'max_normal_flip': 1e9}, 'max_normal_flip', id='flip-past-180'),
            pytest.param({'max_normal_flip': -1.0}, 'max_normal_flip', id='flip-negative'),
            # Documented as 0 to 1; above 1 nothing can satisfy it, so the
            # reduction silently does nothing at all.
            pytest.param({'min_triangle_quality': 5.0}, 'min_triangle_quality', id='quality'),
            pytest.param({'weld_tolerance': -1.0}, 'weld_tolerance', id='weld'),
            pytest.param({'boundary_weight': -3.0}, 'boundary_weight', id='boundary'),
            pytest.param(
                {'certify': True, 'certify_samples': 0}, 'certify_samples', id='no-samples'
            ),
            # NumPy would wrap this round and lock the *last* point instead.
            pytest.param({'locked': [-1]}, 'locked', id='locked-negative'),
            pytest.param({'locked': [[0, 1]]}, 'locked', id='locked-nested'),
            pytest.param({'locked': [0.5]}, 'locked', id='locked-not-integers'),
            pytest.param({'drop_components_below': -0.1}, 'drop_components_below', id='drop'),
        ],
    )
    def test_an_option_outside_its_range_is_refused(self, named, says):
        with pytest.raises(DecimateError, match=says):
            SimplifyOptions(target_ratio=0.5, **named)


class TestRefusedInput:
    def test_a_locked_index_past_the_end_is_refused(self):
        positions, indices = shapes.octahedron()
        with pytest.raises(DecimateError, match='6 vertices'):
            simplify(
                {'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.5, locked=[99])
            )

    def test_the_message_counts_the_callers_vertices_not_the_welded_points(self):
        """A hard-edged cube is 24 vertices over 8 points, and `locked` is by vertex."""
        attributes, indices = shapes.cube_with_hard_normals()
        with pytest.raises(DecimateError, match='24 vertices'):
            simplify(attributes, indices, SimplifyOptions(target_ratio=0.5, locked=[24]))


class TestLockedVertices:
    @staticmethod
    def _seamed():
        """A flat patch whose every position is written twice, one copy per chart.

        The copies are interleaved, so the welded points are numbered
        differently from the vertices on every row after the first.
        """
        positions, indices = shapes.grid(9)
        doubled = np.repeat(positions, 2, axis=0)
        uv = ((positions[:, [0, 2]] + 1.0) * 0.5).astype('f4')
        charts = np.repeat(uv, 2, axis=0)
        charts[1::2] += 10.0
        faces = 2 * np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        right = positions[:, 0] > 0.0
        across = right[faces // 2].all(axis=1)
        faces[across] += 1
        return {'POSITION': doubled, 'TEXCOORD_0': charts}, faces.reshape(-1).astype(np.uint32)

    @pytest.mark.parametrize('vertex', [80, 81], ids=['first-copy', 'second-copy'])
    def test_a_locked_vertex_is_the_callers_vertex(self, vertex):
        attributes, indices = self._seamed()
        held = attributes['POSITION'][vertex].astype('d')
        result = simplify(attributes, indices, SimplifyOptions(target_count=4, locked=[vertex]))
        out = result.attributes['POSITION'].astype('d')
        assert np.min(np.linalg.norm(out - held, axis=1)) < 1e-9

    @pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf], ids=['nan', 'inf', '-inf'])
    def test_a_position_that_is_not_finite_is_refused(self, bad):
        """It is accepted by everything downstream and reaches the output."""
        positions, indices = shapes.octahedron()
        broken = positions.astype('d')
        broken[2, 1] = bad
        with pytest.raises(DecimateError, match='row 2'):
            simplify({'POSITION': broken}, indices, SimplifyOptions(target_ratio=0.5))

    def test_indices_that_are_not_integers_are_refused(self):
        """A float index is truncated by the cast, naming a vertex nobody asked for."""
        positions, indices = shapes.octahedron()
        with pytest.raises(DecimateError, match='integer'):
            simplify(
                {'POSITION': positions},
                indices.astype('f8') + 0.7,
                SimplifyOptions(target_ratio=0.5),
            )

    def test_an_attribute_that_is_not_an_array_is_refused(self):
        positions, indices = shapes.octahedron()
        with pytest.raises(DecimateError, match='WEIGHT'):
            simplify(
                {'POSITION': positions, 'WEIGHT': np.float32(1.0)},
                indices,
                SimplifyOptions(target_ratio=0.5),
            )


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

    def test_the_budget_filters_candidates_rather_than_ending_the_reduction(self):
        """A draw with nothing affordable in it is a failed draw, not the end.

        Eight edges are sampled out of thousands, so the cheapest of the eight
        is over budget long before the cheapest on the mesh is. Stopping there
        would end wherever the draw happened to fall -- which is a different
        answer for every seed and several times too many triangles. So the
        sampled reduction has to land near where the heap does, and has to land
        in the same place whatever the seed.
        """
        positions, indices = shapes.icosphere(3)
        heap = _triangles(
            simplify({'POSITION': positions}, indices, SimplifyOptions(target_error=0.02))
        )
        sampled = [
            _triangles(
                simplify(
                    {'POSITION': positions},
                    indices,
                    SimplifyOptions(target_error=0.02, schedule='multiple-choice', seed=seed),
                )
            )
            for seed in range(8)
        ]
        assert min(sampled) >= heap
        assert max(sampled) < 1.5 * heap
        assert max(sampled) - min(sampled) < 0.25 * heap

    def test_the_work_to_reach_the_floor_is_linear_in_the_edges(self, monkeypatch):
        """Once nothing is left to contract, the sampler stops rather than drawing on.

        A pair drawn and passed over cannot change until one of its ends survives
        a contraction, so it leaves the pool until then, and the reduction ends
        when the pool is empty. Each draw prices one batch, so counting batches
        counts the work.
        """
        batches = 0
        price = reduction._Engine.candidates

        def counted(engine, pairs):
            nonlocal batches
            batches += 1
            return price(engine, pairs)

        monkeypatch.setattr(reduction._Engine, 'candidates', counted)
        positions, indices = shapes.icosphere(3)
        edges = topology.build(positions, indices).edges()
        result = simplify(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_count=0, schedule='multiple-choice'),
        )
        assert result.triangle_count == 4
        assert batches < 4 * len(edges)

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
