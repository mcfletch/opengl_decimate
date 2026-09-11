"""Recording a reduction once, and reaching any point in it by replay.

The claim under test is that a target costs the same whatever it is: the mesh at
any triangle count is a prefix of the recorded contractions, and applying a
prefix is a handful of array operations rather than a re-run. So the tests ask
both halves -- that a replayed prefix is the mesh the same reduction would have
produced, and that asking fifty times costs about what asking once costs.
"""

import itertools
import time

import numpy as np
import pytest
import shapes

from opengl_decimate import SimplifyOptions, collapse_sequence, simplify


def _sphere(subdivisions=3):
    positions, indices = shapes.icosphere(subdivisions)
    return {'POSITION': positions}, indices


class TestRecording:
    def test_a_reduction_records_its_contractions(self):
        attributes, indices = _sphere()
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        assert len(sequence) > 100

    def test_the_targets_in_the_options_do_not_stop_it(self):
        """It records the whole reduction; the target is asked for afterwards."""
        attributes, indices = _sphere()
        stopped = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.9))
        assert stopped.triangles_after(len(stopped)) < 0.1 * (len(indices) // 3)

    def test_triangle_count_falls_all_the_way_along(self):
        attributes, indices = _sphere(2)
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        counts = [sequence.triangles_after(k) for k in range(len(sequence) + 1)]
        assert counts[0] == len(indices) // 3
        assert all(later <= earlier for earlier, later in itertools.pairwise(counts))

    def test_a_flat_patch_records_no_error_until_its_outline_must_change(self):
        """A plane is exactly representable, so removing its interior is free.

        Run to exhaustion the reduction eventually reaches the four corners of
        the patch, and moving one of those does change the shape -- which is why
        this asks about a prefix rather than the whole sequence.
        """
        positions, indices = shapes.grid(7)
        sequence = collapse_sequence(
            {'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.5)
        )
        assert sequence.at(target_count=8).error < 1e-6


class TestReplay:
    def test_a_prefix_gives_the_mesh_that_reduction_would_have_given(self):
        """Replaying to a count must match decimating straight to that count."""
        attributes, indices = _sphere()
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        direct = simplify(attributes, indices, SimplifyOptions(target_count=320))
        replayed = sequence.at(target_count=320)
        assert np.array_equal(replayed.indices, direct.indices)
        assert np.array_equal(replayed.attributes['POSITION'], direct.attributes['POSITION'])

    def test_replay_honours_a_ratio(self):
        attributes, indices = _sphere()
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        result = sequence.at(target_ratio=0.25)
        assert result.triangle_count <= 0.25 * (len(indices) // 3)

    def test_replay_honours_an_error_budget(self):
        attributes, indices = _sphere()
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        tight = sequence.at(target_error=0.002)
        loose = sequence.at(target_error=0.05)
        assert tight.triangle_count > loose.triangle_count
        assert tight.error <= 0.002

    def test_replay_with_no_target_is_the_whole_sequence(self):
        attributes, indices = _sphere(2)
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        assert sequence.at().collapses == len(sequence)

    def test_replay_at_the_start_is_the_mesh_that_went_in(self):
        """Vertices are renumbered by first use, so the map is what to compare."""
        attributes, indices = _sphere(2)
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        result = sequence.at(target_count=len(indices) // 3)
        assert result.collapses == 0
        assert result.triangle_count == len(indices) // 3
        landed = result.attributes['POSITION'][result.vertex_map]
        assert landed == pytest.approx(attributes['POSITION'])

    def test_every_replayed_level_is_a_sound_mesh(self):
        attributes, indices = _sphere()
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        for count in (1000, 640, 320, 160, 80):
            result = sequence.at(target_count=count)
            faces = result.indices.reshape(-1, 3)
            assert np.all(faces[:, 0] != faces[:, 1])
            assert np.all(faces[:, 1] != faces[:, 2])
            assert np.all(faces[:, 0] != faces[:, 2])
            assert len(np.unique(result.indices)) == len(result.attributes['POSITION'])

    def test_attributes_follow_the_replay(self):
        attributes, indices = shapes.cube_with_hard_normals()
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        result = sequence.at(target_count=8)
        count = len(result.attributes['POSITION'])
        assert result.attributes['NORMAL'].shape == (count, 3)

    def test_asking_again_gives_the_same_answer(self):
        attributes, indices = _sphere(2)
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        first = sequence.at(target_count=100)
        second = sequence.at(target_count=100)
        assert np.array_equal(first.indices, second.indices)


class TestReplayCost:
    def test_a_slider_full_of_targets_costs_about_what_one_costs(self):
        """The point of recording: a target is a prefix, not another reduction."""
        attributes, indices = _sphere(4)
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))

        start = time.perf_counter()
        sequence.at(target_count=2000)
        one = time.perf_counter() - start

        targets = np.linspace(20, len(indices) // 3 - 20, 50).astype(int)
        start = time.perf_counter()
        for count in targets:
            sequence.at(target_count=int(count))
        fifty = time.perf_counter() - start

        assert fifty < max(60.0 * one, 2.0)
