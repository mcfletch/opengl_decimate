"""Recording a reduction once, and reaching any point in it by replay.

The claim under test is that a target costs the same whatever it is: the mesh at
any triangle count is a prefix of the recorded contractions, and applying a
prefix is a handful of array operations rather than a re-run. So the tests ask
both halves -- that a replayed prefix is the mesh the same reduction would have
produced, and that asking fifty times costs about what asking once costs.
"""

import itertools
import tracemalloc

import numpy as np
import pytest
import shapes

from opengl_decimate import (
    CollapseSequence,
    SimplifyOptions,
    collapse_sequence,
    reduction,
    simplify,
)


def _sphere(subdivisions=3):
    positions, indices = shapes.icosphere(subdivisions)
    return {'POSITION': positions}, indices


class TestRecording:
    def test_a_reduction_records_its_contractions(self):
        attributes, indices = _sphere()
        sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
        assert len(sequence) > 100

    @pytest.mark.parametrize(
        'options',
        [
            pytest.param(SimplifyOptions(target_ratio=0.9), id='ratio'),
            pytest.param(SimplifyOptions(target_count=1000), id='count'),
            pytest.param(SimplifyOptions(target_error=0.002), id='error'),
            pytest.param(None, id='none-given'),
        ],
    )
    def test_no_target_in_the_options_stops_it(self, options):
        """It records the whole reduction; the target is asked for afterwards.

        An error budget is the one a caller reaches for most naturally here,
        since at least one target has to be named to build the options at all --
        and it used to be the one target the recording loops still honoured, so
        every later ask came back unreduced.
        """
        attributes, indices = _sphere()
        recorded = collapse_sequence(attributes, indices, options)
        assert recorded.triangles_after(len(recorded)) < 0.1 * (len(indices) // 3)
        assert recorded.at(target_count=20).triangle_count <= 20

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


class TestBuildingOneByHand:
    """`CollapseSequence` is a public dataclass, so its defaults have to work.

    A caller holding a recorded reduction of their own -- read back from a file,
    say -- constructs one directly, and everything the reduction would have
    filled in has to have an answer.
    """

    def test_the_fields_a_reduction_fills_in_have_defaults(self):
        positions, indices = shapes.tetrahedron()
        faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        empty = np.zeros(0, dtype=np.int64)
        sequence = CollapseSequence(
            points=positions.astype('d'),
            faces=faces,
            corners=faces,
            vertex_point=np.arange(len(positions), dtype=np.int64),
            attributes={'POSITION': positions},
            dying=empty,
            surviving=empty,
            placement=np.zeros((0, 3), dtype='d'),
            deviation=np.zeros(0, dtype='d'),
            removed_at=np.full(len(faces), -1, dtype=np.int64),
        )
        assert sequence.origin.tolist() == [0.0, 0.0, 0.0]
        assert sequence.input_faces == len(faces)
        assert sequence.at().triangle_count == len(faces)


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
    """A target is a prefix replay, not another reduction.

    Both halves are asserted without a clock -- what a run takes is a fact about
    the machine, and a bound on it is a flake waiting for a loaded CI runner.
    What the claim is really about is that no reduction happens and that nothing
    accumulates, and both of those can be asked directly.
    """

    def test_asking_for_a_target_runs_no_reduction(self, monkeypatch):
        attributes, indices = _sphere(2)
        sequence = collapse_sequence(attributes, indices)
        reductions = []
        engine = reduction._Engine  # noqa: SLF001 counts how often a reduction is built

        class Counted(engine):  # type: ignore[valid-type, misc]
            def __init__(self, *args, **named):
                reductions.append(1)
                super().__init__(*args, **named)

        monkeypatch.setattr(reduction, '_Engine', Counted)
        # The reduction the sequence was recorded with is the only one.
        simplify(attributes, indices, SimplifyOptions(target_count=100))
        assert len(reductions) == 1
        for count in np.linspace(20, len(indices) // 3 - 20, 50).astype(int):
            sequence.at(target_count=int(count))
        assert len(reductions) == 1

    def test_fifty_asks_in_a_row_allocate_what_one_does(self):
        """Nothing is cached or held between asks, so a slider does not grow."""
        attributes, indices = _sphere(4)
        sequence = collapse_sequence(attributes, indices)

        def peak_over(count):
            tracemalloc.start()
            try:
                for _ in range(count):
                    sequence.at(target_count=2000)
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        one = peak_over(1)
        fifty = peak_over(50)
        assert fifty < 1.5 * one, 'fifty asks peaked at %.2f MB against %.2f MB for one' % (
            fifty / 1e6,
            one / 1e6,
        )
