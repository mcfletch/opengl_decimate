"""The compiled reduction against the NumPy one.

The NumPy loop is the readable statement of the algorithm and the compiled one
is what makes a scan reducible. They are the same algorithm, so they have to be
the same answer -- and the only way that stays true is to run both on the same
meshes and compare, which is what this does.

Where the accelerator did not build, the comparisons skip and the rest of the
suite still holds the NumPy path to its contract.
"""

import threading
import time

import numpy as np
import pytest
import shapes

from opengl_decimate import SimplifyOptions, collapse_sequence, native, simplify, topology

pytestmark = pytest.mark.skipif(not native.ACCELERATED, reason='accelerator not built')


def _both(attributes, indices, options):
    """The same reduction through each path."""
    compiled = simplify(attributes, indices, options)
    saved = native.ACCELERATED
    native.ACCELERATED = False
    try:
        pure = simplify(attributes, indices, options)
    finally:
        native.ACCELERATED = saved
    return compiled, pure


class TestTheTwoPathsAgree:
    @pytest.mark.parametrize(
        'mesh',
        [
            pytest.param(shapes.icosphere(3), id='sphere'),
            pytest.param(shapes.grid(20, bump=0.4), id='patch'),
            pytest.param(shapes.grid(12), id='flat-patch'),
            pytest.param(shapes.octahedron(), id='octahedron'),
            pytest.param(shapes.tube(noise=1e-5), id='noisy-tube'),
        ],
    )
    def test_they_reduce_to_the_same_mesh(self, mesh):
        positions, indices = mesh
        compiled, pure = _both({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.4))
        assert compiled.triangle_count == pure.triangle_count
        assert compiled.attributes['POSITION'] == pytest.approx(
            pure.attributes['POSITION'], abs=1e-9
        )
        assert np.array_equal(compiled.indices, pure.indices)

    def test_they_agree_with_attributes_carried(self):
        attributes, indices = shapes.cube_with_hard_normals()
        compiled, pure = _both(attributes, indices, SimplifyOptions(target_ratio=0.6))
        assert np.array_equal(compiled.indices, pure.indices)
        assert np.array_equal(compiled.attributes['NORMAL'], pure.attributes['NORMAL'])

    def test_they_agree_on_a_locked_boundary(self):
        positions, indices = shapes.grid(14)
        compiled, pure = _both(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.3, lock_boundary=True),
        )
        assert compiled.triangle_count == pure.triangle_count
        assert compiled.attributes['POSITION'] == pytest.approx(
            pure.attributes['POSITION'], abs=1e-9
        )

    def test_they_agree_on_the_probabilistic_metric(self):
        positions, indices = shapes.grid(16)
        compiled, pure = _both(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.3, metric='probabilistic'),
        )
        assert np.array_equal(compiled.indices, pure.indices)
        assert compiled.attributes['POSITION'] == pytest.approx(
            pure.attributes['POSITION'], abs=1e-9
        )

    def test_they_agree_on_endpoint_placement(self):
        positions, indices = shapes.icosphere(2)
        compiled, pure = _both(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.4, placement='endpoint'),
        )
        assert np.array_equal(compiled.indices, pure.indices)

    def test_they_agree_on_an_error_budget(self):
        positions, indices = shapes.icosphere(3)
        compiled, pure = _both({'POSITION': positions}, indices, SimplifyOptions(target_error=0.01))
        assert compiled.triangle_count == pure.triangle_count

    def test_they_agree_on_a_quality_floor(self):
        positions, indices = shapes.grid(16, bump=0.3)
        compiled, pure = _both(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.3, min_triangle_quality=0.3),
        )
        assert compiled.triangle_count == pure.triangle_count

    def test_they_record_the_same_sequence(self):
        positions, indices = shapes.icosphere(2)
        options = SimplifyOptions(target_ratio=0.5)
        compiled = collapse_sequence({'POSITION': positions}, indices, options)
        saved = native.ACCELERATED
        native.ACCELERATED = False
        try:
            pure = collapse_sequence({'POSITION': positions}, indices, options)
        finally:
            native.ACCELERATED = saved
        assert len(compiled) == len(pure)
        assert np.array_equal(compiled.dying, pure.dying)
        assert np.array_equal(compiled.surviving, pure.surviving)
        assert compiled.deviation == pytest.approx(pure.deviation, abs=1e-12)

    def test_a_replayed_prefix_matches_too(self):
        positions, indices = shapes.icosphere(3)
        sequence = collapse_sequence(
            {'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.5)
        )
        direct = simplify({'POSITION': positions}, indices, SimplifyOptions(target_count=400))
        assert np.array_equal(sequence.at(target_count=400).indices, direct.indices)

    @pytest.mark.parametrize('valence', [63, 64, 65, 1023, 1024, 1025, 2000])
    def test_they_agree_at_any_valence(self, valence):
        """A point's neighbourhood has no size the compiled path gives up at.

        The buffers the compiled reducer walks a neighbourhood in used to be
        fixed, and a point with more faces than they held had its candidates
        passed over -- silently, and only on that path. The sizes here straddle
        the buffers' starting size and the old ceiling.
        """
        positions, indices = shapes.fan(valence)
        options = SimplifyOptions(target_count=1, locked=list(range(1, valence + 1)))
        compiled, pure = _both({'POSITION': positions}, indices, options)
        assert compiled.triangle_count == pure.triangle_count
        assert np.array_equal(compiled.indices, pure.indices)

    def test_they_agree_with_a_tolerance_weld(self):
        positions, indices = shapes.nearly_coincident(24, spread=1e-7)
        compiled, pure = _both(
            {'POSITION': positions},
            indices,
            SimplifyOptions(target_ratio=0.4, weld_tolerance=1e-5),
        )
        assert compiled.triangle_count == pure.triangle_count
        assert compiled.attributes['POSITION'] == pytest.approx(
            pure.attributes['POSITION'], abs=1e-9
        )

    def test_they_agree_on_named_locks(self):
        """`locked` is the option the cluster-boundary workflow is built on."""
        positions, indices = shapes.grid(14, bump=0.3)
        held = list(range(0, len(positions), 7))
        compiled, pure = _both(
            {'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.3, locked=held)
        )
        assert compiled.triangle_count == pure.triangle_count
        assert np.array_equal(compiled.indices, pure.indices)

    @pytest.mark.parametrize(
        'maker', [shapes.bowtie, shapes.nonmanifold_edge], ids=['bowtie', 'non-manifold']
    )
    def test_they_agree_on_a_surface_that_is_not_one(self, maker):
        positions, indices = maker()
        compiled, pure = _both({'POSITION': positions}, indices, SimplifyOptions(target_count=1))
        assert np.array_equal(compiled.indices, pure.indices)

    def test_they_agree_past_the_queue_and_log_capacities(self):
        """Both grow from 1024, so a mesh smaller than that never grows either."""
        positions, indices = shapes.icosphere(4)
        compiled, pure = _both({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.1))
        assert compiled.collapses > 1024
        assert compiled.triangle_count == pure.triangle_count
        assert compiled.attributes['POSITION'] == pytest.approx(
            pure.attributes['POSITION'], abs=1e-9
        )


class TestTheCompiledPathKeepsTheInvariants:
    """Whatever it agrees with, it still has to produce a surface."""

    def test_a_closed_surface_stays_closed(self):
        positions, indices = shapes.icosphere(3)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.2))
        faces = result.indices.reshape(-1, 3)
        pairs = np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0),
            axis=1,
        )
        _, counts = np.unique(pairs, axis=0, return_counts=True)
        assert np.all(counts == 2)

    def test_it_keeps_the_orientation(self):
        positions, indices = shapes.icosphere(3)
        result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.2))
        corners = result.attributes['POSITION'].astype('d')[result.indices.reshape(-1, 3)]
        volume = np.einsum('ij,ij->i', corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum()
        assert volume > 0

    def test_it_refuses_a_bowtie_and_a_non_manifold_edge(self):
        for maker in (shapes.bowtie, shapes.nonmanifold_edge):
            positions, indices = maker()
            result = simplify({'POSITION': positions}, indices, SimplifyOptions(target_count=1))
            faces = result.indices.reshape(-1, 3)
            assert np.all(faces[:, 0] != faces[:, 1])
            assert np.all(faces[:, 1] != faces[:, 2])

    def test_it_leaves_the_mesh_consistent_for_whoever_asks_next(self):
        """The compiled loop rewrites the faces, so stale adjacency must go."""
        positions, indices = shapes.icosphere(2)
        mesh = topology.build(positions, indices)
        assert mesh.vertex_faces is not None  # build it, so it could go stale
        assert mesh.adjacency_built
        native.reduce_mesh(
            mesh,
            np.zeros((mesh.vertex_count, 10)),
            np.ones(mesh.vertex_count),
            mesh.classify(),
            target_faces=mesh.face_count // 2,
            placement='optimal',
            max_normal_flip=90.0,
            min_triangle_quality=0.0,
            error_limit=None,
        )
        assert not mesh.adjacency_built
        for index in np.flatnonzero(mesh.alive):
            for point in mesh.faces[index]:
                assert index in mesh.vertex_faces[point]


class TestItSharesTheInterpreter:
    def test_the_loop_does_not_shut_the_interpreter_out(self):
        """The contraction loop releases the GIL, so a worker thread is usable.

        Asserted as a *share* of the reduction rather than as a duration, so it
        is a fact about the code and not about the machine: if the loop held the
        GIL, the watching thread would run not at all for its whole length and
        the share would be 1. Released, the longest it waits is a scheduling
        quantum.
        """
        positions, indices = shapes.grid(220, bump=0.4)
        finished = threading.Event()

        def reduce_it():
            simplify({'POSITION': positions}, indices, SimplifyOptions(target_ratio=0.1))
            finished.set()

        worker = threading.Thread(target=reduce_it)
        ticks = []
        start = time.perf_counter()
        worker.start()
        while not finished.is_set():
            ticks.append(time.perf_counter())
        worker.join()
        elapsed = time.perf_counter() - start

        gaps = np.diff(np.asarray(ticks))
        assert len(gaps), 'the reduction finished before the watcher ticked twice'
        assert float(gaps.max()) < 0.5 * elapsed


class TestTheBuildContract:
    def test_the_accelerator_can_be_switched_off_by_the_environment(self, monkeypatch):
        """The suite runs both paths, and tox needs a way to demand each one."""
        monkeypatch.setenv('OPENGL_DECIMATE_NO_ACCEL', '1')
        assert native._load() is None
        monkeypatch.delenv('OPENGL_DECIMATE_NO_ACCEL')
        assert native._load() is not None

    def test_the_compiled_reducer_agrees_on_what_the_classes_mean(self):
        """It compares against the numbers, so the numbers have to be these.

        A reordering of `VertexClass` that did not reach the `.pyx` would not
        fail to build or to run -- it would quietly start locking the wrong
        points, which is the kind of defect that reaches a release.
        """
        assert native.vertex_class_values() == (0, 1, 2)
