"""Asking a mesh how far it will go before spending a reduction on it.

Some models cannot be decimated, and the reason is always a property of the
model rather than of the reducer. A canopy of separate leaf cards has as many
pieces as cards and every piece keeps a triangle; a scan of lace has a handle
through every hole and no contraction closes one; an atlas of thousands of small
charts is mostly seam. A caller who knows which of those they have knows whether
to reach for a reduction or for a different tool.
"""

import numpy as np
import pytest
import shapes

from opengl_decimate import SimplifyOptions, native, simplify, survey


def _cards(count=50, seed=0):
    """Loose quads, no two sharing a vertex -- what a leaf canopy welds to."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-10.0, 10.0, size=(count, 3))
    corner = np.array([(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)])
    positions = (centres[:, None, :] + corner[None, :, :]).reshape(-1, 3).astype('f4')
    base = 4 * np.arange(count)[:, None]
    faces = np.concatenate(
        [base + np.array([[0, 1, 2]]), base + np.array([[0, 2, 3]])], axis=0
    ).astype(np.uint32)
    return {'POSITION': positions}, faces.reshape(-1)


def _torus(around=24, across=12, big=3.0, small=1.0):
    """One closed surface with one handle through it."""
    angle = np.arange(around) * (2.0 * np.pi / around)
    ring = np.arange(across) * (2.0 * np.pi / across)
    grid_u, grid_v = np.meshgrid(angle, ring, indexing='ij')
    radius = big + small * np.cos(grid_v)
    positions = np.stack(
        [radius * np.cos(grid_u), small * np.sin(grid_v), radius * np.sin(grid_u)], axis=-1
    ).reshape(-1, 3)
    index = np.arange(around * across).reshape(around, across)
    right, down = np.roll(index, -1, axis=0), np.roll(index, -1, axis=1)
    faces = np.concatenate(
        [
            np.stack([index, right, np.roll(right, -1, axis=1)], axis=-1).reshape(-1, 3),
            np.stack([index, np.roll(right, -1, axis=1), down], axis=-1).reshape(-1, 3),
        ]
    )
    return {'POSITION': positions.astype('f4')}, faces.reshape(-1).astype(np.uint32)


class TestWhatStopsAReduction:
    def test_a_closed_shell_can_go_to_four_triangles_and_no_further(self):
        attributes = {'POSITION': shapes.icosphere(3)[0]}
        report = survey(attributes, shapes.icosphere(3)[1])
        assert report.pieces == 1
        assert report.handles == 0
        assert report.open_pieces == 0
        assert report.floor == 4

    def test_a_torus_is_one_piece_with_one_handle(self):
        attributes, indices = _torus()
        report = survey(attributes, indices)
        assert report.pieces == 1
        assert report.handles == 1

    def test_loose_cards_keep_a_triangle_each(self):
        """The canopy case: as many pieces as cards, and no reduction touches them.

        Each card is its own component with its own border, so the floor is the
        number of cards -- decimation cannot take a canopy below the count of
        the leaves in it, whatever target it is given.
        """
        attributes, indices = _cards(50)
        report = survey(attributes, indices)
        assert report.pieces == 50
        assert report.open_pieces == 50
        assert report.floor == 50
        assert report.reducible < 0.6

    def test_a_welded_surface_is_reducible(self):
        attributes = {'POSITION': shapes.grid(17)[0]}
        report = survey(attributes, shapes.grid(17)[1])
        assert report.pieces == 1
        assert report.floor == 1
        assert report.reducible > 0.99

    def test_a_seam_is_counted_and_a_plain_mesh_has_none(self):
        """A chart boundary is a duplicated vertex, not merely a jump in uv.

        Two vertices at one position carrying different coordinates is what
        makes a seam; one vertex whose coordinate happens to differ from its
        neighbour's is just a texture.
        """
        positions, indices = shapes.grid(9)
        uv = ((positions[:, [0, 2]] + 1.0) * 0.5).astype('f4')
        assert survey({'POSITION': positions, 'TEXCOORD_0': uv}, indices).seam_share == 0.0

        faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        right = positions[:, 0] > 0.0
        copy_of = np.arange(len(positions)) + len(positions)
        doubled = np.concatenate([positions, positions])
        charts = np.concatenate([uv, uv + np.array([10.0, 0.0], dtype='f4')])
        moved = right[faces].all(axis=1)
        faces[moved] = copy_of[faces[moved]]
        seamed = {'POSITION': doubled, 'TEXCOORD_0': charts}
        assert survey(seamed, faces.reshape(-1).astype(np.uint32)).seam_share > 0.0

    def test_welding_is_what_the_answer_is_about(self):
        """The counts describe the welded surface, not the vertex soup.

        Two cards written with separate vertices but placed edge to edge are one
        piece once welded, and a triangle that welding leaves with a repeated
        corner covers no area, so it is not one of the triangles there is to
        reduce.
        """
        corner = np.array([(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)])
        joined = np.concatenate([corner, corner + np.array([2.0, 0.0, 0.0])]).astype('f4')
        # A sliver: two of its corners are the same place, so welding drops it.
        sliver = np.array([(-1.0, -1.0, 0.0), (-1.0, -1.0, 0.0), (1.0, 1.0, 0.0)], dtype='f4')
        positions = np.concatenate([joined, sliver])
        faces = np.array([(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (8, 9, 10)], dtype=np.uint32)
        report = survey({'POSITION': positions}, faces.reshape(-1))
        assert report.triangles == 5
        assert report.welded_away == 1
        assert report.pieces == 1
        assert report.points == 6


class TestTheFloorIsWhereAReductionStops:
    """The floor is a promise about the reducer, so it is held against one.

    Each mesh is reduced with a target of nothing, on both schedules and on both
    reducers, and has to stop exactly where the survey said it would.
    """

    @pytest.mark.parametrize(
        'mesh',
        [
            pytest.param(shapes.grid(10), id='open-patch'),
            pytest.param(shapes.grid(2), id='two-triangles'),
            pytest.param(shapes.icosphere(2), id='closed-shell'),
            pytest.param(_cards(6), id='loose-cards'),
        ],
    )
    @pytest.mark.parametrize('schedule', ['heap', 'multiple-choice'])
    @pytest.mark.parametrize('compiled', [True, False], ids=['compiled', 'numpy'])
    def test_an_exhaustive_reduction_stops_at_the_floor(
        self, mesh, schedule, compiled, monkeypatch
    ):
        if compiled and not native.ACCELERATED:
            pytest.skip('accelerator not built')
        monkeypatch.setattr(native, 'ACCELERATED', compiled)
        attributes, indices = mesh
        if not isinstance(attributes, dict):
            attributes = {'POSITION': attributes}
        result = simplify(attributes, indices, SimplifyOptions(target_count=0, schedule=schedule))
        assert result.triangle_count == survey(attributes, indices).floor


class TestHowTheAtlasIsCutUp:
    """A texture cannot describe a triangle larger than the chart it sits in.

    An atlas is a set of charts -- connected pieces of surface laid out flat --
    and a texture coordinate is only meaningful inside one. Reduce past the
    point where an output triangle spans more surface than a chart holds and
    there is no coordinate that describes it, whatever the reducer does. That is
    a property of how the model was unwrapped, measurable before a reduction.
    """

    @staticmethod
    def _striped(side=17, stripes=4):
        """A patch cut into vertical chart stripes, each its own piece of atlas."""
        positions, indices = shapes.grid(side)
        faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        positions = np.asarray(positions, dtype='f4')
        column = np.arange(len(positions)) // side
        band = np.minimum(column * stripes // side, stripes - 1)
        # Every band gets its own copy of every vertex, so no vertex is shared
        # across a boundary -- which is what makes it a chart boundary.
        copies, remap = [positions], {}
        base = len(positions)
        for step in range(1, stripes):
            remap[step] = base + np.arange(len(positions))
            copies.append(positions)
            base += len(positions)
        doubled = np.concatenate(copies)
        flat = ((doubled[:, [0, 2]] + 1.0) * 0.5).astype('f4')
        for step in range(1, stripes):
            flat[remap[step]] += (step * 10.0, 0.0)
        face_band = band[faces].min(axis=1)
        for step in range(1, stripes):
            moved = face_band == step
            faces[moved] = remap[step][faces[moved]]
        return (
            {'POSITION': doubled, 'TEXCOORD_0': flat},
            faces.reshape(-1).astype(np.uint32),
        )

    def test_one_chart_where_the_model_is_unwrapped_whole(self):
        positions, indices = shapes.grid(17)
        uv = ((np.asarray(positions)[:, [0, 2]] + 1.0) * 0.5).astype('f4')
        report = survey({'POSITION': positions, 'TEXCOORD_0': uv}, indices)
        assert report.charts == 1
        assert report.median_chart == report.triangles

    def test_a_mesh_carrying_no_texture_has_no_atlas_to_measure(self):
        positions, indices = shapes.icosphere(2)
        report = survey({'POSITION': positions}, indices)
        assert report.charts == 0
        assert report.texture_floor == 0

    def test_the_stripes_are_counted(self):
        attributes, indices = self._striped(17, 4)
        report = survey(attributes, indices)
        assert report.charts == 4
        assert report.median_chart == report.triangles // 4

    def test_a_finely_cut_atlas_puts_a_floor_under_what_the_texture_survives(self):
        """More charts is a higher floor, for the same model and triangle count."""
        coarse = survey(*self._striped(33, 2))
        fine = survey(*self._striped(33, 16))
        assert coarse.triangles == fine.triangles
        assert fine.charts > coarse.charts
        assert fine.texture_floor > coarse.texture_floor
