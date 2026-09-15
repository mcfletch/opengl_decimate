"""Attributes staying where the surface put them.

A contraction moves a point. The corner sitting on that point has to move with
it, or the triangle ends up drawn at one place and textured from another --
which reads as a stretched texture and as shading that does not match the
surface. These tests ask the two questions that catches: does a carried value
still belong to the position it is on, and does a texture seam still separate
the two charts it was drawn to separate.
"""

import numpy as np
import shapes

from opengl_decimate import SimplifyOptions, collapse_sequence, simplify


def _linear_uv(positions):
    """A texture applied flat across the patch: uv is the position, rescaled."""
    return ((np.asarray(positions)[:, [0, 2]] + 1.0) * 0.5).astype('f4')


def _dome_normal(positions):
    """The exact normal of ``shapes.grid(bump=1.0)``, which is a cosine dome."""
    at = np.asarray(positions, dtype='d')
    half = np.pi / 2.0
    slope_x = -half * np.sin(at[:, 0] * half) * np.cos(at[:, 2] * half)
    slope_z = -half * np.cos(at[:, 0] * half) * np.sin(at[:, 2] * half)
    normal = np.stack([-slope_x, np.ones(len(at)), -slope_z], axis=1)
    return (normal / np.linalg.norm(normal, axis=1)[:, None]).astype('f4')


def _texel_density(attributes, indices):
    """sqrt(uv area / world area), per triangle.

    On a flat patch under a linear map this is the same number everywhere and
    stays that number however far the patch is reduced: both areas grow
    together. It is the texel-per-metre figure a renderer samples at, so a
    level whose density has drifted is a level whose texture is magnified.
    """
    faces = np.asarray(indices).reshape(-1, 3)
    world = np.asarray(attributes['POSITION'], dtype='d')[faces]
    uv = np.asarray(attributes['TEXCOORD_0'], dtype='d')[faces]
    world_area = 0.5 * np.linalg.norm(
        np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0]), axis=1
    )
    edge_a, edge_b = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    uv_area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    usable = world_area > 1e-12
    return np.sqrt(uv_area[usable] / world_area[usable])


class TestAnAttributeStaysOnItsPoint:
    def test_a_flat_patch_keeps_its_texel_density(self):
        """The one case where the answer is known exactly.

        A plane textured by a linear map has a single texel density, and no
        reduction of a plane can change it: every triangle, at every level, is
        a piece of the same map. A level that comes back at a fraction of the
        source density is a level whose texture is magnified by the reciprocal.
        """
        positions, indices = shapes.grid(33)
        attributes = {'POSITION': positions, 'TEXCOORD_0': _linear_uv(positions)}
        source = float(np.median(_texel_density(attributes, indices)))

        sequence = collapse_sequence(attributes, indices)
        for target in (512, 128, 32, 8):
            result = sequence.at(target_count=target)
            density = _texel_density(result.attributes, result.indices)
            assert np.allclose(density, source, rtol=1e-5), (
                '%d triangles: density %r against %r'
                % (result.triangle_count, np.unique(np.round(density, 6)), source)
            )

    def test_a_carried_value_belongs_to_the_position_it_is_on(self):
        """Every output vertex carries the uv its own position deserves.

        ``endpoint`` placement puts every surviving point back on an input
        point, so the map can be evaluated at the output position and compared
        with what came back -- no interpolation, no tolerance for a point that
        moved.
        """
        positions, indices = shapes.grid(17)
        attributes = {'POSITION': positions, 'TEXCOORD_0': _linear_uv(positions)}
        result = simplify(
            attributes, indices, SimplifyOptions(target_count=64, placement='endpoint')
        )
        assert np.allclose(
            result.attributes['TEXCOORD_0'], _linear_uv(result.attributes['POSITION']), atol=1e-6
        )

    def test_a_carried_normal_still_belongs_to_the_surface(self):
        """Every vertex of a reduced dome carries the dome's normal at its own place.

        The dome has a normal in closed form, so there is a right answer at
        every point of it and no tolerance is needed: with ``endpoint``
        placement each output vertex is an input vertex, and the normal it
        carries is either the one measured there or one measured somewhere
        else. Shading that turns jagged at a coarse level is this, vertex by
        vertex.
        """
        positions, indices = shapes.grid(33, bump=1.0)
        attributes = {'POSITION': positions, 'NORMAL': _dome_normal(positions)}
        result = simplify(
            attributes, indices, SimplifyOptions(target_count=256, placement='endpoint')
        )

        carried = np.asarray(result.attributes['NORMAL'], dtype='d')
        deserved = np.asarray(_dome_normal(result.attributes['POSITION']), dtype='d')
        apart = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', carried, deserved), -1.0, 1.0)))
        assert float(np.max(apart)) < 1.0, (
            '%d of %d normals belong somewhere else, worst %.1f degrees out'
            % (int(np.sum(apart > 1.0)), len(apart), float(np.max(apart)))
        )

    def test_no_vertex_is_split_that_carries_nothing_to_split_it(self):
        """A reduced closed surface comes back with the vertices it needs.

        An output vertex is a distinct combination of point and attributes. A
        corner reading a vertex the collapse left behind is a corner carrying
        values nothing else at its point carries, so it becomes a vertex of its
        own -- and a level ends up with several times the vertices its triangles
        need, paid for in the vertex buffer and in every transform of it. Euler
        gives the count for a closed surface exactly.
        """
        positions, indices = shapes.icosphere(4)
        normals = positions / np.linalg.norm(positions, axis=1)[:, None]
        attributes = {'POSITION': positions, 'NORMAL': normals.astype('f4')}
        for target in (2000, 500, 200):
            result = simplify(attributes, indices, SimplifyOptions(target_count=target))
            needed = result.triangle_count // 2 + 2
            assert len(result.attributes['POSITION']) == needed, (
                '%d triangles came back with %d vertices, not %d'
                % (result.triangle_count, len(result.attributes['POSITION']), needed)
            )


class TestASeamStaysASeam:
    """A point drawn twice, once per chart, keeps both charts apart.

    The patch below is textured in two halves: the vertices along the middle
    column are doubled, the left copy finishing the left chart at ``u = 1`` and
    the right copy starting the right chart at ``u = 0``. Welding merges the
    two copies into one *point* -- they are at the same position -- and they
    stay two *corners*, which is what a seam is.
    """

    @staticmethod
    def _two_charts(side=17):
        positions, indices = shapes.grid(side)
        faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        positions = np.asarray(positions, dtype='f4')
        column = np.arange(len(positions)) // side  # grid() runs x slowest
        middle = side // 2

        # The right chart gets its own copy of every vertex on the seam column
        # and of everything right of it, so no vertex is shared across the seam.
        right = column >= middle
        copy_of = np.full(len(positions), -1, dtype=np.int64)
        copy_of[right] = len(positions) + np.arange(int(right.sum()))
        positions = np.concatenate([positions, positions[right]])

        uv = np.zeros((len(positions), 2), dtype='f4')
        along = (positions[:, 2] + 1.0) * 0.5
        left_u = np.clip((positions[:, 0] + 1.0) / 1.0, 0.0, 1.0)
        right_u = np.clip(positions[:, 0], 0.0, 1.0)
        uv[:, 1] = along
        uv[: len(column), 0] = left_u[: len(column)]
        uv[len(column) :, 0] = right_u[len(column) :]

        on_right = column[faces].min(axis=1) >= middle
        faces[on_right] = copy_of[faces[on_right]]
        return (
            {'POSITION': positions, 'TEXCOORD_0': uv},
            faces.reshape(-1).astype(np.uint32),
        )

    @staticmethod
    def _crossings(attributes, indices):
        """Triangles whose ``u`` outruns the distance they cover.

        Both charts run ``u`` with the surface at the same rate, so inside
        either of them a triangle's ``u`` range is its ``x`` range. Only a
        triangle reading one corner from each chart can do more than that, and
        what it does is sample a band right across the texture -- the visible
        tear a seam exists to prevent.
        """
        faces = np.asarray(indices).reshape(-1, 3)
        across = np.asarray(attributes['POSITION'], dtype='d')[faces][:, :, 0]
        along = np.asarray(attributes['TEXCOORD_0'], dtype='d')[faces][:, :, 0]
        return np.ptp(along, axis=1) > np.ptp(across, axis=1) + 1e-6

    def test_the_source_mesh_has_a_seam_to_keep(self):
        """The fixture is what it claims: one position, two charts, no crossings."""
        attributes, indices = self._two_charts()
        sequence = collapse_sequence(attributes, indices)
        copies = np.bincount(sequence.vertex_point, minlength=len(sequence.points))
        assert int(np.sum(copies > 1)) > 8, 'the seam did not weld into shared points'
        assert not np.any(self._crossings(attributes, indices))

    def test_no_triangle_ends_up_spanning_both_charts(self):
        """Each surviving triangle still samples from one chart or the other."""
        attributes, indices = self._two_charts()
        sequence = collapse_sequence(attributes, indices)
        for target in (256, 64, 16):
            result = sequence.at(target_count=target)
            crossed = self._crossings(result.attributes, result.indices)
            assert not np.any(crossed), '%d triangles: %d of them cross the seam' % (
                result.triangle_count,
                int(np.sum(crossed)),
            )
