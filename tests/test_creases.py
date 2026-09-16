"""Keeping the edges a shape is read by, when the normals are recomputed.

Recomputing normals is what stops a decimated level faceting: a carried normal
is the one measured at its vertex's own place and is the more accurate of the
two, but its error is uncorrelated between neighbours, and shading shows the
*gradient*. Accumulating the surface's own normals is coherent by construction.

Accumulating them per *point* is too coarse, though. A hard edge is one position
the model draws twice, once per side, and averaging across it turns a cube into
a ball. So the recomputation accumulates per **smoothing group**: corners at a
point that the input drew with the same normal, and -- with ``crease_angle`` --
that are not separated by a fold. That is the input's own marking of what is
hard, used to keep it hard.
"""

import numpy as np
import pytest
import shapes

from opengl_decimate import DecimateError, SimplifyOptions, simplify


def _sharpness(result):
    """How axis-aligned the normals are: 1 for a hard cube, 0.577 for a rounded one."""
    normals = np.asarray(result.attributes['NORMAL'], dtype='d')
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]
    return float(np.min(np.max(np.abs(normals), axis=1)))


def _cylinder(around=64, rings=6):
    """A drum: a curved wall and two flat caps, with a hard rim between them.

    No ``NORMAL`` at all, which is the case ``crease_angle`` is for -- there is
    no marking to inherit, so the fold itself has to say where the edge is.
    """
    angle = np.arange(around) * (2.0 * np.pi / around)
    ring = np.stack([np.cos(angle), np.zeros(around), np.sin(angle)], axis=1)
    rows = [ring + (0.0, height, 0.0) for height in (-1.0, 0.0, 1.0)]
    faces = []
    for band in range(2):
        for corner in range(around):
            step = (corner + 1) % around
            low, high = band * around, (band + 1) * around
            faces += [
                (low + corner, high + corner, low + step),
                (low + step, high + corner, high + step),
            ]
    base = 3 * around
    for cap, height in enumerate((-1.0, 1.0)):
        edge = (0 if height < 0 else 2) * around
        for step in range(1, rings + 1):
            rows.append(ring * (1.0 - step / rings) + (0.0, height, 0.0))
        for step in range(rings):
            here = base + (cap * rings + step) * around
            previous = edge if step == 0 else here - around
            for corner in range(around):
                after = (corner + 1) % around
                if height < 0:
                    faces += [
                        (previous + corner, previous + after, here + corner),
                        (previous + after, here + after, here + corner),
                    ]
                else:
                    faces += [
                        (previous + corner, here + corner, previous + after),
                        (previous + after, here + corner, here + after),
                    ]
    return (
        {'POSITION': np.concatenate(rows).astype('f4')},
        np.asarray(faces, dtype=np.uint32).reshape(-1),
    )


class TestRecomputingKeepsWhatWasHard:
    def test_a_hard_cube_does_not_come_back_round(self):
        """Every normal of a cube points along an axis, recomputed or carried.

        Averaging the three faces at a corner gives 0.577 on each axis, which
        is a ball wearing a cube's silhouette -- the single most visible way to
        get this wrong.
        """
        attributes, indices = shapes.cube_with_hard_normals()
        carried = simplify(attributes, indices, SimplifyOptions(target_ratio=1.0))
        recomputed = simplify(
            attributes, indices, SimplifyOptions(target_ratio=1.0, recompute_normals=True)
        )
        assert _sharpness(carried) == pytest.approx(1.0)
        assert _sharpness(recomputed) == pytest.approx(1.0), (
            'the cube came back rounded: worst axis component %.3f' % (_sharpness(recomputed),)
        )

    def test_a_smooth_surface_still_gets_one_normal_per_point(self):
        """Nothing is split that the input did not split.

        A sphere's recomputed normals point away from the centre, which they
        only can if every face at a point contributed to one normal.
        """
        positions, indices = shapes.icosphere(3)
        attributes = {'POSITION': positions}
        result = simplify(
            attributes, indices, SimplifyOptions(target_count=500, recompute_normals=True)
        )
        at = np.asarray(result.attributes['POSITION'], dtype='d')
        outward = at / np.linalg.norm(at, axis=1)[:, None]
        normals = np.asarray(result.attributes['NORMAL'], dtype='d')
        apart = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', normals, outward), -1.0, 1.0)))
        assert float(np.max(apart)) < 12.0

    def test_a_texture_seam_does_not_become_a_crease(self):
        """The rule is the input's *normals*, not everything it carries.

        A seam splits the texture coordinate and nothing else, and a level whose
        shading creased along every seam would be worse than one that never
        recomputed at all.
        """
        positions, indices = shapes.grid(17)
        uv = ((np.asarray(positions)[:, [0, 2]] + 1.0) * 0.5).astype('f4')
        seamed = {'POSITION': positions, 'TEXCOORD_0': uv}
        result = simplify(
            seamed, indices, SimplifyOptions(target_count=200, recompute_normals=True)
        )
        normals = np.asarray(result.attributes['NORMAL'], dtype='d')
        # The patch is flat, so every normal is the same one.
        assert np.allclose(normals, normals[0], atol=1e-6)

    def test_a_fold_is_kept_hard_where_the_angle_asks_for_it(self):
        """With no input normals, the fold itself says where the edge is."""
        attributes, indices = _cylinder()
        rounded = simplify(
            attributes,
            indices,
            SimplifyOptions(target_ratio=1.0, recompute_normals=True, crease_angle=180.0),
        )
        kept = simplify(
            attributes,
            indices,
            SimplifyOptions(target_ratio=1.0, recompute_normals=True, crease_angle=40.0),
        )

        # The rim is where the two surfaces meet, so it is the only place the
        # answer differs: held, it carries a vertex per side and the cap's side
        # points straight up; rounded, one vertex carries the average of both
        # and the cap turns over at its edge.
        def cap_side_of_rim(result):
            normals = np.asarray(result.attributes['NORMAL'], dtype='d')
            at = np.asarray(result.attributes['POSITION'], dtype='d')
            radius = np.linalg.norm(at[:, [0, 2]], axis=1)
            on_rim = (np.abs(np.abs(at[:, 1]) - 1.0) < 1e-4) & (radius > 0.95)
            return float(np.max(np.abs(normals[on_rim][:, 1])))

        assert cap_side_of_rim(kept) == pytest.approx(1.0, abs=1e-4), (
            'the rim rounded the cap over: best cap-side normal %.3f' % (cap_side_of_rim(kept),)
        )
        assert cap_side_of_rim(rounded) < 0.9, 'the fixture does not exercise the option'
        assert len(kept.attributes['POSITION']) > len(rounded.attributes['POSITION']), (
            'a hard rim needs a vertex per side, so holding it costs vertices'
        )

    def test_the_angle_is_refused_where_it_is_not_an_angle(self):
        with pytest.raises(DecimateError):
            SimplifyOptions(target_count=10, crease_angle=-1.0)
        with pytest.raises(DecimateError):
            SimplifyOptions(target_count=10, crease_angle=181.0)
