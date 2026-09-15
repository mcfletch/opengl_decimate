"""What to ask for, and what it means.

One :class:`SimplifyOptions` covers a whole reduction. Everything it holds is
either a target (how far to go), a statement about the input (how noisy, how
much to weld), or a limit on what a contraction may do to the surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from opengl_decimate.types import DecimateError

__all__ = ['SimplifyOptions', 'METRICS', 'SCHEDULES', 'PLACEMENTS']

#: How the error of a contraction is measured. ``quadric`` is the squared
#: distance to the incident triangle planes; ``probabilistic`` is the expected
#: squared distance when those planes are read as noisy samples, which is what a
#: reconstructed surface is.
METRICS = ('quadric', 'probabilistic')

#: How the next contraction is chosen. ``heap`` takes the cheapest candidate on
#: the whole mesh, which is the best quality available and is sequential by
#: construction. ``multiple-choice`` takes the cheapest of a few sampled
#: candidates, which gives up the global ordering -- and a little quality with
#: it -- in exchange for needing no global structure at all.
SCHEDULES = ('heap', 'multiple-choice')

#: Where the surviving point goes. ``optimal`` is the minimiser of the summed
#: quadric, which is the more accurate. ``endpoint`` keeps whichever end of the
#: edge costs less, so every point in the result is a point of the input and the
#: attributes it carries were measured there rather than inferred.
PLACEMENTS = ('optimal', 'endpoint')

#: Noise used by the probabilistic metric where a caller selects it and leaves
#: one at ``None``. The position figure is a fraction of the model's bounding-box
#: diagonal, so it means the same thing whatever units the model is in; the
#: normal figure is a length in normal-vector units, where the vector itself is
#: unit length. Each field defaults on its own, so naming one leaves the other
#: at this rather than at zero.
DEFAULT_POSITION_NOISE = 0.001
DEFAULT_NORMAL_NOISE = 0.001


@dataclass(frozen=True)
class SimplifyOptions:
    """Everything one reduction needs to know.

    At least one target is required, and where more than one is given the
    reduction stops at whichever is reached first.

    :param target_ratio: share of the input triangles to keep, in ``(0, 1]``.
    :param target_count: triangles to stop at.
    :param target_error: deviation to stop at, a length in model units.
    :param metric: one of :data:`METRICS`.
    :param position_noise: probabilistic metric only -- how far the surface's
        points are trusted, as a fraction of the bounding-box diagonal. It
        enters the quadric as a constant added to every cost, so it raises the
        deviation the reduction reports without changing which contraction it
        chooses. ``None`` uses :data:`DEFAULT_POSITION_NOISE`.
    :param normal_noise: probabilistic metric only -- how far its normals are
        trusted, in units of the unit normal. This is the term that makes the
        metric behave differently: it is what conditions the 3x3 solve, so a
        single plane has a minimum and a noisy neighbourhood is well behaved.
        ``None`` uses :data:`DEFAULT_NORMAL_NOISE`; zero gives the classical
        metric back whatever ``position_noise`` says.
    :param boundary_weight: how hard the edge of an open surface is held. Zero
        lets a border wander; raising it holds the outline at the cost of
        keeping triangles along it.
    :param lock_boundary: hold the border exactly. What a cluster group's
        simplification needs, so its neighbours still meet it.
    :param locked: further points to hold, by index into the welded points --
        which are the caller's own vertex indices unless vertices were welded.
    :param max_normal_flip: degrees a face's normal may turn. Past this it has
        been folded over rather than moved.
    :param min_triangle_quality: the shape floor, from 0 (any shape) to 1
        (equilateral only).
    :param placement: one of :data:`PLACEMENTS`.
    :param weld_tolerance: merge input vertices within this distance first.
    :param drop_components_below: remove whole connected pieces smaller than
        this share of the model's bounding-box diagonal, before reducing. A
        scan arrives with the subject and whatever else was in the room, and
        every crumb is a closed shell with a floor of four triangles -- so a
        target of a few hundred is spent on crumbs unless they go. ``0.01`` is
        about a pixel where the whole model covers a hundred. The largest piece
        is never dropped.
    :param schedule: one of :data:`SCHEDULES`.
    :param candidates: how many edges ``multiple-choice`` samples per step.
    :param seed: fixes the sampling, so a run is reproducible.
    :param recompute_normals: replace ``NORMAL`` with the normals of the surface
        that is left, rather than carrying the input's. The result carries a
        ``NORMAL`` whether or not the input did, so a mesh of positions alone
        comes back with one. Normals are accumulated per point, so vertices
        split by a texture seam still share a normal and the seam stays a
        texture seam rather than becoming a crease.
    :param certify: measure the deviation of the result from the input and put
        it in ``measured_error``. Off by default: it samples both surfaces and
        measures each sample against the other, which costs about what the
        reduction did.
    :param certify_samples: points taken from each surface when certifying.
        Raising it tightens the estimate of the worst case, and costs in
        proportion; the root-mean-square settles long before the maximum does.
    """

    target_ratio: float | None = None
    target_count: int | None = None
    target_error: float | None = None
    metric: str = 'quadric'
    position_noise: float | None = None
    normal_noise: float | None = None
    boundary_weight: float = 1.0
    lock_boundary: bool = False
    locked: Sequence[int] | None = None
    max_normal_flip: float = 90.0
    min_triangle_quality: float = 0.0
    placement: str = 'optimal'
    weld_tolerance: float = 0.0
    drop_components_below: float = 0.0
    schedule: str = 'heap'
    candidates: int = 8
    seed: int = 0
    recompute_normals: bool = False
    certify: bool = False
    certify_samples: int = 4000

    def __post_init__(self) -> None:
        if self.target_ratio is None and self.target_count is None and self.target_error is None:
            raise DecimateError(
                'no target given: set one of target_ratio, target_count or target_error'
            )
        if self.target_ratio is not None and not 0.0 < self.target_ratio <= 1.0:
            raise DecimateError('target_ratio must be in (0, 1], got %r' % (self.target_ratio,))
        if self.target_count is not None and self.target_count < 0:
            raise DecimateError('target_count must not be negative, got %r' % (self.target_count,))
        if self.target_error is not None and self.target_error < 0.0:
            raise DecimateError('target_error must not be negative, got %r' % (self.target_error,))
        if self.metric not in METRICS:
            raise DecimateError('metric must be one of %r, got %r' % (METRICS, self.metric))
        if self.schedule not in SCHEDULES:
            raise DecimateError('schedule must be one of %r, got %r' % (SCHEDULES, self.schedule))
        if self.placement not in PLACEMENTS:
            raise DecimateError(
                'placement must be one of %r, got %r' % (PLACEMENTS, self.placement)
            )
        if self.candidates < 1:
            raise DecimateError('candidates must be at least one, got %r' % (self.candidates,))
        for name in ('position_noise', 'normal_noise'):
            value = getattr(self, name)
            if value is not None and not value >= 0.0:
                raise DecimateError('%s must not be negative, got %r' % (name, value))
        if not 0.0 <= self.max_normal_flip <= 180.0:
            raise DecimateError(
                'max_normal_flip is in degrees and must be in [0, 180], got %r'
                % (self.max_normal_flip,)
            )
        if not 0.0 <= self.min_triangle_quality <= 1.0:
            raise DecimateError(
                'min_triangle_quality must be in [0, 1] -- 0 admits any shape, 1 only '
                'equilateral -- got %r' % (self.min_triangle_quality,)
            )
        if not self.weld_tolerance >= 0.0:
            raise DecimateError(
                'weld_tolerance must not be negative, got %r' % (self.weld_tolerance,)
            )
        if not self.drop_components_below >= 0.0:
            raise DecimateError(
                'drop_components_below is a share of the model and must not be negative,'
                ' got %r' % (self.drop_components_below,)
            )
        if not self.boundary_weight >= 0.0:
            raise DecimateError(
                'boundary_weight must not be negative, got %r' % (self.boundary_weight,)
            )
        if self.certify and self.certify_samples < 1:
            raise DecimateError(
                'certify_samples must be at least one, got %r' % (self.certify_samples,)
            )
        if self.locked is not None:
            held = np.asarray(self.locked)
            if held.size and (held.ndim != 1 or not np.issubdtype(held.dtype, np.integer)):
                raise DecimateError('locked must be a flat sequence of point indices')
            if held.size and int(held.min()) < 0:
                # NumPy would wrap a negative index round to the far end and
                # lock a point the caller never named.
                raise DecimateError(
                    'locked indices must not be negative, got %r' % (int(held.min()),)
                )

    def noise(self, diagonal: float) -> tuple[float, float]:
        """The position and normal noise to accumulate quadrics with.

        The classical metric has none. The probabilistic one uses what the
        caller named, each field defaulting on its own so that naming one does
        not silently zero the other -- and zeroing ``normal_noise`` is the
        classical metric under another name. The position figure is scaled by
        the model's ``diagonal``, so it is a length.

        >>> SimplifyOptions(target_ratio=0.5, metric='probabilistic').noise(1.0)
        (0.001, 0.001)
        >>> SimplifyOptions(
        ...     target_ratio=0.5, metric='probabilistic', position_noise=0.01
        ... ).noise(1.0)
        (0.01, 0.001)
        >>> SimplifyOptions(target_ratio=0.5).noise(1.0)
        (0.0, 0.0)
        """
        if self.metric != 'probabilistic':
            return 0.0, 0.0
        position = DEFAULT_POSITION_NOISE if self.position_noise is None else self.position_noise
        normal = DEFAULT_NORMAL_NOISE if self.normal_noise is None else self.normal_noise
        return position * diagonal, normal
