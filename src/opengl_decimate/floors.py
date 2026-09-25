"""How far a mesh will reduce, asked before a reduction is spent on it.

A decimation can only merge the ends of an edge, and three properties of a model
decide how many edges there are to merge. All three are cheap to measure and
none of them changes as the reduction runs, so the answer is available up front.

The first is the pieces. Every connected piece reduces on its own and keeps at
least one triangle -- four if it is closed. A canopy of separate leaf cards is
as many pieces as it has cards, so no target takes it below the count of the
leaves.

The second is the handles. A tunnel through the surface survives every
contraction, because the link condition preserves topology. A scan of feathers
or lace arrives with hundreds and each costs the triangles it takes to go round
it.

The third is the atlas. An unwrap cuts the surface into charts and lays them flat on the
image, duplicating the vertices along each cut -- so every vertex belongs to
exactly one chart, and a triangle samples the part of the image it is sitting on
only while all three of its corners are in the same one. A contraction across a
cut leaves a triangle whose three coordinates point at three unrelated places in
the atlas, and what it draws is the stripe between them. How soon that starts
happening is set by how finely the model was cut: ``charts`` counts the pieces
and ``texture_floor`` is the triangle count they imply. ``seam_share`` counts
the edges leading onto a seam, which ``lock_seams`` holds, and says what that
option would cost.

:func:`survey` measures all three and puts a floor on what any reduction can
reach. A caller who finds that floor near the triangle count they started with
has a model that wants a different operation -- an imposter, a re-authored
atlas, foliage baked onto larger cards -- rather than a lower target.

    from opengl_decimate import survey

    report = survey(attributes, indices)
    if report.reducible < 0.5:
        ...  # decimation will not get you where you are going
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opengl_decimate import topology
from opengl_decimate.corners import copies_per_point
from opengl_decimate.options import SimplifyOptions
from opengl_decimate.types import AttributeMap, IndexArray

__all__ = ['Survey', 'survey']


@dataclass(frozen=True)
class Survey:
    """What a mesh is made of, and how far that lets it reduce."""

    #: Triangles the caller handed in.
    triangles: int
    #: Distinct positions they cover, once coincident vertices are welded.
    points: int
    #: Triangles welding dropped for leaving with a repeated corner. On a clean
    #: export this is zero; on a scan it is the gap between what the caller
    #: counted and what there was to reduce.
    welded_away: int
    #: Connected pieces of surface.
    pieces: int
    #: How many of those have a border, so are not closed shells.
    open_pieces: int
    #: Triangles in the largest piece, which is the subject where the rest are
    #: specks ``drop_components_below`` would take.
    largest_piece: int
    #: Handles through the surface, from Euler's formula.
    handles: int
    #: The share of edges whose ends are drawn at different numbers of texture
    #: coordinates: the edges leading onto a seam from the chart beside it,
    #: which ``lock_seams`` does not contract. An edge running along a seam has
    #: the same number at both ends and is not counted. Zero for a mesh
    #: carrying no texture coordinates.
    seam_share: float
    #: Pieces the texture atlas is cut into. Zero for a mesh carrying no
    #: texture coordinates, which has no atlas to measure.
    charts: int = 0
    #: Triangles in the middle chart, and in the smaller quarter of them.
    median_chart: int = 0
    small_chart: int = 0
    #: Triangles ``drop_components_below`` removed, where the options asked for
    #: it. They are in ``triangles`` and in none of the counts after it.
    dropped_away: int = 0

    @property
    def surveyed(self) -> int:
        """The triangles the survey describes: what welding and dropping left."""
        return self.triangles - self.welded_away - self.dropped_away

    @property
    def floor(self) -> int:
        """The fewest triangles any reduction of this mesh can reach.

        A lower bound, and only from the pieces: each open piece keeps at least
        one triangle and each closed one at least four. Handles and seams push
        the real floor above this and are reported separately, because neither
        turns into a triangle count a caller could rely on.
        """
        return self.open_pieces + 4 * (self.pieces - self.open_pieces)

    @property
    def texture_floor(self) -> int:
        """Below this many triangles the model starts sampling across its atlas.

        A triangle samples the part of the image it sits on only while all three
        of its corners are in one chart. Once an output triangle is larger than
        a chart it cannot keep them there, and its three coordinates point at
        three unrelated places in the atlas -- what it draws is the stripe
        between them. The reduction is unharmed and so is the silhouette; the
        unwrap is what has run out.

        The estimate is the triangle count at which one output triangle grows to
        the size of a chart in the *smaller quarter* of them: past that, a
        quarter of the model is being drawn from charts too small to describe
        it. Zero where there is no atlas to run out of.

        How visible that is depends on how much of the surface those small
        charts cover. An atlas of a few large charts and a handful of small ones
        loses only the handful, and reads well below the figure; one whose
        charts are *uniformly* small has nowhere to hide. Compare the floor with
        ``median_chart`` to tell those apart.
        """
        if not self.small_chart:
            return 0
        return self.surveyed // self.small_chart

    @property
    def reducible(self) -> float:
        """The share of the triangles a reduction could in principle remove.

        One for a model nothing stops, zero for one already at its floor. Read
        it before choosing a target: a canopy of leaf cards comes back near
        zero, and no target will improve on that.
        """
        live = self.surveyed
        return max(0.0, (live - self.floor) / live) if live else 0.0


def survey(
    attributes: AttributeMap, indices: IndexArray, options: SimplifyOptions | None = None
) -> Survey:
    """Measure what would stop a reduction of this mesh, without running one.

    ``attributes`` and ``indices`` are the same arrays
    :func:`~opengl_decimate.simplify` takes, checked the same way. Texture
    coordinates are read where they are there, to find the atlas seams; nothing
    else is. ``options`` are the ones the reduction will be given: their
    ``weld_tolerance`` and ``drop_components_below`` decide which mesh is
    reduced, so the survey measures that mesh. Nothing else in them is read.

    The cost is the weld and one pass for the adjacency -- a fraction of a
    reduction, and the same fraction whatever target was in mind.
    """
    mesh = topology.build(
        topology.positions_of(attributes),
        indices,
        0.0 if options is None else options.weld_tolerance,
        0.0 if options is None else options.drop_components_below,
    )
    faces = mesh.faces
    welded_away = mesh.input_faces - len(faces) - mesh.dropped_faces
    if not len(faces):
        return Survey(
            triangles=mesh.input_faces,
            points=len(mesh.positions),
            welded_away=welded_away,
            pieces=0,
            open_pieces=0,
            largest_piece=0,
            handles=0,
            seam_share=0.0,
        )

    used = np.unique(faces)
    pairs = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, times = np.unique(pairs, axis=0, return_counts=True)

    # Every point gets a label, an unused one its own, so only the labels of
    # points some triangle uses name a piece of the surface.
    labels = topology.components(faces, len(mesh.positions))
    piece_of = labels[faces[:, 0]]
    names, sizes = np.unique(piece_of, return_counts=True)
    where = np.searchsorted(names, piece_of)

    # Euler per piece: V - E + F is 2 - 2g for a closed one, and a piece with a
    # border has no handles to count -- its characteristic answers a different
    # question, so only closed pieces contribute.
    per_piece = np.zeros((len(names), 3), dtype=np.int64)
    np.add.at(per_piece[:, 0], np.searchsorted(names, labels[used]), 1)
    np.add.at(per_piece[:, 1], np.searchsorted(names, labels[edges[:, 0]]), 1)
    np.add.at(per_piece[:, 2], where, 1)
    bordered = np.zeros(len(names), dtype=bool)
    if np.any(times == 1):
        bordered[np.searchsorted(names, labels[edges[times == 1][:, 0]])] = True
    characteristic = per_piece[:, 0] - per_piece[:, 1] + per_piece[:, 2]
    handles = np.maximum(0, (2 - characteristic[~bordered]) // 2)

    copies = copies_per_point(
        mesh.vertex_point, mesh.corners, dict(attributes), len(mesh.positions)
    )
    held = float(np.count_nonzero(copies[edges[:, 0]] != copies[edges[:, 1]]))

    mapped = next(
        (value for name, value in attributes.items() if name.startswith('TEXCOORD')), None
    )
    charts = median_chart = small_chart = 0
    if mapped is not None:
        per_face = topology.atlas_charts(faces, mesh.corners, mapped)
        _, held_by = np.unique(per_face, return_counts=True)
        charts = len(held_by)
        median_chart = int(np.median(held_by))
        small_chart = max(1, int(np.percentile(held_by, 25)))

    return Survey(
        triangles=mesh.input_faces,
        points=len(mesh.positions),
        welded_away=welded_away,
        pieces=len(names),
        open_pieces=int(np.count_nonzero(bordered)),
        largest_piece=int(sizes.max()),
        handles=int(handles.sum()),
        seam_share=held / len(edges),
        charts=charts,
        median_chart=median_chart,
        small_chart=small_chart,
        dropped_away=mesh.dropped_faces,
    )
