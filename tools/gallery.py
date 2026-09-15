#!/usr/bin/env python
"""Regenerate the pictures and the measurements in ``docs/GALLERY.md``.

Every number and every image under ``docs/gallery/`` comes from this script, so
a change to the reducer is answered by running it again rather than by editing a
table::

    tools/gallery.py                  # every subject
    tools/gallery.py --only bust      # one of them
    tools/gallery.py --no-render      # the measurements, without a GPU

**What it needs beyond the package.** ``opengl_decimate`` depends on NumPy and
nothing else, and that does not change: this is a documentation tool, not part
of the library. It reads glTF, merges a model's primitives and renders the
levels through **OpenGLContext**, and fetches the CC0 subjects through
**OpenGLContext-editor**'s Poly Haven client. Both are siblings of this package
in the workspace it is developed in, and neither is imported by anything a user
installs.

The subjects are CC0. :data:`SUBJECTS` records where each came from and who made
it, and :func:`credits` writes that into the page -- CC0 asks for none of it, and
it costs a line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, 'docs')
PICTURES = os.path.join(DOCS, 'gallery')

#: Where a downloaded subject is kept. Nothing is fetched twice, and nothing
#: fetched lands in the repository: the pictures are the artefact, not the art.
CACHE = os.environ.get(
    'OPENGL_DECIMATE_GALLERY_CACHE',
    os.path.join(os.path.expanduser('~'), '.cache', 'opengl_decimate', 'gallery'),
)

#: Raw URLs for the images, so the README renders on PyPI as well as on GitHub.
RAW = 'https://raw.githubusercontent.com/mcfletch/opengl_decimate/main/docs/gallery'

#: ``oglc-view`` frames a model by putting a sphere of its radius across the
#: field of view, which is this many radii back at margin one. The probe is told
#: a distance directly, so it needs the same number to agree with the captures.
FIT = 2.6

#: A bundled CC0 studio HDRI, so a level is lit the way an asset is looked at
#: rather than by one lamp in the dark.
ENVIRONMENT = 'studio_small_03'

#: The chain a game would actually ship, in triangles. Past the top of it a
#: renderer with a world to draw is not going to spend the triangles on one
#: prop however close the player stands, and below the bottom an imposter -- a
#: billboard of the thing -- costs less than any mesh.
LEVELS = (32_000, 8_000, 4_000, 2_000, 1_000, 500)

#: Screen-space error a level is placed at: the distance where its measured
#: deviation from the source projects to this many pixels. One pixel is the
#: rule a streaming renderer uses, and it is the honest answer to "how far away
#: does this level have to be", because a length in model units means nothing
#: until it is turned into pixels.
SSE_PIXELS = 1.0

#: The camera's vertical field of view, matching ``LODProbe`` and the viewer.
FOVY = 45.0

#: The closest anything is drawn, in radii beyond the surface. The source and
#: the finest level are both drawn here, so the reference and the top of the
#: chain can be compared directly; a level whose screen-space error puts it
#: nearer than this is usable right up to the object.
NEAREST = 0.25

#: And the furthest, past which the object is a smudge whatever it is made of.
FURTHEST = 32.0

#: Pixels per cell in a contact sheet.
CELL = 256

#: Frames timed per level, after a warm-up. The median is reported.
FRAMES = 90


@dataclass(frozen=True)
class Subject:
    """One model in the gallery, and where it came from."""

    slug: str
    title: str
    #: ``polyhaven:<asset>`` to fetch, or a path relative to the workspace root.
    source: str
    credit: str
    #: Degrees about Y, chosen so the model faces the camera. The probe is told
    #: degrees and ``oglc-view`` radians, so this is converted at each call
    #: rather than kept twice.
    rotation: float = 0.0
    note: str = ''
    #: Components below this share of the model's diagonal go before reducing.
    drop_components_below: float = 0.0
    #: Hold the texture atlas exactly, for a subject whose atlas is fragmented
    #: enough that the alternative is a level whose texture has stopped
    #: describing it. Costs the triangles the seam network needs.
    lock_seams: bool = False


SUBJECTS = (
    Subject(
        slug='cliff',
        title='Coastal cliff',
        source='polyhaven:coastal_cliff_04',
        credit=(
            "'Coastal Cliff 04' by Rob Tuytel and Rico Cilliers, from "
            '[Poly Haven](https://polyhaven.com/a/coastal_cliff_04), CC0'
        ),
        rotation=90.0,
        note=(
            'A photogrammetry scan of rock. There is no flat region to spend the '
            'first million triangles on and no symmetry to exploit: what the '
            'reduction keeps is silhouette.'
        ),
    ),
    Subject(
        slug='ruffs',
        title='Lekking ruffs',
        source='tmp/lekking_ruffs.glb',
        credit=(
            "'Lekking ruffs', inventory MP 045, from the Krystyna and Włodzimierz Tomek "
            'Natural Science Museum in Ciężkowice, Poland. Digitised by the Regional '
            'Digitalisation Lab, Małopolska Institute of Culture in Kraków, for the '
            '[Virtual Museums of Małopolska]'
            '(https://muzea.malopolska.pl/en/objects-list/2250) project, CC0'
        ),
        rotation=135.0,
        note=(
            'A museum scan exported as twenty-five primitives, each stopping at the '
            '65,535 vertices a 16-bit index can name -- so the reduction sees one '
            'surface only because the primitives are merged and welded first. Nine '
            'of its thirteen components are specks the photogrammetry left behind, '
            'and `drop_components_below` takes them: 572 triangles, which is fifteen '
            'per cent of what the coarsest rung has to spend.\n\n'
            'This is the subject whose atlas decides the answer, so it is the one '
            'reduced with `lock_seams`. A third of its edges are on a chart '
            'boundary, and the chain stops where the seam network is all that is '
            'left. Without the lock it reaches two thousand triangles, and by eight '
            'thousand each triangle is already sampling two and a half times the '
            'texture it should -- the levels are there, but the bird is not on them. '
            'See **Where a chain stops** below.'
        ),
        drop_components_below=0.01,
        lock_seams=True,
    ),
    Subject(
        slug='rocks',
        title='Coastal land rocks',
        source='polyhaven:coast_land_rocks_02',
        credit=(
            "'Coast Land Rocks 02' by Rob Tuytel and Rico Cilliers, from "
            '[Poly Haven](https://polyhaven.com/a/coast_land_rocks_02), CC0'
        ),
        rotation=180.0,
        note='Several separate boulders in one mesh, so the reduction has to spend across them.',
    ),
    Subject(
        slug='tree',
        title='Island tree',
        source='polyhaven:island_tree_03',
        credit=(
            "'Island Tree 03' by Rob Tuytel and Rico Cilliers, from "
            '[Poly Haven](https://polyhaven.com/a/island_tree_03), CC0'
        ),
        rotation=135.0,
        note=(
            'The hard case. A canopy of alpha-cut leaf cards barely welds at all -- '
            '2,085,320 triangles over 1,598,940 points -- because almost no two cards '
            'share a vertex, and a card cannot be reduced below the triangles that '
            'carry its cutout.'
        ),
    ),
    Subject(
        slug='bust',
        title='Marble bust',
        source='polyhaven:marble_bust_01',
        credit=(
            "'Marble Bust 01' by Rico Cilliers, from "
            '[Poly Haven](https://polyhaven.com/a/marble_bust_01), CC0'
        ),
        rotation=45.0,
        note='Small enough to read every triangle, and a face is where an error is obvious.',
    ),
)


@dataclass
class Group:
    """One material's share of a subject, and the reduction recorded for it."""

    material: Any
    attributes: dict
    indices: Any
    source_triangles: int
    sequence: Any


@dataclass
class Level:
    """One rung: what it holds, what it cost to make, what it costs to draw."""

    index: int
    triangles: int
    vertices: int
    error: float
    measured: float | None
    replay_ms: float
    draw_ms: float = 0.0
    fps: float = 0.0
    #: Radii beyond the surface at which this level's deviation is worth one
    #: pixel -- the distance a renderer would switch to it at.
    safe_at: float = float('inf')
    #: What the swap does to the picture there, from the engine's measurement:
    #: the share of the object's pixels whose outline moved, and whose shading
    #: changed.
    outline: float = 0.0
    shading: float = 0.0
    #: Where it is actually drawn in the gallery: its safe distance, or
    #: :data:`NEAREST` for the finest level and the source.
    shown_at: float = NEAREST
    note: str = ''
    images: dict = field(default_factory=dict)


@dataclass
class Reduction:
    """A whole subject: its source mesh, its levels and what they cost."""

    subject: Subject
    source_triangles: int
    source_vertices: int
    primitives: int
    materials: int
    welded_points: int
    load_s: float
    reduce_s: float
    contractions: int
    peak_mb: float
    levels: list = field(default_factory=list)
    radius: float = 1.0
    centre: Any = None
    #: Where the reduction ran out of legal contractions, if it did before the
    #: ladder did.
    floor: int = 0
    #: Connected pieces the welded source is in, handles through them, and the
    #: share of its edges an atlas seam holds -- see :func:`shape_of`. Between
    #: them they say where a floor comes from.
    pieces: int = 0
    handles: int = 0
    seam_share: float = 0.0
    groups: list = field(default_factory=list)
    #: Materials already written out with external textures, by the id of the
    #: material they came from, so a subject's maps are written once.
    textures: dict = field(default_factory=dict)


def shape_of(groups: list) -> tuple[int, int, float]:
    """Pieces, handles, and the share of edges a seam holds.

    The three reasons a chain stops, and they are properties of the model rather
    than of the reducer.

    Euler's formula gives the first two: a closed surface of ``p`` pieces and
    ``g`` handles has ``V - E + F = 2p - 2g``. Pieces are separable -- the small
    ones are what ``drop_components_below`` takes. Handles are not: contracting
    an edge under the link condition preserves the surface's topology by
    construction, so a tunnel cannot be closed by any number of contractions and
    each one costs the triangles it takes to go round it. A scan of feathers or
    foliage arrives with hundreds.

    The third is the texture atlas. An edge whose ends are drawn at different
    numbers of texture coordinates runs off a seam into a chart; a reduction
    crosses it without tearing the texture, but the coordinate the seam carries
    slides as the merged point moves, and ``lock_seams`` is what refuses it at
    the price of the triangles the seam network needs.
    """
    from opengl_decimate import corners, topology

    pieces = handles = 0
    held = total = 0
    for group in groups:
        mesh = group.sequence
        faces = mesh.faces
        if not len(faces):
            continue
        used = np.unique(faces)
        edges = np.unique(
            np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
            axis=0,
        )
        # Every point gets a label, an unused one its own, so only the labels of
        # points some triangle uses are components of the surface.
        labelled = topology.components(faces, len(mesh.points))
        parts = len(np.unique(labelled[used]))
        characteristic = len(used) - len(edges) + len(faces)
        pieces += parts
        handles += max(0, (2 * parts - characteristic) // 2)

        copies = corners.copies_per_point(
            mesh.vertex_point, mesh.corners, group.attributes, len(mesh.points)
        )
        held += int(np.count_nonzero(copies[edges[:, 0]] != copies[edges[:, 1]]))
        total += len(edges)
    return pieces, handles, (held / total if total else 0.0)


def peak_rss_mb() -> float:
    """High-water mark of this process's resident memory."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes; macOS reports bytes.
    return peak / (1024.0 if sys.platform != 'darwin' else 1024.0 * 1024.0)


def fetch(source: str) -> str:
    """The path to a subject's glTF, downloading it the first time.

    A Poly Haven asset is taken at its smallest published texture resolution:
    the mesh is the same bytes at every one of them, and this measures geometry.
    """
    if not source.startswith('polyhaven:'):
        path = os.path.join(os.path.dirname(ROOT), source)
        if not os.path.exists(path):
            raise SystemExit('%s is not here; the gallery needs it' % (path,))
        return path

    slug = source.split(':', 1)[1]
    sys.path.insert(0, os.path.join(os.path.dirname(ROOT), 'openglcontext-editor', 'src'))
    from OpenGLContext_editor.assets import polyhaven

    entry = polyhaven.files(slug)['gltf']['1k']['gltf']
    directory = os.path.join(CACHE, slug)
    document = _save(entry['url'], os.path.join(directory, os.path.basename(entry['url'])))
    for relative, named in (entry.get('include') or {}).items():
        _save(named['url'], os.path.join(directory, relative))
    return document


def _save(url: str, path: str) -> str:
    if os.path.exists(path) and os.path.getsize(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print('   fetching %s' % (os.path.basename(path),), flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'opengl_decimate gallery'})
    with urllib.request.urlopen(request) as response, open(path + '.part', 'wb') as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    os.replace(path + '.part', path)
    return path


def load(path: str) -> tuple[list, int]:
    """A model as one merged mesh per material, and how many primitives it was.

    Grouped rather than merged whole, because a level has to be *drawn*: a mesh
    draws with one material, so that is as far as the pieces can be brought
    together without losing what the model looks like.
    """
    from OpenGLContext.loaders.assets import merged_by_material, shapes
    from OpenGLContext.loaders.gltf import load_gltf

    scene = load_gltf(path, max_resource_bytes=None)
    grouped = merged_by_material(scene.group)
    if not grouped:
        raise SystemExit('%s holds no triangles' % (path,))
    return grouped, sum(1 for _ in shapes(scene.group))


def ladder(triangles: int) -> list:
    """``(triangles, note)`` for each level, the source first.

    A fixed chain rather than a share of the source, because what a renderer can
    afford is a triangle count and not a proportion: thirty-two thousand is
    thirty-two thousand whether the scan arrived with two hundred thousand
    triangles or two million.

    The source rides above the chain as a reference. It is not a level anything
    would ship -- past the top of the chain a renderer with a world to draw is
    not spending the triangles on one prop -- but it is the picture the rest are
    judged against.
    """
    levels = [(triangles, 'the source, for reference')]
    levels += [(count, '') for count in LEVELS if count < triangles]
    if len(levels) > 1:
        levels[1] = (levels[1][0], 'the finest a game would ship')
        levels[-1] = (levels[-1][0], 'past here, an imposter')
    return levels


def reduce_subject(subject: Subject, certify: bool) -> Reduction:
    """Decimate one subject once per material, and read every level off it."""
    from OpenGLContext.meshlod.chain import bounding_sphere

    from opengl_decimate import SimplifyOptions, collapse_sequence, topology
    from opengl_decimate import certify as certification

    print('== %s ==' % (subject.title,), flush=True)
    path = fetch(subject.source)
    start = time.perf_counter()
    grouped, primitives = load(path)
    load_s = time.perf_counter() - start

    options = SimplifyOptions(
        target_ratio=1.0,
        drop_components_below=subject.drop_components_below,
        lock_seams=subject.lock_seams,
    )
    source_triangles = sum(len(indices) // 3 for _m, _a, indices in grouped)
    welded = sum(
        topology.build(
            attributes['POSITION'], indices, 0.0, options.drop_components_below
        ).vertex_count
        for _m, attributes, indices in grouped
    )
    print(
        '   %s triangles in %d primitive%s over %d material%s, welded to %s points'
        % (
            f'{source_triangles:,}',
            primitives,
            '' if primitives == 1 else 's',
            len(grouped),
            '' if len(grouped) == 1 else 's',
            f'{welded:,}',
        ),
        flush=True,
    )

    # One reduction per material. Every rung below is a prefix replay of those
    # recordings, which is the whole point of `collapse_sequence`.
    start = time.perf_counter()
    groups = [
        Group(
            material=material,
            attributes=attributes,
            indices=indices,
            source_triangles=len(indices) // 3,
            sequence=collapse_sequence(attributes, indices, options),
        )
        for material, attributes, indices in grouped
    ]
    reduce_s = time.perf_counter() - start
    contractions = sum(len(group.sequence) for group in groups)
    print(
        '   reduced in %.1f s, %s contractions recorded' % (reduce_s, f'{contractions:,}'),
        flush=True,
    )

    whole = np.concatenate([group.attributes['POSITION'] for group in groups])
    centre, radius = bounding_sphere(whole)
    out = Reduction(
        subject=subject,
        source_triangles=source_triangles,
        source_vertices=len(whole),
        primitives=primitives,
        materials=len(groups),
        welded_points=welded,
        load_s=load_s,
        reduce_s=reduce_s,
        contractions=contractions,
        peak_mb=peak_rss_mb(),
        radius=float(radius),
        centre=centre,
        groups=groups,
    )
    out.pieces, out.handles, out.seam_share = shape_of(groups)

    for index, (count, note) in enumerate(ladder(source_triangles)):
        start = time.perf_counter()
        # A target is shared out across the materials in proportion to what each
        # brought, so one level is one triangle budget for the whole model.
        share = count / max(1, source_triangles)
        results = [
            group.sequence.at(target_count=max(4, round(group.source_triangles * share)))
            for group in groups
        ]
        replay_ms = (time.perf_counter() - start) * 1000.0
        triangles = sum(result.triangle_count for result in results)
        if out.levels and triangles >= out.levels[-1].triangles:
            # The reduction has run out of legal contractions, and every rung
            # below this one would be the same mesh again. Where a subject stops
            # and why is worth a row; nine copies of it are not.
            out.floor = out.levels[-1].triangles
            print('   floor reached at %s triangles' % (f'{out.floor:,}',), flush=True)
            break
        measured = None
        if certify and index:
            measured = max(
                certification.surface_deviation(
                    group.attributes['POSITION'],
                    group.indices,
                    result.attributes['POSITION'],
                    result.indices,
                    samples=4000,
                ).max
                for group, result in zip(groups, results, strict=True)
            )
        out.levels.append(
            Level(
                index=index,
                triangles=triangles,
                vertices=sum(len(r.attributes['POSITION']) for r in results),
                error=max(result.error for result in results),
                measured=measured,
                replay_ms=replay_ms,
                note=note,
            )
        )
        out.levels[-1].results = results  # type: ignore[attr-defined]
        print(
            '   level %d: %9s triangles  error %.5f  replay %6.1f ms'
            % (index, f'{triangles:,}', out.levels[-1].error, replay_ms),
            flush=True,
        )
    out.peak_mb = peak_rss_mb()
    return out


#: Which texture channels hold colour rather than measurements, and so are
#: written as sRGB.
COLOUR_CHANNELS = ('baseColor', 'emissive', 'sheenColor', 'specularColor')


def externalise(material: Any, where: str, slug: str, seen: dict) -> Any:
    """A copy of ``material`` whose textures are files beside the document.

    The writer embeds a texture into every document that names one, re-encoded
    as PNG -- so a subject's seven levels would carry seven copies of its maps,
    and a museum scan's came to four hundred megabytes apiece. Written once and
    named by relative ``uri``, the levels hold geometry and nothing else.
    """
    import copy

    from OpenGLContext.loaders.gltf.writer import ExternalImage

    if material is None:
        return None
    key = id(material)
    if key in seen:
        return seen[key]
    swapped = {}
    for channel, texture in (getattr(material, 'textures', None) or {}).items():
        image = getattr(texture, 'image', None)
        if image is None:
            continue
        name = '%s-%d-%s.png' % (slug, len(seen), channel)
        path = os.path.join(where, name)
        if not os.path.exists(path):
            image.save(path)
        swapped[channel] = ExternalImage(uri=name, srgb=channel in COLOUR_CHANNELS)
    copied = copy.copy(material)
    copied.textures = swapped
    seen[key] = copied
    return copied


def write_level(reduction: Reduction, level: Level, where: str) -> str:
    """One level as a ``.glb``, its materials naming textures written beside it."""
    from OpenGLContext.loaders.gltf.writer import SceneNode, write_glb
    from OpenGLContext.scenegraph.pbrmesh import PBRMesh

    built = []
    for group, result in zip(reduction.groups, level.results, strict=True):
        if not result.triangle_count:
            continue
        built.append(
            PBRMesh(
                positions=result.attributes['POSITION'],
                normals=result.attributes.get('NORMAL'),
                texcoords=result.attributes.get('TEXCOORD_0'),
                indices=result.indices,
                material=externalise(
                    group.material, where, reduction.subject.slug, reduction.textures
                ),
            )
        )
    path = os.path.join(where, '%s-l%d.glb' % (reduction.subject.slug, level.index))
    write_glb([SceneNode(mesh=built)], path)
    return path


def capture(model: str, png: str, yaw: float, margin: float, size: int) -> bool:
    """Draw a level the way the engine draws it: materials, textures and all.

    Through ``oglc-view`` rather than the bare probe, because what this row of
    the gallery answers is *what does it look like* -- and the answer to that
    involves the material as much as the geometry. The probe's flat shading is
    the other row, where the triangles are the subject.
    """
    run = subprocess.run(
        [
            sys.executable,
            '-m',
            'OpenGLContext.bin.view',
            model,
            '--capture',
            png,
            '--size',
            '%dx%d' % (size, size),
            '--yaw',
            '%g' % (yaw,),
            '--margin',
            '%g' % (margin,),
            '--no-rotate',
            '--no-cameras',
            '--no-shadows',
            '--frames',
            '6',
            '--environment',
            ENVIRONMENT,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, 'OPENGLCONTEXT_HIDDEN': '1', 'OPENGLCONTEXT_NO_VSYNC': '1'},
    )
    if run.returncode or not os.path.exists(png):
        print('   drawing %s failed: %s' % (os.path.basename(png), run.stderr[-300:]), flush=True)
        return False
    return True


def merged_level(level: Level) -> tuple:
    """One level's material groups as a single ``(positions, normals, indices)``.

    For measuring and for the triangle view, where the materials are not the
    question and one array pair is easier to hand to the probe.
    """
    import numpy as np

    positions = np.concatenate([result.attributes['POSITION'] for result in level.results])
    normals = np.concatenate([result.attributes['NORMAL'] for result in level.results])
    offset, joined = 0, []
    for result in level.results:
        joined.append(np.asarray(result.indices).reshape(-1) + offset)
        offset += len(result.attributes['POSITION'])
    return positions, normals, np.concatenate(joined).astype(np.uint32)


def serving_distance(deviation: float, radius: float) -> float:
    """Where a level's error is worth one pixel, in radii beyond the surface.

    A level's deviation is a length in model units, and a length says nothing
    about whether anyone can see it. What decides is how many pixels it covers,
    which is the distance: a camera of vertical field ``FOVY`` rendering to
    ``CELL`` pixels sees ``2 * distance * tan(FOVY / 2) / CELL`` model units per
    pixel, so the error is worth :data:`SSE_PIXELS` at

        distance = deviation * CELL / (SSE_PIXELS * 2 * tan(FOVY / 2))

    This is the rule a streaming renderer switches on, and it is what puts each
    level in this gallery where a game would have chosen it.
    """
    per_pixel = 2.0 * math.tan(math.radians(FOVY) / 2.0) / CELL
    distance = deviation / max(per_pixel * SSE_PIXELS, 1e-12)
    return distance / max(radius, 1e-12) - 1.0


def choose_distances(reduction: Reduction, probe: Any) -> None:
    """Place each level where a renderer would, and measure what it costs there.

    Placement is screen-space error -- see :func:`serving_distance`. What the
    swap actually does to the picture is then measured at that distance by the
    engine's own machinery, split into the outline that moved and the shading
    that changed, because the two want different remedies: a moved outline needs
    triangles, changed shading needs a normal map.
    """
    from OpenGLContext.meshlod.quality import pop_breakdown

    source = merged_level(reduction.levels[0])
    # A chain has to be ordered: a coarser level is never usable closer than a
    # finer one, whatever a sampled deviation happened to measure. Taking the
    # running maximum is what a renderer needs to switch on.
    furthest = 0.0
    for index, level in enumerate(reduction.levels):
        deviation = level.measured if level.measured is not None else level.error
        furthest = max(furthest, serving_distance(deviation, reduction.radius))
        level.safe_at = furthest
        if index <= 1:
            level.shown_at = NEAREST
        else:
            level.shown_at = min(FURTHEST, max(NEAREST, level.safe_at))
        if index == 0:
            level.safe_at = 0.0
            continue
        positions, normals, indices = merged_level(level)
        at = (1.0 + level.shown_at) * reduction.radius
        before = probe.render(
            source[0],
            source[1],
            source[2],
            at,
            reduction.radius,
            reduction.centre,
            rotation=reduction.subject.rotation,
        )
        after = probe.render(
            positions,
            normals,
            indices,
            at,
            reduction.radius,
            reduction.centre,
            rotation=reduction.subject.rotation,
        )
        level.outline, level.shading = pop_breakdown(before, after)
        print(
            '   level %d: one pixel past %.1f radii, drawn at %.2f; outline %.2f%%,'
            ' shading %.1f%%'
            % (index, level.safe_at, level.shown_at, 100 * level.outline, 100 * level.shading),
            flush=True,
        )


def render_subject(reduction: Reduction, probe: Any, models: str) -> None:
    """Draw every level twice: as the game would, and as triangles."""
    from OpenGLContext import capture as capturing

    subject = reduction.subject
    os.makedirs(PICTURES, exist_ok=True)
    choose_distances(reduction, probe)
    for level in reduction.levels:
        name = '%s-l%d' % (subject.slug, level.index)
        model = write_level(reduction, level, models)
        # Two textured views of the same level. Close up, so the levels can be
        # compared with each other and with the source; and at the distance a
        # renderer would have switched to this level, where how *small* it is is
        # the answer rather than an accident of framing.
        for key, where in (('shaded', NEAREST), ('served', level.shown_at)):
            png = '%s-%s.png' % (name, key)
            if capture(
                model,
                os.path.join(PICTURES, png),
                math.radians(subject.rotation),
                (1.0 + where) / FIT,
                CELL,
            ):
                level.images[key] = png

        # And the same level's triangles, close up and flat, whatever distance
        # it is used at -- which is the point of putting them side by side.
        positions, normals, indices = merged_level(level)
        close = (1.0 + NEAREST) * reduction.radius
        image = probe.render(
            positions,
            normals,
            indices,
            close,
            reduction.radius,
            reduction.centre,
            rotation=subject.rotation,
            edges=True,
        )
        capturing.save_png(os.path.join(PICTURES, name + '-edges.png'), image)
        level.images['edges'] = name + '-edges.png'
        cost = probe.frame_cost(
            positions,
            normals,
            indices,
            close,
            reduction.radius,
            reduction.centre,
            rotation=subject.rotation,
            frames=FRAMES,
        )
        level.draw_ms, level.fps = cost.median_ms, cost.fps
        print(
            '   level %d drawn: %6.3f ms  %8.0f fps' % (level.index, cost.median_ms, cost.fps),
            flush=True,
        )


def _read(path: str) -> Any:
    """A written PNG back as an ``(n, n, 3)`` array, for the contact sheet."""
    import numpy as np
    from OpenGLContext.capture import ensure_pillow

    image = ensure_pillow()
    if image is None:
        return np.zeros((CELL, CELL, 3), dtype=np.uint8)
    return np.asarray(image.open(path).convert('RGB'))


def table(reduction: Reduction) -> str:
    """The measurements for one subject, as a markdown table."""
    lines = [
        '| Triangles | Of source | `result.error` | Measured | Replay | Draw | Frame rate'
        ' | One pixel past | Outline | Shading |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for level in reduction.levels:
        share = level.triangles / max(1, reduction.source_triangles)
        lines.append(
            '| %s%s | %s | %.5f | %s | %.0f ms | %.2f ms | %s | %s | %s | %s |'
            % (
                f'{level.triangles:,}',
                ' *(%s)*' % (level.note,) if level.note else '',
                (
                    'source'
                    if level.index == 0
                    else (
                        '%.2f%%' % (100.0 * share) if share < 0.01 else '%.1f%%' % (100.0 * share)
                    )
                ),
                level.error,
                '--' if level.measured is None else '%.5f' % (level.measured,),
                level.replay_ms,
                level.draw_ms,
                '--' if not level.fps else f'{level.fps:,.0f}',
                '--' if level.index == 0 else '%.1f r' % (level.safe_at,),
                '--' if level.index == 0 else '%.2f%%' % (100 * level.outline,),
                '--' if level.index == 0 else '%.1f%%' % (100 * level.shading,),
            )
        )
    return '\n'.join(lines)


def rows(reduction: Reduction) -> str:
    """One row per level: what a player sees, and the triangles behind it."""
    out = [
        '<table>',
        '<tr><th align="left">Level</th><th align="left">Close up</th>'
        '<th align="left">Where it is used</th>'
        '<th align="left">Its triangles</th></tr>',
    ]
    for level in reduction.levels:
        if level.index == 0:
            said = 'the source, drawn where the finest level is'
        else:
            said = (
                'one pixel of error past <b>%.1f radii</b><br>drawn there;'
                ' outline %.2f%%, shading %.1f%%'
                % (level.safe_at, 100 * level.outline, 100 * level.shading)
            )
        out.append(
            '<tr><td valign="top" width="140"><b>%s</b> tri<br><sub>%s%s</sub></td>'
            '<td><img src="gallery/%s" width="250" alt="%s at %s triangles"></td>'
            '<td><img src="gallery/%s" width="250" alt="%s at %s triangles, %s radii away">'
            '</td>'
            '<td><img src="gallery/%s" width="250" alt="%s triangles of %s"></td></tr>'
            % (
                f'{level.triangles:,}',
                ('<b>%s</b><br>' % (level.note,)) if level.note else '',
                said,
                level.images.get('shaded', ''),
                reduction.subject.title,
                f'{level.triangles:,}',
                level.images.get('served', ''),
                reduction.subject.title,
                f'{level.triangles:,}',
                '%.1f' % (level.shown_at,),
                level.images.get('edges', ''),
                f'{level.triangles:,}',
                reduction.subject.title,
            )
        )
    out.append('</table>')
    return '\n'.join(out)


def page(reductions: list, described: str) -> str:
    """The whole of ``docs/GALLERY.md``."""
    out = [
        '# What a reduction looks like',
        '',
        'Every picture and every number on this page is written by'
        ' [`tools/gallery.py`](../tools/gallery.py), so they describe the reducer'
        ' as it stands rather than as it once was. Run it again after a change:',
        '',
        '```bash',
        'tools/gallery.py',
        '```',
        '',
        described,
        '',
        '## How to read it',
        '',
        'Each subject is decimated **once**. Every level below it is a prefix of'
        ' that one recording replayed, which is why the reduction is counted in'
        ' seconds and each level in milliseconds -- see'
        ' [`collapse_sequence`](API.md#collapse_sequenceattributes-indices-optionsnone---collapsesequence).',
        '',
        'The chain is the one a game would ship -- %s triangles -- rather than a'
        ' share of whatever the scan happened to arrive with. What a renderer can'
        ' afford is a count, not a proportion. The source sits above it as a'
        ' reference: not a level anything would draw, but the picture the rest'
        ' are judged against.' % (', '.join(f'{count:,}' for count in LEVELS),),
        '',
        '**Each level is drawn where a renderer would have chosen it.** The rule'
        ' is screen-space error: a level is placed at the distance where its'
        ' *measured* deviation from the source projects to **one pixel**, which'
        ' is what a streaming renderer switches on. A length in model units says'
        ' nothing about whether anyone can see it; a pixel does.',
        '',
        "Three views of each level. **Close up** is the level with the model's"
        ' own materials and textures, lit by a CC0 studio environment, at %g'
        ' radii -- the same distance for every level and for the source, so they'
        ' can be compared with each other. **Where it is used** is the same level'
        ' at the distance the rule above puts it, in the same frame: how small it'
        ' is there is the answer, not an accident of framing. **Its triangles** is'
        ' the level close up again, flat-shaded with its edges on, so what the'
        ' decimation did is visible beside what it produced.' % (NEAREST,),
        '',
        "The **outline** and **shading** figures are the engine's own measurement"
        ' of the swap where it is drawn:'
        ' [`pop_breakdown`](../../openglcontext/OpenGLContext/meshlod/quality.py)'
        " splits the share of the object's pixels that change into the part whose"
        ' outline moved and the part that merely shaded differently. The two want'
        ' different remedies -- a moved outline needs triangles, changed shading'
        ' needs a normal map baked from the fine mesh -- and one number for both'
        ' would hide which is happening.',
        '',
        "- **`result.error`** is the reducer's own figure: an area-weighted"
        ' root-mean-square distance to the planes it has been through, not a bound.',
        '- **Measured** is `certify.surface_deviation`, the sampled two-sided'
        ' Hausdorff distance from the source surface -- the number a switching'
        " distance is built on. Both are in the model's own units.",
        '- **Replay** is what `at()` cost to produce that level.',
        '- **Draw** and **frame rate** are the level rendered into a %d x %d'
        ' framebuffer with the geometry resident, timed with `glFinish` around'
        ' each frame and reported as the median of %d. No swap and no'
        ' compositor, so the number is what the triangles cost rather than what'
        ' a monitor allowed. It is one object on an idle card: read the ratios'
        ' between the rows, not the absolute rate.' % (CELL, CELL, FRAMES),
        '',
        "Distances are in radii of the model's own bounding sphere, measured"
        ' beyond its surface: %g is close enough to fill the view, %g is far'
        ' enough that the object is a smudge.' % (NEAREST, FURTHEST),
        '',
    ]
    for reduction in reductions:
        subject = reduction.subject
        out += [
            '## %s' % (subject.title,),
            '',
            subject.note,
            '',
            '| | |',
            '|---|---|',
            '| Source | %s triangles, %s vertices, %d primitive%s over %d material%s |'
            % (
                f'{reduction.source_triangles:,}',
                f'{reduction.source_vertices:,}',
                reduction.primitives,
                '' if reduction.primitives == 1 else 's',
                reduction.materials,
                '' if reduction.materials == 1 else 's',
            ),
            '| Welded to | %s points |' % (f'{reduction.welded_points:,}',),
            '| Reduced in | %.1f s, %s contractions |'
            % (reduction.reduce_s, f'{reduction.contractions:,}'),
        ]
        if reduction.floor:
            out.append(
                '| Floor | %s triangles, where no contraction is left that keeps'
                ' the surface a surface |' % (f'{reduction.floor:,}',)
            )
        out += [
            '| Credit | %s |' % (subject.credit,),
            '',
            table(reduction),
            '',
        ]
        out += [rows(reduction), '']
    out += closing(reductions)
    return '\n'.join(out) + '\n'


def where_it_stopped(reduction: Reduction) -> str:
    """How far down the ladder a subject actually got.

    Three answers. It reached the last rung; it ran out of contractions part
    way, which is the ``floor``; or it produced a level for every rung and not
    one of them was the count that rung asked for -- which is the answer a
    canopy of leaf cards gives, and reads as success unless it is said.
    """
    if reduction.floor:
        return '%s tri' % (f'{reduction.floor:,}',)
    reached = reduction.levels[-1].triangles
    if reached > LEVELS[-1] * 1.05:
        return '%s tri, where %s was asked for' % (f'{reached:,}', f'{LEVELS[-1]:,}')
    return 'reached the bottom of the chain'


def closing(reductions: list) -> list:
    """What the tables above do not say: where a chain stops, and on what box."""
    stopped = [r for r in reductions if r.floor]
    out = [
        '## Where a chain stops',
        '',
        'Some subjects above run out of ladder before they run out of rungs. A'
        ' reduction stops where no contraction is left that keeps the surface a'
        ' surface, and three properties of the *model* decide where that is.'
        ' `opengl_decimate.survey` measures all three off any mesh, before a'
        ' reduction is spent on it.',
        '',
        '**Pieces.** Every connected piece reduces on its own and each has a'
        ' floor of its own -- a closed shell cannot go below four triangles --'
        ' so a scan that arrived with the subject and two hundred crumbs spends'
        ' four triangles on each crumb however coarse a target it is given.'
        ' `drop_components_below` takes the pieces smaller than a given share of'
        " the model's diagonal, and never the largest.",
        '',
        '**Handles.** A tunnel through the surface cannot be closed at all.'
        ' Contracting an edge under the link condition preserves topology by'
        ' construction -- that is what the condition is for -- so every handle'
        ' survives to the end and costs the triangles it takes to go round it.'
        ' A scan of feathers, foliage or lace arrives with hundreds, and no'
        ' option in this package will remove one: closing a tunnel is a'
        ' different operation from contracting an edge.',
        '',
        '**Seams.** A reduction crosses the boundary of a texture chart without'
        ' tearing it -- each side keeps reading from its own chart -- but the'
        ' coordinate a seam carries slides as the merged point moves, so a model'
        ' that is mostly seam is a model whose texture drifts as it coarsens.'
        ' `lock_seams` refuses those contractions, at the price of the triangles'
        ' the seam network needs. The share of edges on a seam is what says'
        ' whether that price is worth paying.',
        '',
    ]
    out += [
        '| Subject | Pieces | Handles | Edges held by a seam | Floor |',
        '|---|---:|---:|---:|---:|',
    ]
    out += [
        '| %s | %s | %s | %.1f%% | %s |'
        % (
            r.subject.title,
            f'{r.pieces:,}',
            f'{r.handles:,}',
            100.0 * r.seam_share,
            where_it_stopped(r),
        )
        for r in reductions
    ]
    out += ['']
    if stopped:
        out += [
            'A subject that stops is not a subject the reducer gave up on: every'
            ' contraction left would have cost the model one of the three'
            ' properties above. What such an asset needs is a different'
            ' operation -- an imposter, a re-authored atlas, foliage baked to'
            ' larger cards -- rather than a lower target.',
            '',
        ]
    out += [
        '## What this box is',
        '',
        'The draw times and frame rates above are one machine. It is a fast one,'
        ' and a reader sizing a budget for players should read the *ratios*'
        ' between the rows rather than the absolute numbers: what a level costs'
        ' relative to the one above it is a property of the triangles, and it'
        ' carries across hardware. The rate does not.',
        '',
        'Where a GPU is fast enough, per-triangle cost stops being what the'
        ' frame is made of: below some count the draw is bounded by fixed'
        ' per-call work and the times flatten out, so the coarsest rungs look'
        ' free. On an integrated part, or a phone, they are not -- the curve'
        ' keeps falling, and the rungs this page shows as indistinguishable are'
        ' the difference between a frame and a stutter. Size a chain against the'
        ' slowest machine meant to draw it.',
        '',
    ]
    return out


#: The subject the README leads with, and the levels it shows.
HEADLINE = 'cliff'

START, END = '<!-- gallery:start -->', '<!-- gallery:end -->'


def readme_block(reductions: list) -> str:
    """The sample the README carries, between its markers.

    Images are named by absolute URL rather than by relative path, because the
    README is also the project's PyPI page and a relative path resolves to
    nothing there.
    """
    lead = next((r for r in reductions if r.subject.slug == HEADLINE), reductions[0])
    shown = [lead.levels[0]] + [
        lvl for lvl in lead.levels if lvl.note and lvl is not lead.levels[0]
    ]
    if len(shown) < 4:
        shown = lead.levels[:: max(1, len(lead.levels) // 4)][:4]
    lines = [
        START,
        '',
        '## What it does to a scan',
        '',
        '`%s`, %s triangles of photogrammetry, decimated **once** in %.1f s -- every'
        ' level below is a prefix of that one recording replayed.'
        % (lead.subject.source.split(':')[-1], f'{lead.source_triangles:,}', lead.reduce_s),
        '',
        '<table><tr>%s</tr></table>'
        % (
            ''.join(
                '<td align="center"><img src="%s/%s" width="190" alt="%s at %s triangles">'
                '<br><sub><b>%s</b> tri</sub></td>'
                % (
                    RAW,
                    level.images.get('shaded', ''),
                    lead.subject.title,
                    f'{level.triangles:,}',
                    f'{level.triangles:,}',
                )
                for level in shown
            ),
        ),
        '',
    ]
    lines += [
        # Named per row rather than in the heading, because a subject whose
        # atlas or topology stops it never reaches the top of the chain and a
        # column headed with that count would be saying so of every row.
        '| Subject | Source | Reduced in | Draw at source | Finest shipped | Draw there |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for reduction in reductions:
        fine = next(
            (lvl for lvl in reduction.levels if lvl.triangles <= LEVELS[0]), reduction.levels[-1]
        )
        lines.append(
            '| %s | %s tri | %.1f s | %.2f ms | %s tri | %.2f ms |'
            % (
                reduction.subject.title,
                f'{reduction.source_triangles:,}',
                reduction.reduce_s,
                reduction.levels[0].draw_ms,
                f'{fine.triangles:,}',
                fine.draw_ms,
            )
        )
    lines += [
        '',
        '[**docs/GALLERY.md**](docs/GALLERY.md) has every level of every subject, each'
        ' drawn from touching distance out to barely visible, with what it cost to make'
        ' and what it costs to draw.',
        '',
        END,
    ]
    return '\n'.join(lines)


def write_readme(reductions: list) -> None:
    """Replace the block between the markers in ``README.md``."""
    path = os.path.join(ROOT, 'README.md')
    with open(path, encoding='utf-8') as handle:
        text = handle.read()
    if START not in text or END not in text:
        print('README.md carries no %s marker; leaving it alone' % (START,))
        return
    head, rest = text.split(START, 1)
    _stale, tail = rest.split(END, 1)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(head + readme_block(reductions) + tail)
    print('wrote %s' % (path,))


def credits(reductions: list) -> str:
    """The provenance block, which the page carries because it should."""
    lines = ['Every subject is **CC0**. Credit is given because the work was given away.', '']
    for reduction in reductions:
        lines.append('- %s' % (reduction.subject.credit,))
    return '\n'.join(lines)


def describe_machine() -> str:
    """What rendered these, since a frame rate means nothing without it."""
    try:
        from OpenGLContext.testing.glcontext import describe_gl

        return str(describe_gl())
    except Exception:
        return 'no GL context'


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--only', action='append', default=[], help='slug of a subject to do, repeatable'
    )
    parser.add_argument(
        '--no-render', action='store_true', help='measure the reduction, draw nothing'
    )
    parser.add_argument(
        '--no-certify',
        action='store_true',
        help='skip the measured deviation, which is the slow part',
    )
    parser.add_argument('--cell', type=int, default=CELL)
    options = parser.parse_args(argv)

    wanted = [s for s in SUBJECTS if not options.only or s.slug in options.only]
    if not wanted:
        raise SystemExit('no subject matches %r' % (options.only,))

    os.environ.setdefault('OPENGLCONTEXT_HIDDEN', '1')
    os.environ.setdefault('OPENGLCONTEXT_NO_VSYNC', '1')
    os.environ.setdefault('OPENGLCONTEXT_BACKEND', 'glfw')

    reductions = [reduce_subject(subject, not options.no_certify) for subject in wanted]
    # One high-water mark for the run, not one per subject: `ru_maxrss` only
    # ever rises, so a per-subject figure would be the largest subject's number
    # printed against every subject after it.
    biggest = max(reductions, key=lambda r: r.source_triangles)
    machine = (
        'Measured on %s. Reducing every subject took %.1f GB of resident memory at'
        ' its highest, against %s triangles of %s -- the loaded model, its'
        ' textures and the recorded sequence together.'
        % (
            describe_machine(),
            max(r.peak_mb for r in reductions) / 1024.0,
            f'{biggest.source_triangles:,}',
            biggest.subject.title.lower(),
        )
        if not options.no_render
        else ''
    )

    if not options.no_render:
        from OpenGLContext.meshlod.quality import LODProbe
        from OpenGLContext.testing.glcontext import hidden_window

        # Each level is written here as a glb so `oglc-view` can draw it with
        # its own materials. They are a step on the way to the pictures, not
        # something the repository wants, so they live in the cache.
        models = os.path.join(CACHE, 'levels')
        os.makedirs(models, exist_ok=True)
        with (
            hidden_window('opengl_decimate gallery', size=(64, 64), profile='core'),
            LODProbe(size=options.cell) as probe,
        ):
            for reduction in reductions:
                print('== drawing %s ==' % (reduction.subject.title,), flush=True)
                render_subject(reduction, probe, models)
    else:
        for reduction in reductions:
            for level in reduction.levels:
                level.images.setdefault('edges', '')

    whole = len(reductions) == len(SUBJECTS)
    os.makedirs(DOCS, exist_ok=True)
    if whole:
        described = '\n\n'.join(part for part in (machine, credits(reductions)) if part)
        with open(os.path.join(DOCS, 'GALLERY.md'), 'w', encoding='utf-8') as handle:
            handle.write(page(reductions, described))
        print('wrote %s' % (os.path.join(DOCS, 'GALLERY.md'),))
        if not options.no_render:
            write_readme(reductions)
    else:
        # A partial run has pictures for some subjects and nothing for the
        # rest, and a page written from it would quietly drop the others.
        print(
            'did %s of %d subjects; the pictures are written and GALLERY.md is'
            ' left as it was. Run without --only to rewrite the page.'
            % (', '.join(r.subject.slug for r in reductions), len(SUBJECTS))
        )

    with open(os.path.join(PICTURES, 'measurements.json'), 'w', encoding='utf-8') as handle:
        json.dump(
            [
                {
                    'subject': r.subject.slug,
                    'source_triangles': r.source_triangles,
                    'reduce_s': r.reduce_s,
                    'levels': [
                        {
                            'triangles': level.triangles,
                            'error': level.error,
                            'measured': level.measured,
                            'replay_ms': level.replay_ms,
                            'draw_ms': level.draw_ms,
                            'fps': level.fps,
                        }
                        for level in r.levels
                    ],
                }
                for r in reductions
            ],
            handle,
            indent=2,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
