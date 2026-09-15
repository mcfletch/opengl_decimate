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
import os
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

#: Where the camera sits, in radii of the model's own bounding sphere, measured
#: from its centre. ``1.0`` is touching that sphere, which is as close as a view
#: of the whole object gets; past that the range runs out to a smudge, which is
#: where a renderer stops being able to tell one level from another.
DISTANCES = (1.02, 1.5, 2.5, 5.0, 14.0, 40.0)

#: What a game's *ordinary* finest level wants. A scan hands over a million
#: triangles and a renderer with a world to draw will not spend them on one
#: prop at arm's length, let alone on forty of them.
#:
#: The source above it is still worth keeping, and worth looking at here: it is
#: the level for standing right next to the thing and looking into the
#: crevices, which is exactly the case the coarser rungs cannot serve and the
#: case a player puts to a model perhaps once.
HIGH = 10_000

#: The coarsest mesh worth drawing. Past this an imposter -- a billboard of the
#: thing -- is cheaper than any triangles, so the chain stops here.
FLOOR = 120

#: Rungs between :data:`HIGH` and :data:`FLOOR`, and between the source and
#: :data:`HIGH` where the source is far above it. Geometric either way, so each
#: step is the same proportion of the one above and two subjects' sheets read
#: side by side.
RUNGS = 6
APPROACH = 2

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
    #: Degrees about Y, chosen so the model faces the camera.
    rotation: float = 0.0
    note: str = ''


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
            'surface only because the primitives are merged and welded first. It is '
            'also 664 separate shells: the birds, and hundreds of specks the '
            'photogrammetry left behind. Each shell has a floor of its own, which is '
            'why this chain stops where it does rather than at the coarsest rung.'
        ),
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
    note: str = ''
    images: dict = field(default_factory=dict)


@dataclass
class Reduction:
    """A whole subject: its source mesh, its levels and what they cost."""

    subject: Subject
    source_triangles: int
    source_vertices: int
    primitives: int
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


def load(path: str) -> tuple[dict, Any, int]:
    """One model as a single merged mesh, with how many primitives it was."""
    from OpenGLContext.loaders.assets import merged_mesh, shapes
    from OpenGLContext.loaders.gltf import load_gltf

    scene = load_gltf(path, max_resource_bytes=None)
    merged = merged_mesh(scene.group)
    if merged is None:
        raise SystemExit('%s holds no triangles' % (path,))
    attributes, indices = merged
    return attributes, indices, sum(1 for _ in shapes(scene.group))


def _between(top: int, bottom: int, rungs: int) -> list:
    """``rungs`` counts descending from just below ``top`` to ``bottom``."""
    if top <= bottom or rungs < 1:
        return []
    step = (bottom / top) ** (1.0 / rungs)
    counts, count = [], float(top)
    for _ in range(rungs):
        count *= step
        counts.append(max(bottom, round(count)))
    return counts


def ladder(triangles: int) -> list:
    """``(triangles, note)`` for each level, the source first.

    The span a renderer spends most of its time in is :data:`HIGH` -- what an
    ordinary finest level can afford -- down to :data:`FLOOR`, where an imposter
    takes over, and most of the rungs walk it. The source sits above that as the
    level to swap in when somebody walks up to the thing, and a couple of rungs
    cover the descent from it.
    """
    levels = [(triangles, 'walk up to it')]
    if triangles > HIGH:
        levels += [(count, '') for count in _between(triangles, HIGH, APPROACH)[:-1]]
        levels.append((HIGH, "a game's ordinary finest"))
    top = min(triangles, HIGH)
    levels += [(count, '') for count in _between(top, FLOOR, RUNGS)]
    if levels[-1][0] <= FLOOR:
        levels[-1] = (levels[-1][0], 'past here, an imposter')
    return levels


def reduce_subject(subject: Subject, certify: bool) -> Reduction:
    """Decimate one subject once, and read every level off the recording."""
    from OpenGLContext.meshlod.chain import bounding_sphere

    from opengl_decimate import SimplifyOptions, collapse_sequence, topology
    from opengl_decimate import certify as certification

    print('== %s ==' % (subject.title,), flush=True)
    path = fetch(subject.source)
    start = time.perf_counter()
    attributes, indices, primitives = load(path)
    load_s = time.perf_counter() - start

    source_triangles = len(indices) // 3
    welded = topology.build(attributes['POSITION'], indices)
    print(
        '   %s triangles in %d primitive%s, welded to %s points'
        % (
            f'{source_triangles:,}',
            primitives,
            '' if primitives == 1 else 's',
            f'{welded.vertex_count:,}',
        ),
        flush=True,
    )

    # One reduction. Every rung below is a prefix replay of this recording,
    # which is the whole point of `collapse_sequence`.
    start = time.perf_counter()
    sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=1.0))
    reduce_s = time.perf_counter() - start
    print(
        '   reduced in %.1f s, %s contractions recorded' % (reduce_s, f'{len(sequence):,}'),
        flush=True,
    )

    centre, radius = bounding_sphere(attributes['POSITION'])
    out = Reduction(
        subject=subject,
        source_triangles=source_triangles,
        source_vertices=len(attributes['POSITION']),
        primitives=primitives,
        welded_points=welded.vertex_count,
        load_s=load_s,
        reduce_s=reduce_s,
        contractions=len(sequence),
        peak_mb=peak_rss_mb(),
        radius=float(radius),
        centre=centre,
    )

    original = np.asarray(attributes['POSITION'], dtype='d')
    for index, (count, note) in enumerate(ladder(source_triangles)):
        start = time.perf_counter()
        result = sequence.at(target_count=count)
        replay_ms = (time.perf_counter() - start) * 1000.0
        if out.levels and result.triangle_count >= out.levels[-1].triangles:
            # The reduction has run out of legal contractions, and every rung
            # below this one would be the same mesh again. Where a subject stops
            # and why is worth a row; nine copies of it are not.
            out.floor = out.levels[-1].triangles
            print('   floor reached at %s triangles' % (f'{out.floor:,}',), flush=True)
            break
        measured = None
        if certify and index:
            measured = certification.surface_deviation(
                original,
                indices,
                result.attributes['POSITION'],
                result.indices,
                samples=4000,
            ).max
        out.levels.append(
            Level(
                index=index,
                triangles=result.triangle_count,
                vertices=len(result.attributes['POSITION']),
                error=result.error,
                measured=measured,
                replay_ms=replay_ms,
                note=note,
            )
        )
        out.levels[-1].result = result  # type: ignore[attr-defined]
        print(
            '   level %d: %9s triangles  error %.5f  replay %6.1f ms'
            % (index, f'{result.triangle_count:,}', result.error, replay_ms),
            flush=True,
        )
    out.peak_mb = peak_rss_mb()
    return out


def render_subject(reduction: Reduction, probe: Any) -> None:
    """Draw every level at every distance, and time each one."""
    from OpenGLContext import capture

    subject = reduction.subject
    os.makedirs(PICTURES, exist_ok=True)
    for level in reduction.levels:
        result = level.result
        positions = result.attributes['POSITION']
        normals = result.attributes.get('NORMAL')
        close = DISTANCES[0] * reduction.radius
        for label, edges in (('shaded', False), ('edges', True)):
            image = probe.render(
                positions,
                normals,
                result.indices,
                close,
                reduction.radius,
                reduction.centre,
                rotation=subject.rotation,
                edges=edges,
            )
            name = '%s-l%d-%s.png' % (subject.slug, level.index, label)
            capture.save_png(os.path.join(PICTURES, name), image)
            level.images[label] = name
        level.images['distance'] = []
        for step in DISTANCES:
            image = probe.render(
                positions,
                normals,
                result.indices,
                step * reduction.radius,
                reduction.radius,
                reduction.centre,
                rotation=subject.rotation,
            )
            level.images['distance'].append(image)
        cost = probe.frame_cost(
            positions,
            normals,
            result.indices,
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


def sheets(reduction: Reduction) -> dict:
    """Lay the renders out as two contact sheets; return their file names."""
    from OpenGLContext import contactsheet
    from OpenGLContext.capture import ensure_pillow

    subject = reduction.subject
    written = {}
    if ensure_pillow() is None:
        return written

    rows = [
        ('%s tri' % f'{level.triangles:,}', level.images['distance']) for level in reduction.levels
    ]
    columns = ['%g r' % step for step in DISTANCES]
    path = os.path.join(PICTURES, '%s-distance.png' % (subject.slug,))
    contactsheet.tile(
        path, '%s -- every level, close to barely visible' % (subject.title,), rows, columns
    )
    written['distance'] = os.path.basename(path)
    return written


def table(reduction: Reduction) -> str:
    """The measurements for one subject, as a markdown table."""
    lines = [
        '| Level | Triangles | Of source | `result.error` | Measured | Replay | Draw | Frame rate | |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for level in reduction.levels:
        share = level.triangles / max(1, reduction.source_triangles)
        lines.append(
            '| %d | %s | %s | %.5f | %s | %.0f ms | %.2f ms | %s | %s |'
            % (
                level.index,
                f'{level.triangles:,}',
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
                '**%s**' % (level.note,) if level.note else '',
            )
        )
    return '\n'.join(lines)


def _strip(reduction: Reduction, kind: str, width: int) -> str:
    """One row of a subject's levels, each captioned with its triangle count."""
    cells = ''.join(
        '<td align="center"><img src="gallery/%s" width="%d" alt="%s at %s triangles">'
        '<br><sub><b>%s</b> tri%s</sub></td>'
        % (
            level.images.get(kind, ''),
            width,
            reduction.subject.title,
            f'{level.triangles:,}',
            f'{level.triangles:,}',
            '<br>%s' % (level.note,) if level.note else '',
        )
        for level in reduction.levels
    )
    return '<table><tr>%s</tr></table>' % (cells,)


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
        'The chain each subject is cut into is the one a renderer wants rather'
        ' than an even split of the source. The **source** stays at the top: it'
        ' is the level to swap in when somebody walks up to the thing and looks'
        ' into the crevices, and nothing coarser can serve that. Below it,'
        ' **%s triangles** is what an ordinary finest level can afford when'
        ' there is a world to draw as well, and the rungs walk from there down'
        ' to **%s**, past which an imposter costs less than any mesh.'
        % (f'{HIGH:,}', f'{FLOOR:,}'),
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
        "The distance sheets put the camera from %g to %g radii of the model's"
        ' own bounding sphere away from its centre -- %g being as close as a view'
        ' of the whole object gets. The rightmost column is the object at the'
        ' size it covers when a renderer is deciding whether anyone would'
        ' notice, and the point of the sheet is how far left you have to read'
        ' before the rows stop agreeing.' % (DISTANCES[0], DISTANCES[-1], DISTANCES[0]),
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
            '| Source | %s triangles, %s vertices, %d primitive%s |'
            % (
                f'{reduction.source_triangles:,}',
                f'{reduction.source_vertices:,}',
                reduction.primitives,
                '' if reduction.primitives == 1 else 's',
            ),
            '| Welded to | %s points |' % (f'{reduction.welded_points:,}',),
            '| Reduced in | %.1f s, %s contractions |'
            % (reduction.reduce_s, f'{reduction.contractions:,}'),
            '| Peak process memory | %.0f MB, the loaded model included |' % (reduction.peak_mb,),
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
        out += ['### What it looks like', '', _strip(reduction, 'shaded', 240), '']
        out += [
            '<details><summary>and the triangles producing it</summary>',
            '',
            _strip(reduction, 'edges', 240),
            '',
            '</details>',
            '',
        ]
        if reduction.sheet.get('distance'):
            out += [
                '### Close to barely visible',
                '',
                '![%s at every level and distance](gallery/%s)'
                % (subject.title, reduction.sheet['distance']),
                '',
            ]
    return '\n'.join(out) + '\n'


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
        '| Subject | Source | Reduced in | Draw at source | Draw at %s |' % (f'{HIGH:,}',),
        '|---|---:|---:|---:|---:|',
    ]
    for reduction in reductions:
        fine = next(
            (lvl for lvl in reduction.levels if lvl.triangles <= HIGH), reduction.levels[-1]
        )
        lines.append(
            '| %s | %s tri | %.1f s | %.2f ms | %.2f ms |'
            % (
                reduction.subject.title,
                f'{reduction.source_triangles:,}',
                reduction.reduce_s,
                reduction.levels[0].draw_ms,
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
    machine = 'Measured on %s.' % (describe_machine(),) if not options.no_render else ''

    if not options.no_render:
        from OpenGLContext.meshlod.quality import LODProbe
        from OpenGLContext.testing.glcontext import hidden_window

        with (
            hidden_window('opengl_decimate gallery', size=(64, 64), profile='core'),
            LODProbe(size=options.cell) as probe,
        ):
            for reduction in reductions:
                print('== drawing %s ==' % (reduction.subject.title,), flush=True)
                render_subject(reduction, probe)
                reduction.sheet = sheets(reduction)  # type: ignore[attr-defined]
    else:
        for reduction in reductions:
            for level in reduction.levels:
                level.images.setdefault('edges', '')
            reduction.sheet = {}  # type: ignore[attr-defined]

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
