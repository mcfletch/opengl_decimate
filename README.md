# opengl_decimate — mesh decimation over glTF-shaped NumPy arrays

Reduce a triangle mesh to fewer triangles, keeping the shape and the attributes
it carries. A mesh arrives as a mapping of glTF attribute semantics to NumPy
arrays plus a triangle index array, and leaves the same way, so nothing stands
between a loader, a generator, a decimator and a vertex buffer.

```python
import numpy as np
from opengl_decimate import simplify, SimplifyOptions

result = simplify(
    {'POSITION': positions, 'NORMAL': normals, 'TEXCOORD_0': uvs},
    indices,
    SimplifyOptions(target_ratio=0.25),
)

result.attributes['POSITION']   # (v, 3) float32
result.indices                  # (t * 3,) uint32
result.error                    # the quadrics' own estimate of the deviation
result.vertex_map               # input vertex -> output vertex
```

NumPy is the only dependency.

<!-- gallery:start -->

## What it does to a scan

`coastal_cliff_04`, 1,537,926 triangles of photogrammetry, decimated **once** in 11.3 s -- every level below is a prefix of that one recording replayed.

<table><tr><td align="center"><img src="https://raw.githubusercontent.com/mcfletch/opengl_decimate/main/docs/gallery/cliff-l0-shaded.png" width="190" alt="Coastal cliff at 1,537,926 triangles"><br><sub><b>1,537,926</b> tri</sub></td><td align="center"><img src="https://raw.githubusercontent.com/mcfletch/opengl_decimate/main/docs/gallery/cliff-l1-shaded.png" width="190" alt="Coastal cliff at 32,000 triangles"><br><sub><b>32,000</b> tri</sub></td><td align="center"><img src="https://raw.githubusercontent.com/mcfletch/opengl_decimate/main/docs/gallery/cliff-l2-shaded.png" width="190" alt="Coastal cliff at 8,000 triangles"><br><sub><b>8,000</b> tri</sub></td><td align="center"><img src="https://raw.githubusercontent.com/mcfletch/opengl_decimate/main/docs/gallery/cliff-l3-shaded.png" width="190" alt="Coastal cliff at 4,000 triangles"><br><sub><b>4,000</b> tri</sub></td></tr></table>

| Subject | Source | Reduced in | Draw at source | Finest shipped | Draw there |
|---|---:|---:|---:|---:|---:|
| Coastal cliff | 1,537,926 tri | 11.3 s | 0.19 ms | 32,000 tri | 0.04 ms |
| Lekking ruffs | 547,647 tri | 6.3 s | 0.22 ms | 32,000 tri | 0.10 ms |
| Coastal land rocks | 1,291,146 tri | 9.6 s | 0.17 ms | 32,000 tri | 0.04 ms |
| Marble bust | 17,456 tri | 0.1 s | 0.04 ms | 17,456 tri | 0.04 ms |

[**docs/GALLERY.md**](docs/GALLERY.md) has every level of every subject, each drawn from touching distance out to barely visible, with what it cost to make and what it costs to draw.

<!-- gallery:end -->

## What it does

**Quadric error metrics.** Each point carries a quadric accumulated from the
planes of the triangles around it, weighted by their area. Merging two points
costs the summed quadric evaluated where it is smallest, and that smallest place
is one 3x3 solve. The classical metric (Garland & Heckbert) and the
**probabilistic** one (Trettner & Kobbelt) share the storage and the solve: the
second reads each plane as a noisy sample and minimises the *expected* squared
distance, which makes a single plane solvable and a reconstructed surface behave.

**Rules that keep the result a surface.** Before any contraction: the link
condition, so an edge never ends up with three triangles on it; a duplicate-face
check, so a small closed shape cannot fold onto itself; a normal-flip test, so a
fan is never turned inside out; and an optional triangle-shape floor. Points are
classified once as manifold, border or locked, and a border point moves only
along its border, so the outline of an open patch stays where it was.

**Attributes are carried, never invented.** Every value in the output is a value
that was in the input, measured at the place the surface now is: a contraction
hands the corners it merges to the end of the edge the merged point came to rest
nearest, so a texture coordinate and a normal travel with the position they
describe. Two corners at the same point with different normals or texture
coordinates stay two vertices, so a UV seam is still a seam and a hard edge is
still hard. A corner takes the copy nearest in attribute space and the end drawn
at more texture coordinates keeps them, so neither side of a seam is ever left
reading the chart next door; `lock_seams` additionally holds the seam's *line*,
where the atlas is fragmented enough for that to be worth its triangles.
`recompute_normals` replaces `NORMAL` with the normals of the surface that is
left, which is what a level a game ships wants: a carried normal is the one
measured at its own vertex and is the *more* accurate of the two, but its error
is uncorrelated between neighbours and shading shows the gradient, so coarse
levels facet. Recomputing accumulates per smoothing group rather than per point,
so `crease_angle` keeps the edges the model was built hard across and a cube does
not come back round.

**A recorded reduction, replayed to any target.** `collapse_sequence` runs the
whole reduction once and records it; `at()` then reaches any triangle count by
replaying a prefix — four array operations, not another decimation. A target
costs a replay rather than a reduction, and costs the same however far along the
sequence it is, which is what an editor's target slider needs.

```python
from opengl_decimate import collapse_sequence

sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
coarse = sequence.at(target_count=5000)      # immediate
coarser = sequence.at(target_count=500)      # also immediate
within = sequence.at(target_error=0.01)      # or by deviation
```

**A measured error, not only an estimate.** `result.error` is what the quadrics
predicted: an area-weighted root-mean-square distance to the planes, which is the
right thing to steer a reduction by and is not a bound on anything. `certify=True`,
or `opengl_decimate.certify.surface_deviation`, samples both surfaces in both
directions and measures the worst place either is from the other — which is the
number a level of detail's error bound should be built on, and runs about twice
`result.error`. One direction alone cannot see a hole.

**How far a model will go, before spending a reduction on it.** Some meshes
cannot be decimated, and the reason is always a property of the mesh: a canopy
of separate leaf cards is as many pieces as it has cards and every piece keeps a
triangle; lace has a handle through every hole, and no contraction closes one;
an atlas of thousands of small charts is mostly seam. `survey` measures all
three and puts a floor on what any target can reach.

```python
from opengl_decimate import survey

report = survey(attributes, indices)
report.pieces, report.handles, report.seam_share
report.floor                                 # no target goes below this
if report.reducible < 0.5:
    ...  # this model wants a different operation, not a lower target
```

## Targets

At least one, and the reduction stops at whichever is reached first.

| Option | Meaning |
|---|---|
| `target_ratio` | share of the input triangles to keep, in `(0, 1]`. A share of what the caller handed in, whatever welding merged first |
| `target_count` | triangles to stop at |
| `target_error` | deviation to stop at, a length in model units — the quadrics' estimate, not the measured deviation |

## Options

| Option | Default | What it does |
|---|---|---|
| `metric` | `'quadric'` | `'probabilistic'` reads the surface as noisy samples |
| `position_noise` | `0.001` | probabilistic: how far points are trusted, as a fraction of the bounding-box diagonal. Raises the reported error; does not change what gets contracted |
| `normal_noise` | `0.001` | probabilistic: how far normals are trusted. The term that conditions the solve, so this is the one that changes the reduction |
| `boundary_weight` | `1.0` | how hard an open surface's edge is held |
| `lock_boundary` | `False` | hold the border exactly |
| `locked` | `None` | further points to hold, indexed into the **welded points** — the caller's own vertex indices unless vertices were welded |
| `max_normal_flip` | `90.0` | degrees a face's normal may turn |
| `min_triangle_quality` | `0.0` | shape floor, 0 (any) to 1 (equilateral only) |
| `placement` | `'optimal'` | `'endpoint'` keeps every point exactly where it was |
| `weld_tolerance` | `0.0` | merge input vertices within this distance first |
| `schedule` | `'heap'` | `'multiple-choice'` samples instead of ordering |
| `candidates` | `8` | how many edges `multiple-choice` draws per step |
| `seed` | `0` | fixes the sampling, so a run repeats exactly |
| `recompute_normals` | `False` | recompute `NORMAL` for the surface that is left, adding it where the input had none |
| `certify` | `False` | measure the deviation into `measured_error` |
| `certify_samples` | `4000` | points taken from each surface when certifying |

## Two schedules

`heap` takes the cheapest candidate anywhere on the mesh, from a lazily-updated
priority queue. It is the better quality, and it is sequential by construction:
every contraction changes the price of its neighbours, so the next choice depends
on the last.

`multiple-choice` takes the cheapest of a few candidates drawn at random. There
is no global ordering to maintain, every step costs the same as every other, and
the quality loss is small. Giving up the global ordering is what makes a
reduction divisible, so this is the schedule a parallel or GPU implementation is
built on.

## Input it will take

`POSITION` is required; every other attribute must have the same number of rows.
Vertices sharing a position are welded so the reduction sees a surface rather
than a vertex list, and a triangle left with a repeated corner by that welding is
dropped. Non-manifold edges, bowties and isolated points are locked rather than
refused, so a mesh with some bad places is reduced everywhere else.

`weld_tolerance` merges vertices that are merely close, which is what a scan
needs — the same corner reconstructed twice differs in the last several bits. The
exact weld is one vectorised sort; a tolerance weld groups into cells and joins
across the twenty-seven cells a point can have a partner in, and costs more.
Pick a tolerance smaller than the typical spacing between points: the work is the
number of pairs sharing a cell, so a tolerance that sweeps a cluster into one
cell costs the square of that cluster's size.

A model far from the origin is handled where it is. Quadrics are accumulated
about the model's own bounding-box centre, so a georeferenced mesh at 6.4e6
reduces to exactly what the same mesh at the origin does — and `POSITION` comes
back in the float dtype it arrived in, since a `float32` ulp out there is half a
metre.

## Limits

- Attributes follow their corner and are not part of the error metric, so a
  texture can slide slightly where a point moves a long way. `placement='endpoint'`
  removes that entirely, at some cost in geometric accuracy.
- Only the `heap` schedule has the compiled contraction loop. `multiple-choice`
  runs the NumPy implementation whatever is installed, which is around thirty
  times slower per face — worth knowing, since the schedule is otherwise the
  scalable one of the two.
- Where no compiler was available at install time the NumPy contraction loop is
  used for both schedules. It is the same reduction and the same answer, at a
  speed that suits a model rather than a scan. A wheel carries the compiled one.
- **An open surface cannot be reduced past its own boundaries.** Each border loop
  has a floor of three vertices, so a mesh of many small open shells stops well
  above any target. A tree trunk with 246 branch stubs stops at 738 triangles
  whatever it is asked for, because that is three vertices per loop. Joining
  points that are not edges is what gets below it, and is not implemented.
- A reduction run to exhaustion will take a closed surface down to nothing. Give
  it a target, or lock what has to stay.

## Documentation

- [docs/GALLERY.md](docs/GALLERY.md) — five CC0 scans taken down through a level
  chain, each level drawn from touching distance out to barely visible, with what
  it cost to make and what it costs to draw. Regenerated by
  [`tools/gallery.py`](tools/gallery.py).
- [docs/API.md](docs/API.md) — every entry point, with its units and defaults.
- [docs/ALGORITHM.md](docs/ALGORITHM.md) — what each step does and why.

## Licence

MIT. See [LICENSE](LICENSE).
