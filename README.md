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
result.error                    # deviation from the input, in model units
result.vertex_map               # input vertex -> output vertex
```

NumPy is the only dependency.

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
that was in the input, taken from the corner it belonged to. Two corners at the
same point with different normals or texture coordinates stay two vertices, so a
UV seam is still a seam and a hard edge is still hard. `recompute_normals`
replaces `NORMAL` with the normals of the surface that is left, where carrying
the input's is not what you want.

**A recorded reduction, replayed to any target.** `collapse_sequence` runs the
whole reduction once and records it; `at()` then reaches any triangle count by
replaying a prefix — four array operations, not another decimation. Asking for
fifty targets in a row costs about what asking for one costs, which is what an
editor's target slider needs.

```python
from opengl_decimate import collapse_sequence

sequence = collapse_sequence(attributes, indices, SimplifyOptions(target_ratio=0.5))
coarse = sequence.at(target_count=5000)      # immediate
coarser = sequence.at(target_count=500)      # also immediate
within = sequence.at(target_error=0.01)      # or by deviation
```

**A measured error, not only an estimate.** `result.error` is what the quadrics
predicted. Quadrics measure distance to *planes*, and planes extend past the
triangles that made them, so the number drifts optimistic. `certify=True`, or
`opengl_decimate.certify.surface_deviation`, samples both surfaces in both
directions and measures what actually happened — which is the number a level of
detail's error bound should be built on. One direction alone cannot see a hole.

## Targets

At least one, and the reduction stops at whichever is reached first.

| Option | Meaning |
|---|---|
| `target_ratio` | share of the input triangles to keep, in `(0, 1]` |
| `target_count` | triangles to stop at |
| `target_error` | deviation to stop at, a length in model units |

## Options

| Option | Default | What it does |
|---|---|---|
| `metric` | `'quadric'` | `'probabilistic'` reads the surface as noisy samples |
| `position_noise` | `0.0` | probabilistic: how far points are trusted, as a fraction of the bounding-box diagonal |
| `normal_noise` | `0.0` | probabilistic: how far normals are trusted |
| `boundary_weight` | `1.0` | how hard an open surface's edge is held |
| `lock_boundary` | `False` | hold the border exactly |
| `locked` | `None` | further points to hold, by index |
| `max_normal_flip` | `90.0` | degrees a face's normal may turn |
| `min_triangle_quality` | `0.0` | shape floor, 0 (any) to 1 (equilateral only) |
| `placement` | `'optimal'` | `'endpoint'` keeps every point exactly where it was |
| `weld_tolerance` | `0.0` | merge input vertices within this distance first |
| `schedule` | `'heap'` | `'multiple-choice'` samples instead of ordering |
| `candidates` | `8` | how many edges `multiple-choice` draws per step |
| `seed` | `0` | fixes the sampling, so a run repeats exactly |
| `recompute_normals` | `False` | recompute `NORMAL` for the surface that is left |
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

## Limits

- Attributes follow their corner and are not part of the error metric, so a
  texture can slide slightly where a point moves a long way. `placement='endpoint'`
  removes that entirely, at some cost in geometric accuracy.
- Everything runs in Python and NumPy. The batch work — accumulating quadrics,
  pricing candidates, replaying a prefix — is whole-array; the contraction loop
  is not, so a reduction of a few hundred thousand triangles is minutes rather
  than seconds.
- **An open surface cannot be reduced past its own boundaries.** Each border loop
  has a floor of three vertices, so a mesh of many small open shells stops well
  above any target. A tree trunk with 246 branch stubs stops at 738 triangles
  whatever it is asked for, because that is three vertices per loop. Joining
  points that are not edges is what gets below it, and is not implemented.
- A reduction run to exhaustion will take a closed surface down to nothing. Give
  it a target, or lock what has to stay.

## Documentation

- [docs/API.md](docs/API.md) — every entry point, with its units and defaults.
- [docs/ALGORITHM.md](docs/ALGORITHM.md) — what each step does and why.

## Licence

MIT. See [LICENSE](LICENSE).
