# API

Every public entry point, what it takes and what it hands back. Arrays are
NumPy; attribute names are glTF's.

## `simplify(attributes, indices, options) -> SimplifyResult`

Reduce a mesh to the target `options` names.

- `attributes` — a mapping of glTF semantic to array. `POSITION` is required and
  must be `(v, 3)`; every other attribute must have `v` rows. Anything the
  package does not recognise is carried through unchanged.
- `indices` — a flat triangle list, any integer dtype, length a multiple of three.
- `options` — a [`SimplifyOptions`](#simplifyoptions).

## `collapse_sequence(attributes, indices, options=None) -> CollapseSequence`

Run the same reduction but record every contraction and stop at nothing. The
targets in `options` are not used here — none of them, `target_error` included —
so `options` may be left out entirely where the defaults will do. They are what
`CollapseSequence.at` is asked for afterwards. This is the call to make once when
a model is loaded.

## `SimplifyOptions`

A frozen dataclass. At least one target is required; where more than one is
given the reduction stops at whichever is reached first.

### Targets

| Field | Type | Default | Meaning |
|---|---|---|---|
| `target_ratio` | `float \| None` | `None` | share of input triangles to keep, in `(0, 1]` |
| `target_count` | `int \| None` | `None` | triangles to stop at |
| `target_error` | `float \| None` | `None` | deviation to stop at, a length in model units. This is the quadric's own figure — an area-weighted root-mean-square distance to the planes, not a bound on the surface. It runs around half the deviation `certify` measures, so a level of detail's switching distance should be built on `measured_error` rather than on this |

### The error metric

| Field | Type | Default | Meaning |
|---|---|---|---|
| `metric` | `str` | `'quadric'` | `'quadric'` or `'probabilistic'` |
| `position_noise` | `float \| None` | `None` | probabilistic only: how far the surface's points are trusted, as a fraction of the bounding-box diagonal. It enters the quadric as a constant added to every cost, so it raises the reported deviation and does not change which contraction is chosen |
| `normal_noise` | `float \| None` | `None` | probabilistic only: how far its normals are trusted, in units of the unit normal. This is the term that conditions the 3x3 solve, so it is the one that changes the reduction; at zero the probabilistic metric is the classical one with an offset |
| `boundary_weight` | `float` | `1.0` | how hard an open surface's edge is held in place. Zero lets a border wander; raising it holds the outline at the cost of keeping triangles along it |

Each defaults on its own: `None` means 0.001 of the diagonal and 0.001
respectively, so naming one leaves the other at its default rather than at
zero.

### What may move

| Field | Type | Default | Meaning |
|---|---|---|---|
| `lock_boundary` | `bool` | `False` | hold the border exactly |
| `locked` | `Sequence[int] \| None` | `None` | further points to hold, indexed into the welded points — which are the caller's own vertex indices unless vertices were welded |
| `max_normal_flip` | `float` | `90.0` | degrees a face's normal may turn before the contraction is refused |
| `min_triangle_quality` | `float` | `0.0` | shape floor: 0 admits any shape, 1 only equilateral |
| `placement` | `str` | `'optimal'` | `'optimal'` minimises the summed quadric; `'endpoint'` keeps whichever end of the edge costs less, so every output point is an input point |

### How it runs

| Field | Type | Default | Meaning |
|---|---|---|---|
| `weld_tolerance` | `float` | `0.0` | merge input vertices within this distance before reducing |
| `schedule` | `str` | `'heap'` | `'heap'` or `'multiple-choice'` |
| `candidates` | `int` | `8` | edges `multiple-choice` draws per step |
| `seed` | `int` | `0` | fixes the sampling, so a run repeats exactly |
| `recompute_normals` | `bool` | `False` | replace `NORMAL` with the normals of the surface that is left, adding the attribute where the input carried none. Accumulated per point, so vertices split by a texture seam share a normal |
| `certify` | `bool` | `False` | measure the deviation of the result and put it in `measured_error` |
| `certify_samples` | `int` | `4000` | points taken from each surface when certifying |

## `SimplifyResult`

| Field | Type | Meaning |
|---|---|---|
| `attributes` | `dict[str, ndarray]` | the reduced mesh. Every attribute keeps the dtype it arrived with, `POSITION` included — a `POSITION` that was not a float dtype comes back float32 |
| `indices` | `ndarray` | flat triangle list, `uint32` |
| `error` | `float` | the reduction's own estimate of the deviation, in model units: an area-weighted root-mean-square distance to the planes, not a bound |
| `measured_error` | `float \| None` | the *measured* deviation, where `certify` was on |
| `vertex_map` | `ndarray` | for every vertex of the input, the output vertex it became or was merged into; `-1` where its surface is gone |
| `collapses` | `int` | contractions applied |
| `input_triangles` | `int` | triangles the caller handed in, which is what `target_ratio` is a share of |
| `welded_away` | `int` | input triangles welding dropped before the reduction began, because their corners landed on the same point |
| `triangle_count` | `int` | triangles in the result |

Output vertices are numbered by first use in the index stream, so the caller's
vertex order is not preserved. `vertex_map` is the correspondence between the
two — and the correspondence a geomorph needs to lerp one level toward the next.

## `CollapseSequence`

`len(sequence)` is how many contractions were recorded.

### `at(target_count=None, target_ratio=None, target_error=None) -> SimplifyResult`

The mesh at the first point in the sequence meeting every target given. With no
target, the whole sequence. Applying a prefix is a handful of array operations,
so a target costs a replay rather than a reduction, and costs the same however
far along the sequence it is. What it does scale with is the mesh handed back,
which has to be assembled.

### `triangles_after(steps) -> int`

Triangles left once `steps` contractions are applied.

### `steps_for(target_count=None, target_ratio=None, target_error=None) -> int`

The fewest contractions satisfying the targets, without building the mesh.

### `state(steps) -> (positions, roots, faces, corners)`

The mesh after `steps` contractions, in the package's internal form: welded point
positions, the point each original point has become, and the live faces with the
input vertex each corner reads its attributes from — which is a vertex measured
where the corner now sits, not the vertex it started as.

## `opengl_decimate.certify`

### `surface_deviation(reference_positions, reference_indices, candidate_positions, candidate_indices, samples=4000, seed=0) -> Deviation`

How far two surfaces are from each other, measured in both directions.
`Deviation.max` is the sampled Hausdorff distance — the number an error bound has
to cover — and `Deviation.rms` is the root-mean-square over the samples. Raising
`samples` tightens the worst case; the root-mean-square settles quickly.

### `distance_to_mesh(points, positions, indices) -> ndarray`

Exact distance from each point to the nearest triangle: the closest point of a
triangle may be inside it, along an edge or at a corner, and all three are found.
A mesh with no triangles is infinitely far from everywhere.

### `sample_surface(positions, indices, count, seed=0) -> ndarray`

`count` points spread over a mesh in proportion to triangle area.

## `opengl_decimate.topology`

### `build(positions, indices, tolerance=0.0, drop_below=0.0, carried=None) -> Topology`

The welded surface. `Topology.classify(lock_boundary=False, locked=None)` returns
a `VertexClass` per point: `MANIFOLD`, `BORDER` or `LOCKED`.

`drop_below` removes whole connected components smaller than that share of the
model's bounding-box diagonal; the largest is never dropped. `carried` is what
the vertices hold besides their positions, read only to fill `Topology.copies` —
how many different sets of values each point is drawn with, which is where the
surface's seams are. Without it every point counts as drawn once.

`Topology.positions` are held relative to `Topology.origin` — see
`local_origin` below. `CollapseSequence.state` adds it back, so a caller of
`simplify` or `at()` never meets it.

### `weld_positions(positions, tolerance=0.0) -> (points, vertex_point)`

Merge vertices occupying the same point. Points are numbered by first
appearance, so a mesh with no coincident vertices welds to itself.

A positive `tolerance` merges anything within that distance, which is what a
scan needs. It should be smaller than the typical spacing between points: the
comparison is against every point in the twenty-seven cells of side `tolerance`
around each one, so a tolerance that sweeps a large cluster into one cell costs
the square of that cluster's size.

### `local_origin(points) -> ndarray`

The point quadrics are accumulated about. The model's own bounding-box centre on
each axis where the model sits at least twice its own half-extent from the world
origin, and zero elsewhere — so a georeferenced model is measured where float64
still has the digits, and a model near the origin is not moved at all. See
[ALGORITHM.md](ALGORITHM.md) §2.

## `opengl_decimate.quadrics`

`plane_quadric`, `triangle_quadrics`, `scatter`, `accumulate`, `evaluate` and
`minimize` operate on `(n, 10)` arrays of packed symmetric 4x4s. See
[ALGORITHM.md](ALGORITHM.md) for the form and
[the module docstring](../src/opengl_decimate/quadrics.py) for the coefficient
order.

## `DecimateError`

Raised where the input does not describe a triangle mesh, or where the options
do not describe a reduction. A subclass of `ValueError`.

What that covers:

- `POSITION` missing, not `(n, 3)`, or carrying a value that is not finite; an
  attribute with a different number of rows, or that is not one row per vertex;
  an index array whose length is not a multiple of three, or that names a vertex
  the mesh does not have.
- Every field of `SimplifyOptions` outside its documented range — no target at
  all, a ratio outside `(0, 1]`, a negative count, error, noise, tolerance or
  boundary weight, a `max_normal_flip` outside `[0, 180]`, a
  `min_triangle_quality` outside `[0, 1]`, an unknown `metric`, `schedule` or
  `placement`, fewer than one candidate, fewer than one certify sample, or a
  negative `locked` index.
- A `locked` index past the last welded point — raised when the mesh is
  classified, since that is where the point count is known. The count in the
  message is of *welded points*, which is fewer than the caller's vertices
  wherever vertices were welded.

## `opengl_decimate.corners`

Which vertex a corner reads its attributes from, once its point has moved.
`copies_per_point(vertex_point, corners, attributes, point_count)` counts how
many different sets of carried values each point is drawn with — one almost
everywhere, two where a seam runs. `corner_moves(...)` records, for each
contraction, which copies it lets go and which they hand over to, and `follow`
closes a prefix of those moves by pointer-jumping. `CollapseSequence` does all
three; a caller of `simplify` or `at()` never meets them.

## `opengl_decimate.spatial`

The uniform grid both the tolerance weld and `certify` look nearby with.
`CellGrid(cells)` buckets items by integer cell coordinate and `members(cells)`
hands back every item in the cells named, in blocks. `shell(radius)` is the
offsets exactly that many cells out, and `shell_size(radius)` how many there
would be. Used by this package rather than offered as a general spatial index.
