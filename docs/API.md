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

## `collapse_sequence(attributes, indices, options) -> CollapseSequence`

Run the same reduction but record every contraction and stop at nothing. The
targets in `options` are not used here; they are what `CollapseSequence.at` is
asked for afterwards. This is the call to make once when a model is loaded.

## `SimplifyOptions`

A frozen dataclass. At least one target is required; where more than one is
given the reduction stops at whichever is reached first.

### Targets

| Field | Type | Default | Meaning |
|---|---|---|---|
| `target_ratio` | `float \| None` | `None` | share of input triangles to keep, in `(0, 1]` |
| `target_count` | `int \| None` | `None` | triangles to stop at |
| `target_error` | `float \| None` | `None` | deviation to stop at, a length in model units |

### The error metric

| Field | Type | Default | Meaning |
|---|---|---|---|
| `metric` | `str` | `'quadric'` | `'quadric'` or `'probabilistic'` |
| `position_noise` | `float` | `0.0` | probabilistic only: how far the surface's points are trusted, as a fraction of the bounding-box diagonal |
| `normal_noise` | `float` | `0.0` | probabilistic only: how far its normals are trusted, in units of the unit normal |
| `boundary_weight` | `float` | `1.0` | how hard an open surface's edge is held in place. Zero lets a border wander; raising it holds the outline at the cost of keeping triangles along it |

Selecting `'probabilistic'` without naming either noise uses 0.001 of the
diagonal and 0.001 respectively.

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
| `recompute_normals` | `bool` | `False` | replace `NORMAL` with the normals of the surface that is left |
| `certify` | `bool` | `False` | measure the deviation of the result and put it in `measured_error` |
| `certify_samples` | `int` | `4000` | points taken from each surface when certifying |

## `SimplifyResult`

| Field | Type | Meaning |
|---|---|---|
| `attributes` | `dict[str, ndarray]` | the reduced mesh. `POSITION` is float32; every other attribute keeps the dtype it arrived with |
| `indices` | `ndarray` | flat triangle list, `uint32` |
| `error` | `float` | the reduction's own estimate of the deviation, in model units |
| `measured_error` | `float \| None` | the *measured* deviation, where `certify` was on |
| `vertex_map` | `ndarray` | for every vertex of the input, the output vertex it became or was merged into; `-1` where its surface is gone |
| `collapses` | `int` | contractions applied |
| `triangle_count` | `int` | triangles in the result |

Output vertices are numbered by first use in the index stream, so the caller's
vertex order is not preserved. `vertex_map` is the correspondence between the
two — and the correspondence a geomorph needs to lerp one level toward the next.

## `CollapseSequence`

`len(sequence)` is how many contractions were recorded.

### `at(target_count=None, target_ratio=None, target_error=None) -> SimplifyResult`

The mesh at the first point in the sequence meeting every target given. With no
target, the whole sequence. Applying a prefix is a handful of array operations,
so this costs the same whatever it is asked for.

### `triangles_after(steps) -> int`

Triangles left once `steps` contractions are applied.

### `steps_for(target_count=None, target_ratio=None, target_error=None) -> int`

The fewest contractions satisfying the targets, without building the mesh.

### `state(steps) -> (positions, roots, faces, corners)`

The mesh after `steps` contractions, in the package's internal form: welded point
positions, the point each original point has become, and the live faces with the
input vertex each corner came from.

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

### `build(positions, indices, tolerance=0.0) -> Topology`

The welded surface. `Topology.classify(lock_boundary=False, locked=None)` returns
a `VertexClass` per point: `MANIFOLD`, `BORDER` or `LOCKED`.

### `weld_positions(positions, tolerance=0.0) -> (points, vertex_point)`

Merge vertices occupying the same point. Points are numbered by first
appearance, so a mesh with no coincident vertices welds to itself.

## `opengl_decimate.quadrics`

`plane_quadric`, `triangle_quadrics`, `scatter`, `accumulate`, `evaluate` and
`minimize` operate on `(n, 10)` arrays of packed symmetric 4x4s. See
[ALGORITHM.md](ALGORITHM.md) for the form and
[the module docstring](../src/opengl_decimate/quadrics.py) for the coefficient
order.

## `DecimateError`

Raised where the input does not describe a triangle mesh, or where the options
do not describe a reduction. A subclass of `ValueError`.
