# Changelog

Notable changes to `opengl_decimate`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/); `0.x` makes no
compatibility promise.

## [0.2.0a1] - unreleased

The first published release. `0.1.0a1` was the version string of the
development builds and was never published; the changes listed under Changed
are against those builds, for anyone who installed one from source.

### Added

- `simplify(attributes, indices, options)`: reduce a mesh held as glTF-shaped
  NumPy arrays to a triangle count, a share of its triangles or an error budget,
  and hand it back in the same shape.
- `collapse_sequence(attributes, indices, options=None)`: record the whole
  reduction once; `CollapseSequence.at()` reaches any target by replaying a
  prefix of it.
- Classical and probabilistic quadric metrics (`metric`, `position_noise`,
  `normal_noise`), optimal and endpoint placement, the `heap` and
  `multiple-choice` schedules, a border held by constraint planes
  (`boundary_weight`, `lock_boundary`), named vertices held (`locked`), a
  normal-flip limit and a triangle-shape floor.
- Welding: exact by default, and within `weld_tolerance` for scans.
- `drop_components_below`: remove connected pieces smaller than a share of the
  model's bounding-box diagonal before reducing. `SimplifyResult.dropped_away`
  and `welded_away` count what was removed before the reduction began.
- Attributes carried with their corners, so a UV seam stays a seam and a hard
  edge stays hard; `lock_seams` holds each seam on its own line.
- `recompute_normals` and `crease_angle`: normals of the reduced surface,
  accumulated per smoothing group.
- `certify` and `opengl_decimate.certify`: the measured two-sided deviation of
  a result from its input (`surface_deviation`), and the nearest triangle to a
  point (`nearest_triangle`, `distance_to_mesh`).
- `survey(attributes, indices, options=None)`: the pieces, handles and atlas
  charts of a mesh, and the floors they put under any reduction of it.
- A compiled reducer (`_reduce_native`, Cython) for the `heap` schedule and the
  corner handover, making the same contractions in the same order as the NumPy
  loop. Wheels carry it; `opengl_decimate.native.ACCELERATED` says whether it is
  in use, and `OPENGL_DECIMATE_NO_ACCEL=1` switches it off.

### Changed

- `locked` names vertices of the arrays handed in, not welded points.
- The modules defining `simplify` and `survey` are `opengl_decimate.reduction`
  and `opengl_decimate.floors`. The package-level names are unchanged.
- An open piece keeps its last triangle, as `Survey.floor` reports; it was
  reduced to nothing.
- The `multiple-choice` schedule stops when no drawn pair can be contracted,
  rather than after twenty failed draws per edge.
- `certify` measures against the input less the pieces `drop_components_below`
  removed.
- `survey` checks its input as `simplify` does, and takes the options the
  reduction will use.
- A quadric's minimum is used where the system's condition number is below
  1e10 and the minimum lies within an edge length of the edge's midpoint.
- Index arrays that are not of an integer type, and meshes too large for the
  compiled reducer's 32-bit indices, are refused with `DecimateError`.
