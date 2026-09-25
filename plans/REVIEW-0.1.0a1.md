# Review: `opengl_decimate` 0.1.0a1

A full read of the package against its own documentation, plus measurement of
every claim that could be measured. Every finding below was reproduced on this
machine unless the entry says otherwise; each one names the file and line it is
about, the observation that establishes it, and a concrete fix.

Environment for every measurement in this document: Linux x86-64, CPython
3.12.3, NumPy 2.5.3, the compiled reducer built and active
(`native.ACCELERATED is True`). The suite is green (161 passed), `ruff check`,
`ruff format --check` and `mypy src/opengl_decimate` all pass, and the coverage
gate reports 100%.

## Summary table

Severity: **B** blocks a release · **M** wrong or misleading behaviour a user
will meet · **m** smaller defect, mostly documentation or hygiene.

Status, filled in as the work proceeds: **todo** · **fixed** · **partly** (the
finding held but not the fix it proposed — the entry says what was done
instead) · **rejected** (the finding does not hold; the entry says why).

| # | Sev | Status | Area | Issue | Where |
|---|---|---|---|---|---|
| [1](#1) | B | fixed | Correctness | Quadrics are built in absolute coordinates, so a model far from the origin loses the metric: at ECEF scale the error floor is 0.147 in model units and a 10% target returns 23% of the triangles | `quadrics.py:103-114` |
| [2](#2) | B | fixed | Correctness | The compiled reducer silently refuses every contraction at a vertex of valence > 1024 and disagrees with the NumPy path, which the project's contract says it cannot | `_reduce_native.pyx:46-47,341-368` |
| [3](#3) | B | fixed | Portability | `cdef long[::1]` bound to `np.int64` arrays cannot work on Windows (LLP64), where wheels are published | `_reduce_native.pyx:715-719` |
| [4](#4) | B | fixed | Packaging | The published sdist omits `tests/shapes.py`; six of seven test modules fail collection from it | no `MANIFEST.in` |
| [5](#5) | B | fixed | Correctness | `collapse_sequence` honours `target_error` although it documents that it ignores every target; with one set it can record zero contractions | `simplify.py:71-80,303-304` |
| [6](#6) | M | fixed | Correctness | `position_noise` alone is a provable no-op: `metric='probabilistic'` then produces bit-identical output to `'quadric'` | `options.py:132-136`, `quadrics.py:105-114` |
| [7](#7) | M | partly | Correctness | `target_error` under-delivers by about 2x against the deviation that actually occurs | `simplify.py:258` |
| [8](#8) | M | fixed | Correctness | `multiple-choice` with `target_error` ends the whole reduction on one unlucky draw: 2.4x too many triangles, ±30% with the seed | `simplify.py:368-369` |
| [9](#9) | M | fixed | Correctness | `recompute_normals` accumulates per output vertex, so a UV seam becomes a shading seam; it also invents `NORMAL` where none was given | `sequence.py:227-228,248-258` |
| [10](#10) | M | fixed | Correctness | `target_ratio` is a share of the post-weld face count, not of the input triangles as documented | `simplify.py:181`, `sequence.py:106-109` |
| [11](#11) | M | fixed | Error handling | Six of eighteen option fields are validated; `locked`, `certify_samples`, `max_normal_flip`, `min_triangle_quality`, `weld_tolerance` and `boundary_weight` are not, and non-finite positions are accepted and propagated | `options.py:103-123`, `topology.py:401-419` |
| [12](#12) | M | fixed | Performance | `certify` is a brute-force point-against-every-triangle scan: 200x-470x the cost of the reduction it measures, against a documented "about what the reduction did" | `certify.py:64-95` |
| [13](#13) | M | fixed | Performance | `weld_tolerance` is quadratic in the occupancy of one cell — 40 s for 8000 clustered points — in exactly the case it is documented for | `topology.py:93-113` |
| [14](#14) | M | partly | Testing | The 100% coverage gate measures 887 Python statements and none of the 820-line compiled reducer that every wheel runs by default | `tox.ini:60-66` |
| [15](#15) | M | fixed | Testing | The test matrix is `ubuntu-latest` only, while the release builds and publishes Windows and macOS wheels | `.github/workflows/test.yml:21` |
| [16](#16) | m | fixed | Performance | `multiple-choice` has no compiled path: 12.9k faces/s against 300k/s for the heap | `simplify.py:105-111` |
| [17](#17) | m | fixed | Performance | The compiled loop never releases the GIL; a watcher thread runs at 10% of normal throughput for the duration | `_reduce_native.pyx:620-688` |
| [18](#18) | m | fixed | Testing | No memory-safety gate over hand-managed `malloc`/`realloc`; the `realloc` failure paths leak the block they were growing | `_reduce_native.pyx:62-141,556-568` |
| [19](#19) | m | fixed | Testing | The differential tests top out at 1280 faces and never reach the scratch ceiling, `weld_tolerance`, `locked`, or a non-manifold input under the compiled path | `tests/test_native.py:34-114` |
| [20](#20) | m | fixed | Testing | Two wall-clock assertions in the suite | `tests/test_sequence.py:139`, `tests/test_topology.py:171` |
| [21](#21) | m | fixed | Testing | `test_it_still_gets_the_same_answer` checks `distance_to_mesh` against the primitive it is built from | `tests/test_certify.py:195-214` |
| [22](#22) | m | fixed | Testing | The declared `numpy>=1.24` floor is never exercised | `pyproject.toml:47` |
| [23](#23) | m | fixed | Documentation | The package docstring says `result.error` is the measured deviation; it is the estimate | `__init__.py:15` |
| [24](#24) | m | fixed | Documentation | "asking fifty targets costs about what asking for one costs" (32x-34x) and "`at()` costs the same whether ninety per cent or two per cent" (40x); the two statements also contradict each other | `README.md:52-54`, `sequence.py:16-18` |
| [25](#25) | m | fixed | Documentation | Three different accounts of how the heap detects a stale entry, none matching the compiled code | `docs/ALGORITHM.md` §5, `_reduce_native.pyx:24-29` |
| [26](#26) | m | fixed | Documentation | "Above zero, `A` is positive definite" — only the normal variance reaches `A` | `docs/ALGORITHM.md` §2 |
| [27](#27) | m | fixed | Documentation | `DecimateError` is documented as covering options that do not describe a reduction; several do not raise it | `docs/API.md` |
| [28](#28) | m | fixed | API surface | `INTEGER_ATTRIBUTES` and `MutableAttributeMap` are exported and documented with a rule that nothing implements | `types.py:33,41` |
| [29](#29) | m | fixed | Documentation | The README options table gives the noise defaults as `0.0`, hiding the 0.001 fallback that API.md documents | `README.md:87-88` |
| [30](#30) | m | fixed | Documentation | The README does not say that `locked` indexes welded points rather than the caller's vertices | `README.md:91` |
| [31](#31) | m | fixed | Documentation | The union-find docstring's reason for omitting path compression does not hold: roots are not minima and lookups are not one hop | `topology.py:46-68` |
| [32](#32) | m | fixed | Documentation | Found while working the list: the README's *Limits* still says the contraction loop is Python and that a few hundred thousand triangles takes minutes | `README.md:134-137` |

## What holds up

Recorded first because several of the findings below are about documentation
around machinery that is sound, and the distinction matters.

- **The surface invariants hold.** Across 112 configurations — seven shapes
  including a bowtie, a non-manifold edge, a hard-edged cube and two disjoint
  shells, crossed with both schedules, four ratios and both placements — the
  output contained no degenerate triangle, no duplicate face, no edge carrying
  more than two faces, no non-finite position and no unused vertex.
- **The two reduction paths agree.** On six meshes the suite does not use
  (bumped and near-planar grids, a noisy sphere, a mesh with a deliberate
  non-manifold face, the probabilistic metric) at three ratios each, the
  compiled and NumPy reductions produced identical index arrays and positions
  to 1e-9. Finding [2](#2) is the one construction where they part.
- **The probabilistic quadric derivation is right.** `test_matches_a_direct_expectation`
  checks it against a 400,000-draw Monte-Carlo estimate, which is the right kind
  of test for that claim.
- **Classification scales.** 2.87M faces classified in 1.75 s, and the
  adjacency sets are genuinely not built until something walks the surface.
- **The compiled reduction is quick.** 266k-322k faces/s, 1.6M faces to 10% in
  6.1 s at 974 MB peak RSS.
- **The gates that exist are real gates.** `ruff`, `ruff format --check`, `mypy`
  with `disallow_untyped_defs`, a `py.typed`, a tox matrix that asserts each
  env actually got the reducer state it asked for, and a release workflow that
  refuses to guess when PyPI does not answer.

---

## Blockers

### 1. Quadrics in absolute coordinates lose the metric far from the origin {#1}

`plane_quadric` (`quadrics.py:103-114`) builds the constant term as
`c = (n·p)²` from the point's absolute coordinates. For a model at Earth-centred
coordinates — `p ~ 6.4e6`, which is what 3D Tiles and any ECEF-referenced world
uses — `c ~ 4e13`, while the squared distance the quadric exists to report is
around `1e-6`. That is nineteen decimal digits of dynamic range inside a type
that carries sixteen, and the cancellation eats the answer.

The same unit sphere (`icosphere(4)`, 5120 triangles), translated along X, asked
for `target_ratio=0.1`:

| translation | triangles returned | `result.error` | measured deviation |
|---|---|---|---|
| 0 | 512 | 0.00583 | 0.01347 |
| 1e2 | 512 | 0.00574 | 0.01112 |
| 1e4 | 512 | 0.00544 | 0.01208 |
| 1e6 | 512 | 0.01434 | 0.05220 |
| 6.4e6 | **1200** | **0.60909** | 0.30299 |

At ECEF scale the reduction does not reach its target at all — 1200 triangles
where 512 were asked for — and the reported error exceeds the deviation that
actually occurred by 2x in the other direction, so it is not usable as a bound
either.

This is not the `float32` output storage. Accumulating in float64 throughout and
asking each vertex's own quadric to score its own position — which must be zero —
gives:

| translation | worst self-cost (should be 0) | as a length |
|---|---|---|
| 0 | 8.7e-18 | 2.5e-08 |
| 1e4 | 9.3e-10 | 2.6e-04 |
| 1e6 | 9.5e-06 | 2.4e-02 |
| 6.4e6 | 3.7e-04 | **1.5e-01** |

So at ECEF scale the metric's own noise floor is 0.147 model units, which is
larger than most level-of-detail error budgets before any decimation has
happened.

**Fix.** Subtract a local origin — the bounding-box centre is enough — before
accumulating, and add it back when a placement is read out. That is one
subtraction per point and one addition per placement, it is exact, and it moves
the whole computation into the range where `float64` has the digits. The origin
belongs on the `Topology` or the engine so that `CollapseSequence` replay uses
the same one. `SimplifyResult.attributes['POSITION']` being forced to `float32`
(`sequence.py:221`) is a second, independent loss at these coordinates — a
`float32` ulp at 6.4e6 is 0.5 — and is worth either following the input dtype or
saying plainly in the docs that a far-from-origin model must be translated by the
caller first.

### 2. The compiled reducer gives up at valence 1024, and then disagrees with the NumPy path {#2}

`SCRATCH = 1024` (`_reduce_native.pyx:46-47`) sizes the fixed neighbourhood
buffers. `_ring` (`:341-368`), `_affected` (`:404-436`) and `_faces_on_edge`
(`:322-339`) return `-1` when a neighbourhood exceeds it, and every caller in
`run` responds with `continue` — the candidate is passed over silently, with no
warning, no counter and no diagnostic. The NumPy path has no such ceiling.

Reproduction: an open fan of *n* triangles around one apex with every rim vertex
in `locked`, so the only contraction available is apex into rim. `target_count=1`:

| apex valence | compiled | NumPy |
|---|---|---|
| 1000 | 998 | 998 |
| 1023 | 1021 | 1021 |
| 1024 | 1022 | 1022 |
| **1025** | **1025** | **1023** |
| 1100 | 1100 | 1098 |
| 2000 | 2000 | 1998 |

The step is exactly at the constant. Past it the compiled path returns the mesh
unreduced and says nothing.

The exposure is narrower than the constant suggests — valence does not grow
during an ordinary reduction (peak valence measured at 7-9 while taking a 19,602-face
grid and a 5120-face sphere down to nothing), so this needs a high-valence vertex
*in the input* whose neighbours cannot themselves be reduced. But that is a real
shape: a fan-triangulated n-gon, a lathe pole, a CAD hub — and the "neighbours
cannot move" half is precisely the `locked` cluster-boundary workflow that
`options.py:64-65` says the option exists for.

This also contradicts the project's stated contract. `tox.ini:2-5` says the two
reducers "are required to produce the same contractions in the same order for the
same mesh", `native.py:5-7` says "the same algorithm, the same order, the same
result", and no test approaches the constant.

**Fix.** The scratch buffers are already `malloc`ed members of `Reducer`
(`:290-295`); give `_ring`, `_affected` and `_faces_on_edge` a `realloc` when
they run out, and the ceiling disappears. If a bound is wanted anyway, it should
raise rather than return a quietly different mesh. Separately, `_contract`
(`:581-584`) relies on `moving_count < SCRATCH` being guaranteed by the earlier
`_affected` check and documents that in a comment; since the consequence of the
comment being wrong is silent face-list corruption, that invariant deserves an
assertion rather than a sentence.

### 3. The compiled reducer cannot run on Windows {#3}

*Reasoned from the generated C, not reproduced here — this container is Linux.
The mechanism is standard and the consequence is release-blocking, so it is
listed as one.*

`_log` (`_reduce_native.pyx:709-730`) allocates `np.int64` arrays and binds them
to `cdef long[::1]` memoryviews:

```
cdef cnp.ndarray dying = np.empty(self.log_size, dtype=np.int64)
...
cdef long[::1] out_dying = dying
```

Cython emits a type descriptor carrying `sizeof(long)`
(`_reduce_native.c:3867`: `__Pyx_TypeInfo_long = { "long", NULL, sizeof(long), ... }`)
and checks it against the buffer's itemsize at acquisition. On Win64, which is
LLP64, `sizeof(long)` is 4 while `np.int64` is 8, so the binding raises
`ValueError: Buffer dtype mismatch, expected 'long' but got 'long long'`.
`_log` is called unconditionally at the end of `run`, so this is every reduction,
not a corner.

Either outcome is a release problem: the `wheels · windows-latest` job's own
`test-command` fails and blocks the publish, or — if the wheel test is ever
relaxed — every `simplify()` on Windows raises. `long` is also used for
`Heap.serial` and `Reducer.removed_at`, where 32 bits is merely tight rather
than wrong.

**Fix.** Use `cnp.int64_t` (or `long long`) for every declaration that has to
match a NumPy `int64`, and for the `removed_at`/`serial` pools. Then add
`windows-latest` and `macos-latest` to the test matrix, per finding [15](#15),
so this class of thing is caught before the release job.

### 4. The published sdist cannot run its own test suite {#4}

There is no `MANIFEST.in`. Setuptools' default sdist template picks up
`tests/test*.py` but not `tests/shapes.py`, which every test module except
`test_quadrics.py` imports by name. Building the sdist and running its tests the
way a distribution packager would:

```
$ python -m build --sdist
$ tar xzf dist/opengl_decimate-0.1.0a1.tar.gz && cd opengl_decimate-0.1.0a1
$ python -m pytest -q tests -o addopts=
E   ModuleNotFoundError: No module named 'shapes'
ERROR tests/test_certify.py
ERROR tests/test_collapse.py
ERROR tests/test_native.py
ERROR tests/test_sequence.py
ERROR tests/test_simplify.py
ERROR tests/test_topology.py
6 errors in 0.27s
```

`tox.ini` uses `package = sdist` but runs `pytest -o testpaths=tests` against the
*checkout's* tests, and cibuildwheel's `test-command` uses `{project}/tests`,
which is also the checkout — so neither gate sees this. `tox.ini` and `docs/` are
missing from the sdist too, so `tox` cannot be run from it either.

**Fix.** A `MANIFEST.in` with `recursive-include tests *.py`, `include tox.ini`
and `recursive-include docs *.md`, plus a CI step that unpacks the built sdist
and runs the suite from inside it rather than from the checkout beside it.

### 5. `collapse_sequence` honours `target_error` while documenting that it ignores every target {#5}

`collapse_sequence` says "Record the whole reduction, ignoring the targets in
`options`" (`simplify.py:78-79`), and `docs/API.md` repeats it: "The targets in
`options` are not used here." `_reduce` sets `limit = 0` for the exhaust case
(`:104`) but leaves `self.error_limit = options.target_error` (`:143`), and both
loops break on it — `simplify.py:303-304` for NumPy, `_reduce_native.pyx:647-648`
for the compiled path.

`icosphere(3)`, 1280 triangles:

| options passed | contractions recorded | floor reached |
|---|---|---|
| `target_ratio=0.9` | 638 | 4 triangles |
| `target_count=1000` | 638 | 4 triangles |
| `target_error=0.002` | **0** | **1280 triangles** |

With `target_error=0.002` every later ask is silently unsatisfiable:

```
sequence.at(target_count=1000) -> 1280 triangles
sequence.at(target_count=400)  -> 1280 triangles
sequence.at(target_count=20)   -> 1280 triangles
```

The API makes this easy to hit: `SimplifyOptions.__post_init__` requires at least
one target even for a call whose whole contract is that targets do not apply, so
a caller has to supply one, and `target_error` is a natural thing to reach for.

**Fix.** Set `error_limit = None` when `exhaust` is true, so the docstring
becomes true. Better still, give `collapse_sequence` a signature that does not
demand a target — either accept `options=None` or split the reduction-shape
options (metric, noise, locks, placement, weld, schedule) from the stopping
targets, which are two different kinds of thing already.

---

## Correctness and behaviour

### 6. `position_noise` alone is a no-op, so `metric='probabilistic'` can silently be the classical metric {#6}

Two mechanisms compound.

`SimplifyOptions.noise` (`options.py:132-136`):

```python
if not self.position_noise and not self.normal_noise:
    return DEFAULT_POSITION_NOISE * diagonal, DEFAULT_NORMAL_NOISE
return self.position_noise * diagonal, self.normal_noise
```

Naming either noise suppresses the default for the *other* one:

```
SimplifyOptions(metric='probabilistic')                     -> (0.001, 0.001)
SimplifyOptions(metric='probabilistic', position_noise=0.01) -> (0.01,  0.0  )
SimplifyOptions(metric='probabilistic', normal_noise=0.01)   -> (0.0,   0.01 )
```

And only the normal variance reaches `A`. In `plane_quadric` (`:105-114`), `var_n`
adds `var_n * I` to the matrix and `var_n * p` to the vector; `var_p` adds a
scalar to `c` and nothing else. So with `normal_noise` at zero the matrix is
exactly the classical rank-one `n nᵀ`:

```
position_noise=0.0  normal_noise=0.0  -> single plane solvable: False
position_noise=0.01 normal_noise=0.0  -> single plane solvable: False
position_noise=0.0  normal_noise=0.01 -> single plane solvable: True
position_noise=0.01 normal_noise=0.01 -> single plane solvable: True
```

The `var_p` term is provably inert on the mesh produced, not merely small: it
adds a constant `var_p` to every one of a pair's four candidate costs (so the
`argmin` is unchanged) and `deviation = sqrt(cost/weight + var_p)` is monotone in
`cost/weight` (so the heap order is unchanged). Measured on a noisy sphere:

| options | identical to `metric='quadric'`? |
|---|---|
| `metric='quadric'` | — |
| `metric='probabilistic', position_noise=0.01` | **yes, bit-for-bit** |
| `metric='probabilistic', normal_noise=0.01, position_noise=0.01` | no |
| `metric='probabilistic'` (defaults) | no |

A caller reading the README's `position_noise` row — "probabilistic: how far
points are trusted" — and setting exactly that gets the classical reduction. So
does `tests/test_simplify.py:308-319` (`test_it_survives_a_noisy_surface`),
which passes `metric='probabilistic', position_noise=0.01` and therefore
validates the classical metric while naming the probabilistic one.

**Fix.** Default each noise independently — `position_noise if position_noise
else DEFAULT_POSITION_NOISE * diagonal`, and the same for the normal — so naming
one never zeroes the other. Say in the docs for both fields that `normal_noise`
is the term that conditions the solve and `position_noise` shifts the reported
cost. Change the test to name `normal_noise` so it exercises what it claims.

### 7. `target_error` under-delivers by about 2x {#7}

`docs/ALGORITHM.md` §3 says the deviation "is a length in model units rather than
an area-weighted square, which is what makes `target_error` mean something a
caller can picture". §8 then says it "drifts optimistic". The second is right and
the size of the drift is consistent. `icosphere(4)`, measured against
`certify.surface_deviation` at 4000 samples:

| asked for | triangles | `result.error` | measured max deviation | ratio to the budget |
|---|---|---|---|---|
| ≤ 0.005 | 592 | 0.00483 | 0.00905 | 1.8x |
| ≤ 0.01 | 272 | 0.00971 | 0.01962 | 2.0x |
| ≤ 0.02 | 134 | 0.01965 | 0.03925 | 2.0x |
| ≤ 0.05 | 50 | 0.04790 | 0.11162 | 2.2x |

The same under-report shows on ratio targets (1.25x-2.35x across
`target_ratio` 0.5 down to 0.02).

Two causes stack. Planes extend past the triangles that made them, which the docs
name. And `try_contract` (`simplify.py:283-284`) accumulates
`weights[surviving] += weights[dying]` while the faces those areas came from are
being removed, so the divisor in `deviation = sqrt(cost / weight)`
(`simplify.py:258`) keeps growing against a surface that is shrinking — which
makes the reported length fall relative to the truth as the reduction proceeds.

It is a consistent factor, not noise, so it is fixable by being stated. But a
caller computing a level-of-detail switching distance from `target_error` will be
wrong by 2x, and the document that says to trust it and the document that says
not to are the same document.

**Fix.** At minimum, say the factor in `docs/API.md` next to `target_error` and
in the README's target table, and point at `certify` as the number to build a
bound on — the package already has the right measurement, it is just not what
`target_error` is checked against. Better: recompute the weight from the live
incident area rather than accumulating it, which removes the second cause
outright and costs one `bincount` over the survivor's ring.

### 8. `multiple-choice` with `target_error` stops on one unlucky draw {#8}

`run_multiple_choice` (`simplify.py:365-373`) walks the sampled candidates in
price order and, on the first one over budget, **returns** from the whole
reduction:

```python
for rank in np.argsort(deviation, kind='stable'):
    if not np.isfinite(deviation[rank]):
        break
    if self.error_limit is not None and deviation[rank] > self.error_limit:
        return
```

With `candidates=8` drawn from thousands of edges, the cheapest of eight exceeds
the budget long before the cheapest on the mesh does. `icosphere(3)`,
`target_error=0.02`:

| seed | triangles left |
|---|---|
| 0 | 338 |
| 1 | 284 |
| 2 | 314 |
| 3 | 278 |
| 4 | 282 |
| 5 | 318 |
| 6 | 304 |
| 7 | 362 |
| *heap, same budget* | **140** |

So the sampling schedule leaves 2-2.6x more triangles than the budget allows, and
where it stops moves ±30% with the seed. `test_the_sampling_schedule_honours_an_error_budget`
(`tests/test_simplify.py:337-349`) only asserts that a tight budget keeps more
triangles than a loose one, which stays true throughout.

**Fix.** `continue` rather than `return` — count it as a failed draw against the
existing `failures`/`budget` machinery (`:352,374-377`), which already exists to
decide when nothing is left to do. That turns the budget into a filter on
candidates rather than a global stop.

### 9. `recompute_normals` turns an attribute seam into a shading seam {#9}

`_emit` splits output vertices by *point plus attributes* (`sequence.py:209-216`)
and only then calls `_surface_normals` over the already-split vertex array
(`:227-228`). `_surface_normals` accumulates each face's normal onto its three
output vertices (`:254`), so a vertex that was split for a texture reason
receives only the faces on its own side of the split.

A sphere with a UV seam that has no geometric meaning at all
(`TEXCOORD_0[:,0] = (x > 0)`), reduced to `target_ratio=0.3` with
`recompute_normals=True`, worst recomputed normal against the true sphere normal:

| input | output vertices | worst angular error |
|---|---|---|
| POSITION only | 194 | 3.7° |
| POSITION + the UV seam | 205 | **11.3°** |

The seam is invisible in the geometry and visible in the shading.

Separately, `recompute_normals=True` on a mesh carrying no `NORMAL` *adds* one:

```
input attributes : ['POSITION']
output attributes: ['NORMAL', 'POSITION']
```

The option is documented as "replace `NORMAL` with the normals of the surface
that is left", and a caller who passes it while iterating `result.attributes`
into vertex-buffer bindings gets a semantic they did not declare.

**Fix.** Accumulate the face normals per *point* — `out_point` is already in
hand at `sequence.py:218` — and gather to the output vertices, so vertices at one
position share one normal. And either skip the recompute when `NORMAL` was not in
the input, or document that the option adds the attribute.

### 10. `target_ratio` is a share of the welded faces, not of the input triangles {#10}

`_face_limit` uses `int(target_ratio * self.mesh.face_count)` (`simplify.py:181`)
and `steps_for` uses `int(target_ratio * len(self.faces))` (`sequence.py:106-109`).
Both are the face count *after* welding has merged coincident vertices and
`build` has dropped triangles left with a repeated corner (`topology.py:423-425`).
The README says "share of the input triangles to keep" and `docs/API.md` says
"share of input triangles to keep".

A 300-triangle mesh, 100 of whose triangles collapse under welding:

```
input triangles 300, faces after welding 200
target_ratio=0.5 -> 100 triangles; half the input triangles would be 150
```

The gap is the size of whatever welding removed, which on scanned or
badly-exported data is not small. There is a second, smaller effect in the same
place: a single contraction removes up to two faces, so the count can undershoot
the limit by one or two.

**Fix.** Keep the input triangle count on the `Topology` and take the ratio
against that, or change both documents to say "of the triangles that describe a
surface after welding" and say what welding removed — `SimplifyResult` has no
field that reports it today, which is worth adding either way.

### 11. Option validation stops after six fields {#11}

`SimplifyOptions.__post_init__` (`options.py:103-123`) checks the three targets,
`metric`, `schedule`, `placement` and `candidates`. The other six fields are not
checked, and `topology.build` does not check that positions are finite. What a
caller gets:

| given | result |
|---|---|
| `locked=[99]` on a 6-point mesh | `IndexError: index 99 is out of bounds for axis 0 with size 6` |
| `locked=[-1]` | accepted; NumPy wraps, so the *last* point is locked silently |
| `locked=[20]` on a 24-vertex, 8-point cube | `IndexError: ... size 8` — a size the caller never supplied |
| `certify_samples=0` | `measured_error` comes back `inf` |
| `certify_samples=-5` | `ValueError: negative dimensions are not allowed` |
| `min_triangle_quality=5.0` | accepted; refuses every contraction, so the mesh comes back whole |
| `max_normal_flip=1e9` | accepted; `cos(radians(1e9))` is an arbitrary threshold |
| `weld_tolerance=-1.0` | accepted; silently means "exact" |
| `boundary_weight=-3.0` | accepted; a negative-weight quadric |
| an attribute that is a scalar | `TypeError: object of type 'numpy.float32' has no len()` |
| a `NaN` in `POSITION` | accepted; propagates to the output, `result.error` reports 0.0 |

`docs/API.md` says `DecimateError` is "Raised where the input does not describe a
triangle mesh, or where the options do not describe a reduction", and the package
goes to some trouble to make the checked cases raise it with a readable message.
The unchecked ones surface NumPy's internals instead, and three of them
(`locked=[-1]`, `min_triangle_quality=5.0`, a `NaN` position) produce a wrong
answer with no error at all.

The `min_triangle_quality` case is the most costly to debug: the documented range
is 0 to 1, a value above 1 is refused by no one, and the symptom is a reduction
that silently does nothing.

**Fix.** Extend `__post_init__` to the remaining six fields with the same
`DecimateError` shape already used; validate `locked` against the point count
inside `Topology.classify` (which is where the count is known) and reject
negative indices rather than wrapping; check `np.isfinite(positions).all()` in
`build` and raise `DecimateError` naming the first offending row; and wrap the
`len(value)` in `_check` (`simplify.py:88-92`) so a non-array attribute raises
`DecimateError` too.

### 12. `certify` costs 200x-470x the reduction it measures {#12}

`distance_to_mesh` (`certify.py:64-95`) tests every sample against every triangle
of the reference. The blocking (`:78-80`) bounds the *memory* — which the module
docstring and `TestMeasuringABigSurface` are both about — but the work is still
the product of the two counts, and there is no spatial index.

`options.py:78-79` says certify is "Off by default: it samples both surfaces, so
it costs about what the reduction did." Measured, `target_ratio=0.25`, 4000
samples:

| reference triangles | reduce | certify | ratio |
|---|---|---|---|
| 5,120 | 0.024 s | 4.76 s | 201x |
| 20,480 | 0.038 s | 17.93 s | 467x |
| 81,920 | 0.179 s | 69.95 s | 391x |

Certify time is linear in the reference triangle count (4x triangles, 3.9x time),
so the 1.5M-triangle scan the package is written for extrapolates to roughly 21
minutes for one `certify=True` call — set by a single boolean on an options
object, with nothing in the API to suggest the cost.

**Fix.** A uniform grid over the reference triangles turns the inner loop from
every triangle into the handful in the sample's own cell and its neighbours; the
cell-and-27-neighbours machinery already exists in `weld_positions`. Until then,
say the cost in `docs/API.md` against `certify` and `certify_samples` — "O(samples
x reference triangles)" is a one-line change that lets a caller size it — and
consider lowering `certify_samples` from 4000, since the cost is linear in it too.

### 13. `weld_tolerance` is quadratic in cell occupancy {#13}

`weld_positions` (`topology.py:93-113`) buckets into cells of side `tolerance`
and then, for every point, walks all 27 neighbouring buckets comparing against
every member. Within a single bucket that is O(k²) Python-level comparisons.
Points drawn from a tight cluster so that they share one cell:

| points in one cell | time |
|---|---|
| 2,000 | 2.50 s |
| 4,000 | 10.13 s |
| 8,000 | 39.96 s |

Exactly 4x per doubling. A caller who picks a tolerance larger than the local
point spacing — which is the likely mistake, and the failure mode is a hang
rather than a bad answer — will not get their reduction back.

Even well conditioned it is a Python loop: against the vectorised exact weld, on
a grid,

| points | exact | tolerance 1e-6 |
|---|---|---|
| 10,000 | 0.001 s | 0.056 s |
| 40,000 | 0.002 s | 0.202 s |
| 90,000 | 0.005 s | 0.459 s |
| 160,000 | 0.009 s | 0.875 s |

which extrapolates to about 9 s for the 1.5M points the README names as the
target case — tolerable, but 100x the exact path.

**Fix.** Vectorise the within-cell comparison (build the per-cell index arrays
once, then one `np.add.reduce` over pair differences per cell) to remove the
Python constant; and bound the pathological case, either by refusing a tolerance
that leaves a cell with more than some number of members with a `DecimateError`
that says what to pick instead, or by subdividing over-full cells. Documenting
the requirement — "tolerance should be smaller than the typical point spacing" —
is worth doing regardless, since the current text ("costs more") does not suggest
a hang.

---

## Testing and CI

### 14. The coverage gate does not see the code that ships {#14}

`tox.ini:60-66` enforces `--cov-fail-under=100`. Run:

```
TOTAL   887   0   100%
Required test coverage of 100% reached. Total coverage: 100.00%
```

887 statements, all Python. `_reduce_native.pyx` is 820 lines and contributes
none of them. It is the path that runs in every published wheel — `_reduce`
dispatches to it whenever `schedule == 'heap'` and the extension is present
(`simplify.py:105-106`), and the wheel test command asserts the extension is
present — and it is the only code in the package doing manual `malloc`,
`realloc`, `free` and raw pointer arithmetic.

So the number that gates a merge describes the fallback implementation, and the
default implementation is unmeasured. Finding [2](#2) is what an unmeasured
branch looks like: `_ring`'s overflow return is never taken by any test.

**Fix.** Cython supports coverage through `linetrace=True` and
`define_macros=[('CYTHON_TRACE_NOGIL', '1')]` plus the `Cython.Coverage` plugin;
a separate `coverage-accel` tox env building with linetrace would bring the
reducer under the same gate. If that is judged too slow to gate on, a branch
inventory in the `.pyx` with a test named against each refusal path
(`_ring` overflow, `_affected` overflow, `_faces_on_edge` overflow, the locked
pairs, `_price` returning -1) would cover the decisions that matter.

### 15. The test matrix is Linux only; wheels are published for three platforms {#15}

`.github/workflows/test.yml:21` is `os: [ubuntu-latest]`. `release.yml` builds
wheels on `ubuntu-latest`, `windows-latest` and `macos-latest`. The only thing
that runs a test on Windows or macOS is cibuildwheel's `test-command`, which runs
inside the release job — so the first time a Windows problem is seen is during a
release, on the version that was going to be published. Finding [3](#3) is exactly
that class of problem.

**Fix.** Add `windows-latest` and `macos-latest` to the `test` job matrix, at
least on the 3.12 row, so a pull request sees them.

### 16. `multiple-choice` has no compiled path {#16}

`_reduce` (`simplify.py:105-111`) uses the compiled reducer only for
`schedule == 'heap'`; `multiple-choice` always runs the NumPy loop:

| faces | `multiple-choice` | heap (compiled) |
|---|---|---|
| 6,962 | 12.8k faces/s | ~300k faces/s |
| 28,322 | 12.9k faces/s | ~300k faces/s |
| 64,082 | 12.4k faces/s | ~300k faces/s |

The README sells the schedule on scale — "every step costs the same as every
other", "the schedule a parallel or GPU implementation is built on" — and as
shipped it is 23x slower than the schedule it is meant to be the scalable
alternative to. That is a fair thing to ship in an alpha; it is not a fair thing
to leave out of the docs, because a reader will choose it for the wrong reason.

**Fix.** Say in the README and `docs/ALGORITHM.md` §5 that only `heap` is
accelerated today, and what the measured difference is.

### 17. The compiled loop never releases the GIL {#17}

`Reducer.run` (`_reduce_native.pyx:620-688`) has no `nogil` region. The
`pyproject.toml` cibuildwheel comment already notes it ("The reducer holds the
GIL throughout and would need auditing before it is offered to a free-threaded
interpreter"), and the consequence for the threaded case is the same: a watcher
thread measured at **10% of its normal throughput** across a 2.41 s reduction of
977,202 faces. A build tool or editor that decimates on a worker thread stalls
its UI for the duration.

**Fix.** The loop body is C data and typed memoryviews throughout; the only
things that can raise are `heap_push` and `_record`, which grow their buffers.
Give both an error-code return and wrap the body in `with nogil:`, raising from
outside it. That also removes the free-threading blocker the comment names.

### 18. No memory-safety gate over the hand-managed allocations {#18}

`_reduce_native.pyx` does its own `malloc`/`realloc`/`free` for eleven pools and
indexes them with raw pointer arithmetic under
`boundscheck=False, wraparound=False`. Nothing in the gates runs it under ASan or
valgrind, and there is no fuzzing or property testing over mesh shapes.

Two concrete defects in the allocation paths, both on the out-of-memory branch:

- `heap_push` (`:112-124`) uses `p = realloc(p, n)` and then tests `p` for NULL.
  A failed `realloc` returns NULL without freeing the original, so the original
  block is lost. Six pools, same pattern. `_record` (`:557-568`) is identical.
- `heap_init` (`:62-75`) raises `MemoryError` if any of six allocations fails,
  without freeing the ones that succeeded.

These are unreachable in practice on a machine that is not already out of memory,
and they are small. They are worth fixing because they are the parts of the file a
reader uses to calibrate how carefully the rest of it was written.

**Fix.** `tmp = realloc(p, n); if not tmp: free(p); p = NULL; raise` for each
pool, and a `heap_free` before the raise in `heap_init`. For the gate: a tox env
that builds the extension with `-fsanitize=address,undefined` and runs the suite
would be a job of its own — the kind `CLAUDE.md` says belongs in CI rather than
in the pre-commit gate — but it is the gate this file most wants.

### 19. The differential tests never reach the interesting inputs {#19}

`tests/test_native.py` compares the two paths on `icosphere(3)` (1280 faces),
`grid(20)` (722), `grid(12)`, `octahedron` and the hard-normals cube. None of
them:

- approaches `SCRATCH` (finding [2](#2) is invisible to this suite);
- uses `weld_tolerance`, so the tolerance weld is never seen by a differential
  comparison;
- uses `locked` (only `lock_boundary`), which is the option the cluster workflow
  is built on;
- feeds a bowtie or a non-manifold edge through the compiled path *and compares*
  — `test_it_refuses_a_bowtie_and_a_non_manifold_edge` only asserts that no face
  has a repeated corner;
- exceeds 1280 faces, so nothing exercises heap growth, log growth past the
  initial 1024 capacity, or the behaviour at a size where the two paths have room
  to diverge.

I ran the comparison on six meshes outside that set (bumped and near-planar
grids at 3200 faces, a noisy sphere, a mesh with a deliberate non-manifold face,
the probabilistic metric) at three ratios each and found no divergence, so this is
a gap in coverage rather than a known bug behind it — but the gap is where finding
[2](#2) was hiding.

**Fix.** Add the five cases above to `TestTheTwoPathsAgree`, and a case at
valence 1025 specifically, since that boundary is a constant in the source that
nothing currently pins.

### 20. Two wall-clock assertions in the suite {#20}

`tests/test_sequence.py:139` asserts `fifty < max(60.0 * one, 2.0)` and
`tests/test_topology.py:171` asserts `classify()` finishes in 0.5 s. Both bounds
are loose today (measured 32x-34x against a 60x limit; 0.065 s against 0.5 s) and
both are measurements of the machine rather than of the code. On a loaded CI
runner or a slower architecture they become flakes, and a flake in a gate is
eventually a gate somebody switches off.

`tests/test_collapse.py:189-230` and `tests/test_topology.py:212-226` show the
better pattern already in use here — `tracemalloc` peaks, which are facts about
the code — and the file's own docstring says so: "Asserted by watching what is
allocated rather than by timing, so it is a fact about the code rather than about
the machine." The two timing tests are the exception to a rule the project has
already written down.

**Fix.** Replace the replay-cost test with an allocation or operation-count
assertion, and the classification test with one that pins the absence of a
per-corner Python loop rather than a duration — or mark both `@pytest.mark.slow`
and keep them out of the merge gate.

### 21. `distance_to_mesh` is checked against its own primitive {#21}

`test_it_still_gets_the_same_answer` (`tests/test_certify.py:195-214`) compares
`distance_to_mesh` against "the plainest possible statement of the same thing" —
which calls `certify._closest_on_triangles`, the function `distance_to_mesh` is
built from. It validates the blocking, which is what its docstring says it is
for, but it is not an independent check of the distance.

The independent check does exist: the six parametrised cases at
`tests/test_certify.py:24-37` pin the seven-region point-to-triangle result
against distances worked out on paper, which is the right kind of test. Three
regions of the seven are covered by them.

**Fix.** Either extend the parametrised cases to all seven regions, or add a
brute-force oracle that does not share code — sampling each triangle densely and
taking a minimum is slow but exact enough, and only needs to run on a handful of
triangles.

### 22. The declared NumPy floor is never exercised {#22}

`pyproject.toml:47` declares `numpy>=1.24`; the build requires `numpy>=2.0`;
every CI job resolves whatever is newest. Nothing installs 1.24 and runs the
suite. The package uses several APIs whose behaviour moved across the 1.x/2.0
boundary — `np.unique(..., return_inverse=...)` shape semantics most notably,
which the code defends against with `.reshape(-1)` in five places, so the author
clearly knew — but the floor is a promise nothing checks.

**Fix.** A tox env or CI row installing `numpy==1.24.*` and running the suite,
or raise the declared floor to a version the matrix actually tests.

---

## Documentation

These are small individually. Collected, they are the reason a reader cannot take
the docs as the specification, which for a package whose value is that it is a
clean, well-described API is the part that matters most.

### 23. The package docstring describes `result.error` as the measurement {#23}

`__init__.py:15`:

```
result.error                        # measured deviation from the input
```

It is the quadric's estimate. `SimplifyResult.measured_error` is the measured
one, and the README (lines 64-70), `docs/API.md` and `simplify`'s own docstring
all draw the distinction carefully. The front-door docstring — the one a reader
meets first, and the one `--doctest-modules` renders — is the one that does not.

**Fix.** `# the reduction's own estimate of the deviation`, and a second line for
`measured_error`.

### 24. The replay cost claims are wrong, and disagree with each other {#24}

`README.md:52-54`: "Asking for fifty targets in a row costs about what asking for
one costs, which is what an editor's target slider needs."

`sequence.py:16-18` and `docs/ALGORITHM.md` §6: "`at()` costs the same whether it
is asked for ninety per cent of the triangles or two per cent, and asking for
fifty different counts in a row costs fifty times that and no more."

The second half of the second statement is correct and the first half is not;
the README's statement is not:

| mesh | one ask | fifty asks | ratio |
|---|---|---|---|
| 5,120 tri | 3.17 ms | 101.45 ms | 32x |
| 20,480 tri | 9.51 ms | 324.88 ms | 34x |

and `at()` does not cost the same regardless of target, because `_emit` is linear
in what it returns:

| `at(target_count=)` | triangles | time |
|---|---|---|
| 20,000 | 20,000 | 11.58 ms |
| 5,000 | 5,000 | 3.34 ms |
| 1,000 | 1,000 | 0.85 ms |
| 20 | 20 | 0.29 ms |

The underlying claim that matters is true and is worth stating plainly: a target
is a prefix replay, not another reduction — 3.17 ms against the 24 ms the
reduction itself took, and independent of how many contractions are being
replayed over.

**Fix.** Say "a target costs a replay, not a reduction", give the two numbers,
and drop both of the stronger forms.

### 25. Three accounts of how a stale heap entry is detected, none matching the compiled code {#25}

- `docs/ALGORITHM.md` §5: "a stamp per pair, bumped whenever the pair is
  re-priced, and a pop whose stamp is stale is discarded". This describes
  `run_heap`'s `stamps` dict (`simplify.py:294,310-313`) — the NumPy path.
- `_reduce_native.pyx:24-29`: "an entry popped later whose price no longer
  matches what it was queued at has been superseded ... No map from edge to
  version is needed, which is the other structure that would not fit."
- The code: a per-*vertex* version counter carried in each heap entry
  (`:194,637,682-685`), bumped in `_contract` (`:616`).

The third is neither of the first two. The `.pyx` docstring in particular argues
against keeping versions in the very file that keeps them, which will cost the
next maintainer real time — the mechanism is the load-bearing part of the loop.

**Fix.** Rewrite the `.pyx` module docstring to describe the per-vertex version,
and add a sentence to `docs/ALGORITHM.md` §5 saying the compiled path uses a
per-vertex version where the NumPy path uses a per-pair stamp, and why the two
give the same order.

### 26. "Above zero, `A` is positive definite" {#26}

`docs/ALGORITHM.md` §2 and the `quadrics.py` module docstring both say that with
the variances above zero, "`A` is positive *definite* rather than rank one, which
is what makes a single plane solvable". Only `s_n` reaches `A` — see finding
[6](#6). With `s_p` above zero and `s_n` at zero, `A` is the classical rank-one
`n nᵀ` and a single plane is not solvable.

**Fix.** "Above zero, `s_n` makes `A` positive definite rather than rank one;
`s_p` contributes to `c` alone." Both places.

### 27. `DecimateError`'s documented scope {#27}

`docs/API.md`: "Raised where the input does not describe a triangle mesh, or
where the options do not describe a reduction." Finding [11](#11) lists ten
inputs and option values that do not raise it. The sentence is the right
contract; the code does not yet meet it.

**Fix.** Finding [11](#11)'s fix makes the sentence true. Until then the sentence
should say which option fields are validated.

### 28. `INTEGER_ATTRIBUTES` and `MutableAttributeMap` are exported but unused {#28}

`types.py:41` defines `INTEGER_ATTRIBUTES = frozenset({'JOINTS_0', 'JOINTS_1'})`
with the comment "Attributes whose values are identifiers rather than quantities.
They are taken from a surviving vertex whole, never averaged: the mean of joint 3
and joint 9 is joint 6, which is some unrelated bone." Both it and
`MutableAttributeMap` (`:33`) are in `types.__all__` and referenced nowhere in the
package or its suite.

The behaviour the comment describes is in fact what happens — *every* attribute
is carried whole, not just these two — so a reader who finds the constant is told
there is a rule being applied selectively when there is not one, and a caller who
imports it (it is exported, so that is an invitation) gets a name that does
nothing.

**Fix.** Delete both, or use `INTEGER_ATTRIBUTES` for something real — a check
that refuses `recompute_normals`-style interpolation on those semantics if one is
ever added. `test_a_carried_attribute_is_always_one_of_the_input_values` already
tests the property the comment claims; the comment belongs on that property's
implementation, which is `_emit`'s corner-gather.

### 29. The README's noise defaults {#29}

`README.md:87-88` gives `position_noise` and `normal_noise` a default of `0.0`.
The actual default when `metric='probabilistic'` and neither is named is
`0.001 * diagonal` and `0.001` (`options.py:40-41,134-135`). `docs/API.md` gets
this right — "Selecting `'probabilistic'` without naming either noise uses 0.001
of the diagonal and 0.001 respectively" — so the README is the copy that drifted.

**Fix.** Say `0.001 when 'probabilistic' is selected` in the README table, and
add the sentence from finding [6](#6) about naming one not zeroing the other.

### 30. The README does not say what `locked` indexes {#30}

`README.md:91` says "further points to hold, by index". `docs/API.md` and
`options.py:64-65` both add the part that matters: "indexed into the welded
points — which are the caller's own vertex indices unless vertices were welded".
A caller working from the README on a hard-edged glTF mesh — 24 vertices over 8
points — gets an `IndexError` naming a size they never supplied (finding
[11](#11)).

**Fix.** Carry the API.md wording into the README row.

### 31. The union-find docstring's justification does not hold {#31}

`topology.py:46-53`: "No path compression, because the trees never get deep
enough to want it: `weld_positions` unions in ascending order and always attaches
the larger root to the smaller, so every group ends up a star rooted at its
lowest member and a lookup is one hop."

`union` attaches `right_root` to `left_root` (`:64-67`), and `left_root` is not
the smaller once a chain exists. The union sequence `(0,5), (2,5), (3,5)` — which
is the ascending order `weld_positions` produces — gives:

```
parent chain: [2, 1, 3, 3, 4, 0]
root of 5: 3          (not the lowest member, which is 0)
hops to reach the root from 5: 3   (not one)
```

The partition is still correct, so nothing is wrong with the result. But the
stated reason for omitting path compression is not true, and the depth is
unbounded in principle — which matters because this is the code path in finding
[13](#13).

**Fix.** Either union by minimum root (`if left_root < right_root: parent[right_root] = left_root else: parent[left_root] = right_root`),
which makes the docstring true; or add path compression and delete the paragraph.

---

## Suggested order of work

*(Followed, in roughly this order. See [Resolution](#resolution) below for what
each finding turned into.)*

1. Findings [3](#3) and [4](#4) first — both are release mechanics, both are
   small, and neither needs a decision.
2. Finding [1](#1) next. It is a few lines (a local origin) and it decides
   whether the package is usable for the georeferenced worlds it would otherwise
   be a natural fit for.
3. Findings [5](#5), [6](#6), [8](#8), [9](#9) — four small behavioural fixes,
   each with an obvious correct form, each currently producing a wrong answer
   quietly.
4. Finding [2](#2), with the test from finding [19](#19) written first.
5. Finding [11](#11) as one pass over `__post_init__` and `build`.
6. The documentation set ([23](#23)-[31](#31)) as one pass; most are single
   sentences, and [24](#24), [25](#25) and [26](#26) are the ones a maintainer
   will be glad of.
7. Findings [12](#12) and [13](#13) when the sizes they matter at are reached;
   both are worth a note in the docs immediately, since the current text
   understates both by a large factor.
8. Findings [14](#14), [15](#15) and [18](#18) as CI work, in that order.

Nothing here is structural. The architecture — welded topology under corner
attributes, quadrics as a `(points, 10)` array, a recorded collapse log replayed
by prefix, a compiled loop held to a readable NumPy statement of the same
algorithm — is a good design, and the invariant testing above says the core of
it works. The findings are concentrated in the edges of the input space, in the
options that are not validated, and in documentation that has drifted ahead of
the code.

---

## Resolution

Every finding was worked. Thirty were fixed as proposed or in a better form;
two held as observations but not as diagnoses, and say below what was done
instead. The suite is 243 passing, `ruff check`, `ruff format --check` and
`mypy src/opengl_decimate` pass, and the coverage gate is back at 100%.

### Blockers

**[1](#1) — fixed.** Quadrics are accumulated about
`topology.local_origin(points)`: the model's bounding-box centre on each axis
where the model sits at least twice its own half-extent from the world origin,
and zero elsewhere. That threshold is not a tuning constant — it is exactly the
condition under which Sterbenz's lemma makes both the subtraction and the
addition that undoes it *exact*, so a model near the origin is not moved by a
rounding for the sake of a problem it does not have, and a georeferenced one
loses nothing at all. The result is stronger than the finding asked for: the
same icosphere at 0 and at 6.4e6 now gives identical triangle counts, identical
indices, an identical `result.error` and an identical metric noise floor
(2.2e-8 in both cases, against 1.5e-1 before). `POSITION` now comes back in the
float dtype it arrived in, which closes the second loss the finding named.

**[2](#2) — fixed.** The fixed `SCRATCH` buffers are now `IntPool`s that grow;
`_ring`, `_affected`, `_faces_on_edge`, `_link_condition`'s opposite set,
`_contract`'s moving list and `run`'s ring all reallocate rather than give up.
`-1` from any of them now means out of memory and nothing else, and is raised
rather than silently skipped. The `_link_condition` cap that could have made
`shared == opposite` compare two differently-truncated counts is gone with it.
`tests/test_native.py` pins agreement at valences 63, 64, 65, 1023, 1024, 1025
and 2000; before the fix 1025 and above diverged, after it none do.

**[3](#3) — fixed.** Every `long` that has to match a NumPy `int64` is
`cnp.int64_t`, including the `serial` and `removed_at` pools. The generated C
now carries `__Pyx_TypeInfo_nn___pyx_t_5numpy_int64_t` where it carried
`__Pyx_TypeInfo_long`. `windows-latest` and `macos-latest` are on the test
matrix (finding [15](#15)), so the next one of these is caught by a pull
request rather than by a release.

**[4](#4) — fixed.** `MANIFEST.in` carries `tests/*.py`, `tox.ini` and
`docs/*.md`. The `install` job now unpacks the built sdist and runs the suite
from *inside* it — the gap was never that the sdist was untested but that
everything testing it ran the checkout's `tests/` instead. Verified: 161 pass
from inside the unpacked archive.

**[5](#5) — fixed.** `_Engine` takes `exhaust` and sets both `face_limit` and
`error_limit` from it, so the meaning of "record everything" lives in one place
rather than half in `_reduce` and half in the loops. `collapse_sequence` also
takes `options=None` now, since requiring a target for a call whose contract is
that targets do not apply was the thing that made the bug easy to hit.

### Correctness and behaviour

**[6](#6) — fixed.** `position_noise` and `normal_noise` default to `None` and
resolve independently, so naming one no longer zeroes the other — and an
explicit `0.0` is now expressible, which it was not. The second half of the
finding is a fact about the mathematics rather than a defect: averaging a fixed
quadratic form over a Gaussian cloud of sample points adds `s_p² tr(A)` and
nothing else, so `position_noise` *cannot* reach `A`. That is now said plainly
in `options.py`, `quadrics.py`, `ALGORITHM.md` §2 and `API.md`, and pinned by
`test_the_normal_noise_is_what_makes_the_metric_different`. The test that named
the wrong field now names `normal_noise`.

**[7](#7) — partly; the measurement holds, the diagnosis does not.** The 2x is
real and reproduced. But the cause is not that "the divisor keeps growing
against a surface that is shrinking": numerator and denominator accumulate in
lockstep, over the *same* set of original planes and the *same* areas that
weighted them. What that ratio is, by construction, is an area-weighted
**root-mean-square** distance — and comparing a root-mean-square against a
sampled *maximum* is where the factor of two comes from, not from drift.
Measured against `certify` at five ratios and four budgets, `result.error` sits
between 0.5x and 0.9x the sampled rms and between 1.25x and 2.35x the sampled
max: it tracks the rms, as an rms should.

So the proposed fix is declined. Recomputing the weight from the live incident
area would leave the numerator summed over every historical plane while the
denominator became a current local area, and the ratio would stop being a mean
of anything. What was done instead is to say what the number *is* — in
`ALGORITHM.md` §3 and §8, `API.md` against both `target_error` and `error`, and
the README — and to point at `measured_error` for the bound a level of detail
needs. The contradiction the finding noted between §3 and §8 is gone: §3 now
defines the quantity and §8 says why a bound is a different one.

**[8](#8) — fixed.** The budget now `break`s out of the ranked candidates
instead of returning, so a draw with nothing affordable in it is a failed draw
against the existing failure budget. `icosphere(3)` at `target_error=0.02` now
lands at 140-154 triangles across eight seeds, against the heap's 140 — where
before it was 278-362, two to two-and-a-half times too many and ±30% with the
seed.

**[9](#9) — fixed.** `_surface_normals` accumulates per *point* and gathers to
the output vertices, so vertices split by an attribute share a normal. On a
sphere with a UV seam of no geometric meaning, the worst recomputed normal is
now 1.86° — exactly what the same mesh without the seam gives, against 5.62°
before. The second half — that the option adds `NORMAL` where the input had
none — is kept, because generating normals for a position-only mesh is a
reasonable thing to ask for; it is now documented in `options.py`, `API.md` and
the README rather than being a surprise.

**[10](#10) — fixed.** `Topology.input_faces` and `CollapseSequence.input_faces`
carry the caller's triangle count, and both `_face_limit` and `steps_for` take
the ratio against it. A 300-triangle mesh with 100 triangles that weld away now
returns 149 for `target_ratio=0.5` rather than 100. `SimplifyResult` gained
`input_triangles` and `welded_away`, which is what makes the difference legible
rather than mysterious.

**[11](#11) — fixed.** All eleven cases in the finding's table now raise
`DecimateError` with a message naming the field and its range. `locked` is
checked for negative indices in `__post_init__` and against the point count in
`Topology.classify`, which is where the count is known — and the message says
*welded points*, since that is what the number is. `build` refuses a
non-finite position, and `_check` refuses an attribute that is not one row per
vertex. Finding [27](#27)'s sentence is now true, and `API.md` lists what it
covers.

**[12](#12) — fixed.** `distance_to_mesh` puts the reference triangles into a
uniform grid and widens a ring at a time until the nearest found is no further
than the ring searched — at which point no triangle outside it can win, so the
answer is **exact**, not approximate. The widening stops where a ring would
cost more than a scan, and the remainder is scanned. Measured against the
brute force on four meshes at four spreads, including a mesh with one huge
triangle among small ones: zero mismatches, maximum difference 0.0.

The effect on the workload the option exists for, at `target_ratio=0.25` and
4000 samples:

| reference triangles | certify before | certify after | reduction |
|---|---|---|---|
| 20,480 | 17.74 s | 0.20 s | 0.04 s |
| 318,402 | (extrapolated ~4.5 min) | 0.45 s | 0.73 s |
| 977,202 | (extrapolated ~14 min) | 1.02 s | 2.74 s |

The documented claim — "it costs about what the reduction did" — is now true,
where it was out by a factor of several hundred.

**[13](#13) — fixed.** The weld is vectorised: cells are sorted once, a lookup
is a `searchsorted`, and the pair comparison is whole-array and blocked so the
memory is bounded by a constant rather than by the fullest cell. 8000 points
clustered into one cell went from 40.91 s to 1.08 s; a 160,000-point grid from
0.850 s to 0.053 s, which is seven times the exact weld rather than a hundred.
The docs now say to pick a tolerance smaller than the typical point spacing and
why. A tolerance too fine for the coordinates to carry — which used to overflow
the cell arithmetic silently — is refused.

The grid this rests on is shared with finding [12](#12) as
`opengl_decimate.spatial`, rather than each growing its own.

### Testing and CI

**[14](#14) — partly; the alternative was taken, and the reason is measured.**
`setup.py` will now build the reducer with `linetrace` on
`OPENGL_DECIMATE_LINETRACE=1`, so the measurement is available to anyone
investigating a particular reduction. It is not a gate: instrumented, the two
smallest test modules — twelve seconds ordinarily — did not finish in ten
minutes, because every line of the contraction loop calls back into Python's
tracer and the `nogil` region has to take the GIL to do it. A gate nobody can
run is not a gate.

What holds the compiled reducer instead is the finding's own second suggestion,
carried out: `tests/test_native.py` now requires the same contractions in the
same order on meshes chosen to reach each decision in the loop — a locked pair,
a border, an error budget, a quality floor, a link condition refused, a
duplicate face refused, a queue and a log grown past their initial capacity, and
a neighbourhood larger than the buffers start at. An agreement test over a
decision is the stronger of the two checks, since a branch can be taken without
being taken correctly. `tox.ini` now says all of this where the gate is
declared, so the number's scope is stated rather than assumed.

One thing the linetrace work turned up on its own: switching it back off left
the traced C in place, because Cython judges staleness from timestamps and
cannot see that a directive changed — a reducer ten times slower with nothing
to say so. `cythonize(force=True)` closes that.

**[15](#15) — fixed.** `windows-latest` and `macos-latest` on the 3.12 row,
which is the row that also runs lint, types and coverage.

**[16](#16) — fixed** (as a documentation finding, which is what it was).
Measured at 12.3k-16.2k faces/s against 543k-584k for the compiled heap. Said in
`ALGORITHM.md` §5 and in the README's *Limits*.

**[17](#17) — fixed.** The contraction loop moved into `_loop`, which runs under
`with nogil:`; `heap_push`, `_record` and `_contract` return error codes instead
of raising, and the `MemoryError` is raised from outside. A watching thread now
runs at 102% of its idle rate during a 977k-face reduction, against 10% before.
Reduction throughput is unchanged at 299k faces/s. The free-threading note in
`pyproject.toml` no longer claims a blocker that is gone, and does not claim a
guarantee nothing has tested either.

**[18](#18) — fixed.** Every growth path — `heap_grow`, `_grow_log`,
`pool_reserve` — takes the `realloc` through a temporary and adopts it only once
it is a block, so a failure loses the growth and not the pool. `heap_init` zeroes
its pointers first and calls `heap_free` before raising.

**[19](#19) — fixed.** Added to `TestTheTwoPathsAgree`: seven valences straddling
the old ceiling, a tolerance weld, named `locked` points, a bowtie and a
non-manifold edge *compared* rather than merely checked for degeneracy, and a
mesh large enough to grow both the queue and the log past 1024.

**[20](#20) — fixed.** Both replaced with facts about the code. The replay-cost
test now asserts that `at()` constructs no `_Engine` at all across fifty asks
(counted by substitution, not timed) and that fifty asks peak within 1.5x of
one's allocation. The classification test asserts that `classify()` on a
178,802-face mesh never builds the per-point face sets — the memory claim that
actually blocks a scan — and that the answer at that size is right, not merely
quick.

**[21](#21) — fixed.** The parametrised cases now cover all seven regions, the
face, three edges and three corners. Added an oracle that shares no code with
the selector: the projection where it lands inside the triangle, and three
segment clamps otherwise.

**[22](#22) — fixed.** A `oldest-numpy` tox env pins `numpy==1.24.*` on 3.10,
asserts the resolved version, and runs the suite; `[gh]` maps it onto the 3.10
CI row. Run here, it passes — 228 tests on NumPy 1.24.4 — so the declared floor
was honest and is now checked. Only the runtime NumPy is pinned, which is also
how a user's install works: an extension compiled against NumPy 2 runs on 1.x,
and that arrangement is part of what the env checks.

### Documentation

**[23](#23), [24](#24), [25](#25), [26](#26), [27](#27), [29](#29), [30](#30),
[32](#32) — fixed**, across `__init__.py`, `sequence.py`, `quadrics.py`, the
`.pyx` module docstring, `README.md`, `docs/API.md` and `docs/ALGORITHM.md`.

On [24](#24): measured at 41x and 144x for fifty asks against one, and `at()` is
linear in what it returns (11.18 ms for 20,000 triangles against 0.30 ms for
20). Both stronger forms are gone. What replaced them is the claim that is true
and is the one that matters — a target costs a replay rather than a reduction
(around 2 ms against 25-62 ms), and costs the same however far along the
sequence it is asked for.

On [25](#25): the `.pyx` docstring now describes the per-vertex version counter
it actually keeps, and says why a version per *edge* — the structure it used to
argue against while keeping something else — would not fit. `ALGORITHM.md` §5
says both paths' spellings and why they agree.

**[28](#28) — fixed.** Both deleted. The rule `INTEGER_ATTRIBUTES`' comment
described is real and applies to *every* attribute, so the comment moved to
`_emit`'s corner gather, where the behaviour is.

**[31](#31) — fixed** by deletion: `_DisjointSet` is gone, since finding
[13](#13)'s vectorised weld uses the whole-array `_connected` the module already
had for walking fans. The docstring that was not true is not there to be true.

### Found while working the list

**[32](#32)** was added to the table: the README's *Limits* still said the
contraction loop was Python and that a few hundred thousand triangles took
minutes, which the compiled reducer had already made untrue.

`_price`'s `if not have_best: return -1.0` is unreachable — two locked ends
return earlier, and any other combination leaves at least one candidate. It is
left in place as a guard on a loop that may grow more cases, but it is not a
branch any test covers and should not be counted as one.
