# How a reduction works

A decimation is a long sequence of one operation: merge the two ends of an edge
into a single point. Everything else is deciding which edge, where the merged
point goes, and whether the result is still a surface.

## 1. Weld and classify

A vertex is not the same thing as a point. A hard-edged export carries one corner
of a cube as three vertices with three different normals, and a decimator that
treats those as three separate places cannot reduce the cube at all.

So [`topology.build`](../src/opengl_decimate/topology.py) welds vertices sharing
a position and works in those points, keeping each corner's original vertex
alongside. Topology is asked of the points; attributes are carried by the corners.

Each point is then classified once:

- **Manifold** — a closed fan of faces around it, free to merge onto any
  neighbour.
- **Border** — on the edge of the surface. May merge only along the border, which
  is what keeps the outline of an open patch where the author put it.
- **Locked** — never moves: an edge with three or more triangles on it, a bowtie
  (faces forming more than one fan around one point), an isolated point, or a
  point the caller named.

The classification is computed once and stays true, because the rules in step 4
admit only contractions that preserve the link of an edge, and those leave every
border a border and every fan a fan. A bowtie is found by joining corners across
every interior edge and asking whether a point's corners ended up in one group.

## 2. Accumulate quadrics

A quadric is a symmetric 4x4 answering one question — for a given point, what is
the summed squared distance to a set of planes? — with two properties that make
decimation possible. Quadrics **add**, so the cost of merging two points is the
sum of their quadrics. And the point minimising one is a single 3x3 solve.

It is stored as its ten distinct coefficients, so a whole mesh's quadrics are one
`(points, 10)` array and every operation is a whole-array operation.

Each point accumulates the planes of its incident triangles, weighted by area —
which stops a dense patch of tiny triangles outvoting the large ones around it,
so the quadric measures deviation of *surface* rather than a count of planes.

**Border constraints.** A border edge has one face, and a plane standing
perpendicular to that face along the edge is a wall the border may slide along
but not leave. Both ends of every border edge accumulate it. Without this the
outline of a patch creeps inward as the patch is reduced.

### The probabilistic form

With `metric='probabilistic'`, each plane is read as a sample from a Gaussian
rather than as a fact, and the quadric holds the *expected* squared distance.
For a plane with mean normal `n`, mean point `p`, and isotropic variances
`s_n²` and `s_p²`, expanding `E[((x - p)·n)²]` over both distributions gives

```
A = n n' + s_n² I
b = (n·p) n + s_n² p
c = (n·p)² + s_n² |p|² + s_p² + 3 s_n² s_p²
```

for `E(x) = x'Ax - 2b'x + c`. With both variances at zero this is exactly the
classical plane quadric. Above zero, `A` is positive *definite* rather than rank
one, which is what makes a single plane solvable and a noisy neighbourhood well
conditioned — the case a reconstructed surface presents everywhere.

The derivation is checked against a Monte-Carlo estimate of the same expectation
in `tests/test_quadrics.py`.

## 3. Price every candidate

For a candidate pair, the summed quadric is minimised. Four placements are
considered — each endpoint, the midpoint, and the minimiser — and the cheapest
admissible one wins. The minimiser is dropped where the 3x3 system is singular,
which is the ordinary case on a flat neighbourhood rather than an error; a locked
endpoint admits only its own position, and two locked endpoints admit nothing.

All of this is one pass over arrays: the batch solve is a vectorised adjugate,
and the four costs are one evaluation over a reshaped stack.

The reported **deviation** is the quadric cost divided by the summed area around
the two points, square-rooted — so it is a length in model units rather than an
area-weighted square, which is what makes `target_error` mean something a caller
can picture.

## 4. Refuse what would break the surface

Four questions, all asked every time, because the cases they catch are not rare
— they are what a dense mesh is made of.

**The link condition.** Contracting is safe exactly when the points joined to
both ends of the edge are precisely the points opposite it. A shared neighbour
anywhere else becomes, after the merge, an edge carrying three triangles.

**No duplicate face.** The link condition alone lets a small closed shape fold
onto itself: a tetrahedron contracts to two triangles back to back, which passes
every local test and encloses nothing.

**No fold.** Every face that has to outlive the contraction is checked against the
proposed placement, before and after. A normal that turns further than
`max_normal_flip` has been folded over rather than moved, and the surface would
render inside out there. A face that collapses to a line is refused outright.

**Shape, optionally.** `min_triangle_quality` refuses a contraction leaving a
triangle thinner than a scale-free quality measure allows — four root three times
the area over the summed squared edge lengths, which is 1 for equilateral and 0
for a line. Off by default: it trades reduction for triangle shape.

## 5. Choose the next one

**`heap`** keeps every candidate in a priority queue ordered by deviation, with
lazy invalidation: a stamp per pair, bumped whenever the pair is re-priced, and a
pop whose stamp is stale is discarded. After a contraction only the survivor's
quadric changed, so only the edges around it are re-priced — one batched call,
not one call per edge.

This is the best quality available and it is sequential by construction: every
contraction changes the price of its neighbours, so the next choice depends on
the last.

**`multiple-choice`** draws a few candidates at random and contracts the cheapest
of them. There is no global ordering to maintain and no queue to keep, every step
costs the same as every other, and the quality loss is small. A pair that is no
longer an edge is dropped when it is drawn rather than hunted for.

Giving up the global ordering is the point: it is what makes a reduction
divisible, and it is the schedule a parallel or GPU implementation is built on.

## 6. Record it

Each applied contraction appends to a log: the two points, the placement, the
deviation, and which faces it removed. The log is the reduction expressed once,
and every target is a prefix of it.

Replaying a prefix of length *k* is four array operations:

- **Which point became which.** `parent` is filled from the prefix in one
  assignment — each point dies at most once — then pointer-jumped until it stops
  changing.
- **Where each survivor sits.** The placement of the last contraction in the
  prefix that it survived, gathered with one scatter-maximum.
- **Which faces are left.** Each face records the step that removed it, so the
  live set is a comparison.
- **The attributes.** They never moved, so there is nothing to recompute.

So `at()` costs the same whether it is asked for ninety per cent of the triangles
or two, and asking fifty times in a row costs fifty times that and no more.

## 7. Hand it back

An output vertex is a distinct combination of *point* and *attributes*: two
corners at the same point carrying the same normal and texture coordinate are one
vertex, two carrying different ones are two. That is what keeps a seam a seam
without any seam-specific machinery.

Vertices are numbered by first use in the index stream. `vertex_map` relates the
input's vertices to the output's.

## 8. Measure what happened

The quadric's own number is an estimate accumulated from planes, and planes
extend past the triangles that produced them, so it drifts optimistic as a
reduction goes on. What a level of detail has to promise is a bound on the
*surface*, which is a different quantity.

[`certify.surface_deviation`](../src/opengl_decimate/certify.py) samples both
surfaces in proportion to area and asks each sample how far it is from the other,
in both directions. One direction alone cannot see a hole: every point of a mesh
with a piece missing is still on the original.

The point-to-triangle distance underneath it is exact. The plane is divided into
the seven regions a nearest point can fall in — the face, three edges, three
corners — and every region is evaluated for every pair with the right one
selected, which makes it one pass over arrays rather than a branch per triangle.

## References

- Garland & Heckbert, *Surface Simplification Using Quadric Error Metrics*,
  SIGGRAPH 1997.
- Hoppe, *Progressive Meshes*, SIGGRAPH 1996 — the recorded sequence and its
  replay.
- Wu & Kobbelt, *Fast Mesh Decimation by Multiple-Choice Techniques*, VMV 2002.
- Trettner & Kobbelt, *Fast and Robust QEF Minimization using Probabilistic
  Quadrics*, Computer Graphics Forum 39(2), 2020.
- Dey, Edelsbrunner, Guha & Nekhayev, *Topology preserving edge contraction*,
  1999 — the link condition.
