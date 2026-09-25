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

Each point is also counted: how many different **texture coordinates** the model
draws it at. One almost everywhere; two where a seam runs, because a chart
boundary is one position drawn twice, at the end of one chart and the start of
the next. Step 4 uses the count.

Texture coordinates and not the rest of what a corner carries, because they are
the values that jump. A normal splits at a crease and the two sides differ by an
angle, so taking the nearer of them is a small error in a direction; a chart
boundary puts the two sides at unrelated ends of the texture, and taking either
draws a band of the whole image across the triangles on the other side.

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

**Where the origin is.** A plane's constant term is the squared plane offset,
taken from the point's own coordinates. For a model at Earth-centred
coordinates — `p` around 6.4e6, which is what 3D Tiles and any ECEF-referenced
world uses — that constant is around 4e13, while the squared distance the
quadric exists to report is around 1e-6: nineteen decimal digits of range inside
a type that carries sixteen, and the cancellation takes the answer with it.

So quadrics are accumulated about the model's own bounding-box centre, and the
shift is undone when a position is handed back. It is taken on an axis only
where the model sits at least twice its own half-extent from the origin — where
absolute coordinates cost something, and where the subtraction and the addition
that undoes it are both exact, so no model pays a rounding on account of a
problem it does not have. A reduction is then independent of where the model is:
the same mesh at the origin and at 6.4e6 gives the same triangles, the same
placements and the same reported error.

`POSITION` comes back in the float dtype it arrived in, for the same reason — a
`float32` ulp at 6.4e6 is half a metre.

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
classical plane quadric.

The two variances do different things, and the difference matters when choosing
them. `s_n` reaches `A`, making it positive *definite* rather than rank one —
which is what makes a single plane solvable and a noisy neighbourhood well
conditioned, the case a reconstructed surface presents everywhere. `s_p` reaches
`c` alone, since averaging a fixed quadratic form over a Gaussian cloud of
sample points adds `s_p² tr(A)` and nothing else: it raises every cost by the
same amount and leaves the choice of contraction where it was. So `normal_noise`
is the option that changes the reduction, and with it at zero the probabilistic
metric is the classical one with an offset on the reported error.

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
area-weighted square, which is what makes `target_error` a figure a caller can
picture.

**What that length is, exactly.** The cost is a sum of squared distances to
planes, each weighted by the area of the triangle that contributed it; the
divisor is the sum of those same areas. Both accumulate together as points
merge, so the ratio stays a *mean*: the deviation is the area-weighted
root-mean-square distance to the planes the surface has been through. That is
the right quantity to steer a reduction by, and it is **not a bound** — a
root-mean-square never is, and planes extend past the triangles that made them,
which pulls it down further. Measured against `certify` on an icosphere it runs
a little above the sampled root-mean-square and around half the sampled
maximum; a level of detail's switching distance needs the maximum, so build it
on [`certify`](#8-measure-what-happened) rather than on this.

## 4. Refuse what would break the surface

Five questions, all asked every time, because the cases they catch are not rare
— they are what a dense mesh is made of.

**The link condition.** Contracting is safe exactly when the points joined to
both ends of the edge are precisely the points opposite it. A shared neighbour
anywhere else becomes, after the merge, an edge carrying three triangles. An
open surface is treated as closed by a virtual point joined to every border
point, which adds one refusal: a border edge whose triangle has its other two
edges on the border as well. That triangle is the last one of its piece, so an
open piece keeps at least one triangle and a closed one at least four.

**No duplicate face.** The link condition alone lets a small closed shape fold
onto itself: a tetrahedron contracts to two triangles back to back, which passes
every local test and encloses nothing.

**No fold.** Every face that has to outlive the contraction is checked against the
proposed placement, before and after. A normal that turns further than
`max_normal_flip` has been folded over rather than moved, and the surface would
render inside out there. A face that collapses to a line is refused outright.

**The seam, where `lock_seams` asks for it.** A point drawn at more than one
texture coordinate may then merge only with a point drawn at the same number,
which is the same shape of rule as the border one and holds the seam network
exactly where the author drew it. It is off by default, because step 6 already
keeps each side of a seam reading from its own chart and holding the *line* as
well costs the triangles the seam network needs — on a coastal cliff scan, the
difference between reaching five hundred triangles and stopping at two thousand.
What it buys is a seam that does not slide, which is worth its triangles where
the atlas is thousands of small charts and the sliding is most of the surface.

**Shape, optionally.** `min_triangle_quality` refuses a contraction leaving a
triangle thinner than a scale-free quality measure allows — four root three times
the area over the summed squared edge lengths, which is 1 for equilateral and 0
for a line. Off by default: it trades reduction for triangle shape.

## 5. Choose the next one

**`heap`** keeps every candidate in a priority queue ordered by deviation, with
lazy invalidation: an entry that has been superseded is discarded when it is
popped rather than hunted down when it goes stale. After a contraction only the
survivor's quadric changed, so only the edges around it are re-priced — one
batched call, not one call per edge.

The two implementations spell the staleness test differently, because they are
sized for different meshes. The NumPy path keeps a stamp per *pair* in a dict,
bumped whenever the pair is re-priced. The compiled path keeps a version per
*point*, bumped when its quadric changes, and each queued entry names the two
versions it was priced against — which costs one integer per point instead of a
hash map the size of the queue, and is the difference between fitting a scan in
memory and not. The orders agree because a pair goes stale under either test on
exactly the same event: a contraction at one of its ends.

This is the best quality available and it is sequential by construction: every
contraction changes the price of its neighbours, so the next choice depends on
the last.

**`multiple-choice`** draws a few candidates at random and contracts the cheapest
of them. There is no global ordering to maintain and no queue to keep, every step
costs the same as every other, and the quality loss is small. A pair that is no
longer an edge is dropped when it is drawn rather than hunted for.

Giving up the global ordering is the point: it is what makes a reduction
divisible, and it is the schedule a parallel or GPU implementation is built on.

**Only `heap` has the compiled loop.** `multiple-choice` runs the NumPy
implementation whether or not the accelerator is installed, so as it stands it
is around thirty times slower per face than the schedule it is the scalable
alternative to. The divisibility is what it is for; the throughput is not yet
there to go with it.

## 6. Record it

Each applied contraction appends to a log: the two points, the placement, the
deviation, and which faces it removed. The log is the reduction expressed once,
and every target is a prefix of it.

A merged point sits at the placement, which is somewhere on or near the edge,
and the two ends were measured in different places — so the corners take up the
copies of whichever end the placement came to rest nearest, and the other end's
copies are let go. Which end that is has little to do with which *index*
survived: the placement is chosen from the edge before the legality tests say
which end may die.

Two rules sit over that, and between them are why a seam survives a reduction
without being locked. The end drawn at **more** texture coordinates wins
outright, whatever the distance says, because it is the only one with a
coordinate to give each side of the seam it is on. And where the winning end
carries several copies, the corner takes the one nearest in attribute space,
which is the copy on its own side of the seam. So a triangle is never left
reading the chart next door; what it can do is read a coordinate measured a
little along from where it now is, which is what `lock_seams` refuses.

These handovers are recorded alongside the contractions, in
[`opengl_decimate.corners`](../src/opengl_decimate/corners.py), so that they
replay as a prefix too.

Replaying a prefix of length *k* is four array operations:

- **Which point became which.** `parent` is filled from the prefix in one
  assignment — each point dies at most once — then pointer-jumped until it stops
  changing.
- **Where each survivor sits.** The placement of the last contraction in the
  prefix that it survived, gathered with one scatter-maximum.
- **Which faces are left.** Each face records the step that removed it, so the
  live set is a comparison.
- **Which vertex each corner reads.** The handovers are filled in from the
  prefix the same way, then pointer-jumped: a corner handed to a copy whose own
  end is given up later is handed on again.

So a target costs a replay rather than a reduction — a few milliseconds against
the tens the reduction itself took — and the cost does not grow with how far
along the sequence the target is. What it does scale with is the mesh handed
back, since step 7 has to assemble it: asking for twenty thousand triangles
costs a good deal more than asking for twenty. Fifty asks cost fifty replays,
which is still a fraction of one reduction.

## 7. Hand it back

An output vertex is a distinct combination of *point* and *attributes*: two
corners at the same point carrying the same normal and texture coordinate are one
vertex, two carrying different ones are two. That is what keeps a seam a seam
without any seam-specific machinery.

Vertices are numbered by first use in the index stream. `vertex_map` relates the
input's vertices to the output's.

**Normals, where they are recomputed.** `recompute_normals` is what a level a
game ships wants. A carried normal is the one measured at its own vertex and is
the more accurate of the two — on the marble bust at 8,000 triangles it sits 1.9
degrees from the source surface against 5.6 for a recomputed one — but its error
is uncorrelated between neighbours, and shading shows the *gradient*. So coarse
levels facet: the three corners of a triangle disagree by 51 degrees on a cliff
scan at 8,000 where the source's own triangles disagree by 15. Recomputed, they
disagree by 23, which is the surface.

They are accumulated per **smoothing group**, not per point. Per point is too
coarse — a hard edge is one position the model draws twice, once per side, and
averaging over it turns a cube into a ball. Per output *vertex* is too fine —
an output vertex is split by attributes, so a texture seam carrying no geometric
meaning would come back as a crease. The group is between the two: corners
joined across the interior edges the model is smooth across, labelled by
connected component, which walks each point's fan and stops where the fan is
hard. Where the input carries normals they say where that is; where it carries
none, the fold between two faces does. `crease_angle` is the threshold, and the
group is part of what makes a vertex distinct, because an edge needs a vertex
per side to be hard at all.

## 8. Measure what happened

The quadric's own number is an area-weighted root-mean-square distance to the
planes the surface has been through — see [step 3](#3-price-every-candidate).
What a level of detail has to promise is a bound on the *surface*: a maximum
rather than a mean, over the triangles rather than over their planes. Those are
two different quantities, and the second has to be measured.

[`certify.surface_deviation`](../src/opengl_decimate/certify.py) samples both
surfaces in proportion to area and asks each sample how far it is from the other,
in both directions. One direction alone cannot see a hole: every point of a mesh
with a piece missing is still on the original.

The point-to-triangle distance underneath it is exact. The plane is divided into
the seven regions a nearest point can fall in — the face, three edges, three
corners — and every region is evaluated for every pair with the right one
selected, which makes it one pass over arrays rather than a branch per triangle.

**Finding the triangle to measure against.** Testing every sample against every
triangle is the product of the two counts, which on a scan is billions. So the
triangles go into a uniform grid — each registered in every cell its bounding
box covers — and a sample is measured against the handful in the cells around
it, widening a ring at a time until the nearest found is no further away than
the ring searched. At that point no triangle outside the ring can beat it, so
the answer is exact rather than approximate. The widening stops where a ring
would cost more than measuring against every triangle, and the few samples still
unsettled are measured that way instead.

The same grid answers the other question that needs one: which points a
`weld_tolerance` merges. Both live in
[`opengl_decimate.spatial`](../src/opengl_decimate/spatial.py).

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
