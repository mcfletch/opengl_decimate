# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""The contraction loop, compiled.

The loop itself is what stops a scan being reducible in Python, and it stops it
twice over. It is slow -- a contraction is a dozen tiny array operations whose
overhead dwarfs their arithmetic -- and it is enormous, because the structures
it needs are Python objects: a set of faces per point is half a gigabyte before
a 1.5M-triangle mesh has been touched, and a priority queue of tuples holding
arrays is as much again.

Neither is inherent. Here the same algorithm keeps:

* the faces on a point as a doubly-linked list over a fixed pool of ``3F``
  incidences, so a point's faces are walked without allocating and a
  contraction moves them between lists in constant time;
* the queue as parallel arrays with a serial number for ties, so an entry is a
  few dozen bytes rather than a tuple of boxed objects.

Everything around the loop -- welding, classification, accumulating the
quadrics, assembling the output -- stays in NumPy, where it is already whole-
array work. This takes the mesh as arrays, mutates them in place, and hands
back the log of what it did. The loop runs with the GIL released, so a
reduction on a worker thread leaves the rest of the process at full speed.

Staleness is tracked without a map from edge to version. A contraction changes
the price of exactly the edges touching the survivor, and those are re-priced and
re-queued when it happens. Rather than a version per *edge* -- a hash map the
size of the queue, which is the structure that would not fit -- each *point*
carries one, bumped when its quadric changes, and every queued entry names the
two versions it was priced against. A pop whose versions have moved on has been
superseded by an entry already in the queue, and is dropped.

The NumPy path in :mod:`opengl_decimate.reduction` reaches the same order with a
stamp per pair, which it can afford because it is not the path a scan goes
through. The two agree because a pair is stale under one exactly when it is
stale under the other: both are bumped by the same event, a contraction at one
of the pair's ends.
"""

import numpy as np

cimport numpy as cnp
from libc.math cimport fabs, isfinite, sqrt
from libc.limits cimport INT_MAX
from libc.stdlib cimport free, malloc, qsort, realloc

cnp.import_array()

#: Below this a triangle has no meaningful normal. Matches ``collapse._TINY``.
cdef double TINY = 1e-30

# Anything counted here, and anything that has to bind to a NumPy ``int64``, is
# ``cnp.int64_t`` rather than C ``long``. Windows is LLP64, where ``long`` is
# thirty-two bits: a memoryview declared ``long[::1]`` carries a type descriptor
# holding ``sizeof(long)``, and acquiring an ``int64`` buffer through it raises
# ``Buffer dtype mismatch``. The counters would also wrap at two billion, which
# a mesh large enough to need this reducer can reach.

#: What a working buffer starts at. A point on an ordinary surface has a
#: handful of faces, so this is already far more than most neighbourhoods need;
#: a lathe pole or a fan-triangulated n-gon has as many as it has, and the
#: buffer grows to meet it.
cdef enum:
    SCRATCH = 64


# A growable run of ints, reused by every candidate rather than reallocated.
cdef struct IntPool:
    int* data
    Py_ssize_t capacity


cdef int pool_reserve(IntPool* pool, Py_ssize_t wanted) noexcept nogil:
    """Make room for ``wanted`` ints. 0 on success, -1 where there is no memory.

    ``realloc`` returns NULL without freeing what it was asked to grow, so the
    result is only adopted once it is known to be a block -- otherwise a failure
    would lose the pool as well as the growth.
    """
    cdef Py_ssize_t size = pool.capacity
    cdef int* grown
    if wanted <= pool.capacity:
        return 0
    if size < 1:
        size = 1
    while size < wanted:
        size *= 2
    grown = <int*> realloc(pool.data, size * sizeof(int))
    if grown == NULL:
        return -1
    pool.data = grown
    pool.capacity = size
    return 0


cdef inline int pool_append(IntPool* pool, Py_ssize_t at, int value) noexcept nogil:
    """Write ``value`` at ``at``, growing first. 0 on success, -1 out of memory."""
    if at >= pool.capacity and pool_reserve(pool, at + 1) < 0:
        return -1
    pool.data[at] = value
    return 0


cdef struct Heap:
    double* cost
    int* left
    int* right
    int* left_version
    int* right_version
    cnp.int64_t* serial
    Py_ssize_t size
    Py_ssize_t capacity
    cnp.int64_t counter


cdef int heap_init(Heap* heap, Py_ssize_t capacity) except -1:
    heap.size = 0
    heap.capacity = capacity
    heap.counter = 0
    # Set before anything is allocated, so a partial failure is something
    # `heap_free` can be handed rather than a mix of blocks and stack rubbish.
    heap.cost = NULL
    heap.left = NULL
    heap.right = NULL
    heap.serial = NULL
    heap.left_version = NULL
    heap.right_version = NULL
    heap.cost = <double*> malloc(capacity * sizeof(double))
    heap.left = <int*> malloc(capacity * sizeof(int))
    heap.right = <int*> malloc(capacity * sizeof(int))
    heap.serial = <cnp.int64_t*> malloc(capacity * sizeof(cnp.int64_t))
    heap.left_version = <int*> malloc(capacity * sizeof(int))
    heap.right_version = <int*> malloc(capacity * sizeof(int))
    if not heap.cost or not heap.left or not heap.right or not heap.serial \
            or not heap.left_version or not heap.right_version:
        heap_free(heap)
        raise MemoryError('could not allocate the contraction queue')
    return 0


cdef void heap_free(Heap* heap) noexcept nogil:
    free(heap.cost)
    free(heap.left)
    free(heap.right)
    free(heap.serial)
    free(heap.left_version)
    free(heap.right_version)
    heap.cost = NULL
    heap.left = NULL
    heap.right = NULL
    heap.serial = NULL
    heap.left_version = NULL
    heap.right_version = NULL


cdef inline bint heap_before(Heap* heap, Py_ssize_t a, Py_ssize_t b) noexcept nogil:
    """Cheaper first, and for equal prices the one queued first."""
    if heap.cost[a] != heap.cost[b]:
        return heap.cost[a] < heap.cost[b]
    return heap.serial[a] < heap.serial[b]


cdef inline void heap_swap(Heap* heap, Py_ssize_t a, Py_ssize_t b) noexcept nogil:
    cdef double cost = heap.cost[a]
    cdef int left = heap.left[a], right = heap.right[a]
    cdef int left_version = heap.left_version[a], right_version = heap.right_version[a]
    cdef cnp.int64_t serial = heap.serial[a]
    heap.cost[a] = heap.cost[b]; heap.left[a] = heap.left[b]
    heap.right[a] = heap.right[b]; heap.serial[a] = heap.serial[b]
    heap.left_version[a] = heap.left_version[b]
    heap.right_version[a] = heap.right_version[b]
    heap.cost[b] = cost; heap.left[b] = left
    heap.right[b] = right; heap.serial[b] = serial
    heap.left_version[b] = left_version; heap.right_version[b] = right_version


cdef int heap_grow(Heap* heap) noexcept nogil:
    """Double the queue. 0 on success, -1 where there is no memory.

    Each block is grown through a temporary and only adopted once it is one: a
    failed ``realloc`` returns NULL and leaves the original allocated, so
    assigning the result straight back would lose the queue it was growing.
    """
    cdef Py_ssize_t wanted = heap.capacity * 2
    cdef double* cost
    cdef int* pointer
    cdef cnp.int64_t* serial
    cost = <double*> realloc(heap.cost, wanted * sizeof(double))
    if cost == NULL:
        return -1
    heap.cost = cost
    pointer = <int*> realloc(heap.left, wanted * sizeof(int))
    if pointer == NULL:
        return -1
    heap.left = pointer
    pointer = <int*> realloc(heap.right, wanted * sizeof(int))
    if pointer == NULL:
        return -1
    heap.right = pointer
    serial = <cnp.int64_t*> realloc(heap.serial, wanted * sizeof(cnp.int64_t))
    if serial == NULL:
        return -1
    heap.serial = serial
    pointer = <int*> realloc(heap.left_version, wanted * sizeof(int))
    if pointer == NULL:
        return -1
    heap.left_version = pointer
    pointer = <int*> realloc(heap.right_version, wanted * sizeof(int))
    if pointer == NULL:
        return -1
    heap.right_version = pointer
    heap.capacity = wanted
    return 0


cdef int heap_push(
    Heap* heap, double cost, int left, int right, int left_version, int right_version
) noexcept nogil:
    cdef Py_ssize_t child, parent
    if heap.size == heap.capacity and heap_grow(heap) < 0:
        return -1
    child = heap.size
    heap.cost[child] = cost
    heap.left[child] = left
    heap.right[child] = right
    heap.left_version[child] = left_version
    heap.right_version[child] = right_version
    heap.serial[child] = heap.counter
    heap.counter += 1
    heap.size += 1
    while child > 0:
        parent = (child - 1) // 2
        if heap_before(heap, child, parent):
            heap_swap(heap, child, parent)
            child = parent
        else:
            break
    return 0


cdef void heap_pop(
    Heap* heap, double* cost, int* left, int* right, int* left_version, int* right_version
) noexcept nogil:
    cdef Py_ssize_t parent = 0, child
    cost[0] = heap.cost[0]
    left[0] = heap.left[0]
    right[0] = heap.right[0]
    left_version[0] = heap.left_version[0]
    right_version[0] = heap.right_version[0]
    heap.size -= 1
    if heap.size > 0:
        heap.cost[0] = heap.cost[heap.size]
        heap.left[0] = heap.left[heap.size]
        heap.right[0] = heap.right[heap.size]
        heap.left_version[0] = heap.left_version[heap.size]
        heap.right_version[0] = heap.right_version[heap.size]
        heap.serial[0] = heap.serial[heap.size]
    while True:
        child = 2 * parent + 1
        if child >= heap.size:
            break
        if child + 1 < heap.size and heap_before(heap, child + 1, child):
            child += 1
        if heap_before(heap, child, parent):
            heap_swap(heap, child, parent)
            parent = child
        else:
            break


cdef class Reducer:
    """One mesh, reduced in place, with the log of every contraction."""

    cdef double[:, ::1] positions
    cdef int[:, ::1] faces
    cdef unsigned char[::1] alive
    cdef double[:, ::1] quadrics
    cdef double[::1] weights
    cdef signed char[::1] kinds
    cdef int[::1] copies
    cdef int[::1] charts

    # Faces on a point: `head[v]` is an incidence, `nxt`/`prv` chain them. An
    # incidence is `face * 3 + slot`, so the point it belongs to is always
    # `faces[face][slot]` and moving a face between points is two list splices.
    cdef int* head
    cdef int* nxt
    cdef int* prv
    # Bumped whenever a point's quadric changes, which is whenever it survives a
    # contraction. A queued entry naming a version that has moved on has been
    # superseded by the one queued at the time it moved, and is passed over --
    # which is what a map from edge to version would otherwise be needed for.
    cdef int* version
    # Marks a point as met by the current walk: `stamp[p] == epoch`. Starting a
    # walk is one increment, so deduplicating a neighbourhood or intersecting
    # two costs what the neighbourhoods hold rather than their product.
    cdef int* stamp
    cdef int epoch

    cdef Py_ssize_t point_count
    cdef Py_ssize_t face_count
    cdef Py_ssize_t live_faces

    cdef int placement_mode
    cdef double max_flip_cos
    cdef double min_quality
    cdef double error_limit

    # Scratch, reused by every candidate rather than reallocated. Each grows to
    # whatever neighbourhood it is handed: a fan-triangulated n-gon, a lathe
    # pole or a CAD hub has a point of any valence at all, and a fixed ceiling
    # would mean passing those candidates over -- a different reduction from the
    # NumPy path's, arrived at silently.
    cdef IntPool ring_a
    cdef IntPool ring_b
    cdef IntPool opposite
    cdef IntPool touched
    cdef IntPool on_edge
    cdef IntPool moving
    cdef IntPool ring
    cdef IntPool pairs

    # The log.
    cdef int* log_dying
    cdef int* log_surviving
    cdef double* log_place
    cdef double* log_deviation
    cdef Py_ssize_t log_size
    cdef Py_ssize_t log_capacity
    cdef cnp.int64_t* removed_at

    def __cinit__(self):
        self.head = NULL
        self.nxt = NULL
        self.prv = NULL
        self.version = NULL
        self.ring_a.data = NULL; self.ring_a.capacity = 0
        self.ring_b.data = NULL; self.ring_b.capacity = 0
        self.opposite.data = NULL; self.opposite.capacity = 0
        self.touched.data = NULL; self.touched.capacity = 0
        self.on_edge.data = NULL; self.on_edge.capacity = 0
        self.moving.data = NULL; self.moving.capacity = 0
        self.ring.data = NULL; self.ring.capacity = 0
        self.pairs.data = NULL; self.pairs.capacity = 0
        self.stamp = NULL
        self.log_dying = NULL
        self.log_surviving = NULL
        self.log_place = NULL
        self.log_deviation = NULL
        self.removed_at = NULL

    def __dealloc__(self):
        free(self.head); free(self.nxt); free(self.prv); free(self.version)
        free(self.stamp); free(self.pairs.data)
        free(self.ring_a.data); free(self.ring_b.data); free(self.opposite.data)
        free(self.touched.data); free(self.on_edge.data)
        free(self.moving.data); free(self.ring.data)
        free(self.log_dying); free(self.log_surviving)
        free(self.log_place); free(self.log_deviation)
        free(self.removed_at)

    def __init__(
        self,
        double[:, ::1] positions,
        int[:, ::1] faces,
        unsigned char[::1] alive,
        double[:, ::1] quadrics,
        double[::1] weights,
        signed char[::1] kinds,
        int[::1] copies,
        int[::1] charts,
        int placement_mode,
        double max_flip_cos,
        double min_quality,
        double error_limit,
    ):
        cdef Py_ssize_t i, slot
        self.positions = positions
        self.faces = faces
        self.alive = alive
        self.quadrics = quadrics
        self.weights = weights
        self.kinds = kinds
        self.copies = copies
        self.charts = charts
        self.placement_mode = placement_mode
        self.max_flip_cos = max_flip_cos
        self.min_quality = min_quality
        self.error_limit = error_limit

        self.point_count = positions.shape[0]
        self.face_count = faces.shape[0]
        self.live_faces = 0

        self.head = <int*> malloc(self.point_count * sizeof(int))
        self.nxt = <int*> malloc(3 * self.face_count * sizeof(int))
        self.prv = <int*> malloc(3 * self.face_count * sizeof(int))
        self.removed_at = <cnp.int64_t*> malloc(self.face_count * sizeof(cnp.int64_t))
        self.version = <int*> malloc(self.point_count * sizeof(int))
        self.stamp = <int*> malloc(self.point_count * sizeof(int))
        if not self.head or not self.nxt or not self.prv or not self.removed_at \
                or not self.version or not self.stamp:
            raise MemoryError('could not allocate the adjacency')
        for i in range(self.point_count):
            self.head[i] = -1
            self.version[i] = 0
            self.stamp[i] = 0
        self.epoch = 0
        for i in range(self.face_count):
            self.removed_at[i] = -1
            if alive[i]:
                self.live_faces += 1
                for slot in range(3):
                    self._attach(faces[i, slot], <int> (3 * i + slot))

        if (pool_reserve(&self.ring_a, SCRATCH) < 0
                or pool_reserve(&self.ring_b, SCRATCH) < 0
                or pool_reserve(&self.opposite, SCRATCH) < 0
                or pool_reserve(&self.touched, SCRATCH) < 0
                or pool_reserve(&self.on_edge, SCRATCH) < 0
                or pool_reserve(&self.moving, SCRATCH) < 0
                or pool_reserve(&self.ring, SCRATCH) < 0
                or pool_reserve(&self.pairs, 2 * SCRATCH) < 0):
            raise MemoryError('could not allocate the working buffers')

        self.log_capacity = 1024
        self.log_dying = <int*> malloc(self.log_capacity * sizeof(int))
        self.log_surviving = <int*> malloc(self.log_capacity * sizeof(int))
        self.log_place = <double*> malloc(3 * self.log_capacity * sizeof(double))
        self.log_deviation = <double*> malloc(self.log_capacity * sizeof(double))
        if not self.log_dying or not self.log_surviving or not self.log_place \
                or not self.log_deviation:
            raise MemoryError('could not allocate the contraction log')
        self.log_size = 0

    cdef inline void _attach(self, int point, int incidence) noexcept nogil:
        self.nxt[incidence] = self.head[point]
        self.prv[incidence] = -1
        if self.head[point] >= 0:
            self.prv[self.head[point]] = incidence
        self.head[point] = incidence

    cdef inline void _detach(self, int point, int incidence) noexcept nogil:
        if self.prv[incidence] >= 0:
            self.nxt[self.prv[incidence]] = self.nxt[incidence]
        else:
            self.head[point] = self.nxt[incidence]
        if self.nxt[incidence] >= 0:
            self.prv[self.nxt[incidence]] = self.prv[incidence]

    cdef int _faces_on_edge(self, int a, int b) noexcept nogil:
        """Fill ``on_edge`` with the live faces using both ends; return how many.

        ``-1`` where the buffer could not be grown to hold them, which is out of
        memory and nothing else.
        """
        cdef int incidence = self.head[a], face, found = 0
        while incidence >= 0:
            face = incidence // 3
            if (self.faces[face, 0] == b or self.faces[face, 1] == b
                    or self.faces[face, 2] == b):
                if pool_append(&self.on_edge, found, face) < 0:
                    return -1
                found += 1
            incidence = self.nxt[incidence]
        return found

    cdef int _count_on_edge(self, int a, int b) noexcept nogil:
        """How many live faces use both ends, without disturbing ``on_edge``."""
        cdef int incidence = self.head[a], face, found = 0
        while incidence >= 0:
            face = incidence // 3
            if (self.faces[face, 0] == b or self.faces[face, 1] == b
                    or self.faces[face, 2] == b):
                found += 1
            incidence = self.nxt[incidence]
        return found

    cdef int _next_epoch(self) noexcept nogil:
        """Start a walk: a stamp no point carries yet."""
        cdef Py_ssize_t i
        if self.epoch == INT_MAX:
            for i in range(self.point_count):
                self.stamp[i] = 0
            self.epoch = 0
        self.epoch += 1
        return self.epoch

    cdef int _ring(self, int point, IntPool* out) noexcept nogil:
        """Fill ``out`` with the points joined to ``point``; return how many.

        Each point found is stamped with this walk's epoch, which is what
        removes the duplicates, and the stamps stay for the caller to test
        membership against until the next walk begins.
        """
        cdef int incidence = self.head[point], face, slot, other
        cdef int found = 0
        cdef int epoch = self._next_epoch()
        while incidence >= 0:
            face = incidence // 3
            for slot in range(3):
                other = self.faces[face, slot]
                if other == point or self.stamp[other] == epoch:
                    continue
                self.stamp[other] = epoch
                if pool_append(out, found, other) < 0:
                    return -1
                found += 1
            incidence = self.nxt[incidence]
        return found

    cdef int _link_condition(self, int a, int b, int edge_faces) noexcept nogil:
        """The points joined to both ends must be exactly those opposite the edge.

        1 where the contraction keeps the surface a surface, 0 where it does
        not, -1 where a buffer could not be grown.
        """
        cdef int count_a = self._ring(a, &self.ring_a)
        cdef int count_b = self._ring(b, &self.ring_b)
        cdef int i, shared = 0
        cdef int opposite = 0, face, slot, point, k
        cdef bint counted

        if count_a < 0 or count_b < 0:
            return -1

        # `ring_b` was walked last, so its points carry the current epoch.
        for i in range(count_a):
            if self.stamp[self.ring_a.data[i]] == self.epoch:
                shared += 1
        # Distinct points opposite the edge. Two faces sharing one is a pillow,
        # and leaving it to the count is what refuses it.
        for i in range(edge_faces):
            face = self.on_edge.data[i]
            for slot in range(3):
                point = self.faces[face, slot]
                if point == a or point == b:
                    continue
                counted = False
                for k in range(opposite):
                    if self.opposite.data[k] == point:
                        counted = True
                        break
                if not counted:
                    if pool_append(&self.opposite, opposite, point) < 0:
                        return -1
                    opposite += 1
        if shared != opposite:
            return 0
        # The border form, as in `collapse.link_condition`: a border edge whose
        # triangle has its other two edges on the border too is the last
        # triangle of its piece.
        if edge_faces == 1 and opposite == 1:
            point = self.opposite.data[0]
            if self._count_on_edge(a, point) == 1 and self._count_on_edge(b, point) == 1:
                return 0
        return 1

    cdef int _affected(self, int a, int b, int edge_faces) noexcept nogil:
        """Fill ``touched`` with the faces that outlive a contraction of (a, b)."""
        cdef int incidence, face, i, found = 0
        cdef int end
        cdef bint on
        for end in range(2):
            incidence = self.head[a] if end == 0 else self.head[b]
            while incidence >= 0:
                face = incidence // 3
                on = False
                for i in range(edge_faces):
                    if self.on_edge.data[i] == face:
                        on = True
                        break
                if not on:
                    if pool_append(&self.touched, found, face) < 0:
                        return -1
                    found += 1
                incidence = self.nxt[incidence]
        return found

    cdef int _would_duplicate(self, int a, int b, int count) noexcept nogil:
        """Two surviving faces landing on the same three points.

        Once ``a`` is ``b``, every face that outlives the contraction has ``b``
        as one corner, so it is named by its other two. Those pairs are sorted
        and a repeat is a duplicate face. 1 where there is one, 0 where not,
        -1 where the buffer could not be grown.
        """
        cdef int i, s, fi, point, low, high, found
        if pool_reserve(&self.pairs, 2 * count) < 0:
            return -1
        for i in range(count):
            fi = self.touched.data[i]
            found = 0
            low = high = -1
            for s in range(3):
                point = self.faces[fi, s]
                if point == a or point == b:
                    continue
                if found == 0:
                    low = point
                else:
                    high = point
                found += 1
            if low > high:
                low, high = high, low
            self.pairs.data[2 * i] = low
            self.pairs.data[2 * i + 1] = high
        qsort(self.pairs.data, count, 2 * sizeof(int), _pair_order)
        for i in range(1, count):
            if (self.pairs.data[2 * i] == self.pairs.data[2 * i - 2]
                    and self.pairs.data[2 * i + 1] == self.pairs.data[2 * i - 1]):
                return 1
        return 0

    cdef bint _would_distort(self, int a, int b, double* place, int count) noexcept nogil:
        """A face turned past the limit, thinned past the floor, or vanished."""
        cdef int i, s, face, point
        cdef double before[3][3]
        cdef double after[3][3]
        cdef double old[3]
        cdef double new[3]
        cdef double old_len, new_len, cosine, quality
        for i in range(count):
            face = self.touched.data[i]
            for s in range(3):
                point = self.faces[face, s]
                before[s][0] = self.positions[point, 0]
                before[s][1] = self.positions[point, 1]
                before[s][2] = self.positions[point, 2]
                if point == a or point == b:
                    after[s][0] = place[0]
                    after[s][1] = place[1]
                    after[s][2] = place[2]
                else:
                    after[s][0] = before[s][0]
                    after[s][1] = before[s][1]
                    after[s][2] = before[s][2]
            _normal(before, old)
            _normal(after, new)
            old_len = sqrt(old[0] * old[0] + old[1] * old[1] + old[2] * old[2])
            new_len = sqrt(new[0] * new[0] + new[1] * new[1] + new[2] * new[2])
            if new_len <= TINY:
                if old_len > TINY:
                    return True
            elif old_len > TINY:
                cosine = (old[0] * new[0] + old[1] * new[1] + old[2] * new[2]) \
                    / (old_len * new_len)
                if cosine < self.max_flip_cos:
                    return True
            if self.min_quality > 0.0:
                quality = _quality(after)
                if quality < self.min_quality:
                    return True
        return False

    cdef double _price(self, int a, int b, double* place) noexcept nogil:
        """The deviation contracting (a, b) would cost, and where it would land.

        -1 where nothing may be placed, or where the price is not finite: a
        quadric whose value overflows prices nothing, as the NumPy reducer's
        ``isfinite`` test has it.
        """
        cdef double q[10]
        cdef int i
        cdef double best = 0.0, cost
        cdef double candidate[3]
        cdef double optimum[3]
        cdef bint have_best = False, solved
        optimum[0] = 0.0; optimum[1] = 0.0; optimum[2] = 0.0
        cdef bint locked_a = self.kinds[a] == 2
        cdef bint locked_b = self.kinds[b] == 2

        if locked_a and locked_b:
            return -1.0
        for i in range(10):
            q[i] = self.quadrics[a, i] + self.quadrics[b, i]

        solved = _minimise(q, optimum)
        for i in range(4):
            if i == 0:
                candidate[0] = self.positions[a, 0]
                candidate[1] = self.positions[a, 1]
                candidate[2] = self.positions[a, 2]
                if locked_b:
                    continue
            elif i == 1:
                candidate[0] = self.positions[b, 0]
                candidate[1] = self.positions[b, 1]
                candidate[2] = self.positions[b, 2]
                if locked_a:
                    continue
            elif i == 2:
                if self.placement_mode != 0 or locked_a or locked_b:
                    continue
                candidate[0] = 0.5 * (self.positions[a, 0] + self.positions[b, 0])
                candidate[1] = 0.5 * (self.positions[a, 1] + self.positions[b, 1])
                candidate[2] = 0.5 * (self.positions[a, 2] + self.positions[b, 2])
            else:
                if not solved or self.placement_mode != 0 or locked_a or locked_b:
                    continue
                if not self._near_the_edge(a, b, optimum):
                    continue
                candidate[0] = optimum[0]
                candidate[1] = optimum[1]
                candidate[2] = optimum[2]
            cost = _evaluate(q, candidate)
            if not have_best or cost < best:
                best = cost
                have_best = True
                place[0] = candidate[0]
                place[1] = candidate[1]
                place[2] = candidate[2]
        if not have_best:
            return -1.0
        if best < 0.0:
            best = 0.0
        cdef double weight = self.weights[a] + self.weights[b]
        if weight < TINY:
            weight = TINY
        cdef double deviation = sqrt(best / weight)
        if not isfinite(deviation):
            return -1.0
        return deviation

    cdef bint _near_the_edge(self, int a, int b, double* point) noexcept nogil:
        """No further from the midpoint than the edge is long, as ``reduction._near_the_edge``."""
        cdef double away[3]
        cdef double edge[3]
        cdef int axis
        for axis in range(3):
            away[axis] = point[axis] - 0.5 * (self.positions[a, axis] + self.positions[b, axis])
            edge[axis] = self.positions[b, axis] - self.positions[a, axis]
        return (
            away[0] * away[0] + away[1] * away[1] + away[2] * away[2]
            <= edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]
        )

    cdef int _grow_log(self) noexcept nogil:
        """Double the log. 0 on success, -1 where there is no memory.

        As in :func:`heap_grow`, each block is adopted only once ``realloc`` has
        returned one, since a failure leaves the original allocated and losing
        the pointer would lose the log.
        """
        cdef Py_ssize_t wanted = self.log_capacity * 2
        cdef int* indices
        cdef double* values
        indices = <int*> realloc(self.log_dying, wanted * sizeof(int))
        if indices == NULL:
            return -1
        self.log_dying = indices
        indices = <int*> realloc(self.log_surviving, wanted * sizeof(int))
        if indices == NULL:
            return -1
        self.log_surviving = indices
        values = <double*> realloc(self.log_place, 3 * wanted * sizeof(double))
        if values == NULL:
            return -1
        self.log_place = values
        values = <double*> realloc(self.log_deviation, wanted * sizeof(double))
        if values == NULL:
            return -1
        self.log_deviation = values
        self.log_capacity = wanted
        return 0

    cdef int _record(self, int dying, int surviving, double* place, double deviation) noexcept nogil:
        if self.log_size == self.log_capacity and self._grow_log() < 0:
            return -1
        self.log_dying[self.log_size] = dying
        self.log_surviving[self.log_size] = surviving
        self.log_place[3 * self.log_size + 0] = place[0]
        self.log_place[3 * self.log_size + 1] = place[1]
        self.log_place[3 * self.log_size + 2] = place[2]
        self.log_deviation[self.log_size] = deviation
        self.log_size += 1
        return 0

    cdef int _contract(self, int dying, int surviving, double* place, int edge_faces) noexcept nogil:
        """Apply the contraction. 0 on success, -1 where there is no memory.

        The incidences to move are collected before any of them is moved: the
        walk is over the very list the move splices them out of.
        """
        cdef int i, slot, incidence, face, point
        cdef cnp.int64_t step = <cnp.int64_t> (self.log_size - 1)
        cdef int moving_count = 0

        for i in range(edge_faces):
            face = self.on_edge.data[i]
            if not self.alive[face]:
                continue
            self.alive[face] = 0
            self.live_faces -= 1
            self.removed_at[face] = step
            for slot in range(3):
                self._detach(self.faces[face, slot], <int> (3 * face + slot))

        incidence = self.head[dying]
        while incidence >= 0:
            if pool_append(&self.moving, moving_count, incidence) < 0:
                return -1
            moving_count += 1
            incidence = self.nxt[incidence]
        for i in range(moving_count):
            incidence = self.moving.data[i]
            face = incidence // 3
            slot = incidence % 3
            self._detach(dying, incidence)
            self.faces[face, slot] = surviving
            self._attach(surviving, incidence)

        self.positions[surviving, 0] = place[0]
        self.positions[surviving, 1] = place[1]
        self.positions[surviving, 2] = place[2]
        for i in range(10):
            self.quadrics[surviving, i] += self.quadrics[dying, i]
        self.weights[surviving] += self.weights[dying]
        self.version[surviving] += 1
        point = dying
        self.head[point] = -1
        return 0

    def run(self, int[:, ::1] edges, Py_ssize_t target_faces):
        """Contract until ``target_faces`` is reached or nothing is left to do.

        The loop itself runs with the GIL released. Everything it touches is a
        typed memoryview or a C pool, so a thread that decimates in the
        background leaves the rest of the process running at full speed --
        which is what a build tool or an editor needs from it.
        """
        cdef Heap heap
        cdef int failure

        # Seeding queues at most one entry per edge, so that is where it starts;
        # the entries re-queued after each contraction grow it from there.
        heap_init(&heap, max(SCRATCH, edges.shape[0]))
        try:
            with nogil:
                failure = self._loop(&heap, edges, target_faces)
        finally:
            heap_free(&heap)
        if failure < 0:
            raise MemoryError('the reducer ran out of memory')
        return self._log()

    cdef int _loop(self, Heap* heap, int[:, ::1] edges, Py_ssize_t target_faces) noexcept nogil:
        """The contraction loop. 0 when it finished, -1 out of memory."""
        cdef double cost, fresh
        cdef int a, b, dying, surviving, edge_faces, affected, i, other
        cdef int queued_a, queued_b, linked, duplicate
        cdef double place[3]
        cdef double current[3]
        cdef int ring_count
        cdef bint can_a, can_b

        if self._queue_all(heap, edges) < 0:
            return -1
        while heap.size > 0 and self.live_faces > target_faces:
            heap_pop(heap, &cost, &a, &b, &queued_a, &queued_b)
            if self.version[a] != queued_a or self.version[b] != queued_b:
                continue
            if self.head[a] < 0 or self.head[b] < 0:
                continue
            edge_faces = self._faces_on_edge(a, b)
            if edge_faces < 0:
                return -1
            if edge_faces == 0:
                continue
            fresh = self._price(a, b, current)
            if fresh < 0.0:
                continue
            if self.error_limit >= 0.0 and fresh > self.error_limit:
                break

            # A point drawn at several texture coordinates sits on a seam, and
            # merges only with a point drawn at the same number. Merging it
            # with a point inside a chart moves the seam off its own line --
            # whichever end dies, since it is the merged point's position that
            # changes -- and the coordinate it carries was measured where the
            # seam used to be.
            if self.copies[a] != self.copies[b]:
                continue
            # And only along a seam: an edge with a different chart on each
            # side. Across a chart, it would take one seam onto another.
            if self.copies[a] > 1 and (
                    edge_faces != 2
                    or self.charts[self.on_edge.data[0]] == self.charts[self.on_edge.data[1]]):
                continue
            can_a = self.kinds[a] != 2 and (
                self.kinds[a] != 1 or edge_faces == 1)
            can_b = self.kinds[b] != 2 and (
                self.kinds[b] != 1 or edge_faces == 1)
            if can_a:
                dying, surviving = a, b
            elif can_b:
                dying, surviving = b, a
            else:
                continue
            linked = self._link_condition(a, b, edge_faces)
            if linked < 0:
                return -1
            if linked == 0:
                continue
            affected = self._affected(a, b, edge_faces)
            if affected < 0:
                return -1
            duplicate = self._would_duplicate(dying, surviving, affected)
            if duplicate < 0:
                return -1
            if duplicate:
                continue
            place[0] = current[0]; place[1] = current[1]; place[2] = current[2]
            if self._would_distort(dying, surviving, place, affected):
                continue

            if self._record(dying, surviving, place, fresh) < 0:
                return -1
            if self._contract(dying, surviving, place, edge_faces) < 0:
                return -1

            ring_count = self._ring(surviving, &self.ring)
            if ring_count < 0:
                return -1
            _sort_ints(self.ring.data, ring_count)
            for i in range(ring_count):
                other = self.ring.data[i]
                fresh = self._price(surviving, other, current)
                if fresh >= 0.0:
                    if surviving < other:
                        if heap_push(heap, fresh, surviving, other,
                                     self.version[surviving], self.version[other]) < 0:
                            return -1
                    else:
                        if heap_push(heap, fresh, other, surviving,
                                     self.version[other], self.version[surviving]) < 0:
                            return -1
        return 0

    cdef int _queue_all(self, Heap* heap, int[:, ::1] edges) noexcept nogil:
        """Price the caller's edges, in the caller's order, into the queue.

        The order matters beyond tidiness: equal-priced edges are separated by
        the serial number they were queued with, and a flat region of a mesh is
        full of edges that cost exactly nothing. Seeding from the same list in
        the same order is what makes this reduction and the NumPy one the same
        reduction rather than merely similar ones.
        """
        cdef Py_ssize_t i
        cdef double cost
        cdef double place[3]
        for i in range(edges.shape[0]):
            cost = self._price(edges[i, 0], edges[i, 1], place)
            if cost >= 0.0:
                if heap_push(heap, cost, edges[i, 0], edges[i, 1],
                             self.version[edges[i, 0]], self.version[edges[i, 1]]) < 0:
                    return -1
        return 0

    cdef _log(self):
        cdef cnp.ndarray dying = np.empty(self.log_size, dtype=np.int64)
        cdef cnp.ndarray surviving = np.empty(self.log_size, dtype=np.int64)
        cdef cnp.ndarray place = np.empty((self.log_size, 3), dtype=np.float64)
        cdef cnp.ndarray deviation = np.empty(self.log_size, dtype=np.float64)
        cdef cnp.ndarray removed = np.empty(self.face_count, dtype=np.int64)
        cdef cnp.int64_t[::1] out_dying = dying
        cdef cnp.int64_t[::1] out_surviving = surviving
        cdef double[:, ::1] out_place = place
        cdef double[::1] out_deviation = deviation
        cdef cnp.int64_t[::1] out_removed = removed
        cdef Py_ssize_t i
        for i in range(self.log_size):
            out_dying[i] = self.log_dying[i]
            out_surviving[i] = self.log_surviving[i]
            out_place[i, 0] = self.log_place[3 * i + 0]
            out_place[i, 1] = self.log_place[3 * i + 1]
            out_place[i, 2] = self.log_place[3 * i + 2]
            out_deviation[i] = self.log_deviation[i]
        for i in range(self.face_count):
            out_removed[i] = self.removed_at[i]
        return dying, surviving, place, deviation, removed


def ends_given_up(
    double[:, ::1] points,
    double[:, ::1] placement,
    cnp.int64_t[::1] dying,
    cnp.int64_t[::1] surviving,
    cnp.int64_t[::1] drawn,
    Py_ssize_t point_count,
):
    """For each contraction, which end's copies are let go and which are kept.

    The loop of :func:`opengl_decimate.corners.ends_given_up_in_python`, with
    the same arithmetic in the same order: the end drawn at more texture
    coordinates wins, and between equals the one whose copies were measured
    nearer the placement, the surviving end on a tie.
    """
    cdef Py_ssize_t steps = dying.shape[0], step, i
    loser_array = np.empty(steps, dtype=np.int64)
    winner_array = np.empty(steps, dtype=np.int64)
    sourced_array = np.arange(point_count, dtype=np.int64)
    cdef cnp.int64_t[::1] loser = loser_array
    cdef cnp.int64_t[::1] winner = winner_array
    cdef cnp.int64_t[::1] sourced = sourced_array
    cdef cnp.int64_t from_dying, from_surviving, keep, give_up, richer
    cdef double x, y, z, near, far
    cdef bint nearer
    with nogil:
        for step in range(steps):
            from_dying = sourced[dying[step]]
            from_surviving = sourced[surviving[step]]
            richer = drawn[from_dying] - drawn[from_surviving]
            if richer:
                nearer = richer > 0
            else:
                x = placement[step, 0]
                y = placement[step, 1]
                z = placement[step, 2]
                near = ((x - points[from_dying, 0]) * (x - points[from_dying, 0])
                        + (y - points[from_dying, 1]) * (y - points[from_dying, 1])
                        + (z - points[from_dying, 2]) * (z - points[from_dying, 2]))
                far = ((x - points[from_surviving, 0]) * (x - points[from_surviving, 0])
                       + (y - points[from_surviving, 1]) * (y - points[from_surviving, 1])
                       + (z - points[from_surviving, 2]) * (z - points[from_surviving, 2]))
                nearer = near < far
            if nearer:
                keep, give_up = from_dying, from_surviving
            else:
                keep, give_up = from_surviving, from_dying
            sourced[surviving[step]] = keep
            winner[step] = keep
            loser[step] = give_up
    return loser_array, winner_array


cdef int _int_order(const void* left, const void* right) noexcept nogil:
    cdef int a = (<const int*> left)[0], b = (<const int*> right)[0]
    return (a > b) - (a < b)


cdef int _pair_order(const void* left, const void* right) noexcept nogil:
    cdef const int* a = <const int*> left
    cdef const int* b = <const int*> right
    if a[0] != b[0]:
        return (a[0] > b[0]) - (a[0] < b[0])
    return (a[1] > b[1]) - (a[1] < b[1])


cdef inline void _sort_ints(int* values, int count) noexcept nogil:
    qsort(values, count, sizeof(int), _int_order)


cdef inline void _normal(double corners[3][3], double* out) noexcept nogil:
    cdef double ax = corners[1][0] - corners[0][0]
    cdef double ay = corners[1][1] - corners[0][1]
    cdef double az = corners[1][2] - corners[0][2]
    cdef double bx = corners[2][0] - corners[0][0]
    cdef double by = corners[2][1] - corners[0][1]
    cdef double bz = corners[2][2] - corners[0][2]
    out[0] = ay * bz - az * by
    out[1] = az * bx - ax * bz
    out[2] = ax * by - ay * bx


cdef inline double _quality(double corners[3][3]) noexcept nogil:
    """Four root three times the area over the summed squared edge lengths."""
    cdef double n[3]
    cdef double lengths = 0.0, delta
    cdef int i, j
    _normal(corners, n)
    cdef double area = 0.5 * sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    for i in range(3):
        j = (i + 1) % 3
        delta = corners[j][0] - corners[i][0]
        lengths += delta * delta
        delta = corners[j][1] - corners[i][1]
        lengths += delta * delta
        delta = corners[j][2] - corners[i][2]
        lengths += delta * delta
    if lengths <= TINY:
        return 0.0
    return 6.928203230275509 * area / lengths


cdef inline double _evaluate(double* q, double* p) noexcept nogil:
    """The cost a packed quadric assigns to a point."""
    return (
        q[0] * p[0] * p[0] + q[4] * p[1] * p[1] + q[7] * p[2] * p[2]
        + 2.0 * (q[1] * p[0] * p[1] + q[2] * p[0] * p[2] + q[5] * p[1] * p[2])
        + 2.0 * (q[3] * p[0] + q[6] * p[1] + q[8] * p[2])
        + q[9]
    )


cdef inline double _largest(
    double a, double b, double c, double d, double e, double f
) noexcept nogil:
    """The largest magnitude of six values."""
    cdef double most = fabs(a)
    if fabs(b) > most: most = fabs(b)
    if fabs(c) > most: most = fabs(c)
    if fabs(d) > most: most = fabs(d)
    if fabs(e) > most: most = fabs(e)
    if fabs(f) > most: most = fabs(f)
    return most


cdef inline bint _minimise(double* q, double* out) noexcept nogil:
    """Solve for the quadric's minimum; False where it has no single one.

    The same test as :func:`opengl_decimate.quadrics.minimize`: the reciprocal
    condition number, largest entry times largest cofactor over the
    determinant, has to exceed 1e-10.
    """
    cdef double a00 = q[0], a01 = q[1], a02 = q[2]
    cdef double a11 = q[4], a12 = q[5], a22 = q[7]
    cdef double b0 = -q[3], b1 = -q[6], b2 = -q[8]
    cdef double c00 = a11 * a22 - a12 * a12
    cdef double c01 = a02 * a12 - a01 * a22
    cdef double c02 = a01 * a12 - a02 * a11
    cdef double c11 = a00 * a22 - a02 * a02
    cdef double c12 = a01 * a02 - a00 * a12
    cdef double c22 = a00 * a11 - a01 * a01
    cdef double determinant = a00 * c00 + a01 * c01 + a02 * c02
    cdef double magnitude = _largest(a00, a01, a02, a11, a12, a22)
    cdef double cofactor = _largest(c00, c01, c02, c11, c12, c22)
    if fabs(determinant) <= 1e-10 * magnitude * cofactor:
        return False
    out[0] = (c00 * b0 + c01 * b1 + c02 * b2) / determinant
    out[1] = (c01 * b0 + c11 * b1 + c12 * b2) / determinant
    out[2] = (c02 * b0 + c12 * b1 + c22 * b2) / determinant
    return True
