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
* the queue as three parallel arrays with a serial number for ties, so an entry
  is twenty-four bytes rather than a tuple of boxed objects.

Everything around the loop -- welding, classification, accumulating the
quadrics, assembling the output -- stays in NumPy, where it is already whole-
array work. This takes the mesh as arrays, mutates them in place, and hands
back the log of what it did.

**Staleness without a second structure.** A contraction changes the price of
exactly the edges touching the survivor, and those are re-priced and re-queued
when it happens. So an entry popped later whose price no longer matches what it
was queued at has been superseded by one already in the queue, and is dropped.
No map from edge to version is needed, which is the other structure that would
not fit.
"""

import numpy as np

cimport numpy as cnp
from libc.math cimport fabs, sqrt
from libc.stdlib cimport free, malloc, realloc

cnp.import_array()

#: Below this a triangle has no meaningful normal. Matches ``collapse._TINY``.
cdef double TINY = 1e-30

#: Largest neighbourhood handled; a point on a sane surface has a handful of
#: faces, and the buffers are sized for far more than that. An enum so it is a
#: compile-time constant and can size a C array.
cdef enum:
    SCRATCH = 1024


cdef struct Heap:
    double* cost
    int* left
    int* right
    int* left_version
    int* right_version
    long* serial
    Py_ssize_t size
    Py_ssize_t capacity
    long counter


cdef int heap_init(Heap* heap, Py_ssize_t capacity) except -1:
    heap.cost = <double*> malloc(capacity * sizeof(double))
    heap.left = <int*> malloc(capacity * sizeof(int))
    heap.right = <int*> malloc(capacity * sizeof(int))
    heap.serial = <long*> malloc(capacity * sizeof(long))
    heap.left_version = <int*> malloc(capacity * sizeof(int))
    heap.right_version = <int*> malloc(capacity * sizeof(int))
    if not heap.cost or not heap.left or not heap.right or not heap.serial \
            or not heap.left_version or not heap.right_version:
        raise MemoryError('could not allocate the contraction queue')
    heap.size = 0
    heap.capacity = capacity
    heap.counter = 0
    return 0


cdef void heap_free(Heap* heap) noexcept:
    free(heap.cost)
    free(heap.left)
    free(heap.right)
    free(heap.serial)
    free(heap.left_version)
    free(heap.right_version)


cdef inline bint heap_before(Heap* heap, Py_ssize_t a, Py_ssize_t b) noexcept:
    """Cheaper first, and for equal prices the one queued first."""
    if heap.cost[a] != heap.cost[b]:
        return heap.cost[a] < heap.cost[b]
    return heap.serial[a] < heap.serial[b]


cdef inline void heap_swap(Heap* heap, Py_ssize_t a, Py_ssize_t b) noexcept:
    cdef double cost = heap.cost[a]
    cdef int left = heap.left[a], right = heap.right[a]
    cdef int left_version = heap.left_version[a], right_version = heap.right_version[a]
    cdef long serial = heap.serial[a]
    heap.cost[a] = heap.cost[b]; heap.left[a] = heap.left[b]
    heap.right[a] = heap.right[b]; heap.serial[a] = heap.serial[b]
    heap.left_version[a] = heap.left_version[b]
    heap.right_version[a] = heap.right_version[b]
    heap.cost[b] = cost; heap.left[b] = left
    heap.right[b] = right; heap.serial[b] = serial
    heap.left_version[b] = left_version; heap.right_version[b] = right_version


cdef int heap_push(
    Heap* heap, double cost, int left, int right, int left_version, int right_version
) except -1:
    cdef Py_ssize_t child, parent
    if heap.size == heap.capacity:
        heap.capacity *= 2
        heap.cost = <double*> realloc(heap.cost, heap.capacity * sizeof(double))
        heap.left = <int*> realloc(heap.left, heap.capacity * sizeof(int))
        heap.right = <int*> realloc(heap.right, heap.capacity * sizeof(int))
        heap.serial = <long*> realloc(heap.serial, heap.capacity * sizeof(long))
        heap.left_version = <int*> realloc(
            heap.left_version, heap.capacity * sizeof(int))
        heap.right_version = <int*> realloc(
            heap.right_version, heap.capacity * sizeof(int))
        if not heap.cost or not heap.left or not heap.right or not heap.serial \
                or not heap.left_version or not heap.right_version:
            raise MemoryError('could not grow the contraction queue')
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
) noexcept:
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

    cdef Py_ssize_t point_count
    cdef Py_ssize_t face_count
    cdef Py_ssize_t live_faces

    cdef int placement_mode
    cdef double max_flip_cos
    cdef double min_quality
    cdef double error_limit

    # Scratch, reused by every candidate rather than reallocated.
    cdef int* ring_a
    cdef int* ring_b
    cdef int* touched
    cdef int* on_edge

    # The log.
    cdef int* log_dying
    cdef int* log_surviving
    cdef double* log_place
    cdef double* log_deviation
    cdef Py_ssize_t log_size
    cdef Py_ssize_t log_capacity
    cdef long* removed_at

    def __cinit__(self):
        self.head = NULL
        self.nxt = NULL
        self.prv = NULL
        self.version = NULL
        self.ring_a = NULL
        self.ring_b = NULL
        self.touched = NULL
        self.on_edge = NULL
        self.log_dying = NULL
        self.log_surviving = NULL
        self.log_place = NULL
        self.log_deviation = NULL
        self.removed_at = NULL

    def __dealloc__(self):
        free(self.head); free(self.nxt); free(self.prv); free(self.version)
        free(self.ring_a); free(self.ring_b)
        free(self.touched); free(self.on_edge)
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
        self.removed_at = <long*> malloc(self.face_count * sizeof(long))
        self.version = <int*> malloc(self.point_count * sizeof(int))
        if not self.head or not self.nxt or not self.prv or not self.removed_at \
                or not self.version:
            raise MemoryError('could not allocate the adjacency')
        for i in range(self.point_count):
            self.head[i] = -1
            self.version[i] = 0
        for i in range(self.face_count):
            self.removed_at[i] = -1
            if alive[i]:
                self.live_faces += 1
                for slot in range(3):
                    self._attach(faces[i, slot], <int> (3 * i + slot))

        self.ring_a = <int*> malloc(SCRATCH * sizeof(int))
        self.ring_b = <int*> malloc(SCRATCH * sizeof(int))
        self.touched = <int*> malloc(SCRATCH * sizeof(int))
        self.on_edge = <int*> malloc(SCRATCH * sizeof(int))
        if not self.ring_a or not self.ring_b or not self.touched or not self.on_edge:
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

    cdef inline void _attach(self, int point, int incidence) noexcept:
        self.nxt[incidence] = self.head[point]
        self.prv[incidence] = -1
        if self.head[point] >= 0:
            self.prv[self.head[point]] = incidence
        self.head[point] = incidence

    cdef inline void _detach(self, int point, int incidence) noexcept:
        if self.prv[incidence] >= 0:
            self.nxt[self.prv[incidence]] = self.nxt[incidence]
        else:
            self.head[point] = self.nxt[incidence]
        if self.nxt[incidence] >= 0:
            self.prv[self.nxt[incidence]] = self.prv[incidence]

    cdef int _faces_on_edge(self, int a, int b) noexcept:
        """Fill ``on_edge`` with the live faces using both ends; return how many.

        ``-1`` where the neighbourhood is larger than the buffers hold, which
        makes the caller pass the candidate over. A point with a thousand faces
        on it is not a surface anybody should be contracting blind.
        """
        cdef int incidence = self.head[a], face, found = 0
        while incidence >= 0:
            face = incidence // 3
            if (self.faces[face, 0] == b or self.faces[face, 1] == b
                    or self.faces[face, 2] == b):
                if found >= SCRATCH:
                    return -1
                self.on_edge[found] = face
                found += 1
            incidence = self.nxt[incidence]
        return found

    cdef int _ring(self, int point, int* out) noexcept:
        """Fill ``out`` with the points joined to ``point``; return how many.

        Duplicates are removed by a linear scan, which is the right shape here:
        a point has a handful of neighbours, and sorting them would cost more
        than comparing them.
        """
        cdef int incidence = self.head[point], face, slot, other, i
        cdef int found = 0
        cdef bint seen
        while incidence >= 0:
            face = incidence // 3
            for slot in range(3):
                other = self.faces[face, slot]
                if other == point:
                    continue
                seen = False
                for i in range(found):
                    if out[i] == other:
                        seen = True
                        break
                if not seen:
                    if found >= SCRATCH:
                        return -1
                    out[found] = other
                    found += 1
            incidence = self.nxt[incidence]
        return found

    cdef bint _link_condition(self, int a, int b, int edge_faces) noexcept:
        """The points joined to both ends must be exactly those opposite the edge."""
        cdef int count_a = self._ring(a, self.ring_a)
        cdef int count_b = self._ring(b, self.ring_b)
        cdef int i, j, shared = 0
        cdef int opposite = 0, face, slot, point, k
        cdef bint counted

        if count_a < 0 or count_b < 0:
            return False

        for i in range(count_a):
            for j in range(count_b):
                if self.ring_a[i] == self.ring_b[j]:
                    shared += 1
                    break
        # Distinct points opposite the edge. Two faces sharing one is a pillow,
        # and leaving it to the count is what refuses it.
        for i in range(edge_faces):
            face = self.on_edge[i]
            for slot in range(3):
                point = self.faces[face, slot]
                if point == a or point == b:
                    continue
                counted = False
                for k in range(opposite):
                    if self.ring_a[SCRATCH // 2 + k] == point:
                        counted = True
                        break
                if not counted and opposite < SCRATCH // 2:
                    self.ring_a[SCRATCH // 2 + opposite] = point
                    opposite += 1
        return shared == opposite

    cdef int _affected(self, int a, int b, int edge_faces) noexcept:
        """Fill ``touched`` with the faces that outlive a contraction of (a, b)."""
        cdef int incidence, face, i, found = 0
        cdef bint on
        incidence = self.head[a]
        while incidence >= 0:
            face = incidence // 3
            on = False
            for i in range(edge_faces):
                if self.on_edge[i] == face:
                    on = True
                    break
            if not on:
                if found >= SCRATCH:
                    return -1
                self.touched[found] = face
                found += 1
            incidence = self.nxt[incidence]
        incidence = self.head[b]
        while incidence >= 0:
            face = incidence // 3
            on = False
            for i in range(edge_faces):
                if self.on_edge[i] == face:
                    on = True
                    break
            if not on:
                if found >= SCRATCH:
                    return -1
                self.touched[found] = face
                found += 1
            incidence = self.nxt[incidence]
        return found

    cdef bint _would_duplicate(self, int a, int b, int count) noexcept:
        """Two surviving faces landing on the same three points."""
        cdef int i, j, fi, fj, s
        cdef int pi[3]
        cdef int pj[3]
        for i in range(count):
            fi = self.touched[i]
            for s in range(3):
                pi[s] = b if self.faces[fi, s] == a else self.faces[fi, s]
            _sort3(pi)
            for j in range(i + 1, count):
                fj = self.touched[j]
                for s in range(3):
                    pj[s] = b if self.faces[fj, s] == a else self.faces[fj, s]
                _sort3(pj)
                if pi[0] == pj[0] and pi[1] == pj[1] and pi[2] == pj[2]:
                    return True
        return False

    cdef bint _would_distort(self, int a, int b, double* place, int count) noexcept:
        """A face turned past the limit, thinned past the floor, or vanished."""
        cdef int i, s, face, point
        cdef double before[3][3]
        cdef double after[3][3]
        cdef double old[3]
        cdef double new[3]
        cdef double old_len, new_len, cosine, quality
        for i in range(count):
            face = self.touched[i]
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

    cdef double _price(self, int a, int b, double* place) noexcept:
        """The deviation contracting (a, b) would cost, and where it would land."""
        cdef double q[10]
        cdef int i
        cdef double best = 0.0, cost
        cdef double candidate[3]
        cdef double optimum[3]
        cdef bint have_best = False, solved
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
        return sqrt(best / weight)

    cdef int _record(self, int dying, int surviving, double* place, double deviation) except -1:
        if self.log_size == self.log_capacity:
            self.log_capacity *= 2
            self.log_dying = <int*> realloc(self.log_dying, self.log_capacity * sizeof(int))
            self.log_surviving = <int*> realloc(
                self.log_surviving, self.log_capacity * sizeof(int))
            self.log_place = <double*> realloc(
                self.log_place, 3 * self.log_capacity * sizeof(double))
            self.log_deviation = <double*> realloc(
                self.log_deviation, self.log_capacity * sizeof(double))
            if not self.log_dying or not self.log_surviving or not self.log_place \
                    or not self.log_deviation:
                raise MemoryError('could not grow the contraction log')
        self.log_dying[self.log_size] = dying
        self.log_surviving[self.log_size] = surviving
        self.log_place[3 * self.log_size + 0] = place[0]
        self.log_place[3 * self.log_size + 1] = place[1]
        self.log_place[3 * self.log_size + 2] = place[2]
        self.log_deviation[self.log_size] = deviation
        self.log_size += 1
        return 0

    cdef void _contract(self, int dying, int surviving, double* place, int edge_faces) noexcept:
        cdef int i, slot, incidence, face, point
        cdef long step = <long> (self.log_size - 1)
        cdef int moving[SCRATCH]
        cdef int moving_count = 0
        # Bounded by construction: the candidate was refused above unless every
        # buffer held its neighbourhood.

        for i in range(edge_faces):
            face = self.on_edge[i]
            if not self.alive[face]:
                continue
            self.alive[face] = 0
            self.live_faces -= 1
            self.removed_at[face] = step
            for slot in range(3):
                self._detach(self.faces[face, slot], <int> (3 * face + slot))

        incidence = self.head[dying]
        while incidence >= 0:
            if moving_count < SCRATCH:
                moving[moving_count] = incidence
                moving_count += 1
            incidence = self.nxt[incidence]
        for i in range(moving_count):
            incidence = moving[i]
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

    def run(self, int[:, ::1] edges, Py_ssize_t target_faces):
        """Contract until ``target_faces`` is reached or nothing is left to do."""
        cdef Heap heap
        cdef double cost, fresh
        cdef int a, b, dying, surviving, edge_faces, affected, i, other
        cdef int queued_a, queued_b
        cdef double place[3]
        cdef double current[3]
        cdef int ring[SCRATCH]
        cdef int ring_count
        cdef bint can_a, can_b

        heap_init(&heap, 1024)
        try:
            self._queue_all(&heap, edges)
            while heap.size > 0 and self.live_faces > target_faces:
                heap_pop(&heap, &cost, &a, &b, &queued_a, &queued_b)
                if self.version[a] != queued_a or self.version[b] != queued_b:
                    continue
                if self.head[a] < 0 or self.head[b] < 0:
                    continue
                edge_faces = self._faces_on_edge(a, b)
                if edge_faces <= 0:
                    continue
                fresh = self._price(a, b, current)
                if fresh < 0.0:
                    continue
                if self.error_limit >= 0.0 and fresh > self.error_limit:
                    break

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
                if not self._link_condition(a, b, edge_faces):
                    continue
                affected = self._affected(a, b, edge_faces)
                if affected < 0:
                    continue
                if self._would_duplicate(dying, surviving, affected):
                    continue
                place[0] = current[0]; place[1] = current[1]; place[2] = current[2]
                if self._would_distort(dying, surviving, place, affected):
                    continue

                self._record(dying, surviving, place, fresh)
                self._contract(dying, surviving, place, edge_faces)

                ring_count = self._ring(surviving, ring)
                _sort_ints(ring, ring_count)
                for i in range(ring_count):
                    other = ring[i]
                    fresh = self._price(surviving, other, current)
                    if fresh >= 0.0:
                        if surviving < other:
                            heap_push(&heap, fresh, surviving, other,
                                      self.version[surviving], self.version[other])
                        else:
                            heap_push(&heap, fresh, other, surviving,
                                      self.version[other], self.version[surviving])
        finally:
            heap_free(&heap)
        return self._log()

    cdef int _queue_all(self, Heap* heap, int[:, ::1] edges) except -1:
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
                heap_push(heap, cost, edges[i, 0], edges[i, 1],
                          self.version[edges[i, 0]], self.version[edges[i, 1]])
        return 0

    cdef _log(self):
        cdef cnp.ndarray dying = np.empty(self.log_size, dtype=np.int64)
        cdef cnp.ndarray surviving = np.empty(self.log_size, dtype=np.int64)
        cdef cnp.ndarray place = np.empty((self.log_size, 3), dtype=np.float64)
        cdef cnp.ndarray deviation = np.empty(self.log_size, dtype=np.float64)
        cdef cnp.ndarray removed = np.empty(self.face_count, dtype=np.int64)
        cdef long[::1] out_dying = dying
        cdef long[::1] out_surviving = surviving
        cdef double[:, ::1] out_place = place
        cdef double[::1] out_deviation = deviation
        cdef long[::1] out_removed = removed
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


cdef inline void _sort_ints(int* values, int count) noexcept:
    """Insertion sort: a point has a handful of neighbours, never a heap of them."""
    cdef int i, j, held
    for i in range(1, count):
        held = values[i]
        j = i - 1
        while j >= 0 and values[j] > held:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = held


cdef inline void _sort3(int* values) noexcept:
    cdef int swap
    if values[0] > values[1]:
        swap = values[0]; values[0] = values[1]; values[1] = swap
    if values[1] > values[2]:
        swap = values[1]; values[1] = values[2]; values[2] = swap
    if values[0] > values[1]:
        swap = values[0]; values[0] = values[1]; values[1] = swap


cdef inline void _normal(double corners[3][3], double* out) noexcept:
    cdef double ax = corners[1][0] - corners[0][0]
    cdef double ay = corners[1][1] - corners[0][1]
    cdef double az = corners[1][2] - corners[0][2]
    cdef double bx = corners[2][0] - corners[0][0]
    cdef double by = corners[2][1] - corners[0][1]
    cdef double bz = corners[2][2] - corners[0][2]
    out[0] = ay * bz - az * by
    out[1] = az * bx - ax * bz
    out[2] = ax * by - ay * bx


cdef inline double _quality(double corners[3][3]) noexcept:
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


cdef inline double _evaluate(double* q, double* p) noexcept:
    """The cost a packed quadric assigns to a point."""
    return (
        q[0] * p[0] * p[0] + q[4] * p[1] * p[1] + q[7] * p[2] * p[2]
        + 2.0 * (q[1] * p[0] * p[1] + q[2] * p[0] * p[2] + q[5] * p[1] * p[2])
        + 2.0 * (q[3] * p[0] + q[6] * p[1] + q[8] * p[2])
        + q[9]
    )


cdef inline bint _minimise(double* q, double* out) noexcept:
    """Solve for the quadric's minimum; False where it has no single one."""
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
    cdef double magnitude = fabs(a00)
    if fabs(a01) > magnitude: magnitude = fabs(a01)
    if fabs(a02) > magnitude: magnitude = fabs(a02)
    if fabs(a11) > magnitude: magnitude = fabs(a11)
    if fabs(a12) > magnitude: magnitude = fabs(a12)
    if fabs(a22) > magnitude: magnitude = fabs(a22)
    if fabs(determinant) <= 1e-10 * magnitude * magnitude * magnitude:
        return False
    out[0] = (c00 * b0 + c01 * b1 + c02 * b2) / determinant
    out[1] = (c01 * b0 + c11 * b1 + c12 * b2) / determinant
    out[2] = (c02 * b0 + c12 * b1 + c22 * b2) / determinant
    return True
