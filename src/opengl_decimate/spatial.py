"""A uniform grid of cells, for the two questions that need to look nearby.

Welding asks which points lie within a tolerance of which. Certifying asks how
far a sample is from the nearest triangle. Both come down to the same thing:
bucket items by integer cell coordinate, then walk the cells around a query
rather than the whole model. Doing that with a dictionary keyed by cell is what
makes either cost the size of the model in Python-level work, so the cells are
sorted once and looking one up is a ``searchsorted``.

:class:`CellGrid` holds the buckets and hands back, for a batch of query cells,
every item in each -- in blocks, because the count is the product of two
occupancies and a caller who picks a coarse cell offers a great many.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from opengl_decimate.types import IndexArray

__all__ = ['CellGrid', 'NEIGHBOURHOOD', 'shell', 'shell_size']

#: A cell coordinate as one comparable value, for a grid too large to pack into
#: an integer. Structured rows compare lexicographically, and unlike a packed
#: key there is no product of the grid's spans to overflow.
_CELL = np.dtype([('x', 'i8'), ('y', 'i8'), ('z', 'i8')])

#: Most ``(query, item)`` pairs handed back at once. What comes back is the same
#: whatever this is; it only bounds the memory one batch asks for.
BLOCK = 1 << 22

#: The twenty-seven cells touching a cell, including itself.
NEIGHBOURHOOD = tuple((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1))


def shell_size(radius: int) -> int:
    """How many cells :func:`shell` would name, without naming them.

    A widening search has to decide whether the next ring is worth walking
    before it walks it, and past a few cells the ring is large.

    >>> [shell_size(radius) for radius in range(4)]
    [1, 26, 98, 218]
    """
    return 1 if radius < 1 else (2 * radius + 1) ** 3 - (2 * radius - 1) ** 3


def shell(radius: int) -> tuple[tuple[int, int, int], ...]:
    """The cell offsets exactly ``radius`` cells away, as a hollow box.

    A search that widens by one cell at a time needs only the cells it has not
    already looked in, which is the surface of the box rather than its volume.

    >>> len(shell(0)), len(shell(1)), len(shell(2))
    (1, 26, 98)
    """
    span = range(-radius, radius + 1)
    return tuple(
        (x, y, z) for x in span for y in span for z in span if max(abs(x), abs(y), abs(z)) == radius
    )


class CellGrid:
    """Items bucketed by cell, with the members of any cell in one gather.

    ``cells`` is one integer coordinate triple per item; an item may be listed
    more than once, which is how a triangle spanning several cells is registered
    in each of them.

    A cell coordinate is turned into a single comparable value so that finding a
    cell is a ``searchsorted`` over an array. Where the grid is small enough to
    pack three coordinates into an ``int64`` without overflowing, that is what
    they are packed into -- an integer ``searchsorted`` is around thirty times
    faster than comparing structured rows, and a query asks for twenty-seven
    cells per item. Where it is not, the rows are compared as rows, which is
    slower and always right.

    >>> import numpy as np
    >>> grid = CellGrid(np.array([[0, 0, 0], [0, 0, 0], [4, 1, 1]]))
    >>> [(q.tolist(), m.tolist()) for q, m in grid.members(np.array([[0, 0, 0]]))]
    [([0, 0], [0, 1])]
    """

    def __init__(self, cells: IndexArray) -> None:
        cells = np.ascontiguousarray(cells, dtype=np.int64)
        # One cell of margin on each side, so a neighbour query is still inside
        # the grid and a packed key is still non-negative.
        self._low = cells.min(axis=0) - 1
        span = [int(reach) + 2 for reach in cells.max(axis=0) - self._low]
        plane = span[1] * span[2]
        #: How to pack a cell coordinate into one integer, or ``None`` where the
        #: grid is too big for the packing to fit.
        self._strides: IndexArray | None = (
            np.asarray([plane, span[2], 1], dtype=np.int64)
            if plane * span[0] <= np.iinfo(np.int64).max
            else None
        )

        keys = self.key(cells)
        self.order = np.argsort(keys, kind='stable')
        ordered = keys[self.order]
        opens = np.empty(len(ordered), dtype=bool)
        opens[0] = True
        opens[1:] = ordered[1:] != ordered[:-1]
        self._starts = np.flatnonzero(opens)
        self._occupied = ordered[self._starts]
        self._sizes = np.diff(np.append(self._starts, len(ordered)))

    def key(self, cells: IndexArray) -> np.ndarray:
        """One comparable value per cell coordinate."""
        strides = self._strides
        if strides is None:
            return np.ascontiguousarray(cells, dtype=np.int64).view(_CELL).ravel()
        return (cells - self._low) @ strides

    def members(
        self, cells: IndexArray, block: int = BLOCK
    ) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Every item in the cell each row of ``cells`` names.

        Yields ``(query, item)`` index pairs -- ``query`` into ``cells``, ``item``
        into the grid's own items -- in batches of at most ``block`` pairs. Rows
        naming an empty cell contribute nothing.
        """
        keys = self.key(cells)
        slot = np.searchsorted(self._occupied, keys)
        within = np.minimum(slot, len(self._occupied) - 1)
        asked = np.flatnonzero((slot < len(self._occupied)) & (self._occupied[within] == keys))
        if not len(asked):
            return
        cell_of = slot[asked]
        starts, sizes = self._starts[cell_of], self._sizes[cell_of]
        running = np.cumsum(sizes)
        cut = 0
        while cut < len(asked):
            end = max(
                cut + 1,
                int(np.searchsorted(running, running[cut] - sizes[cut] + block, side='right')),
            )
            per = sizes[cut:end]
            # The concatenated ``arange(0, n)`` of each cell, without a loop: a
            # ramp over the whole batch, less the running start of the cell each
            # entry belongs to.
            ramp = np.arange(int(per.sum()), dtype=np.int64) - np.repeat(np.cumsum(per) - per, per)
            yield (
                np.repeat(asked[cut:end], per),
                self.order[np.repeat(starts[cut:end], per) + ramp],
            )
            cut = end
