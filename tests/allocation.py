"""The peak memory a block of code allocates, for tests that hold a budget.

``tracemalloc`` is CPython's: PyPy ships the module without the ``_tracemalloc``
it imports, so a test that measures an allocation is skipped there.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

try:
    import tracemalloc
except ImportError:
    tracemalloc = None  # type: ignore[assignment]  # absent on PyPy


class Peak:
    """The most a block had allocated at once, in bytes."""

    bytes = 0


@contextmanager
def traced():
    """Yield a `Peak` that holds the block's peak allocation once it exits."""
    if tracemalloc is None:
        pytest.skip('this interpreter has no tracemalloc to measure allocation with')
    peak = Peak()
    tracemalloc.start()
    try:
        yield peak
        peak.bytes = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
