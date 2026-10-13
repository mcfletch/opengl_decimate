"""The array shapes and names this package speaks.

The vocabulary is glTF's -- ``POSITION``, ``NORMAL``, ``TEXCOORD_0`` -- and the
container is a plain mapping of those names to NumPy arrays, so a mesh crosses
the boundary as arrays rather than as a file or an object graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    'FloatArray',
    'IndexArray',
    'AttributeMap',
    'POSITION',
    'DecimateError',
    'index_width',
]

#: A float array of any shape. NumPy's own generic alias carries no shape, so
#: the shape a parameter wants is said in its docstring.
FloatArray = np.ndarray[Any, np.dtype[np.floating[Any]]]
#: An integer array, used for triangle indices and for vertex remaps.
IndexArray = np.ndarray[Any, np.dtype[np.integer[Any]]]

#: glTF attribute semantic -> array. What a caller hands in and gets back.
AttributeMap = Mapping[str, np.ndarray]

#: The one attribute every mesh must carry.
POSITION = 'POSITION'


class DecimateError(ValueError):
    """The input does not describe a triangle mesh this package can work on."""


def index_width(integers: IndexArray) -> IndexArray:
    """``integers`` as ``numpy.intp``, the width of an index on this platform.

    ``numpy.repeat`` takes its counts and ``numpy.bincount`` its bins at that
    width and refuses a wider one, which the ``int64`` this package keeps its
    indices in is on a 32-bit platform. Where the two widths are the same the
    array is returned as it is.
    """
    return integers.astype(np.intp, copy=False)
