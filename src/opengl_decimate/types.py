"""The array shapes and names this package speaks.

The vocabulary is glTF's -- ``POSITION``, ``NORMAL``, ``TEXCOORD_0`` -- and the
container is a plain mapping of those names to NumPy arrays, so a mesh crosses
the boundary as arrays rather than as a file or an object graph.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np

__all__ = [
    'FloatArray',
    'IndexArray',
    'AttributeMap',
    'MutableAttributeMap',
    'POSITION',
    'INTEGER_ATTRIBUTES',
    'DecimateError',
]

#: A float array of any shape. NumPy's own generic alias carries no shape, so
#: the shape a parameter wants is said in its docstring.
FloatArray = np.ndarray[Any, np.dtype[np.floating[Any]]]
#: An integer array, used for triangle indices and for vertex remaps.
IndexArray = np.ndarray[Any, np.dtype[np.integer[Any]]]

#: glTF attribute semantic -> array. What a caller hands in and gets back.
AttributeMap = Mapping[str, np.ndarray]
MutableAttributeMap = MutableMapping[str, np.ndarray]

#: The one attribute every mesh must carry.
POSITION = 'POSITION'

#: Attributes whose values are identifiers rather than quantities. They are
#: taken from a surviving vertex whole, never averaged: the mean of joint 3 and
#: joint 9 is joint 6, which is some unrelated bone.
INTEGER_ATTRIBUTES = frozenset({'JOINTS_0', 'JOINTS_1'})


class DecimateError(ValueError):
    """The input does not describe a triangle mesh this package can work on."""
