"""Dot products and lengths of 3-vectors, rounded the same on every platform.

``np.einsum`` and the BLAS behind ``np.dot`` sum their products with a fused
multiply-add where the processor has one, so the last bit of a dot product
depends on the machine: x86-64 and arm64 disagree. A reduction orders its
contractions by cost, and a sphere or a grid is full of equal costs that bit
decides between, so the same mesh would reduce to a different mesh on each.

These multiply and add one coordinate at a time, each operation rounded once,
which is also how the compiled reducer computes them.

    >>> import numpy as np
    >>> dot(np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]]))
    array([32.])
"""

from __future__ import annotations

import numpy as np

from opengl_decimate.types import FloatArray

__all__ = ['dot', 'length']


def dot(a: FloatArray, b: FloatArray) -> FloatArray:
    """The dot product along the last axis, which holds three coordinates."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def length(vectors: FloatArray) -> FloatArray:
    """The length of each vector along the last axis."""
    return np.sqrt(dot(vectors, vectors))
