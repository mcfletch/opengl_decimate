"""Triangle mesh decimation over glTF-shaped NumPy arrays.

A mesh arrives as a mapping of glTF attribute semantics to arrays plus a
triangle index array, and leaves the same way, so nothing stands between a
generator, a loader, a decimator and a vertex buffer:

    from opengl_decimate import simplify, SimplifyOptions

    result = simplify(
        {'POSITION': positions, 'NORMAL': normals, 'TEXCOORD_0': uvs},
        indices,
        SimplifyOptions(target_ratio=0.25),
    )
    result.attributes['POSITION']       # the coarser mesh
    result.error                        # the reduction's own estimate of the deviation
    result.measured_error               # the measured deviation, with options.certify on

:func:`collapse_sequence` records the whole reduction once, and
:meth:`~opengl_decimate.sequence.CollapseSequence.at` then reaches any triangle
count by replaying a prefix of it -- which is how an editor's target slider
answers without decimating again.
"""

from opengl_decimate.floors import Survey, survey
from opengl_decimate.options import SimplifyOptions
from opengl_decimate.reduction import collapse_sequence, simplify
from opengl_decimate.sequence import CollapseSequence, SimplifyResult
from opengl_decimate.types import DecimateError

__version__ = '0.2.0a1'

__all__ = [
    # Reducing a mesh
    'simplify',
    'SimplifyOptions',
    'SimplifyResult',
    # Recording a reduction, so any target is reachable without re-running it
    'collapse_sequence',
    'CollapseSequence',
    # Asking how far one will go before spending it
    'survey',
    'Survey',
    # Refusing one
    'DecimateError',
    '__version__',
]
