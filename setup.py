"""Build the optional compiled reducer.

The package is pure Python and installs without a compiler; this only adds the
compiled contraction loop described in ``_reduce_native.pyx``. A build that
fails for any reason -- no compiler, no Cython, an unsupported platform -- is
not an error, because the NumPy implementation of the same reduction is always
there and is what the compiled one is tested against.
"""

import os

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - no Cython
    cythonize = None

# Set OPENGL_DECIMATE_LINETRACE=1 to build the reducer so that `coverage`, with
# the `Cython.Coverage` plugin, can see which of its lines ran.
#
# It is for answering a question about a particular reduction, not for a gate:
# every line of the contraction loop then calls back into Python's tracer, and
# the `nogil` region has to take the GIL to do it, which puts the suite orders
# of magnitude beyond finishing. What holds the compiled reducer to its contract
# is `tests/test_native.py`, which requires it to make the same contractions in
# the same order as the NumPy loop -- see the note in `tox.ini`.
LINETRACE = bool(os.environ.get('OPENGL_DECIMATE_LINETRACE'))


class build_ext(_build_ext):
    """A failed build of an optional accelerator is not a failed install."""

    def run(self):
        try:
            super().run()
        except Exception as error:  # noqa: BLE001 any toolchain failure leaves NumPy  # pragma: no cover
            self.warn(
                'the optional accelerator did not build (%s); the NumPy '
                'reduction will be used instead' % (error,)
            )

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as error:  # noqa: BLE001 any toolchain failure leaves NumPy  # pragma: no cover
            self.warn(
                '%s did not build (%s); the NumPy reduction will be used '
                'instead' % (ext.name, error)
            )


extensions = []
if cythonize is not None:
    macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    directives = {'language_level': '3'}
    if LINETRACE:
        macros += [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
        directives['linetrace'] = True
    extensions = cythonize(
        [
            Extension(
                'opengl_decimate._reduce_native',
                ['src/opengl_decimate/_reduce_native.pyx'],
                include_dirs=[numpy.get_include()],
                define_macros=macros,
            )
        ],
        compiler_directives=directives,
        # Always regenerate. Cython decides staleness from the timestamps of the
        # `.pyx` and the `.c`, which cannot see that the directives changed --
        # so without this, switching `LINETRACE` off leaves the traced C in
        # place and the reducer stays ten times slower with nothing to say so.
        # Regenerating one file costs a second or two of a build that is
        # compiling it anyway.
        force=True,
    )

setup(ext_modules=extensions, cmdclass={'build_ext': build_ext})
