"""Build the optional compiled reducer.

The package is pure Python and installs without a compiler; this only adds the
compiled contraction loop described in ``_reduce_native.pyx``. A build that
fails for any reason -- no compiler, no Cython, an unsupported platform -- is
not an error, because the NumPy implementation of the same reduction is always
there and is what the compiled one is tested against.
"""

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - no Cython
    cythonize = None


class build_ext(_build_ext):
    """A failed build of an optional accelerator is not a failed install."""

    def run(self):
        try:
            super().run()
        except Exception as error:  # pragma: no cover - toolchain
            self.warn(
                'the optional accelerator did not build (%s); the NumPy '
                'reduction will be used instead' % (error,)
            )

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as error:  # pragma: no cover - toolchain
            self.warn(
                '%s did not build (%s); the NumPy reduction will be used '
                'instead' % (ext.name, error)
            )


extensions = []
if cythonize is not None:
    extensions = cythonize(
        [
            Extension(
                'opengl_decimate._reduce_native',
                ['src/opengl_decimate/_reduce_native.pyx'],
                include_dirs=[numpy.get_include()],
                define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            )
        ],
        compiler_directives={'language_level': '3'},
    )

setup(ext_modules=extensions, cmdclass={'build_ext': build_ext})
