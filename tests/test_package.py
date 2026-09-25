"""The package's own namespace.

Every public submodule has to be reachable as ``import opengl_decimate.name``.
A function re-exported under a submodule's name replaces the submodule as the
package attribute, and ``import a.b as c`` reads that attribute, so the
import would hand back the function.
"""

import importlib
import pkgutil
import types

import pytest

import opengl_decimate

SUBMODULES = sorted(
    info.name
    for info in pkgutil.iter_modules(opengl_decimate.__path__)
    if not info.name.startswith('_')
)


@pytest.mark.parametrize('name', SUBMODULES)
def test_a_submodule_imports_as_itself(name):
    module = importlib.import_module('opengl_decimate.' + name)
    assert isinstance(getattr(opengl_decimate, name), types.ModuleType)
    assert getattr(opengl_decimate, name) is module
