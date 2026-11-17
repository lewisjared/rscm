"""
Rust Simple Climate Model (RSCM)

A framework for simple climate models built in Rust.
Models are assembled with [ModelBuilder][rscm.core.ModelBuilder] in [rscm.core][],
and the MAGICC component builders are in [rscm.magicc][].

Submodules load on first attribute access,
so `rscm.core.TimeAxis` works after a bare `import rscm`.
"""

import importlib
import importlib.metadata
import warnings
from types import ModuleType

from ._lib import __version__ as _lib_version

_SUBMODULES = frozenset(
    {
        "calibrate",
        "component",
        "components",
        "config",
        "core",
        "example_components",
        "fair",
        "magicc",
        "names",
        "spec",
        "two_layer",
        "validation",
    }
)

try:
    __version__ = importlib.metadata.version("rscm")
except importlib.metadata.PackageNotFoundError:
    # Running from a source tree that was never installed.
    __version__ = _lib_version

if __version__ != _lib_version:
    warnings.warn(
        f"Version mismatch between rscm and rscm._lib: {__version__} != {_lib_version}"
    )


def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted({*globals(), *_SUBMODULES})
