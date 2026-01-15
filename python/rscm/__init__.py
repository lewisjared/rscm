"""
Rust Simple Climate Model (RSCM)

A framework for simple climate models built it Rust.
"""

import importlib.metadata
import warnings

from ._lib import __version__ as _lib_version

__version__ = importlib.metadata.version("rscm")

if __version__ != _lib_version:
    warnings.warn(
        f"Version mismatch between rscm and rscm._lib: {__version__} != {_lib_version}"
    )
