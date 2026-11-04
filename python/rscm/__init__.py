"""
Rust Simple Climate Model (RSCM)

A framework for simple climate models built in Rust.
Models are assembled with [ModelBuilder][rscm.core.ModelBuilder] in [rscm.core][],
and the MAGICC component builders are in [rscm.magicc][].
"""

import importlib.metadata
import warnings

from ._lib import __version__ as _lib_version

__version__ = importlib.metadata.version("rscm")

if __version__ != _lib_version:
    warnings.warn(
        f"Version mismatch between rscm and rscm._lib: {__version__} != {_lib_version}"
    )
