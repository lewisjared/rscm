r"""
Shared utilities for MAGICC regression data generation scripts.

Provides MAGICC setup and configuration helpers for the regression data generators.

Setup:
    export MAGICC_ROOT=/path/to/magicc-v7.5.3

    uv run --with pymagicc --with scmdata --with "pandas<3" \
        scripts/regression/generate_full_run.py
"""

import os
import platform
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _setup_magicc():
    """Locate MAGICC executable and configure pymagicc."""
    magicc_root_str = os.environ.get("MAGICC_ROOT")
    if not magicc_root_str:
        msg = (
            "MAGICC_ROOT environment variable not set. "
            "Set it to your MAGICC installation directory, e.g.:\n"
            "  export MAGICC_ROOT=/path/to/magicc-v7.5.3"
        )
        raise RuntimeError(msg)

    magicc_root = Path(magicc_root_str)

    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        exe_name = f"magicc-darwin-{machine}"
    elif system == "Linux":
        exe_name = "magicc-linux-x86_64"
    else:
        exe_name = "magicc"

    executable = magicc_root / "bin" / exe_name
    if not executable.exists():
        msg = f"MAGICC executable not found: {executable}"
        raise RuntimeError(msg)

    os.environ["MAGICC_EXECUTABLE_7"] = str(executable)

    from pymagicc import config as pymagicc_config  # noqa: PLC0415

    pymagicc_config.config["EXECUTABLE_7"] = str(executable)

    return magicc_root


MAGICC_ROOT = _setup_magicc()

from pymagicc import MAGICC7  # noqa: E402

#: Root for regression data: ``tests/regression/data/`` relative to repo root.
REGRESSION_DATA_ROOT = (
    Path(__file__).parent.parent.parent / "tests" / "regression" / "data"
)


def output_dir(suite: str) -> Path:
    """Return and create the output directory for a regression suite."""
    d = REGRESSION_DATA_ROOT / suite
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_config(*bases: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """
    Build a MAGICC configuration by merging base dicts and keyword overrides.

    Later bases and overrides win.
    """
    config: dict[str, Any] = {}
    for base in bases:
        config.update(base)
    config.update(overrides)
    return config


@contextmanager
def run_magicc_ctx():
    """Context manager for running MAGICC with proper setup and cleanup."""
    m = MAGICC7(strict=False)
    m.create_copy()
    try:
        yield m
    finally:
        if m.root_dir and Path(m.root_dir).exists():
            shutil.rmtree(m.root_dir)
