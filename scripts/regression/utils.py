r"""
Shared utilities for MAGICC regression data generation scripts.

Provides MAGICC setup and configuration helpers for the regression data generators.

Setup:
    The MAGICC v7.5.3 distribution is read from `DEFAULT_MAGICC_ROOT`
    unless `MAGICC_ROOT` points elsewhere.

    uv run --group regression scripts/regression/generate_full_run.py
"""

import os
import platform
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MAGICC_DATA = Path.home() / ".local/share/magicc"
DEFAULT_MAGICC_ROOT = MAGICC_DATA / "magicc-v7.5.3"


def _setup_magicc():
    """Locate MAGICC executable and configure pymagicc."""
    magicc_root = Path(os.environ.get("MAGICC_ROOT") or DEFAULT_MAGICC_ROOT)
    if not magicc_root.is_dir():
        msg = (
            f"MAGICC distribution not found at {magicc_root}. "
            "Install MAGICC v7.5.3 there or set MAGICC_ROOT to its directory."
        )
        raise RuntimeError(msg)

    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        exe_name = f"magicc-darwin-{machine}"
    elif system == "Linux":
        exe_name = "magicc-linux-x86_64"
    else:
        exe_name = "magicc"

    # The MAGICC v7.5.3 distribution ships a bare `bin/magicc`.
    candidates = [magicc_root / "bin" / n for n in dict.fromkeys((exe_name, "magicc"))]
    executable = next((c for c in candidates if c.exists()), None)
    if executable is None:
        msg = f"MAGICC executable not found, tried: {', '.join(map(str, candidates))}"
        raise RuntimeError(msg)

    os.environ["MAGICC_EXECUTABLE_7"] = str(executable)

    from pymagicc import config as pymagicc_config  # noqa: PLC0415

    pymagicc_config.config["EXECUTABLE_7"] = str(executable)

    return magicc_root


MAGICC_ROOT = _setup_magicc()

#: AR6 probabilistic distribution (600 members), published separately from MAGICC.
AR6_CONFIG = Path(
    os.environ.get("MAGICC_AR6_CONFIG")
    or MAGICC_DATA
    / "magicc-ar6-0fd0f62-f023edb-drawnset"
    / "0fd0f62-derived-metrics-id-f023edb-drawnset.json"
)

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
