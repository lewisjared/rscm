"""Shared helpers for regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent / "data"

METADATA_COLUMNS = [
    "climate_model",
    "model",
    "region",
    "scenario",
    "todo",
    "unit",
    "variable",
]


def load_regression_data(
    suite: str,
    name: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Load regression CSV + JSON config and return long-format DataFrame.

    Parameters
    ----------
    suite
        Subdirectory under data/ (e.g., "ghg_forcing", "ocean_udeb").
    name
        Test name without extension (e.g., "01_concentration_driven").

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Long-format DataFrame with (year, variable, unit, region, value)
        and the config dictionary.
    """
    suite_dir = DATA_DIR / suite
    csv_path = suite_dir / f"{name}.csv"
    config_path = suite_dir / f"{name}_config.json"

    if not csv_path.exists():
        pytest.skip(f"Reference data not found: {csv_path}")

    wide = pd.read_csv(csv_path)
    time_cols = [c for c in wide.columns if c not in METADATA_COLUMNS]

    long = wide.melt(
        id_vars=METADATA_COLUMNS,
        value_vars=time_cols,
        var_name="timestamp",
        value_name="value",
    )
    long["year"] = pd.to_datetime(long["timestamp"]).dt.year.astype(np.float64)
    long = long.drop(columns=["timestamp"])

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    return long, config


def get_variable_values(
    df: pd.DataFrame,
    variable: str,
    region: str = "World",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (years, values) arrays for a variable, sorted by year."""
    subset = df[(df["variable"] == variable) & (df["region"] == region)]
    subset = subset.sort_values("year")

    if len(subset) == 0:
        available = sorted(df["variable"].unique())
        msg = (
            f"Variable '{variable}' not found for region '{region}'."
            f" Available: {available}"
        )
        raise ValueError(msg)

    return subset["year"].values, subset["value"].values


def fourbox_global_mean(values_2d: np.ndarray) -> np.ndarray:
    """
    Area-weighted global mean from FourBox temperature data.

    Uses MAGICC default area fractions:
    NH ocean=0.29, NH land=0.21, SH ocean=0.395, SH land=0.105.
    """
    weights = np.array([0.5 * 0.58, 0.5 * 0.42, 0.5 * 0.79, 0.5 * 0.21])
    return values_2d @ weights


def assert_allclose_phased(  # noqa: PLR0913
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    skip: int = 5,
    shock_end: int = 25,
    converge_start: int = 55,
    shock_rtol: float = 3e-2,
    converge_rtol: float = 2e-2,
    final_rtol: float = 2e-2,
    final_years: int = 20,
    atol: float = 1e-6,
    name: str = "",
) -> None:
    """
    Assert allclose with tighter tolerances as the solution converges.

    Splits the comparison into phases to distinguish step forcing onset
    transient from steady-state agreement. All index parameters are
    relative to the start of the arrays (not calendar years).

    Parameters
    ----------
    actual, expected
        Arrays to compare (same length).
    skip
        Initial indices to skip entirely (forcing onset, index < skip).
    shock_end
        End of shock phase (index). Shock = [skip, shock_end).
    converge_start
        Start of convergence phase (index). Convergence = [converge_start, end).
        Indices in [shock_end, converge_start) are checked with shock_rtol
        (transition period).
    shock_rtol
        Relative tolerance for the shock and transition phases.
    converge_rtol
        Relative tolerance for the convergence phase.
    final_rtol
        Relative tolerance for the last ``final_years`` indices.
    final_years
        Number of trailing indices checked at the tightest tolerance.
    atol
        Absolute tolerance (applied to all phases).
    name
        Label for error messages.
    """
    n = len(actual)
    prefix = f"{name}: " if name else ""

    # Shock phase: [skip, shock_end)
    s_end = min(shock_end, n)
    if skip < s_end:
        npt.assert_allclose(
            actual[skip:s_end],
            expected[skip:s_end],
            rtol=shock_rtol,
            atol=atol,
            err_msg=f"{prefix}shock phase (index {skip}-{s_end})",
        )

    # Transition phase: [shock_end, converge_start)
    c_start = min(converge_start, n)
    if s_end < c_start:
        npt.assert_allclose(
            actual[s_end:c_start],
            expected[s_end:c_start],
            rtol=shock_rtol,
            atol=atol,
            err_msg=f"{prefix}transition phase (index {s_end}-{c_start})",
        )

    # Convergence phase: [converge_start, end)
    if c_start < n:
        npt.assert_allclose(
            actual[c_start:],
            expected[c_start:],
            rtol=converge_rtol,
            atol=atol,
            err_msg=f"{prefix}convergence phase (index {c_start}+)",
        )

    # Final phase: last final_years indices (tightest)
    f_start = max(c_start, n - final_years)
    if f_start < n:
        npt.assert_allclose(
            actual[f_start:],
            expected[f_start:],
            rtol=final_rtol,
            atol=atol,
            err_msg=f"{prefix}final phase (last {n - f_start} indices)",
        )
