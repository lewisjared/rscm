"""Shared helpers for regression tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


@dataclass
class PhaseResult:
    """Metrics for a single comparison phase."""

    name: str
    rtol: float
    max_rel_err: float
    mean_rel_err: float
    n_points: int

    @property
    def passed(self) -> bool:
        return self.max_rel_err <= self.rtol


@dataclass
class PhasedComparisonResult:
    """Result of a phased comparison between actual and expected arrays."""

    label: str
    suite: str = ""
    variable: str = ""
    phases: list[PhaseResult] = field(default_factory=list)
    bias_sign: str = ""

    @property
    def passed(self) -> bool:
        return all(p.passed for p in self.phases)

    def summary_row(self) -> dict:
        """Return a flat dict suitable for tabulation."""
        row: dict = {"test": self.label, "status": "PASS" if self.passed else "FAIL"}
        for p in self.phases:
            row[p.name] = f"{p.max_rel_err:.2%}"
            row[f"{p.name}_threshold"] = f"{p.rtol:.2%}"
            row[f"{p.name}_pass"] = p.passed
        row["bias"] = self.bias_sign
        return row

    def to_csv_rows(self) -> list[dict]:
        """Return a list of flat dicts (one per phase) for CSV output."""
        rows = []
        for p in self.phases:
            if p.n_points > 0:
                rows.append(
                    {
                        "suite": self.suite,
                        "test": self.label,
                        "variable": self.variable,
                        "phase": p.name,
                        "threshold": f"{p.rtol:.4g}",
                        "actual": f"{p.max_rel_err:.4g}",
                        "mean_bias": f"{p.mean_rel_err:.4g}",
                        "n_points": p.n_points,
                        "pass": p.passed,
                        "bias_sign": self.bias_sign,
                    }
                )
        return rows


# Module-level collector: assert_allclose_phased appends results here.
# The conftest session finalizer writes these to CSV.
_collected_results: list[PhasedComparisonResult] = []


def compute_phased_metrics(  # noqa: PLR0913
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
    suite: str = "",
    variable: str = "",
) -> PhasedComparisonResult:
    """
    Compute phased error metrics between actual and expected arrays.

    Returns a ``PhasedComparisonResult`` with per-phase max/mean errors
    and pass/fail status against the given tolerances. Does not raise
    on failure -- use ``assert_allclose_phased`` for test assertions.

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
        Indices in [shock_end, converge_start) are checked with shock_rtol.
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
        Test scenario label.
    suite
        Test suite label (e.g. "ghg_forcing", "ocean_udeb").
    variable
        Variable being compared (e.g. "Surface Temperature", "ERF|CO2").
    """
    n = len(actual)
    if len(expected) != n:
        msg = f"length mismatch: actual={n}, expected={len(expected)}"
        if name:
            msg = f"{name}: {msg}"
        raise AssertionError(msg)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(np.abs(expected) > atol, (actual - expected) / expected, 0.0)

    s_end = min(shock_end, n)
    c_start = min(converge_start, n)
    # Final phase starts from max(skip, n - final_years) so short series
    # still get final-phase coverage even when n < converge_start.
    f_start = max(skip, n - final_years)

    def _phase(label: str, start: int, end: int, rtol: float) -> PhaseResult:
        if start >= end:
            return PhaseResult(label, rtol, 0.0, 0.0, 0)
        chunk = rel_err[start:end]
        return PhaseResult(
            label,
            rtol,
            float(np.max(np.abs(chunk))),
            float(np.mean(chunk)),
            len(chunk),
        )

    # Convergence stops at f_start so it doesn't overlap with the final phase
    phases = [
        _phase("shock", skip, s_end, shock_rtol),
        _phase("transition", s_end, c_start, shock_rtol),
        _phase("converge", c_start, f_start, converge_rtol),
        _phase("final", f_start, n, final_rtol),
    ]

    # Overall bias direction from the convergence phase
    converge_mean = phases[2].mean_rel_err if phases[2].n_points > 0 else 0.0
    bias_sign = "warm" if converge_mean > 0 else "cool"

    return PhasedComparisonResult(
        label=name, suite=suite, variable=variable, phases=phases, bias_sign=bias_sign
    )


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
    suite: str = "",
    variable: str = "",
) -> PhasedComparisonResult:
    """
    Assert allclose with tighter tolerances as the solution converges.

    Computes phased metrics then asserts each phase passes. Returns the
    ``PhasedComparisonResult`` for reporting even when all phases pass.

    See ``compute_phased_metrics`` for parameter descriptions.
    """
    result = compute_phased_metrics(
        actual,
        expected,
        skip=skip,
        shock_end=shock_end,
        converge_start=converge_start,
        shock_rtol=shock_rtol,
        converge_rtol=converge_rtol,
        final_rtol=final_rtol,
        final_years=final_years,
        atol=atol,
        name=name,
        suite=suite,
        variable=variable,
    )

    # Record before asserting so xfail tests still capture metrics
    _collected_results.append(result)

    prefix = f"{name}: " if name else ""
    for phase in result.phases:
        if phase.n_points > 0 and not phase.passed:
            npt.assert_allclose(
                np.array([phase.max_rel_err]),
                np.array([0.0]),
                atol=phase.rtol,
                err_msg=(
                    f"{prefix}{phase.name} phase: max_rel_err={phase.max_rel_err:.4%}"
                    f" exceeds rtol={phase.rtol:.4%}"
                ),
            )

    return result


def assert_allclose_recorded(  # noqa: PLR0913
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    name: str = "",
    suite: str = "",
    variable: str = "",
) -> None:
    """
    Assert allclose and record the result for CSV reporting.

    For simple single-tolerance comparisons (e.g. GHG forcing ERF).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(np.abs(expected) > atol, (actual - expected) / expected, 0.0)

    max_err = float(np.max(np.abs(rel_err)))
    mean_err = float(np.mean(rel_err))
    phase = PhaseResult("all", rtol, max_err, mean_err, len(actual))
    bias = "warm" if mean_err > 0 else "cool"
    result = PhasedComparisonResult(
        label=name, suite=suite, variable=variable, phases=[phase], bias_sign=bias
    )
    _collected_results.append(result)

    npt.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
