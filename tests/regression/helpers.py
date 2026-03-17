"""Shared helpers for regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
