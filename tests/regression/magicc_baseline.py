"""Strict, offline comparison of the legacy SSP245 concentration fixture."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

import rscm._lib as rust_lib
from rscm.core import ModelBuilder, TimeAxis
from rscm.magicc import ClimateUDEBBuilder, GhgForcingBuilder, run_magicc

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).parent / "data" / "ghg_forcing"
CASE = "01_concentration_driven"
GASES = ("CO2", "CH4", "N2O")
ERF = "Effective Radiative Forcing"
OBSERVABLES = (*(f"{ERF}|{gas}" for gas in GASES), ERF, "Surface Temperature")
META = ("climate_model", "model", "region", "scenario", "todo", "unit", "variable")
UNITS = {
    **{
        f"Atmospheric Concentrations|{g}": {"ppm" if g == "CO2" else "ppb"}
        for g in GASES
    },
    **{v: {"W / m^2", "W/m^2"} for v in OBSERVABLES[:-1]},
    "Surface Temperature": {"K"},
}


def load_case(data_dir: Path) -> tuple[list[str], dict, dict, dict]:
    """Read a complete case, rejecting ambiguous selections and malformed values."""
    csv_path = data_dir / f"{CASE}.csv"
    config_path = data_dir / f"{CASE}_config.json"
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (csv_path, config_path)
    }
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        msg = "Configuration must be an object"
        raise TypeError(msg)
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if len(set(header)) != len(header) or not set(META) <= set(header):
            msg = "Duplicate columns or missing metadata columns"
            raise ValueError(msg)
        timestamps = [c for c in header if c not in META]
        dates = pd.to_datetime(timestamps, errors="raise")
        expected = pd.date_range("1750-01-01", "2100-01-01", freq="YS")
        if not dates.equals(expected):
            msg = "Expected exactly the annual January 1 boundaries 1750-2100"
            raise ValueError(msg)
        series = {}
        identities = set()
        for cells in reader:
            if len(cells) != len(header):
                msg = "Row width does not match header"
                raise ValueError(msg)
            row = dict(zip(header, cells, strict=True))
            variable = row["variable"]
            if variable not in UNITS:
                continue
            if variable in series or row["region"] != "World":
                msg = f"Ambiguous selection or wrong region for {variable}"
                raise ValueError(msg)
            identities.add(
                tuple(row[k] for k in ("climate_model", "model", "scenario", "todo"))
            )
            if row["unit"] not in UNITS[variable]:
                msg = f"Unsupported unit for {variable}: {row['unit']}"
                raise ValueError(msg)
            values = np.array([row[t] for t in timestamps], dtype=float)
            if not np.isfinite(values).all():
                msg = f"Nonfinite values for {variable}"
                raise ValueError(msg)
            if variable.startswith("Atmospheric") and np.any(values <= 0):
                msg = f"Nonpositive concentration for {variable}"
                raise ValueError(msg)
            series[variable] = values
    if set(series) != set(UNITS) or len(identities) != 1:
        msg = "Missing required variables or inconsistent scenario/model selection"
        raise ValueError(msg)
    return timestamps, series, config, hashes


def parameters(config: dict, concentrations: dict) -> tuple[dict, dict]:
    """Map this IPCCTAR experiment, retaining explicit baseline assumptions."""
    required = {
        "startyear": 1750,
        "endyear": 2100,
        "core_co2ch4n2o_rfmethod": "IPCCTAR",
        "co2_switchfromconc2emis_year": 5000,
        "ch4_switchfromconc2emis_year": 5000,
        "n2o_switchfromconc2emis_year": 5000,
    }
    if any(config.get(k) != v for k, v in required.items()):
        msg = "Configuration does not describe the supported concentration case"
        raise ValueError(msg)
    forcing = {"method": "Ipcctar", "delq2xco2": config["core_delq2xco2"]}
    for gas in GASES:
        forcing[f"{gas.lower()}_pi"] = float(concentrations[gas][0])
        forcing[f"adjust_{gas.lower()}"] = config.get(
            f"core_rfrapidadjust_{gas.lower()}", 1.0
        )
    climate = {
        "ecs": config["core_climatesensitivity"],
        "rf_2xco2": config["core_delq2xco2"],
    }
    numbers = [v for k, v in forcing.items() if k != "method"]
    numbers.extend(climate.values())
    if not all(
        not isinstance(v, bool)
        and isinstance(v, (float, int))
        and np.isfinite(v)
        and v > 0
        for v in numbers
    ):
        msg = "Mapped parameters must be finite and positive"
        raise ValueError(msg)
    return forcing, climate


def effective_parameters(forcing: dict, climate: dict) -> dict:
    """Read resolved component parameters from existing Rust serialization."""
    model = (
        ModelBuilder()
        .with_time_axis(TimeAxis.from_values(np.array([1750.0, 1751.0])))
        .with_rust_component(GhgForcingBuilder.from_parameters(forcing).build())
        .with_rust_component(ClimateUDEBBuilder.from_parameters(climate).build())
        .with_initial_values({"Surface Temperature": 0.0})
        .build()
    )
    nodes = tomllib.loads(model.to_toml())["components"]["nodes"]
    return {node["type"]: node["parameters"] for node in nodes if "parameters" in node}


def compare_values(
    actual: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute signed and relative errors without hiding zero-reference errors."""
    if actual.shape != reference.shape or not np.isfinite([actual, reference]).all():
        msg = "Comparison requires matching, finite arrays"
        raise ValueError(msg)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        signed = actual - reference
        relative = np.full_like(signed, np.nan)
        np.divide(signed, np.abs(reference), out=relative, where=reference != 0)
    return signed, relative


def build_report(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict, str]:
    """Run the case and return complete comparison data, metadata and summary."""
    timestamps, reference, config, hashes = load_case(data_dir)
    years = np.arange(1750.0, 2101.0)
    concentrations = {
        gas: reference[f"Atmospheric Concentrations|{gas}"] for gas in GASES
    }
    forcing, climate = parameters(config, concentrations)
    residual = reference[ERF] - sum(reference[f"{ERF}|{gas}"] for gas in GASES)
    result = run_magicc(
        years,
        concentrations,
        other_forcing=residual,
        forcing_parameters=forcing,
        climate_parameters=climate,
    )
    rows = []
    summary = [
        "# SSP245 observational baseline",
        "",
        "No scientific parity verdict or accepted temperature tolerance.",
        (
            "All comparisons use unchanged calendar labels; temperature "
            "alignment is provisional."
        ),
        "",
        (
            "| Variable | Points | Maximum absolute error | Year | First "
            "error | Final error | Unit |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for variable in OBSERVABLES:
        actual = result.values[variable]
        expected = reference[variable]
        signed, relative = compare_values(actual, expected)
        for i, year in enumerate(years):
            rows.append(
                {
                    "case": CASE,
                    "variable": variable,
                    "unit": result.units[variable],
                    "raw_reference_timestamp": timestamps[i],
                    "comparison_year": int(year),
                    "actual": actual[i],
                    "reference": expected[i],
                    "signed_error": signed[i],
                    "absolute_error": abs(signed[i]),
                    "relative_error": relative[i],
                    "comparability": "provisional",
                }
            )
        peak = int(np.argmax(np.abs(signed)))
        label = variable.replace("|", r"\|")
        summary.append(
            f"| {label} | {len(years)} | "
            f"{abs(signed[peak]):.10g} | {int(years[peak])} | "
            f"{signed[0]:.10g} | {signed[-1]:.10g} | {result.units[variable]} |"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for pattern in (
            "python/rscm/**/*.py",
            "crates/**/*.rs",
            "Cargo.lock",
            "pyproject.toml",
            "uv.lock",
            "tests/regression/magicc_baseline.py",
            "scripts/regression/compare_magicc_baseline.py",
        )
        for p in sorted(ROOT.glob(pattern))
    }
    metadata = {
        "schema_version": 1,
        "case": CASE,
        "execution_status": "complete",
        "scientific_parity": "not_evaluated",
        "comparability": "provisional",
        "fixture_hashes": hashes,
        "rscm_revision": revision,
        "source_hashes": source_hashes,
        "extension_sha256": hashlib.sha256(
            Path(rust_lib.__file__).read_bytes()
        ).hexdigest(),
        "reference_config_overrides": config,
        "supplied_parameters": {"forcing": forcing, "climate": climate},
        "effective_parameters": effective_parameters(forcing, climate),
        "forcing_contributors": list(result.forcing_contributors),
        "residual_formula": (
            "reference total ERF - reference CO2 ERF - reference CH4 ERF "
            "- reference N2O ERF"
        ),
        "assumptions": [
            (
                "PI concentrations use the first fixture concentrations, "
                "following the existing reference test."
            ),
            (
                "IPCCTAR rapid adjustments default to 1.0 when absent from "
                "the reference overrides."
            ),
            (
                "Climate starts with zero temperature and ocean anomalies, "
                "without spin-up."
            ),
            (
                "All labels are compared directly, without shifting or "
                "dropping endpoints; physical temperature time alignment "
                "remains unverified."
            ),
            (
                "Total ERF includes a reference-derived residual and is not "
                "independent validation of other forcing."
            ),
            "UDEB applies the CO2 regional forcing pattern to total ERF.",
        ],
        "reference_provenance_unknown": [
            "executable version/hash",
            "consumed input hashes",
            "resolved configuration",
            "initial climate state",
        ],
        "relative_error_definition": (
            "(actual-reference)/abs(reference); null at exactly zero reference"
        ),
    }
    return pd.DataFrame(rows), metadata, "\n".join(summary) + "\n"


def write_report(output_dir: Path, data_dir: Path = DATA_DIR) -> None:
    """Publish a complete report into a new directory, rejecting collisions."""
    if output_dir.exists():
        msg = f"Output directory already exists: {output_dir}"
        raise FileExistsError(msg)
    table, metadata, summary = build_report(data_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".magicc-baseline-", dir=output_dir.parent
    ) as stage:
        staged = Path(stage)
        table.to_csv(staged / "comparison.csv", index=False, na_rep="")
        (staged / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (staged / "summary.md").write_text(summary)
        if output_dir.exists():
            msg = f"Output directory already exists: {output_dir}"
            raise FileExistsError(msg)
        staged.rename(output_dir)
