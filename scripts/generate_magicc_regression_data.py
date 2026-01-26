r"""
Generate regression test data for validating Rust MAGICC implementation.

This script creates reference outputs from Fortran MAGICC that can be used
to validate the Rust implementation at key checkpoints:

1. GHG Forcing Module - Concentration -> Forcing
2. Climate Module - Forcing -> Temperature
3. Full Integration - End-to-end validation

Usage:
    # Set MAGICC_ROOT environment variable first:
    export MAGICC_ROOT=/path/to/magicc-v7.5.3

    # Or create a .env file with MAGICC_ROOT=/path/to/magicc-v7.5.3

    uv run --with pymagicc --with scmdata --with pyarrow \
        scripts/generate_magicc_regression_data.py

Output:
    tests/regression_data/magicc/*.parquet
    tests/regression_data/magicc/*.json
"""

import json
import os
import platform
import shutil
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

# Load MAGICC path from environment
# Set MAGICC_ROOT in .env or environment before running
MAGICC_ROOT_STR = os.environ.get("MAGICC_ROOT")
if not MAGICC_ROOT_STR:
    msg = (
        "MAGICC_ROOT environment variable not set. "
        "Set it to your MAGICC installation directory, e.g.:\n"
        "  export MAGICC_ROOT=/path/to/magicc-v7.5.3"
    )
    raise RuntimeError(msg)

MAGICC_ROOT = Path(MAGICC_ROOT_STR)

# Determine executable name based on platform
if platform.system() == "Darwin":
    if platform.machine() == "arm64":
        executable_name = "magicc-darwin-arm64"
    else:
        executable_name = "magicc-darwin-x86_64"
elif platform.system() == "Linux":
    executable_name = "magicc-linux-x86_64"
else:
    executable_name = "magicc"  # Fallback

MAGICC_EXECUTABLE = MAGICC_ROOT / "bin" / executable_name
if not MAGICC_EXECUTABLE.exists():
    msg = f"MAGICC executable not found: {MAGICC_EXECUTABLE}"
    raise RuntimeError(msg)

os.environ["MAGICC_EXECUTABLE_7"] = str(MAGICC_EXECUTABLE)

import scmdata  # noqa: E402
from pymagicc import MAGICC7  # noqa: E402
from pymagicc import config as pymagicc_config  # noqa: E402

# Explicitly set pymagicc config
pymagicc_config.config["EXECUTABLE_7"] = str(MAGICC_EXECUTABLE)


@contextmanager
def run_magicc():
    """Context manager for running MAGICC with proper setup."""
    m = MAGICC7(strict=False)
    m.create_copy()
    try:
        yield m
    finally:
        # Cleanup temp directory
        if m.root_dir and Path(m.root_dir).exists():
            shutil.rmtree(m.root_dir)


OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "regression_data" / "magicc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping from MAGICC DAT_* names to scmdata variable names
MAGICC_VAR_MAP = {
    "DAT_CO2_CONC": "Atmospheric Concentrations|CO2",
    "DAT_CH4_CONC": "Atmospheric Concentrations|CH4",
    "DAT_N2O_CONC": "Atmospheric Concentrations|N2O",
    "DAT_CO2_RF": "Radiative Forcing|CO2",
    "DAT_CO2_ERF": "Effective Radiative Forcing|CO2",
    "DAT_CH4_RF": "Radiative Forcing|CH4",
    "DAT_CH4_ERF": "Effective Radiative Forcing|CH4",
    "DAT_N2O_RF": "Radiative Forcing|N2O",
    "DAT_N2O_ERF": "Effective Radiative Forcing|N2O",
    "DAT_TOTAL_INCLVOLCANIC_RF": "Radiative Forcing",
    "DAT_TOTAL_INCLVOLCANIC_ERF": "Effective Radiative Forcing",
    "DAT_SURFACE_TEMP": "Surface Temperature",
    "DAT_CH4_EMIS": "Emissions|CH4",
    "DAT_CH4A_EMIS": "Emissions|CH4|MAGICC Fossil and Industrial",
    "DAT_CH4N_EMIS": "Emissions|CH4|MAGICC AFOLU",
    "DAT_N2O_EMIS": "Emissions|N2O",
}


def filter_results(
    results: scmdata.ScmRun, out_vars: list[str], region: str = "World"
) -> scmdata.ScmRun:
    """Filter results to only needed variables and region."""
    # Map DAT_* names to scmdata variable names
    wanted_vars = []
    for var in out_vars:
        if var in MAGICC_VAR_MAP:
            wanted_vars.append(MAGICC_VAR_MAP[var])
        else:
            # Try the variable name directly
            wanted_vars.append(var)

    # Filter by variable and region
    filtered = results.filter(variable=wanted_vars, region=region, log_if_empty=False)

    if len(filtered) == 0:
        # Try without region filter in case variables have different regions
        print(f"  Warning: No data for region={region}, trying without filter")
        filtered = results.filter(variable=wanted_vars, log_if_empty=False)

    return filtered


def save_results(results: scmdata.ScmRun, name: str, config: dict, out_vars: list[str]):
    """Save filtered results and configuration."""
    # Filter to only needed variables
    filtered = filter_results(results, out_vars)

    if len(filtered) == 0:
        print(f"  Warning: No matching variables found for {name}")
        avail_vars = results.get_unique_meta("variable")
        print(f"  Available ({len(avail_vars)} total): {avail_vars[:10]}...")
        return

    # Convert to long format DataFrame and save as parquet
    df = filtered.timeseries()

    # Reset index to get metadata as columns, then melt to long format
    df_reset = df.reset_index()

    # Identify year columns (Timestamp objects from scmdata)
    year_cols = [c for c in df_reset.columns if isinstance(c, pd.Timestamp)]

    # Keep only essential metadata: variable, unit, region
    essential_meta = ["variable", "unit", "region"]
    df_slim = df_reset[essential_meta + year_cols].copy()

    # Melt to long format
    df_long = df_slim.melt(id_vars=essential_meta, var_name="year", value_name="value")
    # Convert Timestamp to year integer
    df_long["year"] = pd.to_datetime(df_long["year"]).dt.year

    # Save as parquet (much smaller than CSV)
    parquet_path = OUTPUT_DIR / f"{name}.parquet"
    df_long.to_parquet(parquet_path, index=False)

    # Save configuration as JSON
    json_path = OUTPUT_DIR / f"{name}_config.json"
    config_clean = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            config_clean[k] = v
        else:
            config_clean[k] = str(v)

    with open(json_path, "w") as f:
        json.dump(config_clean, f, indent=2)

    # Report size
    size_kb = parquet_path.stat().st_size / 1024
    n_rows = len(df_long)
    n_vars = df_long["variable"].nunique()
    print(f"  Saved: {name}.parquet ({size_kb:.1f} KB, {n_rows} rows, {n_vars} vars)")


def test_01_concentration_driven():
    """Test 1: Concentration-driven run with SSP245 concentrations"""
    config = {
        "startyear": 1750,
        "endyear": 2100,
        # Force concentration-driven mode (no emissions)
        "co2_switchfromconc2emis_year": 5000,
        "ch4_switchfromconc2emis_year": 5000,
        "n2o_switchfromconc2emis_year": 5000,
        # Use SSP245 concentration files (available in this MAGICC distribution)
        "file_co2_conc": "SSP245_CO2_CONC.IN",
        "file_ch4_conc": "SSP245_CH4_CONC.IN",
        "file_n2o_conc": "SSP245_N2O_CONC.IN",
        # IPCCTAR forcing method for simpler validation
        "core_co2ch4n2o_rfmethod": "IPCCTAR",
        "core_climatesensitivity": 3.0,
        "core_delq2xco2": 3.71,
        # Disable other forcings for cleaner signal
        "rf_solar_scale": 0.0,
        "rf_volcanic_scale": 0.0,
        # Output settings
        "out_forcing": 1,
        "out_concentrations": 1,
        "out_temperature": 1,
        "out_ascii_binary": "ASCII",
    }

    out_vars = [
        "DAT_CO2_CONC",
        "DAT_CH4_CONC",
        "DAT_N2O_CONC",
        "DAT_CO2_RF",
        "DAT_CO2_ERF",
        "DAT_CH4_RF",
        "DAT_CH4_ERF",
        "DAT_N2O_RF",
        "DAT_N2O_ERF",
        "DAT_TOTAL_INCLVOLCANIC_RF",
        "DAT_TOTAL_INCLVOLCANIC_ERF",
        "DAT_SURFACE_TEMP",
    ]

    with run_magicc() as m:
        m.set_output_variables(write_ascii=True, write_binary=False, parameters=True)
        res = m.run(out_dynamic_vars=out_vars, **config)

    save_results(res, "01_concentration_driven", config, out_vars)
    return res


def test_02_ghg_forcing_olbl():
    """Test 2: GHG Forcing with OLBL method"""
    config = {
        "startyear": 1750,
        "endyear": 2100,
        "co2_switchfromconc2emis_year": 5000,
        "ch4_switchfromconc2emis_year": 5000,
        "n2o_switchfromconc2emis_year": 5000,
        "file_co2_conc": "SSP245_CO2_CONC.IN",
        "file_ch4_conc": "SSP245_CH4_CONC.IN",
        "file_n2o_conc": "SSP245_N2O_CONC.IN",
        "core_co2ch4n2o_rfmethod": "OLBL",
        "core_climatesensitivity": 3.0,
        "core_delq2xco2": 3.71,
        "core_rfrapidadjust_co2": 1.05,
        "core_rfrapidadjust_ch4": 0.86,
        "core_rfrapidadjust_n2o": 0.93,
        "rf_solar_scale": 0.0,
        "rf_volcanic_scale": 0.0,
        "out_forcing": 1,
        "out_concentrations": 1,
        "out_temperature": 1,
        "out_ascii_binary": "ASCII",
    }

    out_vars = [
        "DAT_CO2_CONC",
        "DAT_CH4_CONC",
        "DAT_N2O_CONC",
        "DAT_CO2_RF",
        "DAT_CO2_ERF",
        "DAT_CH4_RF",
        "DAT_CH4_ERF",
        "DAT_N2O_RF",
        "DAT_N2O_ERF",
        "DAT_TOTAL_INCLVOLCANIC_RF",
        "DAT_TOTAL_INCLVOLCANIC_ERF",
        "DAT_SURFACE_TEMP",
    ]

    with run_magicc() as m:
        m.set_output_variables(write_ascii=True, write_binary=False, parameters=True)
        res = m.run(out_dynamic_vars=out_vars, **config)

    save_results(res, "02_ghg_forcing_olbl", config, out_vars)
    return res


def test_03_emissions_driven():
    """Test 3: Emissions-driven run with SSP245"""
    config = {
        "startyear": 1750,
        "endyear": 2100,
        "file_emisscen": "SSP245_EMMS.SCEN7",
        "core_climatesensitivity": 3.0,
        "out_forcing": 1,
        "out_concentrations": 1,
        "out_emissions": 1,
        "out_temperature": 1,
        "out_carboncycle": 1,
        "out_ascii_binary": "ASCII",
    }

    out_vars = [
        "DAT_CO2_CONC",
        "DAT_CH4_CONC",
        "DAT_N2O_CONC",
        "DAT_CO2_RF",
        "DAT_CH4_RF",
        "DAT_N2O_RF",
        "DAT_TOTAL_INCLVOLCANIC_RF",
        "DAT_TOTAL_INCLVOLCANIC_ERF",
        "DAT_SURFACE_TEMP",
    ]

    with run_magicc() as m:
        m.set_output_variables(write_ascii=True, write_binary=False, parameters=True)
        res = m.run(out_dynamic_vars=out_vars, **config)

    save_results(res, "03_emissions_driven", config, out_vars)
    return res


def test_04_ecs_sweep():
    """Test 4: Climate Sensitivity Parameter Sweep"""
    out_vars = [
        "DAT_CO2_CONC",
        "DAT_SURFACE_TEMP",
        "DAT_CO2_ERF",
    ]

    for ecs in [1.5, 2.0, 3.0, 4.0, 4.5]:
        config = {
            "startyear": 1750,
            "endyear": 2100,
            "co2_switchfromconc2emis_year": 5000,
            "ch4_switchfromconc2emis_year": 5000,
            "n2o_switchfromconc2emis_year": 5000,
            "file_co2_conc": "SSP245_CO2_CONC.IN",
            "file_ch4_conc": "SSP245_CH4_CONC.IN",
            "file_n2o_conc": "SSP245_N2O_CONC.IN",
            "core_climatesensitivity": ecs,
            "core_delq2xco2": 3.71,
            "core_co2ch4n2o_rfmethod": "IPCCTAR",
            "rf_solar_scale": 0.0,
            "rf_volcanic_scale": 0.0,
            "out_forcing": 1,
            "out_concentrations": 1,
            "out_temperature": 1,
            "out_ascii_binary": "ASCII",
        }

        with run_magicc() as m:
            m.set_output_variables(
                write_ascii=True, write_binary=False, parameters=True
            )
            res = m.run(out_dynamic_vars=out_vars, **config)

        save_results(res, f"04_ecs_sweep_{ecs}", config, out_vars)


def test_05_co2_only_forcing():
    """Test 5: CO2-only forcing for clean comparison"""
    config = {
        "startyear": 1750,
        "endyear": 2100,
        "co2_switchfromconc2emis_year": 5000,
        "ch4_switchfromconc2emis_year": 5000,
        "n2o_switchfromconc2emis_year": 5000,
        "file_co2_conc": "SSP245_CO2_CONC.IN",
        "file_ch4_conc": "SSP245_CH4_CONC.IN",
        "file_n2o_conc": "SSP245_N2O_CONC.IN",
        # Run only CO2 forcing
        "rf_total_runmodus": "CO2",
        "rf_solar_scale": 0.0,
        "rf_volcanic_scale": 0.0,
        "core_co2ch4n2o_rfmethod": "IPCCTAR",
        "core_climatesensitivity": 3.0,
        "core_delq2xco2": 3.71,
        "out_forcing": 1,
        "out_concentrations": 1,
        "out_temperature": 1,
        "out_ascii_binary": "ASCII",
    }

    out_vars = [
        "DAT_CO2_CONC",
        "DAT_CO2_RF",
        "DAT_CO2_ERF",
        "DAT_SURFACE_TEMP",
        "DAT_TOTAL_INCLVOLCANIC_RF",
        "DAT_TOTAL_INCLVOLCANIC_ERF",
    ]

    with run_magicc() as m:
        m.set_output_variables(write_ascii=True, write_binary=False, parameters=True)
        res = m.run(out_dynamic_vars=out_vars, **config)

    save_results(res, "05_co2_only_forcing", config, out_vars)
    return res


def main():
    """Generate all regression test data."""
    print("Generating RSCM regression test data (v2 - parquet format)...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"MAGICC root: {MAGICC_ROOT}")
    print()

    tests = [
        ("01", "Concentration-driven (IPCCTAR)", test_01_concentration_driven),
        ("02", "GHG Forcing (OLBL)", test_02_ghg_forcing_olbl),
        ("03", "Emissions-driven (SSP245)", test_03_emissions_driven),
        ("04", "Climate Sensitivity Sweep", test_04_ecs_sweep),
        ("05", "CO2-only Forcing", test_05_co2_only_forcing),
    ]

    for test_id, name, func in tests:
        print(f"Running Test {test_id}: {name}...")
        try:
            func()
        except Exception as e:
            print(f"  FAILED: {e}")
            raise

    print()
    print("=" * 60)
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
    print(f"Total parquet size: {total_size / 1024:.1f} KB")
    print()
    print("Files generated:")
    for f in sorted(OUTPUT_DIR.glob("*.parquet")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
