r"""
Shared utilities for MAGICC regression data generation scripts.

Provides MAGICC setup, common configuration blocks, and output helpers
used across regression test generation scripts.

Setup:
    export MAGICC_ROOT=/path/to/magicc-v7.5.3

    uv run --with pymagicc --with scmdata --with "pandas<3" \
        scripts/regression/generate_<suite>.py
"""

import json
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

# Now safe to import pymagicc / scmdata at module level
import scmdata  # noqa: E402
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


#: Mapping from MAGICC internal DAT_* names to scmdata variable names.
MAGICC_VAR_MAP: dict[str, str] = {
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
    "DAT_HEATUPTK_AGGREG": "HEATUPTAKE_EBALANCE_TOTAL",
    "DAT_HEATCONTENT_AGGREG_TOTAL": "Heat Content|Ocean",
    "DAT_UPWELLING_RATE": "UPWELLING_RATE",
    "DAT_CH4_EMIS": "Emissions|CH4",
    "DAT_CH4A_EMIS": "Emissions|CH4|MAGICC Fossil and Industrial",
    "DAT_CH4N_EMIS": "Emissions|CH4|MAGICC AFOLU",
    "DAT_N2O_EMIS": "Emissions|N2O",
    "DAT_CO2_EMIS": "Emissions|CO2",
    "DAT_CO2I_EMIS": "Emissions|CO2|Fossil",
    "DAT_CO2B_EMIS": "Emissions|CO2|Land Use",
    "DAT_NOX_EMIS": "Emissions|NOx",
    "DAT_CO_EMIS": "Emissions|CO",
    "DAT_NMVOC_EMIS": "Emissions|NMVOC",
    "DAT_SOX_EMIS": "Emissions|SOx",
    "DAT_BC_EMIS": "Emissions|BC",
    "DAT_OC_EMIS": "Emissions|OC",
}


#: Concentration-driven mode: bypass all atmospheric chemistry.
CONC_DRIVEN: dict[str, Any] = {
    "co2_switchfromconc2emis_year": 5000,
    "ch4_switchfromconc2emis_year": 5000,
    "n2o_switchfromconc2emis_year": 5000,
    "file_ch4_conc": "CONST_CH4_CONC.IN",
    "file_n2o_conc": "CONST_N2O_CONC.IN",
}

#: Isolate CO2 forcing with IPCCTAR method.
CO2_ONLY: dict[str, Any] = {
    "rf_total_runmodus": "CO2",
    "core_co2ch4n2o_rfmethod": "IPCCTAR",
    "ch4_incl_ch4ox": 0,
}

#: Disable natural variability for reproducibility.
NO_VARIABILITY: dict[str, Any] = {
    "core_amv_apply": 0,
    "core_elnino_apply": 0,
    "rf_solar_scale": 0.0,
    "rf_volcanic_scale": 0.0,
}

#: Default climate parameters.
DEFAULT_CLIMATE: dict[str, Any] = {
    "core_climatesensitivity": 3.0,
    "core_delq2xco2": 3.71,
}

#: All pymagicc output categories.
_ALL_OUTPUTS = dict(
    emissions=True,
    concentrations=True,
    carboncycle=True,
    forcing=True,
    surfaceforcing=True,
    temperature=True,
    sealevel=True,
    parameters=True,
    misc=True,
    lifetimes=True,
    tempoceanlayers=True,
    heatuptake=True,
)


def make_config(*bases: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """
    Build a MAGICC configuration by merging base dicts and keyword overrides.

    Later bases and overrides win. Example::

        config = make_config(
            CONC_DRIVEN,
            CO2_ONLY,
            DEFAULT_CLIMATE,
            startyear=1850,
            endyear=2150,
            file_co2_conc="ABRUPT-2XCO2_CO2_CONC.IN",
        )
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


def run_magicc(config: dict[str, Any]) -> scmdata.ScmRun:
    """
    Run MAGICC7 with the given configuration and return results.

    All output categories are enabled so callers don't need to
    specify out_* flags. Filtering happens at save time.
    """
    with run_magicc_ctx() as m:
        m.set_output_variables(write_ascii=True, write_binary=False, **_ALL_OUTPUTS)
        res = m.run(**config)
    return res


def filter_results(
    results: scmdata.ScmRun,
    out_vars: list[str],
    region: str = "World",
) -> scmdata.ScmRun:
    """Filter results to requested variables and region."""
    wanted_vars = [MAGICC_VAR_MAP.get(v, v) for v in out_vars]

    filtered = results.filter(
        variable=wanted_vars,
        region=region,
        log_if_empty=False,
    )
    if len(filtered) == 0:
        # Try without region filter
        filtered = results.filter(variable=wanted_vars, log_if_empty=False)

    return filtered


def save_results(
    results: scmdata.ScmRun,
    name: str,
    config: dict[str, Any],
    out_vars: list[str],
    *,
    output_dir: Path,
) -> None:
    """
    Save results as CSV timeseries and config as JSON.

    The CSV contains all matching variables from the run.
    The JSON captures the exact configuration used.
    """
    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Filter to requested variables
    filtered = filter_results(results, out_vars)

    if len(filtered) == 0:
        avail_vars = results.get_unique_meta("variable")
        print(f"  Warning: no matching variables for {name}")
        print(f"  Available ({len(avail_vars)} total): {avail_vars[:10]}...")
        return

    # Save timeseries as CSV
    csv_path = out / f"{name}.csv"
    filtered.to_csv(csv_path)

    # Save configuration as JSON
    json_path = out / f"{name}_config.json"
    json_types = (int, float, str, bool, list, dict, type(None))
    config_clean = {
        k: v if isinstance(v, json_types) else str(v) for k, v in config.items()
    }
    with open(json_path, "w") as f:
        json.dump(config_clean, f, indent=2)

    size_kb = csv_path.stat().st_size / 1024
    n_vars = filtered.get_unique_meta("variable")
    print(f"  Saved: {name}.csv ({size_kb:.1f} KB, {len(n_vars)} vars)")


def run_suite(
    suite_name: str,
    tests: list[tuple[str, str, callable]],
    suite_output_dir: Path,
) -> dict[str, scmdata.ScmRun]:
    """
    Run a list of regression tests and report results.

    Parameters
    ----------
    suite_name
        Printed as header.
    tests
        List of ``(test_id, description, callable)`` tuples.
    suite_output_dir
        Directory where this suite writes CSV/JSON files.
    """
    print(f"Generating {suite_name}...")
    print(f"Output directory: {suite_output_dir}")
    print(f"MAGICC root: {MAGICC_ROOT}")
    print()

    results = {}
    for test_id, name, func in tests:
        print(f"Running Test {test_id}: {name}...")
        try:
            results[test_id] = func()
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            raise

    print()
    print("=" * 60)
    total_size = sum(f.stat().st_size for f in suite_output_dir.glob("*.csv"))
    print(f"Total CSV size: {total_size / 1024:.1f} KB")
    print()
    print("Files generated:")
    for f in sorted(suite_output_dir.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")

    return results
