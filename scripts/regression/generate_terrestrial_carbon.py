r"""
Generate terrestrial carbon cycle regression data.

Creates reference outputs from Fortran MAGICC for validating the Rust
implementation of the terrestrial carbon cycle (TERRCARBON2).

Unlike GHG forcing tests, this suite reads CARBONCYCLE.OUT directly
since pymagicc doesn't expose carbon cycle diagnostics through its
standard result object.

Usage:
    export MAGICC_ROOT=/path/to/magicc-v7.5.3

    uv run --with pymagicc --with scmdata --with "pandas<3" \
        scripts/regression/generate_terrestrial_carbon.py

Output:
    tests/regression/data/terrestrial_carbon/*.csv
    tests/regression/data/terrestrial_carbon/*_config.json
"""

import json
import os

import pandas as pd
from utils import (
    DEFAULT_CLIMATE,
    MAGICC_ROOT,
    NO_VARIABILITY,
    filter_results,
    make_config,
    output_dir,
    run_magicc_ctx,
)

OUT = output_dir("terrestrial_carbon")

# Column names from CARBONCYCLE.OUT header (line 21), mapped to clean names.
# Order must match the Fortran output exactly.
CARBONCYCLE_COLUMNS = [
    "YEARS",
    "LAND_POOL",
    "NETATMOSLANDCO2FLUX",
    "PLANT_POOL",
    "DETRITUS_POOL",
    "SOIL_POOL",
    "TOTALDEAD_POOL",
    "ATMOS_POOL",
    "NETCUMUL_EMIS",
    "NOFEED_PLANT_POOL",
    "NOFEED_SOIL_POOL",
    "NOFEED_DETRITUS_POOL",
    "GROSSDEFO_EMIS",
    "REGROWTH_PLANT_FLUX",
    "REGROWTH_DETRITUS_FLUX",
    "REGROWTH_SOIL_FLUX",
    "REGROWTH_TOTAL_FLUX",
    "TERRBIO_AND_FOSSIL_EMIS",
    "NETHUMAN_CO2EMIS",
    "AIR2OCEAN_FLUX",
    "AIR2LAND_FLUX",
    "NETECOEXCH_FLUX",
    "CURRENT_NPP",
    "PLANTRESPIRATION",
    "FACT_FERTILIZATION",
    "NPP_TEMPFEEDBACK",
    "RESP_TEMPFEEDBACK",
    "DETRITUS_TEMPFEEDBACK",
    "SOIL_TEMPFEEDBACK",
    "TURNOVERTIME_PLANT",
    "TURNOVERTIME_DETRITUS",
    "TURNOVERTIME_SOIL",
    "TOTALRESPIRATION",
    "DELTA_TERRPOOLS",
    "CO2I_INVERSE_EMIS",
    "CH4_2CO2_OXIDISATION_EMIS",
    "PF_TOT_TOT_POOL",
    "PF_TOTPLUSEMIS_POOL",
    "PF_MS_AREATHAWED",
    "PF_PEAT_AREATHAWED",
    "PF_TOT_AREATHAWED",
    "NCYCLE_LIMIT_FACTOR",
    "NCYCLE_NPOOL",
    "NCYCLE_FN_PLANTUPTAKE",
    "NCYCLE_FN_RECYCLING",
    "NCYCLE_FN_LOSSES",
    "NCYCLE_FN_DEPOSITION",
]

# Variables we care about for terrestrial carbon parity
TERR_VARS = [
    "PLANT_POOL",
    "DETRITUS_POOL",
    "SOIL_POOL",
    "CURRENT_NPP",
    "TOTALRESPIRATION",
    "GROSSDEFO_EMIS",
    "REGROWTH_TOTAL_FLUX",
    "FACT_FERTILIZATION",
    "NPP_TEMPFEEDBACK",
    "RESP_TEMPFEEDBACK",
    "DETRITUS_TEMPFEEDBACK",
    "SOIL_TEMPFEEDBACK",
    "TURNOVERTIME_PLANT",
    "TURNOVERTIME_DETRITUS",
    "TURNOVERTIME_SOIL",
    "DELTA_TERRPOOLS",
    "AIR2LAND_FLUX",
    "NOFEED_PLANT_POOL",
    "NOFEED_SOIL_POOL",
    "NOFEED_DETRITUS_POOL",
    "NETHUMAN_CO2EMIS",
    "TERRBIO_AND_FOSSIL_EMIS",
    "CO2B_EMIS",
]

# Map CARBONCYCLE column names to RSCM-style variable names for the CSV
VAR_NAME_MAP = {
    "PLANT_POOL": "Carbon Pool|Plant",
    "DETRITUS_POOL": "Carbon Pool|Detritus",
    "SOIL_POOL": "Carbon Pool|Soil",
    "CURRENT_NPP": "Net Primary Production",
    "TOTALRESPIRATION": "Respiration|Terrestrial",
    "GROSSDEFO_EMIS": "Emissions|CO2|Gross Deforestation",
    "REGROWTH_TOTAL_FLUX": "Carbon Flux|Regrowth",
    "FACT_FERTILIZATION": "CO2 Fertilization Factor",
    "NPP_TEMPFEEDBACK": "Temperature Factor|NPP",
    "RESP_TEMPFEEDBACK": "Temperature Factor|Respiration",
    "DETRITUS_TEMPFEEDBACK": "Temperature Factor|Detritus",
    "SOIL_TEMPFEEDBACK": "Temperature Factor|Soil",
    "TURNOVERTIME_PLANT": "Turnover Time|Plant",
    "TURNOVERTIME_DETRITUS": "Turnover Time|Detritus",
    "TURNOVERTIME_SOIL": "Turnover Time|Soil",
    "DELTA_TERRPOOLS": "Carbon Flux|Terrestrial",
    "AIR2LAND_FLUX": "Carbon Flux|Air to Land",
    "NOFEED_PLANT_POOL": "No-Feedback Pool|Plant",
    "NOFEED_SOIL_POOL": "No-Feedback Pool|Soil",
    "NOFEED_DETRITUS_POOL": "No-Feedback Pool|Detritus",
    "NETHUMAN_CO2EMIS": "Emissions|CO2|Net Human",
    "TERRBIO_AND_FOSSIL_EMIS": "Emissions|CO2|Terrestrial Bio and Fossil",
    "CO2B_EMIS": "Emissions|CO2|Land Use",
}

# Unit mapping
VAR_UNIT_MAP = {
    "Carbon Pool|Plant": "GtC",
    "Carbon Pool|Detritus": "GtC",
    "Carbon Pool|Soil": "GtC",
    "Net Primary Production": "GtC/yr",
    "Respiration|Terrestrial": "GtC/yr",
    "Emissions|CO2|Gross Deforestation": "GtC/yr",
    "Carbon Flux|Regrowth": "GtC/yr",
    "CO2 Fertilization Factor": "1",
    "Temperature Factor|NPP": "1",
    "Temperature Factor|Respiration": "1",
    "Temperature Factor|Detritus": "1",
    "Temperature Factor|Soil": "1",
    "Turnover Time|Plant": "yr",
    "Turnover Time|Detritus": "yr",
    "Turnover Time|Soil": "yr",
    "Carbon Flux|Terrestrial": "GtC/yr",
    "Carbon Flux|Air to Land": "GtC/yr",
    "No-Feedback Pool|Plant": "GtC",
    "No-Feedback Pool|Soil": "GtC",
    "No-Feedback Pool|Detritus": "GtC",
    "Emissions|CO2|Net Human": "GtC/yr",
    "Emissions|CO2|Terrestrial Bio and Fossil": "GtC/yr",
}


def parse_carboncycle_out(out_dir: str) -> pd.DataFrame:
    """Parse CARBONCYCLE.OUT from a MAGICC output directory."""
    fpath = os.path.join(out_dir, "CARBONCYCLE.OUT")
    if not os.path.exists(fpath):
        msg = f"CARBONCYCLE.OUT not found in {out_dir}. Set out_carboncycle=1."
        raise FileNotFoundError(msg)

    with open(fpath) as f:
        lines = f.readlines()

    # Find first data row (after header)
    # The header specifies THISFILE_FIRSTDATAROW
    first_data = None
    for line in lines:
        if "THISFILE_FIRSTDATAROW" in line:
            first_data = int(line.split("=")[1].split(",")[0].strip())
            break

    if first_data is None:
        first_data = 23  # Default from observed output

    # Parse data lines (1-indexed in Fortran, 0-indexed here)
    data_lines = lines[first_data - 1 :]
    rows = []
    for raw_line in data_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        values = stripped.split()
        if len(values) >= len(CARBONCYCLE_COLUMNS):
            rows.append([float(v) for v in values[: len(CARBONCYCLE_COLUMNS)]])

    df = pd.DataFrame(rows, columns=CARBONCYCLE_COLUMNS)
    df["YEARS"] = df["YEARS"].astype(int)
    return df


def carboncycle_to_regression_csv(
    cc_df: pd.DataFrame,
    name: str,
    config: dict,
    std_results=None,
    variables: list[str] | None = None,
) -> None:
    """Convert CARBONCYCLE DataFrame to regression CSV format (matching helpers.py).

    If ``std_results`` is provided, also includes CO2 concentration,
    surface temperature, and land-use emissions from the standard MAGICC
    output as driving inputs for RSCM parity comparisons.
    """
    if variables is None:
        variables = TERR_VARS

    # Build wide-format CSV matching the existing regression data format
    rows = []
    for col in variables:
        var_name = VAR_NAME_MAP.get(col, col)
        unit = VAR_UNIT_MAP.get(var_name, "unknown")

        row = {
            "climate_model": "MAGICC7",
            "model": "unspecified",
            "region": "World",
            "scenario": name,
            "todo": "SET",
            "unit": unit,
            "variable": var_name,
        }

        for _, data_row in cc_df.iterrows():
            year = int(data_row["YEARS"])
            timestamp = f"{year}-01-01 00:00:00"
            row[timestamp] = data_row[col]

        rows.append(row)

    # Add driving inputs from standard MAGICC output
    if std_results is not None:
        driving_vars = [
            "DAT_CO2_CONC",
            "DAT_SURFACE_TEMP",
            "DAT_CO2B_EMIS",
        ]
        driving_units = {
            "Atmospheric Concentrations|CO2": "ppm",
            "Surface Temperature": "K",
            "Emissions|CO2|Land Use": "GtC/yr",
        }
        filtered = filter_results(std_results, driving_vars)
        for scm_var in filtered.get_unique_meta("variable"):
            sub = filtered.filter(variable=scm_var, region="World")
            if len(sub) == 0:
                sub = filtered.filter(variable=scm_var)
            if len(sub) == 0:
                continue
            ts = sub.timeseries().iloc[0]
            unit = driving_units.get(scm_var, str(sub.get_unique_meta("unit")[0]))
            row = {
                "climate_model": "MAGICC7",
                "model": "unspecified",
                "region": "World",
                "scenario": name,
                "todo": "SET",
                "unit": unit,
                "variable": scm_var,
            }
            for dt_idx, val in ts.items():
                timestamp = f"{dt_idx.year}-01-01 00:00:00"
                row[timestamp] = val
            rows.append(row)

    wide_df = pd.DataFrame(rows)
    csv_path = OUT / f"{name}.csv"
    wide_df.to_csv(csv_path, index=False)

    # Save config
    json_types = (int, float, str, bool, list, dict, type(None))
    config_clean = {
        k: v if isinstance(v, json_types) else str(v) for k, v in config.items()
    }
    json_path = OUT / f"{name}_config.json"
    with open(json_path, "w") as f:
        json.dump(config_clean, f, indent=2)

    n_total = len(variables) + (3 if std_results else 0)
    size_kb = csv_path.stat().st_size / 1024
    print(f"  Saved: {name}.csv ({size_kb:.1f} KB, {n_total} vars)")


def _parse_dat_file_global(out_dir: str, filename: str) -> dict[int, float]:
    """Parse a DAT_*.OUT file and return {year: global_value}."""
    fpath = os.path.join(out_dir, filename)
    if not os.path.exists(fpath):
        return {}

    with open(fpath) as f:
        lines = f.readlines()

    first_data = 18
    for line in lines:
        if "THISFILE_FIRSTDATAROW" in line:
            first_data = int(line.split("=")[1].split(",")[0].strip())
            break

    result = {}
    for line in lines[first_data - 1 :]:
        parts = line.split()
        if len(parts) >= 2:  # noqa: PLR2004
            try:
                year = int(float(parts[0]))
                val = float(parts[1])  # First data column = GLOBAL
                result[year] = val
            except ValueError:
                continue
    return result


def run_with_carboncycle(config: dict) -> tuple:
    """Run MAGICC7 with carbon cycle output enabled.

    Returns (carboncycle_df, standard_results) so callers can extract
    both CARBONCYCLE.OUT diagnostics and standard DAT_* outputs
    (CO2 concentration, temperature, land-use emissions).
    """
    with run_magicc_ctx() as m:
        m.set_output_variables(
            write_ascii=True,
            write_binary=False,
            emissions=True,
            concentrations=True,
            carboncycle=True,
            forcing=True,
            temperature=True,
        )
        res = m.run(out_carboncycle=1, **config)
        out_dir = os.path.join(m.root_dir, "out")
        cc_df = parse_carboncycle_out(out_dir)

        # Also parse raw land-use emissions (not in CARBONCYCLE.OUT)
        co2b = _parse_dat_file_global(out_dir, "DAT_CO2B_EMIS.OUT")
        if co2b:
            cc_df["CO2B_EMIS"] = cc_df["YEARS"].map(co2b).fillna(0.0)

    return cc_df, res


def test_01_pi_steady_state():
    """Pre-industrial steady state: constant CO2, no temperature anomaly."""
    config = make_config(
        DEFAULT_CLIMATE,
        NO_VARIABILITY,
        startyear=1750,
        endyear=1850,
        co2_switchfromconc2emis_year=5000,
        ch4_switchfromconc2emis_year=5000,
        n2o_switchfromconc2emis_year=5000,
        file_co2_conc="1850_CO2_CONC.IN",
        file_ch4_conc="CONST_CH4_CONC.IN",
        file_n2o_conc="CONST_N2O_CONC.IN",
    )

    cc_df, res = run_with_carboncycle(config)
    carboncycle_to_regression_csv(cc_df, "01_pi_steady_state", config, std_results=res)


def test_02_co2_fertilization_only():
    """SSP245 CO2 concentrations, temperature feedbacks disabled."""
    config = make_config(
        DEFAULT_CLIMATE,
        NO_VARIABILITY,
        startyear=1750,
        endyear=2100,
        co2_switchfromconc2emis_year=5000,
        ch4_switchfromconc2emis_year=5000,
        n2o_switchfromconc2emis_year=5000,
        file_co2_conc="SSP245_CO2_CONC.IN",
        file_ch4_conc="CONST_CH4_CONC.IN",
        file_n2o_conc="CONST_N2O_CONC.IN",
        co2_tempfeedback_switch=0,
        co2_fertilization_method=1.10,
    )

    cc_df, res = run_with_carboncycle(config)
    carboncycle_to_regression_csv(
        cc_df, "02_co2_fertilization_only", config, std_results=res
    )


def test_03_co2_and_temperature():
    """SSP245 CO2 concentrations with temperature feedbacks enabled."""
    config = make_config(
        DEFAULT_CLIMATE,
        NO_VARIABILITY,
        startyear=1750,
        endyear=2100,
        co2_switchfromconc2emis_year=5000,
        ch4_switchfromconc2emis_year=5000,
        n2o_switchfromconc2emis_year=5000,
        file_co2_conc="SSP245_CO2_CONC.IN",
        file_ch4_conc="CONST_CH4_CONC.IN",
        file_n2o_conc="CONST_N2O_CONC.IN",
        co2_tempfeedback_switch=1,
        co2_fertilization_method=1.10,
    )

    cc_df, res = run_with_carboncycle(config)
    carboncycle_to_regression_csv(
        cc_df, "03_co2_and_temperature", config, std_results=res
    )


def test_04_emissions_driven():
    """Full emissions-driven SSP245 with all feedbacks."""
    config = make_config(
        DEFAULT_CLIMATE,
        NO_VARIABILITY,
        startyear=1750,
        endyear=2100,
        co2_tempfeedback_switch=1,
        co2_fertilization_method=1.10,
    )

    cc_df, res = run_with_carboncycle(config)
    carboncycle_to_regression_csv(cc_df, "04_emissions_driven", config, std_results=res)


def test_05_gifford_fertilization():
    """SSP245 concentrations with Gifford (method=2.0) fertilization."""
    config = make_config(
        DEFAULT_CLIMATE,
        NO_VARIABILITY,
        startyear=1750,
        endyear=2100,
        co2_switchfromconc2emis_year=5000,
        ch4_switchfromconc2emis_year=5000,
        n2o_switchfromconc2emis_year=5000,
        file_co2_conc="SSP245_CO2_CONC.IN",
        file_ch4_conc="CONST_CH4_CONC.IN",
        file_n2o_conc="CONST_N2O_CONC.IN",
        co2_tempfeedback_switch=0,
        co2_fertilization_method=2.0,
    )

    cc_df, res = run_with_carboncycle(config)
    carboncycle_to_regression_csv(
        cc_df, "05_gifford_fertilization", config, std_results=res
    )


def test_06_resp_method2():
    """SSP245 concentrations with respiration method 2."""
    config = make_config(
        DEFAULT_CLIMATE,
        NO_VARIABILITY,
        startyear=1750,
        endyear=2100,
        co2_switchfromconc2emis_year=5000,
        ch4_switchfromconc2emis_year=5000,
        n2o_switchfromconc2emis_year=5000,
        file_co2_conc="SSP245_CO2_CONC.IN",
        file_ch4_conc="CONST_CH4_CONC.IN",
        file_n2o_conc="CONST_N2O_CONC.IN",
        co2_tempfeedback_switch=1,
        co2_fertilization_method=1.10,
        co2_plantboxresp_method=2,
        co2_plantboxresp_fertscale=0.5,
    )

    cc_df, res = run_with_carboncycle(config)
    carboncycle_to_regression_csv(cc_df, "06_resp_method2", config, std_results=res)


TESTS = [
    ("01", "PI Steady State", test_01_pi_steady_state),
    ("02", "CO2 Fertilization Only", test_02_co2_fertilization_only),
    ("03", "CO2 + Temperature Feedbacks", test_03_co2_and_temperature),
    ("04", "Emissions-Driven (Full)", test_04_emissions_driven),
    ("05", "Gifford Fertilization", test_05_gifford_fertilization),
    ("06", "Respiration Method 2", test_06_resp_method2),
]

if __name__ == "__main__":
    print("Generating terrestrial carbon regression data...")
    print(f"Output directory: {OUT}")
    print(f"MAGICC root: {MAGICC_ROOT}")
    print()

    for test_id, name, func in TESTS:
        print(f"Running Test {test_id}: {name}...")
        try:
            func()
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            raise

    print()
    print("=" * 60)
    total_size = sum(f.stat().st_size for f in OUT.glob("*.csv"))
    print(f"Total CSV size: {total_size / 1024:.1f} KB")
