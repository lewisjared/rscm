r"""
Generate GHG forcing and chemistry regression data.

Creates reference outputs from Fortran MAGICC for validating the Rust
implementation of forcing calculations, chemistry modules, and climate
step response.

Usage:
    export MAGICC_ROOT=/path/to/magicc-v7.5.3

    uv run --with pymagicc --with scmdata --with "pandas<3" \
        scripts/generate_magicc_regression_data.py

Output:
    tests/regression_data/magicc/*.csv
    tests/regression_data/magicc/*_config.json
"""

from magicc_regression_utils import (
    CO2_ONLY,
    CONC_DRIVEN,
    DEFAULT_CLIMATE,
    NO_VARIABILITY,
    make_config,
    run_magicc,
    run_suite,
    save_results,
)


def test_01_concentration_driven():
    """Concentration-driven run with SSP245 concentrations (IPCCTAR)."""
    config = make_config(
        CONC_DRIVEN,
        DEFAULT_CLIMATE,
        startyear=1750,
        endyear=2100,
        file_co2_conc="SSP245_CO2_CONC.IN",
        file_ch4_conc="SSP245_CH4_CONC.IN",
        file_n2o_conc="SSP245_N2O_CONC.IN",
        core_co2ch4n2o_rfmethod="IPCCTAR",
        rf_solar_scale=0.0,
        rf_volcanic_scale=0.0,
    )

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

    res = run_magicc(config)
    save_results(res, "01_concentration_driven", config, out_vars)
    return res


def test_02_ghg_forcing_olbl():
    """GHG forcing with OLBL method (rapid adjustments)."""
    config = make_config(
        CONC_DRIVEN,
        DEFAULT_CLIMATE,
        startyear=1750,
        endyear=2100,
        file_co2_conc="SSP245_CO2_CONC.IN",
        file_ch4_conc="SSP245_CH4_CONC.IN",
        file_n2o_conc="SSP245_N2O_CONC.IN",
        core_co2ch4n2o_rfmethod="OLBL",
        core_rfrapidadjust_co2=1.05,
        core_rfrapidadjust_ch4=0.86,
        core_rfrapidadjust_n2o=0.93,
        rf_solar_scale=0.0,
        rf_volcanic_scale=0.0,
    )

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

    res = run_magicc(config)
    save_results(res, "02_ghg_forcing_olbl", config, out_vars)
    return res


def test_03_emissions_driven():
    """Emissions-driven run with SSP245."""
    config = make_config(
        DEFAULT_CLIMATE,
        startyear=1750,
        endyear=2100,
        file_emisscen="SSP245_EMMS.SCEN7",
    )

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
        "DAT_CO2_EMIS",
        "DAT_CO2I_EMIS",
        "DAT_CO2B_EMIS",
        "DAT_CH4_EMIS",
        "DAT_N2O_EMIS",
        "Emissions|NOx|MAGICC Fossil and Industrial",
        "Emissions|NOx|MAGICC AFOLU",
        "Emissions|CO|MAGICC Fossil and Industrial",
        "Emissions|CO|MAGICC AFOLU",
        "Emissions|NMVOC|MAGICC Fossil and Industrial",
        "Emissions|NMVOC|MAGICC AFOLU",
        "Emissions|SOx|MAGICC Fossil and Industrial",
        "Emissions|SOx|MAGICC AFOLU",
        "Emissions|BC|MAGICC Fossil and Industrial",
        "Emissions|BC|MAGICC AFOLU",
        "Emissions|OC|MAGICC Fossil and Industrial",
        "Emissions|OC|MAGICC AFOLU",
    ]

    res = run_magicc(config)
    save_results(res, "03_emissions_driven", config, out_vars)
    return res


def test_04_ecs_sweep():
    """Climate sensitivity parameter sweep (2xCO2 step response)."""
    out_vars = [
        "DAT_CO2_CONC",
        "DAT_CO2_ERF",
        "DAT_SURFACE_TEMP",
    ]

    for ecs in [1.5, 2.0, 3.0, 4.0, 4.5]:
        config = make_config(
            CONC_DRIVEN,
            CO2_ONLY,
            NO_VARIABILITY,
            startyear=1750,
            endyear=2100,
            file_co2_conc="SSP245_CO2_CONC.IN",
            file_ch4_conc="SSP245_CH4_CONC.IN",
            file_n2o_conc="SSP245_N2O_CONC.IN",
            core_climatesensitivity=ecs,
            core_delq2xco2=3.71,
        )

        res = run_magicc(config)
        save_results(res, f"04_ecs_sweep_{ecs}", config, out_vars)


def test_05_co2_only_forcing():
    """CO2-only forcing for clean comparison."""
    config = make_config(
        CONC_DRIVEN,
        CO2_ONLY,
        DEFAULT_CLIMATE,
        startyear=1750,
        endyear=2100,
        file_co2_conc="SSP245_CO2_CONC.IN",
        file_ch4_conc="SSP245_CH4_CONC.IN",
        file_n2o_conc="SSP245_N2O_CONC.IN",
        rf_solar_scale=0.0,
        rf_volcanic_scale=0.0,
    )

    out_vars = [
        "DAT_CO2_CONC",
        "DAT_CO2_RF",
        "DAT_CO2_ERF",
        "DAT_SURFACE_TEMP",
        "DAT_TOTAL_INCLVOLCANIC_RF",
        "DAT_TOTAL_INCLVOLCANIC_ERF",
    ]

    res = run_magicc(config)
    save_results(res, "05_co2_only_forcing", config, out_vars)
    return res


TESTS = [
    ("01", "Concentration-driven (IPCCTAR)", test_01_concentration_driven),
    ("02", "GHG Forcing (OLBL)", test_02_ghg_forcing_olbl),
    ("03", "Emissions-driven (SSP245)", test_03_emissions_driven),
    ("04", "Climate Sensitivity Sweep", test_04_ecs_sweep),
    ("05", "CO2-only Forcing", test_05_co2_only_forcing),
]


if __name__ == "__main__":
    run_suite("GHG forcing and chemistry regression data", TESTS)
