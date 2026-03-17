r"""
Generate ocean UDEB regression data.

Creates reference outputs from Fortran MAGICC for validating the Rust
upwelling-diffusion energy balance (UDEB) implementation. Tests
progressively enable ocean physics to isolate specific features.

Usage:
    export MAGICC_ROOT=/path/to/magicc-v7.5.3

    uv run --with pymagicc --with scmdata --with "pandas<3" \
        scripts/generate_ocean_regression_data.py

Output:
    tests/regression_data/magicc/ocean_*.csv
    tests/regression_data/magicc/ocean_*_config.json
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

OCEAN_OUT_VARS = [
    "DAT_SURFACE_TEMP",
    "DAT_HEATUPTK_AGGREG",
    "DAT_UPWELLING_RATE",
    "Ocean Temperature|Layer 1",
    "Ocean Temperature|Layer 5",
    "Ocean Temperature|Layer 10",
    "Ocean Temperature|Layer 25",
    "Ocean Temperature|Layer 50",
]

# All ocean physics disabled -- the base for additive tests.
_OCEAN_DISABLED = {
    "core_initial_upwelling_rate": 0,
    "core_landheatcapacity_apply": 0,
    "core_ocn_depthdependent": 0,
    "core_feedback_qsensitivity": 0,
    "core_feedback_cumtsensitivity": 0,
    "core_heatxchange_northsouth": 0,
    "core_verticaldiff_top_dkdt": 0,
}

# Common base: concentration-driven, CO2-only, no variability.
_OCEAN_BASE = {
    **CONC_DRIVEN,
    **CO2_ONLY,
    **DEFAULT_CLIMATE,
    **NO_VARIABILITY,
}


def _ocean_config(**overrides):
    """Build an ocean test config: base + all-disabled + per-test overrides."""
    return make_config(_OCEAN_BASE, _OCEAN_DISABLED, **overrides)


def test_01_diffusion_only():
    """
    Cylindrical ocean, diffusion only.

    All ocean feedbacks disabled (upwelling, ground heat, depth-dependent
    area, time-varying ECS, K_NS, temperature-dependent diffusivity).
    Isolates the pure 1D diffusion + Thomas solver + LAMCALC + land-ocean coupling.
    """
    config = _ocean_config(
        startyear=1850,
        endyear=2150,
        file_co2_conc="ABRUPT-2XCO2_CO2_CONC.IN",
    )

    res = run_magicc(config)
    save_results(res, "ocean_01_diffusion_only", config, OCEAN_OUT_VARS)
    return res


def test_02_constant_upwelling():
    """
    Cylindrical ocean, diffusion + constant upwelling (3.5 m/yr).

    Adds upwelling advection and polar sinking entrainment to Test 1.
    CORE_UPWELLING_VARIABLE_PART=0 keeps upwelling constant.
    """
    config = _ocean_config(
        startyear=1850,
        endyear=2150,
        file_co2_conc="ABRUPT-2XCO2_CO2_CONC.IN",
        core_initial_upwelling_rate=3.5,
        core_upwelling_variable_part=0,
    )

    res = run_magicc(config)
    save_results(res, "ocean_02_constant_upwelling", config, OCEAN_OUT_VARS)
    return res


def test_03_depth_dependent_area():
    """
    Depth-dependent ocean area (hypsometric profile).

    Adds CORE_OCN_DEPTHDEPENDENT=1 to Test 2. Tests the area factor
    calculations (TOPFLOW, BOTTOMFLOW, DIFFFLOW) and their effect on
    diffusion and entrainment.
    """
    config = _ocean_config(
        startyear=1850,
        endyear=2150,
        file_co2_conc="ABRUPT-2XCO2_CO2_CONC.IN",
        core_initial_upwelling_rate=3.5,
        core_upwelling_variable_part=0,
        core_ocn_depthdependent=1,
    )

    res = run_magicc(config)
    save_results(res, "ocean_03_depth_dependent_area", config, OCEAN_OUT_VARS)
    return res


def test_04_variable_upwelling():
    """
    Temperature-dependent upwelling.

    Adds CORE_UPWELLING_VARIABLE_PART=0.7 to Test 3. 70% of upwelling
    varies with surface temperature (positive feedback: warming reduces
    upwelling, reducing ocean heat uptake efficiency).
    """
    config = _ocean_config(
        startyear=1850,
        endyear=2150,
        file_co2_conc="ABRUPT-2XCO2_CO2_CONC.IN",
        core_initial_upwelling_rate=3.5,
        core_upwelling_variable_part=0.7,
        core_ocn_depthdependent=1,
    )

    res = run_magicc(config)
    save_results(res, "ocean_04_variable_upwelling", config, OCEAN_OUT_VARS)
    return res


def test_05_temp_dependent_diffusivity():
    """
    Temperature-dependent vertical diffusivity.

    Adds CORE_VERTICALDIFF_TOP_DKDT=-0.1910108 to Test 2 (cylindrical,
    constant upwelling). Diffusivity decreases with surface warming.
    """
    config = _ocean_config(
        startyear=1850,
        endyear=2150,
        file_co2_conc="ABRUPT-2XCO2_CO2_CONC.IN",
        core_initial_upwelling_rate=3.5,
        core_upwelling_variable_part=0,
        core_verticaldiff_top_dkdt=-0.1910108,
    )

    res = run_magicc(config)
    save_results(res, "ocean_05_temp_dependent_diffusivity", config, OCEAN_OUT_VARS)
    return res


def test_06_ground_heat():
    """
    Ground heat reservoir.

    Adds CORE_LANDHEATCAPACITY_APPLY=1 to Test 2. The ground reservoir
    provides additional thermal inertia for the land fraction.
    """
    config = _ocean_config(
        startyear=1850,
        endyear=2150,
        file_co2_conc="ABRUPT-2XCO2_CO2_CONC.IN",
        core_initial_upwelling_rate=3.5,
        core_upwelling_variable_part=0,
        core_landheatcapacity_apply=1,
        core_landhc_effthickness=300,
        core_heatxchange_landground=0.1,
    )

    res = run_magicc(config)
    save_results(res, "ocean_06_ground_heat", config, OCEAN_OUT_VARS)
    return res


def test_07_interhemispheric_exchange():
    """
    Inter-hemispheric heat exchange.

    Adds CORE_HEATXCHANGE_NORTHSOUTH=0.3115475 to Test 2. Even with
    uniform forcing, asymmetry arises from different NH/SH land fractions.
    """
    config = _ocean_config(
        startyear=1850,
        endyear=2150,
        file_co2_conc="ABRUPT-2XCO2_CO2_CONC.IN",
        core_initial_upwelling_rate=3.5,
        core_upwelling_variable_part=0,
        core_heatxchange_northsouth=0.3115475,
    )

    res = run_magicc(config)
    save_results(res, "ocean_07_interhemispheric_exchange", config, OCEAN_OUT_VARS)
    return res


def test_08_sst_to_sat():
    """
    SST-to-air temperature conversion.

    Short run (1850-1860) with default parameters. Validates the
    quadratic SST-to-SAT conversion over ocean.
    """
    config = make_config(
        _OCEAN_BASE,
        startyear=1850,
        endyear=1860,
        file_co2_conc="ABRUPT-2XCO2_CO2_CONC.IN",
    )

    out_vars = ["DAT_SURFACE_TEMP"]

    res = run_magicc(config)
    save_results(res, "ocean_08_sst_to_sat", config, out_vars)
    return res


def test_09_time_varying_ecs():
    """
    Time-varying ECS (state-dependent feedback).

    Adds cumulative-temperature and forcing-level ECS adjustments
    to Test 3. Effective climate sensitivity changes over time.
    """
    config = _ocean_config(
        startyear=1850,
        endyear=2150,
        file_co2_conc="ABRUPT-2XCO2_CO2_CONC.IN",
        core_initial_upwelling_rate=3.5,
        core_upwelling_variable_part=0,
        core_ocn_depthdependent=1,
        core_feedback_cumtsensitivity=0.08,
        core_feedback_qsensitivity=7.84e-9,
    )

    res = run_magicc(config)
    save_results(res, "ocean_09_time_varying_ecs", config, OCEAN_OUT_VARS)
    return res


def test_10_full_default():
    """
    Full default configuration with 1pctCO2.

    All default ocean parameters enabled. Standard transient experiment
    for TCR validation.
    """
    config = make_config(
        _OCEAN_BASE,
        startyear=1850,
        endyear=2000,
        file_co2_conc="1PCTCO2_CO2_CONC.IN",
    )

    out_vars = [*OCEAN_OUT_VARS, "DAT_HEATCONTENT_AGGREG_TOTAL"]

    res = run_magicc(config)
    save_results(res, "ocean_10_full_default", config, out_vars)
    return res


TESTS = [
    ("01", "Diffusion Only", test_01_diffusion_only),
    ("02", "Constant Upwelling", test_02_constant_upwelling),
    ("03", "Depth-Dependent Area", test_03_depth_dependent_area),
    ("04", "Variable Upwelling", test_04_variable_upwelling),
    ("05", "Temperature-Dependent Diffusivity", test_05_temp_dependent_diffusivity),
    ("06", "Ground Heat Reservoir", test_06_ground_heat),
    ("07", "Inter-Hemispheric Exchange", test_07_interhemispheric_exchange),
    ("08", "SST-to-SAT Conversion", test_08_sst_to_sat),
    ("09", "Time-Varying ECS", test_09_time_varying_ecs),
    ("10", "Full Default (1pctCO2)", test_10_full_default),
]


if __name__ == "__main__":
    run_suite("Ocean UDEB regression data", TESTS)
