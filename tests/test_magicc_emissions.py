"""Execution contracts for the coupled emissions-driven MAGICC runner."""

from dataclasses import replace

import numpy as np
import numpy.testing as npt
import pytest

from rscm.magicc import Emissions, run_magicc_emissions

YEARS = np.arange(1750.0, 1851.0)

# Concentrations at which the default natural emissions alone are in balance
CH4_NATURAL = 182.1908 / (1 / 12.417920434628465 + 1 / 50) / (2.824 * 0.973)
N2O_NATURAL = 8.0345017 * 139.275 / 4.79


def emissions(scale: float = 1.0) -> Emissions:
    ramp = np.linspace(0.0, scale, len(YEARS) - 1)
    ramp = np.r_[ramp, ramp[-1]]
    return Emissions(
        co2_fossil=8.0 * ramp,
        co2_land_use=1.0 * ramp,
        ch4=300.0 * ramp,
        ch4_fossil=120.0 * ramp,
        n2o=7.0 * ramp,
        nox=40.0 * ramp,
        co=900.0 * ramp,
        nmvoc=200.0 * ramp,
    )


def run(emis: Emissions, **kwargs):
    kwargs.setdefault(
        "initial_concentrations", {"CO2": 278.0, "CH4": 722.0, "N2O": 270.0}
    )
    kwargs.setdefault("other_forcing", np.zeros(len(YEARS)))
    return run_magicc_emissions(YEARS, emis, **kwargs)


def test_rising_emissions_raise_concentrations_and_temperature():
    result = run(emissions())
    for gas in ("CO2", "CH4", "N2O"):
        assert (
            result.values[f"Atmospheric Concentration|{gas}"][-1]
            > (result.values[f"Atmospheric Concentration|{gas}"][0])
        )
    assert result.values["Surface Temperature"][-1] > 0.0
    assert result.sources["Atmospheric Concentration|CO2"] == "CO2Budget"
    assert result.sources["Atmospheric Concentration|CH4"] == "CH4Chemistry"
    assert result.sources["Emissions|CO2|Fossil"] == "prescribed"


def test_natural_steady_state_is_preserved():
    initial = {"CO2": 278.0, "CH4": CH4_NATURAL, "N2O": N2O_NATURAL}
    result = run(
        emissions(0.0),
        initial_concentrations=initial,
        parameters={
            "GhgForcing": {f"{gas.lower()}_pi": c for gas, c in initial.items()}
        },
    )
    for gas, value in initial.items():
        npt.assert_allclose(
            result.values[f"Atmospheric Concentration|{gas}"], value, rtol=1e-9
        )
    npt.assert_allclose(result.values["Surface Temperature"], 0.0, atol=1e-12)
    npt.assert_allclose(result.values["Carbon Flux|Ocean"][:-1], 0.0, atol=1e-12)


def test_co2_mass_balance():
    result = run(emissions())
    v = result.values
    sources = v["Emissions|CO2|Fossil"] + v["Emissions|CO2|CH4 Oxidation"]
    sinks = v["Carbon Flux|Terrestrial"] + v["Carbon Flux|Ocean"]
    npt.assert_allclose(
        np.diff(v["Atmospheric Concentration|CO2"]),
        (sources - sinks)[:-1] / 2.123,
        atol=1e-12,
    )
    assert v["Emissions|CO2|CH4 Oxidation"][-2] > 0.0
    assert np.isnan(v["Emissions|CO2|CH4 Oxidation"][-1])


def test_climate_feedback_follows_forcing_configuration():
    default = run(emissions())
    explicit = run(emissions(), parameters={"ClimateUDEB": {"rf_2xco2": 3.71}})
    # A smaller doubling forcing weakens the feedback, so the climate warms more
    assert (
        explicit.values["Surface Temperature"][-1]
        > default.values["Surface Temperature"][-1]
    )


def test_land_use_reduces_land_carbon():
    base = run(emissions())
    deforested = run(replace(emissions(), co2_land_use=np.full(len(YEARS), 3.0)))
    assert (
        deforested.values["Carbon Pool|Plant"][-1]
        < base.values["Carbon Pool|Plant"][-1]
    )
    assert (
        deforested.values["Atmospheric Concentration|CO2"][-1]
        > base.values["Atmospheric Concentration|CO2"][-1]
    )


def test_carbon_conversion_shared_between_budget_and_ocean():
    budget_only = run(emissions(), parameters={"CO2Budget": {"gtc_per_ppm": 2.0}})
    both = run(
        emissions(),
        parameters={
            "CO2Budget": {"gtc_per_ppm": 2.0},
            "OceanCarbon": {"gtc_per_ppm": 2.0},
        },
    )
    npt.assert_array_equal(
        budget_only.values["Atmospheric Concentration|CO2"],
        both.values["Atmospheric Concentration|CO2"],
    )
    with pytest.raises(ValueError, match="gtc_per_ppm"):
        run(
            emissions(),
            parameters={
                "CO2Budget": {"gtc_per_ppm": 2.0},
                "OceanCarbon": {"gtc_per_ppm": 2.1},
            },
        )


def test_ch4_budget_shared_between_chemistry_and_oxidation():
    concentration = "Atmospheric Concentration|CH4"
    chemistry_only = run(emissions(), parameters={"CH4Chemistry": {"ppb_to_tg": 3.0}})
    oxidation_only = run(emissions(), parameters={"CH4Oxidation": {"ppb_to_tg": 3.0}})
    npt.assert_array_equal(
        chemistry_only.values[concentration], oxidation_only.values[concentration]
    )
    with pytest.raises(ValueError, match="natural_emissions"):
        run(
            emissions(),
            parameters={
                "CH4Chemistry": {"natural_emissions": 200.0},
                "CH4Oxidation": {"natural_emissions": 210.0},
            },
        )
    with pytest.raises(ValueError, match="tau_other"):
        run(
            emissions(),
            parameters={
                "CH4Chemistry": {"tau_soil": 100.0},
                "CH4Oxidation": {"tau_other": 50.0},
            },
        )


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="Unknown component"):
        run(emissions(), parameters={"Ozone": {}})
    with pytest.raises(ValueError, match="initial_concentrations"):
        run(emissions(), initial_concentrations={"CO2": 278.0})
    with pytest.raises(ValueError, match="co2_fossil has 100 values, one per interval"):
        run(replace(emissions(), co2_fossil=np.zeros(len(YEARS) - 1)))
    with pytest.raises(ValueError, match="ch4_fossil"):
        run(replace(emissions(), ch4_fossil=emissions().ch4 + 1.0))
    with pytest.raises(ValueError, match="other_forcing"):
        run(emissions(), other_forcing=np.zeros(len(YEARS) - 1))
    with pytest.raises(ValueError, match="years"):
        run_magicc_emissions(
            [1750.0],
            emissions(),
            initial_concentrations={"CO2": 278.0, "CH4": 722.0, "N2O": 270.0},
            other_forcing=[0.0],
        )
