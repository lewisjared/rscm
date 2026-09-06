"""Execution contracts for partial MAGICC runs, independent of Fortran parity."""

import numpy as np
import numpy.testing as npt
import pytest

from rscm.core import InterpolationStrategy, ModelBuilder, TimeAxis, Timeseries
from rscm.magicc import ClimateUDEBBuilder, CO2Emissions, GhgForcingBuilder, run_magicc


def concentrations(n):
    return {
        "CO2": np.full(n, 278.0),
        "CH4": np.full(n, 722.0),
        "N2O": np.full(n, 270.0),
    }


def test_preindustrial_equilibrium():
    years = np.arange(1750.0, 1801.0)
    result = run_magicc(
        years, concentrations(len(years)), other_forcing=np.zeros(len(years))
    )
    npt.assert_array_equal(result.years, years)
    npt.assert_allclose(result.values["Effective Radiative Forcing"], 0.0, atol=1e-12)
    npt.assert_allclose(result.values["Surface Temperature"], 0.0, atol=1e-12)
    assert result.sources["Atmospheric Concentration|CO2"] == "prescribed"


def test_initial_heat_uptake_respects_efficacy():
    result = run_magicc(
        [1750, 1751],
        concentrations(2),
        other_forcing=[1.0, 1.0],
        climate_parameters={"efficacy_apply": 1, "prescribed_efficacy_co2": 2.0},
    )
    npt.assert_allclose(result.values["Heat Uptake"][0], 2.0)


@pytest.mark.parametrize("method", ["Ipcctar", "Olbl"])
def test_forcing_has_same_boundary_as_concentrations(method):
    years = np.arange(2000.0, 2011.0)
    conc = concentrations(len(years))
    conc["CO2"] = np.linspace(400.0, 556.0, len(years))
    result = run_magicc(
        years,
        conc,
        other_forcing=np.zeros(len(years)),
        forcing_parameters={"method": method, "adjust_co2": 1.0},
        forcing_species=("CO2",),
    )
    # Includes nonzero initial forcing and the final input boundary.
    forcing = GhgForcingBuilder.from_parameters({"method": method, "adjust_co2": 1.0})
    expected = [forcing.calculate_forcings(c, 722.0, 270.0)["CO2"] for c in conc["CO2"]]
    npt.assert_allclose(result.values["Effective Radiative Forcing"], expected)
    npt.assert_allclose(result.values["Heat Uptake"][0], expected[0])
    assert result.values["Surface Temperature"][-1] > 0.0


@pytest.mark.parametrize("land_fractions", [(0.42, 0.21), (0.3, 0.25)])
def test_calculated_forcing_matches_prescribed_climate_run(land_fractions):
    years = np.arange(1750.0, 1781.0)
    conc = concentrations(len(years))
    conc["CO2"][1:] = 556.0
    other = np.linspace(0.3, -0.5, len(years))
    nh_land, sh_land = land_fractions
    climate_parameters = {"nh_land_fraction": nh_land, "sh_land_fraction": sh_land}
    result = run_magicc(
        years,
        conc,
        other_forcing=other,
        forcing_parameters={"method": "Ipcctar", "adjust_co2": 1.0},
        forcing_species=("CO2",),
        climate_parameters=climate_parameters,
    )
    expected_erf = 3.71 * np.log2(conc["CO2"] / 278.0) + other
    npt.assert_allclose(result.values["Effective Radiative Forcing"], expected_erf)

    axis = TimeAxis.from_values(years)
    model = (
        ModelBuilder()
        .with_time_axis(axis)
        .with_rust_component(
            ClimateUDEBBuilder.from_parameters(climate_parameters).build()
        )
        .with_exogenous_variable(
            "Effective Radiative Forcing",
            Timeseries(expected_erf, axis, "W/m^2", InterpolationStrategy.Linear),
        )
        .with_initial_values({"Surface Temperature": 0.0})
        .build()
    )
    model.run()
    regional = (
        model.timeseries()
        .get_fourbox_timeseries_by_name("Surface Temperature")
        .values()
    )
    npt.assert_allclose(
        result.values["Surface Temperature"],
        regional @ (np.array([1 - nh_land, nh_land, 1 - sh_land, sh_land]) / 2),
    )


def test_partial_emissions_budget_and_concentration_replay():
    years = np.arange(1750.0, 1801.0)
    conc = concentrations(len(years))
    del conc["CO2"]
    rates = np.linspace(0.0, 12.0, len(years) - 1)
    emissions = CO2Emissions(
        initial_concentration=278.0,
        fossil=rates,
        land_use=np.ones_like(rates),
        land_uptake=np.full_like(rates, 0.5),
        ocean_uptake=np.full_like(rates, 0.25),
    )
    result = run_magicc(
        years, conc, other_forcing=np.zeros(len(years)), co2_emissions=emissions
    )
    expected = 278.0 + np.r_[0.0, np.cumsum(rates + 1.0 - 0.5 - 0.25) / 2.123]
    npt.assert_allclose(result.values["Atmospheric Concentration|CO2"], expected)
    assert result.sources["Atmospheric Concentration|CO2"] == "CO2Budget"
    assert result.sources["Carbon Flux|Ocean"] == "prescribed"

    conc["CO2"] = result.values["Atmospheric Concentration|CO2"]
    replay = run_magicc(years, conc, other_forcing=np.zeros(len(years)))
    for variable in ("Effective Radiative Forcing", "Surface Temperature"):
        npt.assert_allclose(
            result.values[variable], replay.values[variable], atol=1e-12
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, 0.0])
def test_invalid_concentrations_rejected(bad):
    conc = concentrations(3)
    conc["CH4"][1] = bad
    with pytest.raises(ValueError, match="CH4"):
        run_magicc([1750, 1751, 1752], conc, other_forcing=[0, 0, 0])


def test_missing_concentration_rejected():
    with pytest.raises(ValueError, match="N2O"):
        run_magicc(
            [1750, 1751], {"CO2": [278, 278], "CH4": [722, 722]}, other_forcing=[0, 0]
        )


@pytest.mark.parametrize(
    "years", [[1750], [1750, 1750], [1751, 1750], [1750, 1752], [1750, np.nan]]
)
def test_invalid_years_rejected(years):
    with pytest.raises(ValueError, match="years"):
        run_magicc(
            years, concentrations(len(years)), other_forcing=np.zeros(len(years))
        )


def test_conflicting_co2_ownership_rejected():
    emissions = CO2Emissions(278.0, [0], [0], [0], [0])
    with pytest.raises(ValueError, match="CO2"):
        run_magicc(
            [1750, 1751],
            concentrations(2),
            other_forcing=[0, 0],
            co2_emissions=emissions,
        )


def test_interval_driver_length_rejected():
    emissions = CO2Emissions(278.0, [0, 0], [0], [0], [0])
    conc = concentrations(2)
    del conc["CO2"]
    with pytest.raises(ValueError, match="Fossil"):
        run_magicc([1750, 1751], conc, other_forcing=[0, 0], co2_emissions=emissions)


@pytest.mark.parametrize("other", [[0], [0, np.nan], [0, np.inf]])
def test_invalid_residual_forcing_rejected(other):
    with pytest.raises(ValueError, match="other_forcing"):
        run_magicc([1750, 1751], concentrations(2), other_forcing=other)


@pytest.mark.parametrize("species", [(), ("CO2", "CO2"), ("SF6",)])
def test_invalid_forcing_selection_rejected(species):
    with pytest.raises(ValueError, match="forcing_species"):
        run_magicc(
            [1750, 1751],
            concentrations(2),
            other_forcing=[0, 0],
            forcing_species=species,
        )


def test_nonfinite_calculated_output_fails_run():
    conc = concentrations(2)
    conc["CO2"][1] = 556.0
    with pytest.raises(RuntimeError, match="Nonfinite"):
        run_magicc(
            [1750, 1751],
            conc,
            other_forcing=[0, 0],
            forcing_parameters={"method": "Ipcctar", "adjust_co2": 1e308},
        )
