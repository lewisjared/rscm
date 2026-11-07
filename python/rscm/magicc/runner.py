"""Annual concentration and partial CO2 emissions runs using MAGICC components.

This runner deliberately prescribes the processes it cannot yet couple. It does
not claim full MAGICC7 parity or implement inverse emissions or mode switching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rscm._lib.magicc import ClimateUDEBBuilder, CO2BudgetBuilder, GhgForcingBuilder
from rscm.core import (
    GridType,
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    TimeseriesType,
    VariableSchema,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray

    from rscm._lib.magicc import ClimateUDEBParams, GhgForcingParams
    from rscm.core import Model

__all__ = ["CO2Emissions", "MAGICCResult", "run_magicc"]

CONCENTRATION = "Atmospheric Concentration|"
ERF = "Effective Radiative Forcing"
SPECIES = ("CO2", "CH4", "N2O")


@dataclass(frozen=True)
class CO2Emissions:
    """CO2 budget inputs with prescribed carbon uptake.

    ``initial_concentration`` is in ppm. All four arrays contain annual mean
    rates in GtC/yr, one for each supplied year, labelled by the year they start.
    Positive uptake removes carbon from the atmosphere. Provide explicit zeros to
    disable land-use emissions or either sink.
    """

    initial_concentration: float
    fossil: ArrayLike
    land_use: ArrayLike
    land_uptake: ArrayLike
    ocean_uptake: ArrayLike


@dataclass(frozen=True)
class MAGICCResult:
    """Run output and the processes used to produce it.

    ``values`` contains every variable at the year labels,
    with global surface temperature computed using the configured box areas.
    A point, such as a concentration, holds at its label.
    An average, such as a flux, holds over the year its label starts,
    so a modelled average has no value at the last label.
    ``regional_temperature`` retains the four individual surface temperatures.
    ``sources`` identifies calculated and prescribed variables;
    ``forcing_contributors`` identifies the terms included in total ERF.
    """

    years: NDArray[np.float64]
    values: dict[str, NDArray[np.float64]]
    units: dict[str, str]
    regional_temperature: NDArray[np.float64]
    sources: dict[str, str]
    forcing_contributors: tuple[str, ...]


def _series(
    name: str, values: ArrayLike, size: int, *, positive: bool = False
) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        msg = f"{name} must contain {size} finite values"
        raise ValueError(msg)
    if positive and np.any(result <= 0):
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return result


def _average_series(name: str, values: ArrayLike, size: int) -> NDArray[np.float64]:
    """Annual means, one per year label."""
    if np.shape(values) == (size - 1,):
        msg = (
            f"{name} has {size - 1} values, one per interval. "
            f"Give one value per year, {size} in all."
        )
        raise ValueError(msg)
    return _series(name, values, size)


def _validate_axis(
    years: ArrayLike, forcing_species: tuple[str, ...]
) -> NDArray[np.float64]:
    """Validate annual boundaries and the selected forcing gases."""
    time = np.array(years, dtype=np.float64, copy=True)
    if (
        time.ndim != 1
        or len(time) <= 1
        or not np.isfinite(time).all()
        or not np.all(np.diff(time) == 1)
    ):
        msg = "years must contain at least two consecutive annual boundaries"
        raise ValueError(msg)
    if (
        not forcing_species
        or len(set(forcing_species)) != len(forcing_species)
        or not set(forcing_species) <= set(SPECIES)
    ):
        msg = "forcing_species must select CO2, CH4 and/or N2O without duplicates"
        raise ValueError(msg)
    return time


def _prepare_inputs(
    years: ArrayLike,
    concentrations: Mapping[str, ArrayLike],
    other_forcing: ArrayLike,
    co2_emissions: CO2Emissions | None,
    forcing_species: tuple[str, ...],
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Validate boundary data and ownership before constructing components."""
    time = _validate_axis(years, forcing_species)
    required = set(SPECIES) - ({"CO2"} if co2_emissions is not None else set())
    if set(concentrations) != required:
        msg = (
            f"Concentration inputs must be exactly {sorted(required)}; "
            f"missing={sorted(required - set(concentrations))}, "
            f"unexpected={sorted(set(concentrations) - required)}"
        )
        raise ValueError(msg)

    data = {
        CONCENTRATION + gas: _series(gas, concentrations[gas], len(time), positive=True)
        for gas in required
    }
    data[ERF + "|Other"] = _series("other_forcing", other_forcing, len(time))
    return time, data


def run_magicc(  # noqa: PLR0913
    years: ArrayLike,
    concentrations: Mapping[str, ArrayLike],
    *,
    other_forcing: ArrayLike,
    co2_emissions: CO2Emissions | None = None,
    forcing_species: tuple[str, ...] = SPECIES,
    forcing_parameters: GhgForcingParams | None = None,
    climate_parameters: ClimateUDEBParams | None = None,
) -> MAGICCResult:
    """Run GHG forcing, aggregation and UDEB on annual boundaries.

    ``years`` must contain at least two consecutive annual boundaries. Supply
    CO2 in ppm and CH4/N2O in ppb, keyed by species, with one concentration per
    boundary. In partial emissions mode, omit CO2 from ``concentrations`` and
    supply ``co2_emissions`` instead. CH4 and N2O remain prescribed.

    ``other_forcing`` is an explicit residual ERF in W/m² at every boundary,
    covering all forcing outside ``forcing_species``. Supply zeros for an
    experiment that excludes those processes. Do not include calculated GHG
    contributions again in this residual. CO2-only experiments can select
    ``forcing_species=("CO2",)``; all three concentrations are still required
    for absorption overlap calculations.

    Component parameter dictionaries use their Rust names and defaults, except
    that ``rf_2xco2`` defaults to the CO2-doubling ERF of the forcing
    configuration, as in MAGICC7.
    Preindustrial reference concentrations are independent of the start year.
    Climate starts from an unperturbed ocean and zero temperature anomaly.

    Raises ``ValueError`` for invalid or ambiguous inputs and ``RuntimeError``
    for nonfinite run outputs. This is a partial model: carbon sinks and all
    non-GHG forcing are prescribed, with no temperature feedback to carbon.
    """
    time, data = _prepare_inputs(
        years, concentrations, other_forcing, co2_emissions, forcing_species
    )
    units = {CONCENTRATION + gas: "ppm" if gas == "CO2" else "ppb" for gas in SPECIES}
    units[ERF + "|Other"] = "W/m^2"
    sources = {**dict.fromkeys(data, "prescribed"), ERF: "sum"}
    initial: dict[str, float] = {}
    budget = None
    if co2_emissions is not None:
        initial[CONCENTRATION + "CO2"] = float(
            _series(
                "initial CO2", [co2_emissions.initial_concentration], 1, positive=True
            )[0]
        )
        fossil, land_use, land_uptake, ocean_uptake = (
            _average_series(name, values, len(time))
            for name, values in (
                ("Emissions|CO2|Fossil", co2_emissions.fossil),
                ("Emissions|CO2|Land Use", co2_emissions.land_use),
                ("land uptake", co2_emissions.land_uptake),
                ("ocean uptake", co2_emissions.ocean_uptake),
            )
        )
        # The budget takes the net land flux, which includes land-use emissions
        for name, values in {
            "Emissions|CO2|Fossil": fossil,
            "Emissions|CO2|Land Use": land_use,
            "Emissions|CO2|CH4 Oxidation": np.zeros(len(time)),
            "Carbon Flux|Terrestrial": land_uptake - land_use,
            "Carbon Flux|Ocean": ocean_uptake,
        }.items():
            data[name] = values
            units[name] = "GtC/yr"
            sources[name] = "prescribed"
        budget = CO2BudgetBuilder.from_parameters({}).build()
        sources[CONCENTRATION + "CO2"] = "CO2Budget"

    forcing = GhgForcingBuilder.from_parameters(forcing_parameters or {})
    start_co2 = (
        initial[CONCENTRATION + "CO2"]
        if co2_emissions is not None
        else float(data[CONCENTRATION + "CO2"][0])
    )
    initial_erf = forcing.calculate_forcings(
        start_co2,
        float(data[CONCENTRATION + "CH4"][0]),
        float(data[CONCENTRATION + "N2O"][0]),
    )
    for gas, value in initial_erf.items():
        name = ERF + "|" + gas
        initial[name] = value
        units[name] = "W/m^2"
        sources[name] = "GhgForcing"
    initial[ERF] = sum(initial_erf[gas] for gas in forcing_species) + float(
        data[ERF + "|Other"][0]
    )
    if not all(np.isfinite(value) for value in initial.values()):
        msg = "Component parameters produced nonfinite initial forcing"
        raise ValueError(msg)

    climate_params: ClimateUDEBParams = {
        "rf_2xco2": forcing.co2_doubling_forcing(),
        **(climate_parameters or {}),
    }
    climate_builder = ClimateUDEBBuilder.from_parameters(climate_params)
    climate = climate_builder.build()
    initial.update(climate_builder.initial_values(initial[ERF]))
    weights = _climate_weights(climate_params)
    units.update(
        {
            ERF: "W/m^2",
            "Surface Temperature": "K",
            "Sea Surface Temperature": "K",
            "Heat Uptake": "W/m^2",
            "Ocean Heat Content": "J/m^2",
        }
    )
    for name in (
        "Surface Temperature",
        "Sea Surface Temperature",
        "Heat Uptake",
        "Ocean Heat Content",
    ):
        sources[name] = "ClimateUDEB"

    schema = _base_schema()
    for name, unit in units.items():
        if name != ERF:
            schema.add_variable(
                name,
                unit,
                GridType.FourBox if name == "Surface Temperature" else GridType.Scalar,
            )
    if budget is not None:
        schema.add_variable("Emissions|CO2|Net", "GtC/yr")
        schema.add_variable("Airborne Fraction|CO2", "1")
    contributors = (*(ERF + "|" + gas for gas in forcing_species), ERF + "|Other")
    schema.add_aggregate(ERF, "W/m^2", "sum", list(contributors))
    axis = TimeAxis.from_values(time)
    # Components reading global temperature must see the area-weighted mean
    builder = (
        ModelBuilder()
        .with_time_axis(axis)
        .with_schema(schema)
        .with_grid_weights(GridType.FourBox, list(weights))
    )
    # Register the budget before forcing: its current-step concentration must
    # be available before the forcing diagnostic is evaluated.
    if budget is not None:
        builder.with_component(budget)
    builder.with_component(forcing.build()).with_component(climate)
    for name, values in data.items():
        builder.with_prescribed_variable(
            name, Timeseries(values, axis, units[name], InterpolationStrategy.Linear)
        )
    builder.with_initial_values(initial)
    model = builder.build()
    model.run()
    return _collect_result(model, time, units, sources, contributors, weights)


def _collect_result(  # noqa: PLR0913
    model: Model,
    time: NDArray[np.float64],
    units: dict[str, str],
    sources: dict[str, str],
    contributors: tuple[str, ...],
    weights: NDArray[np.float64],
    diagnostics: frozenset[str] = frozenset(),
) -> MAGICCResult:
    """Validate every returned value and retain the four regional temperatures.

    An average has no value at the last label.
    ``diagnostics`` are per-step points, which have no value at the first label.
    """
    collection = model.timeseries()
    temperature = collection.get_four_box_timeseries_by_name("Surface Temperature")
    if temperature is None:
        msg = "Missing Surface Temperature output"
        raise RuntimeError(msg)
    regional = temperature.values().copy()
    values = {}
    for name in units:
        if name != "Surface Temperature":
            series = collection.get_timeseries_by_name(name)
            if series is None:
                msg = f"Missing {name} output"
                raise RuntimeError(msg)
            values[name] = series.values().copy()
    values["Surface Temperature"] = regional @ weights
    for name, output in {**values, "Regional Surface Temperature": regional}.items():
        start = 1 if name in diagnostics else 0
        average = model.timeseries_type(name) == TimeseriesType.Average
        end = len(time) - 1 if average else len(time)
        if not np.isfinite(output[start:end]).all():
            index = start + int(np.argwhere(~np.isfinite(output[start:end]))[0, 0])
            msg = f"Nonfinite {name} at year {time[index]}"
            raise RuntimeError(msg)
    for gas in SPECIES:
        if np.any(values[CONCENTRATION + gas] <= 0):
            msg = f"Nonpositive {gas} concentration calculated during run"
            raise RuntimeError(msg)
    return MAGICCResult(time, values, units, regional, sources, contributors)


def _base_schema() -> VariableSchema:
    """Return a schema declaring the component outputs that no runner uses."""
    schema = VariableSchema()
    for name in ("2xCO2", "CH4 Oxidation Stratospheric H2O"):
        schema.add_variable(f"{ERF}|{name}", "W/m^2")
    schema.add_variable("Equilibrium Climate Sensitivity", "K")
    return schema


def _climate_weights(parameters: ClimateUDEBParams) -> NDArray[np.float64]:
    nh_land = parameters.get("nh_land_fraction", 0.42)
    sh_land = parameters.get("sh_land_fraction", 0.21)
    return np.array([1 - nh_land, nh_land, 1 - sh_land, sh_land]) * 0.5


def _ch4_burdens(
    concentration: float, ppb_to_tg: float, mixbox_size: float
) -> dict[str, float]:
    """Return initial CH4Oxidation pools, with all CH4 natural as in MAGICC7."""
    burden = "Atmospheric Burden|CH4|"
    return {
        **{burden + source: 0.0 for source in ("Fossil", "AFOLU", "Permafrost")},
        burden + "Natural": ppb_to_tg * concentration * mixbox_size,
    }


# CH4Chemistry defaults of the lifetimes combined into ``tau_other``
CH4_OTHER_LIFETIMES = {"tau_soil": 150.0, "tau_strat": 120.0, "tau_trop_cl": 200.0}


def _ch4_tau_other(lifetimes: Mapping[str, float]) -> float:
    """CH4 lifetime against soil, stratospheric and chlorine loss, as in MAGICC7."""
    tau = {**CH4_OTHER_LIFETIMES, **lifetimes}
    return 1.0 / sum(1.0 / tau[k] for k in CH4_OTHER_LIFETIMES)
