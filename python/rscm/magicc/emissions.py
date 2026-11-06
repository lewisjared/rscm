"""Emissions-driven runs coupling MAGICC chemistry, carbon cycle and climate.

CO2, CH4 and N2O concentrations are calculated from emissions, with
temperature feeding back on the carbon cycle and chemistry. Forcing outside
the three gases is prescribed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from rscm._lib.magicc import (
    CH4ChemistryBuilder,
    CH4OxidationBuilder,
    ClimateUDEBBuilder,
    CO2BudgetBuilder,
    GhgForcingBuilder,
    N2OChemistryBuilder,
    OceanCarbonBuilder,
    TerrestrialCarbonBuilder,
)
from rscm.core import (
    GridType,
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    VariableSchema,
)
from rscm.magicc.runner import (
    CH4_OTHER_LIFETIMES,
    CONCENTRATION,
    ERF,
    SPECIES,
    MAGICCResult,
    _average_series,
    _base_schema,
    _ch4_burdens,
    _ch4_tau_other,
    _climate_weights,
    _collect_result,
    _series,
    _validate_axis,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray

    from rscm._lib.magicc import ClimateUDEBParams, GhgForcingParams

__all__ = ["Emissions", "run_magicc_emissions"]

# MAGICC7 preindustrial land pools (CO2_*POOL_INITIAL), the TerrestrialCarbon defaults
DEFAULT_POOLS = {"plant": 884.8584, "detritus": 92.7738, "soil": 1681.525}

#: Components in registration order, which sets the lags between them.
COMPONENTS = {
    "CH4Chemistry": CH4ChemistryBuilder,
    "CH4Oxidation": CH4OxidationBuilder,
    "N2OChemistry": N2OChemistryBuilder,
    "TerrestrialCarbon": TerrestrialCarbonBuilder,
    "OceanCarbon": OceanCarbonBuilder,
    "CO2Budget": CO2BudgetBuilder,
    "GhgForcing": GhgForcingBuilder,
    "ClimateUDEB": ClimateUDEBBuilder,
}
#: Chemistry and the carbon cycle read last year's temperature,
#: and the carbon sinks last year's CO2.
LAG = ("Surface Temperature",)
LAG_FOR = (
    (CONCENTRATION + "CO2", "TerrestrialCarbon"),
    (CONCENTRATION + "CO2", "OceanCarbon"),
)

CH4_PPB_TO_TG = 2.824
CH4_MIXBOX_SIZE = 0.973
CH4_BUDGET_PARAMETERS = ("ppb_to_tg", "mixbox_size", "natural_emissions")

EMISSION_VARIABLES = {
    "co2_fossil": ("Emissions|CO2|Fossil", "GtC/yr"),
    "co2_land_use": ("Emissions|CO2|Land Use", "GtC/yr"),
    "ch4": ("Emissions|CH4", "Mt CH4/yr"),
    "ch4_fossil": ("Emissions|CH4|Fossil", "Mt CH4/yr"),
    "n2o": ("Emissions|N2O", "Mt N/yr"),
    "nox": ("Emissions|NOx", "Mt N/yr"),
    "co": ("Emissions|CO", "Mt CO/yr"),
    "nmvoc": ("Emissions|NMVOC", "Mt NMVOC/yr"),
}

#: Calculated variables with their units and sources.
CALCULATED = {
    "Lifetime|CH4": ("yr", "CH4Chemistry"),
    "Lifetime|CH4|OH": ("yr", "CH4Chemistry"),
    "Lifetime|CH4|Effective": ("yr", "CH4Chemistry"),
    "Atmospheric Burden|CH4|Fossil": ("Mt CH4", "CH4Oxidation"),
    "Atmospheric Burden|CH4|AFOLU": ("Mt CH4", "CH4Oxidation"),
    "Atmospheric Burden|CH4|Natural": ("Mt CH4", "CH4Oxidation"),
    "Atmospheric Burden|CH4|Permafrost": ("Mt CH4", "CH4Oxidation"),
    "Emissions|CO2|CH4 Oxidation": ("GtC/yr", "CH4Oxidation"),
    "Lifetime|N2O": ("yr", "N2OChemistry"),
    "Carbon Pool|Plant": ("GtC", "TerrestrialCarbon"),
    "Carbon Pool|Detritus": ("GtC", "TerrestrialCarbon"),
    "Carbon Pool|Soil": ("GtC", "TerrestrialCarbon"),
    "Carbon Flux|Terrestrial": ("GtC/yr", "TerrestrialCarbon"),
    "Emissions|CO2|Gross Deforestation": ("GtC/yr", "TerrestrialCarbon"),
    "Carbon Flux|Regrowth": ("GtC/yr", "TerrestrialCarbon"),
    "Net Primary Production": ("GtC/yr", "TerrestrialCarbon"),
    "Respiration|Terrestrial": ("GtC/yr", "TerrestrialCarbon"),
    "Ocean Surface pCO2": ("ppm", "OceanCarbon"),
    "Cumulative Ocean Uptake": ("GtC", "OceanCarbon"),
    "Carbon Flux|Ocean": ("GtC/yr", "OceanCarbon"),
    "End-of-Year Atmospheric CO2": ("ppm", "OceanCarbon"),
    "Emissions|CO2|Net": ("GtC/yr", "CO2Budget"),
    "Airborne Fraction|CO2": ("1", "CO2Budget"),
    "Surface Temperature": ("K", "ClimateUDEB"),
    "Sea Surface Temperature": ("K", "ClimateUDEB"),
    "Heat Uptake": ("W/m^2", "ClimateUDEB"),
    "Ocean Heat Content": ("J/m^2", "ClimateUDEB"),
}


#: Per-step points, undefined at the first label.
DIAGNOSTICS = frozenset({"End-of-Year Atmospheric CO2", "Airborne Fraction|CO2"})


@dataclass(frozen=True)
class Emissions:
    """Anthropogenic emissions driving an emissions-driven run.

    Each array contains annual mean rates, one for each supplied year,
    labelled by the year they start.
    ``ch4`` is total anthropogenic CH4
    and ``ch4_fossil`` its fossil and industrial part,
    whose oxidation adds CO2.
    Natural CH4 and N2O emissions are chemistry parameters.
    """

    co2_fossil: ArrayLike
    """Fossil and industrial CO2 (GtC/yr)"""
    co2_land_use: ArrayLike
    """Land-use change CO2 (GtC/yr)"""
    ch4: ArrayLike
    """Anthropogenic CH4 (Mt CH4/yr)"""
    ch4_fossil: ArrayLike
    """Fossil and industrial CH4 (Mt CH4/yr)"""
    n2o: ArrayLike
    """Anthropogenic N2O (Mt N/yr)"""
    nox: ArrayLike
    """NOx (Mt N/yr)"""
    co: ArrayLike
    """CO (Mt CO/yr)"""
    nmvoc: ArrayLike
    """NMVOC (Mt NMVOC/yr)"""


def _validate_parameters(
    parameters: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {name: {} for name in COMPONENTS}
    for name, values in (parameters or {}).items():
        if name not in COMPONENTS:
            msg = f"Unknown component {name!r}, expected one of {sorted(COMPONENTS)}"
            raise ValueError(msg)
        resolved[name] = dict(values)
    # The ocean evolves atmospheric CO2 within each year, so it must match the budget
    _share(resolved, ("CO2Budget", "OceanCarbon"), "gtc_per_ppm")
    resolved["CO2Budget"].setdefault("ocean_end_of_year", True)
    # CH4Oxidation repeats the CH4 budget, so it must use CH4Chemistry's parameters
    for key in CH4_BUDGET_PARAMETERS:
        _share(resolved, ("CH4Chemistry", "CH4Oxidation"), key)
    chemistry, oxidation = resolved["CH4Chemistry"], resolved["CH4Oxidation"]
    lifetimes = {k: chemistry[k] for k in CH4_OTHER_LIFETIMES if k in chemistry}
    tau_other = _ch4_tau_other(lifetimes)
    if "tau_other" in oxidation and not math.isclose(oxidation["tau_other"], tau_other):
        msg = (
            "CH4Oxidation tau_other must match CH4Chemistry's soil, stratospheric "
            f"and chlorine lifetimes ({tau_other})"
        )
        raise ValueError(msg)
    if lifetimes:
        oxidation["tau_other"] = tau_other
    return resolved


def _share(
    resolved: dict[str, dict[str, Any]], components: tuple[str, str], key: str
) -> None:
    """Give both components the one value of ``key`` either of them sets."""
    values = {resolved[name][key] for name in components if key in resolved[name]}
    if len(values) > 1:
        msg = f"{components[0]} and {components[1]} must use the same {key}"
        raise ValueError(msg)
    for value in values:
        for name in components:
            resolved[name][key] = value


def _schema(units: Mapping[str, str], contributors: tuple[str, ...]) -> VariableSchema:
    schema = _base_schema()
    for name, unit in units.items():
        if name == "Surface Temperature":
            schema.add_variable(name, unit, GridType.FourBox)
        elif name != ERF:
            schema.add_variable(name, unit)
    schema.add_aggregate(ERF, "W/m^2", "Sum", list(contributors))
    return schema


def _variables(
    data: Mapping[str, NDArray[np.float64]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Units and sources of every variable, given the prescribed inputs."""
    units = {name: unit for name, unit in EMISSION_VARIABLES.values()}
    units["Emissions|CH4|AFOLU"] = "Mt CH4/yr"
    units[ERF + "|Other"] = "W/m^2"
    sources = {**dict.fromkeys(data, "prescribed"), ERF: "Sum"}
    owners = {"CO2": "CO2Budget", "CH4": "CH4Chemistry", "N2O": "N2OChemistry"}
    for gas, owner in owners.items():
        units[CONCENTRATION + gas] = "ppm" if gas == "CO2" else "ppb"
        sources[CONCENTRATION + gas] = owner
        units[ERF + "|" + gas] = "W/m^2"
        sources[ERF + "|" + gas] = "GhgForcing"
    for name, (unit, source) in CALCULATED.items():
        units[name] = unit
        sources[name] = source
    units[ERF] = "W/m^2"
    return units, sources


def run_magicc_emissions(  # noqa: PLR0913
    years: ArrayLike,
    emissions: Emissions,
    *,
    initial_concentrations: Mapping[str, float],
    other_forcing: ArrayLike,
    forcing_species: tuple[str, ...] = SPECIES,
    parameters: Mapping[str, Mapping[str, Any]] | None = None,
) -> MAGICCResult:
    """Run the coupled MAGICC chain from emissions on annual boundaries.

    ``years`` must contain at least two consecutive annual boundaries.
    ``initial_concentrations`` gives CO2 (ppm), CH4 and N2O (ppb) at the first
    boundary. The ocean starts in equilibrium with that CO2 and the land pools
    at their preindustrial sizes, so the first year should be preindustrial.

    ``other_forcing`` is the ERF in W/m² from everything except the selected
    gases, one value per boundary. ``parameters`` maps component names from
    ``COMPONENTS`` to their Rust parameter dictionaries. ClimateUDEB's
    ``rf_2xco2`` defaults to the CO2-doubling ERF of the forcing configuration,
    as in MAGICC7. A ``gtc_per_ppm`` given for CO2Budget or OceanCarbon applies
    to both.

    As in MAGICC7, chemistry and the carbon cycle use the temperature at the
    start of each year, and forcing uses the concentrations at its end.

    Raises ``ValueError`` for invalid inputs and ``RuntimeError`` for nonfinite
    or nonpositive outputs.
    """
    time = _validate_axis(years, forcing_species)
    if set(initial_concentrations) != set(SPECIES):
        msg = f"initial_concentrations must give exactly {list(SPECIES)}"
        raise ValueError(msg)
    initial_conc = {
        gas: float(_series(gas, [initial_concentrations[gas]], 1, positive=True)[0])
        for gas in SPECIES
    }
    params = _validate_parameters(parameters)

    data = {
        EMISSION_VARIABLES[field.name][0]: _average_series(
            field.name, getattr(emissions, field.name), len(time)
        )
        for field in fields(Emissions)
    }
    if np.any(data["Emissions|CH4|Fossil"] > data["Emissions|CH4"]):
        msg = "ch4_fossil must not exceed total anthropogenic ch4"
        raise ValueError(msg)
    data["Emissions|CH4|AFOLU"] = data["Emissions|CH4"] - data["Emissions|CH4|Fossil"]
    data[ERF + "|Other"] = _series("other_forcing", other_forcing, len(time))
    units, sources = _variables(data)

    forcing = GhgForcingBuilder.from_parameters(
        cast("GhgForcingParams", params["GhgForcing"])
    )
    initial_erf = forcing.calculate_forcings(*(initial_conc[gas] for gas in SPECIES))
    total_erf = sum(initial_erf[gas] for gas in forcing_species) + float(
        data[ERF + "|Other"][0]
    )
    climate_params = cast(
        "ClimateUDEBParams",
        {"rf_2xco2": forcing.co2_doubling_forcing(), **params["ClimateUDEB"]},
    )
    climate = ClimateUDEBBuilder.from_parameters(climate_params)
    terrestrial = params["TerrestrialCarbon"]
    initial = {
        **{CONCENTRATION + gas: value for gas, value in initial_conc.items()},
        **{ERF + "|" + gas: value for gas, value in initial_erf.items()},
        ERF: total_erf,
        **climate.initial_values(total_erf),
        **_ch4_burdens(
            initial_conc["CH4"],
            float(params["CH4Chemistry"].get("ppb_to_tg", CH4_PPB_TO_TG)),
            float(params["CH4Chemistry"].get("mixbox_size", CH4_MIXBOX_SIZE)),
        ),
        "Ocean Surface pCO2": initial_conc["CO2"],
        "Cumulative Ocean Uptake": 0.0,
        **{
            f"Carbon Pool|{pool.title()}": float(
                terrestrial.get(f"{pool}_pool_pi", default)
            )
            for pool, default in DEFAULT_POOLS.items()
        },
    }

    contributors = (*(ERF + "|" + gas for gas in forcing_species), ERF + "|Other")
    axis = TimeAxis.from_values(time)
    weights = _climate_weights(climate_params)
    # Components reading global temperature must see the area-weighted mean
    builder = (
        ModelBuilder()
        .with_time_axis(axis)
        .with_schema(_schema(units, contributors))
        .with_grid_weights(GridType.FourBox, list(weights))
    )
    for name, component_builder in COMPONENTS.items():
        if name == "GhgForcing":
            component = forcing.build()
        elif name == "ClimateUDEB":
            component = climate.build()
        else:
            component = component_builder.from_parameters(params[name]).build()
        builder.with_rust_component(component)
    for variable in LAG:
        builder.with_lag(variable)
    for variable, consumer in LAG_FOR:
        builder.with_lag_for(variable, consumer)
    for name, values in data.items():
        builder.with_exogenous_variable(
            name, Timeseries(values, axis, units[name], InterpolationStrategy.Linear)
        )
    builder.with_initial_values(initial)
    model = builder.build()
    model.run()
    return _collect_result(
        model,
        time,
        units,
        sources,
        contributors,
        weights,
        DIAGNOSTICS,
    )
