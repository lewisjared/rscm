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
    ClimateUdebBuilder,
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
    SPECIES,
    MagiccResult,
    _average_series,
    _base_schema,
    _ch4_burdens,
    _ch4_non_oh_lifetime,
    _climate_weights,
    _collect_result,
    _series,
    _validate_axis,
)
from rscm.names import (
    AIRBORNE_FRACTION_CO2,
    ATMOSPHERIC_BURDEN_CH4_BIOSPHERE,
    ATMOSPHERIC_BURDEN_CH4_FOSSIL,
    ATMOSPHERIC_BURDEN_CH4_NATURAL,
    ATMOSPHERIC_BURDEN_CH4_PERMAFROST,
    ATMOSPHERIC_CONCENTRATIONS,
    CARBON_FLUX_REGROWTH,
    CARBON_POOL_DETRITUS,
    CARBON_POOL_PLANT,
    CARBON_POOL_SOIL,
    CUMULATIVE_CARBON_UPTAKE_OCEAN,
    EFFECTIVE_LIFETIME_CH4,
    EMISSIONS_CH4,
    EMISSIONS_CH4_BIOSPHERE,
    EMISSIONS_CH4_FOSSIL,
    EMISSIONS_CO,
    EMISSIONS_CO2_BIOSPHERE,
    EMISSIONS_CO2_CH4_OXIDATION,
    EMISSIONS_CO2_FOSSIL,
    EMISSIONS_CO2_GROSS_DEFORESTATION,
    EMISSIONS_N2O,
    EMISSIONS_NMVOC,
    EMISSIONS_NOX,
    EMISSIONS_UNITS,
    ERF,
    LIFETIME_CH4,
    LIFETIME_CH4_OH,
    LIFETIME_N2O,
    NET_ATMOSPHERE_TO_LAND_FLUX_CO2,
    NET_ATMOSPHERE_TO_OCEAN_FLUX_CO2,
    NET_EMISSIONS_CO2,
    OCEAN_SURFACE_PARTIAL_PRESSURE_CO2,
    RESPIRATION_LAND,
    SURFACE_AIR_TEMPERATURE_CHANGE,
    child,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray

    from rscm._lib.magicc import ClimateUdebParameters, GhgForcingParameters

__all__ = ["Emissions", "run_magicc_emissions"]

# MAGICC7 preindustrial land pools (CO2_*POOL_INITIAL), the TerrestrialCarbon defaults
DEFAULT_POOLS = {"plant": 884.8584, "detritus": 92.7738, "soil": 1681.525}

# Components in registration order, which sets the lags between them.
COMPONENTS = {
    "CH4Chemistry": CH4ChemistryBuilder,
    "CH4Oxidation": CH4OxidationBuilder,
    "N2OChemistry": N2OChemistryBuilder,
    "TerrestrialCarbon": TerrestrialCarbonBuilder,
    "OceanCarbon": OceanCarbonBuilder,
    "CO2Budget": CO2BudgetBuilder,
    "GhgForcing": GhgForcingBuilder,
    "ClimateUdeb": ClimateUdebBuilder,
}
# Chemistry and the carbon cycle read last year's temperature,
# and the carbon sinks last year's CO2.
_LAG = (SURFACE_AIR_TEMPERATURE_CHANGE,)
_LAG_FOR = (
    (child(ATMOSPHERIC_CONCENTRATIONS, "CO2"), "TerrestrialCarbon"),
    (child(ATMOSPHERIC_CONCENTRATIONS, "CO2"), "OceanCarbon"),
)

CH4_PPB_TO_TG = 2.824
CH4_MIXBOX_SIZE = 0.973
CH4_BUDGET_PARAMETERS = ("tg_per_ppb", "mixing_box_size", "natural_emissions")

EMISSION_VARIABLES = {
    "co2_fossil": (EMISSIONS_CO2_FOSSIL, EMISSIONS_UNITS["CO2"]),
    "co2_biosphere": (EMISSIONS_CO2_BIOSPHERE, EMISSIONS_UNITS["CO2"]),
    "ch4": (EMISSIONS_CH4, EMISSIONS_UNITS["CH4"]),
    "ch4_fossil": (EMISSIONS_CH4_FOSSIL, EMISSIONS_UNITS["CH4"]),
    "n2o": (EMISSIONS_N2O, EMISSIONS_UNITS["N2O"]),
    "nox": (EMISSIONS_NOX, EMISSIONS_UNITS["NOx"]),
    "co": (EMISSIONS_CO, EMISSIONS_UNITS["CO"]),
    "nmvoc": (EMISSIONS_NMVOC, EMISSIONS_UNITS["NMVOC"]),
}

# Calculated variables with their units and sources.
CALCULATED = {
    LIFETIME_CH4: ("yr", "CH4Chemistry"),
    LIFETIME_CH4_OH: ("yr", "CH4Chemistry"),
    EFFECTIVE_LIFETIME_CH4: ("yr", "CH4Chemistry"),
    ATMOSPHERIC_BURDEN_CH4_FOSSIL: ("Mt CH4", "CH4Oxidation"),
    ATMOSPHERIC_BURDEN_CH4_BIOSPHERE: ("Mt CH4", "CH4Oxidation"),
    ATMOSPHERIC_BURDEN_CH4_NATURAL: ("Mt CH4", "CH4Oxidation"),
    ATMOSPHERIC_BURDEN_CH4_PERMAFROST: ("Mt CH4", "CH4Oxidation"),
    EMISSIONS_CO2_CH4_OXIDATION: ("GtC/yr", "CH4Oxidation"),
    LIFETIME_N2O: ("yr", "N2OChemistry"),
    CARBON_POOL_PLANT: ("GtC", "TerrestrialCarbon"),
    CARBON_POOL_DETRITUS: ("GtC", "TerrestrialCarbon"),
    CARBON_POOL_SOIL: ("GtC", "TerrestrialCarbon"),
    NET_ATMOSPHERE_TO_LAND_FLUX_CO2: ("GtC/yr", "TerrestrialCarbon"),
    EMISSIONS_CO2_GROSS_DEFORESTATION: ("GtC/yr", "TerrestrialCarbon"),
    CARBON_FLUX_REGROWTH: ("GtC/yr", "TerrestrialCarbon"),
    "Net Primary Production": ("GtC/yr", "TerrestrialCarbon"),
    RESPIRATION_LAND: ("GtC/yr", "TerrestrialCarbon"),
    OCEAN_SURFACE_PARTIAL_PRESSURE_CO2: ("ppm", "OceanCarbon"),
    CUMULATIVE_CARBON_UPTAKE_OCEAN: ("GtC", "OceanCarbon"),
    NET_ATMOSPHERE_TO_OCEAN_FLUX_CO2: ("GtC/yr", "OceanCarbon"),
    "Ocean Carbon Atmospheric Concentrations|CO2": ("ppm", "OceanCarbon"),
    NET_EMISSIONS_CO2: ("GtC/yr", "CO2Budget"),
    AIRBORNE_FRACTION_CO2: ("1", "CO2Budget"),
    SURFACE_AIR_TEMPERATURE_CHANGE: ("K", "ClimateUdeb"),
    "Sea Surface Temperature": ("K", "ClimateUdeb"),
    "Heat Uptake": ("W/m^2", "ClimateUdeb"),
    "Ocean Heat Content": ("J", "ClimateUdeb"),
}


# Per-step points, undefined at the first label.
DIAGNOSTICS = frozenset(
    {"Ocean Carbon Atmospheric Concentrations|CO2", AIRBORNE_FRACTION_CO2}
)


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
    """Fossil and industrial CO2 (Mt CO2/yr)"""
    co2_biosphere: ArrayLike
    """Biosphere CO2, mostly from land-use change (Mt CO2/yr)"""
    ch4: ArrayLike
    """Anthropogenic CH4 (Mt CH4/yr)"""
    ch4_fossil: ArrayLike
    """Fossil and industrial CH4 (Mt CH4/yr)"""
    n2o: ArrayLike
    """Anthropogenic N2O (kt N2O/yr)"""
    nox: ArrayLike
    """NOx (Mt NO2/yr)"""
    co: ArrayLike
    """CO (Mt CO/yr)"""
    nmvoc: ArrayLike
    """NMVOC (Mt VOC/yr)"""


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
    non_oh_lifetime = _ch4_non_oh_lifetime(lifetimes)
    if "non_oh_lifetime" in oxidation and not math.isclose(
        oxidation["non_oh_lifetime"], non_oh_lifetime
    ):
        msg = (
            "CH4Oxidation non_oh_lifetime must match CH4Chemistry's soil, "
            f"stratospheric and chlorine lifetimes ({non_oh_lifetime})"
        )
        raise ValueError(msg)
    if lifetimes:
        oxidation["non_oh_lifetime"] = non_oh_lifetime
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
        if name == SURFACE_AIR_TEMPERATURE_CHANGE:
            schema.add_variable(name, unit, GridType.FourBox)
        elif name != ERF:
            schema.add_variable(name, unit)
    schema.add_aggregate(ERF, "W/m^2", "sum", list(contributors))
    return schema


def _variables(
    data: Mapping[str, NDArray[np.float64]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Units and sources of every variable, given the prescribed inputs."""
    units = {name: unit for name, unit in EMISSION_VARIABLES.values()}
    units[EMISSIONS_CH4_BIOSPHERE] = EMISSIONS_UNITS["CH4"]
    units[child(ERF, "Other")] = "W/m^2"
    sources = {**dict.fromkeys(data, "prescribed"), ERF: "sum"}
    owners = {"CO2": "CO2Budget", "CH4": "CH4Chemistry", "N2O": "N2OChemistry"}
    for gas, owner in owners.items():
        units[child(ATMOSPHERIC_CONCENTRATIONS, gas)] = "ppm" if gas == "CO2" else "ppb"
        sources[child(ATMOSPHERIC_CONCENTRATIONS, gas)] = owner
        units[child(ERF, gas)] = "W/m^2"
        sources[child(ERF, gas)] = "GhgForcing"
    for name, (unit, source) in CALCULATED.items():
        units[name] = unit
        sources[name] = source
    units[ERF] = "W/m^2"
    return units, sources


def run_magicc_emissions(  # noqa: PLR0913
    time_axis: TimeAxis,
    emissions: Emissions,
    *,
    initial_concentrations: Mapping[str, float],
    other_forcing: ArrayLike,
    forcing_species: tuple[str, ...] = SPECIES,
    parameters: Mapping[str, Mapping[str, Any]] | None = None,
) -> MagiccResult:
    """Run the coupled MAGICC chain from emissions on annual boundaries.

    ``time_axis`` must have at least two annual steps.
    ``initial_concentrations`` gives CO2 (ppm), CH4 and N2O (ppb) at the first
    boundary. The ocean starts in equilibrium with that CO2 and the land pools
    at their preindustrial sizes, so the first year should be preindustrial.

    ``other_forcing`` is the ERF in W/m² from everything except the selected
    gases, one value per boundary. ``parameters`` maps component names from
    ``COMPONENTS`` to their Rust parameter dictionaries. ClimateUdeb's
    ``erf_2xco2`` defaults to the CO2-doubling ERF of the forcing configuration,
    as in MAGICC7. A ``gtc_per_ppm`` given for CO2Budget or OceanCarbon applies
    to both.

    As in MAGICC7, chemistry and the carbon cycle use the temperature at the
    start of each year, and forcing uses the concentrations at its end.

    Raises ``ValueError`` for invalid inputs and ``RuntimeError`` for nonfinite
    or nonpositive outputs.
    """
    time = _validate_axis(time_axis, forcing_species)
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
    if np.any(data[EMISSIONS_CH4_FOSSIL] > data[EMISSIONS_CH4]):
        msg = "ch4_fossil must not exceed total anthropogenic ch4"
        raise ValueError(msg)
    data[EMISSIONS_CH4_BIOSPHERE] = data[EMISSIONS_CH4] - data[EMISSIONS_CH4_FOSSIL]
    data[child(ERF, "Other")] = _series("other_forcing", other_forcing, len(time))
    units, sources = _variables(data)

    forcing = GhgForcingBuilder.from_parameters(
        cast("GhgForcingParameters", params["GhgForcing"])
    )
    initial_erf = forcing.calculate_forcings(*(initial_conc[gas] for gas in SPECIES))
    total_erf = sum(initial_erf[gas] for gas in forcing_species) + float(
        data[child(ERF, "Other")][0]
    )
    climate_params = cast(
        "ClimateUdebParameters",
        {"erf_2xco2": forcing.erf_2xco2(), **params["ClimateUdeb"]},
    )
    climate = ClimateUdebBuilder.from_parameters(climate_params)
    terrestrial = params["TerrestrialCarbon"]
    initial = {
        **{
            child(ATMOSPHERIC_CONCENTRATIONS, gas): value
            for gas, value in initial_conc.items()
        },
        **{child(ERF, gas): value for gas, value in initial_erf.items()},
        ERF: total_erf,
        **climate.initial_values(total_erf),
        **_ch4_burdens(
            initial_conc["CH4"],
            float(params["CH4Chemistry"].get("tg_per_ppb", CH4_PPB_TO_TG)),
            float(params["CH4Chemistry"].get("mixing_box_size", CH4_MIXBOX_SIZE)),
        ),
        OCEAN_SURFACE_PARTIAL_PRESSURE_CO2: initial_conc["CO2"],
        CUMULATIVE_CARBON_UPTAKE_OCEAN: 0.0,
        **{
            f"Carbon Pool|{pool.title()}": float(
                terrestrial.get(f"{pool}_pool_pi", default)
            )
            for pool, default in DEFAULT_POOLS.items()
        },
    }

    contributors = (*(child(ERF, gas) for gas in forcing_species), child(ERF, "Other"))
    weights = _climate_weights(climate_params)
    # Components reading global temperature must see the area-weighted mean
    builder = (
        ModelBuilder()
        .with_time_axis(time_axis)
        .with_schema(_schema(units, contributors))
        .with_grid_weights(GridType.FourBox, list(weights))
    )
    for name, component_builder in COMPONENTS.items():
        if name == "GhgForcing":
            component = forcing.build()
        elif name == "ClimateUdeb":
            component = climate.build()
        else:
            component = component_builder.from_parameters(params[name]).build()
        builder.with_component(component)
    for variable in _LAG:
        builder.with_lag(variable)
    for variable, consumer in _LAG_FOR:
        builder.with_lag_for(variable, consumer)
    for name, values in data.items():
        builder.with_prescribed_variable(
            name,
            Timeseries(values, time_axis, units[name], InterpolationStrategy.Linear),
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
