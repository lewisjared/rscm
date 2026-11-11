"""
Build FaIR components from a FaIR configuration.

A configuration is a dict holding one FaIR config:

- `options`: `ch4_method`, `ghg_method`, `iirf_max`, `br_cl_ods_potential` and `ch4_min`
- `climate`: `ocean_heat_capacity`, `ocean_heat_transfer`, `deep_ocean_efficacy`
  and `gamma_autocorrelation`
- `species_configs`: the species configs without a species dimension
  (`aci_scale`, `lifetime_temperature_sensitivity`)
- `species`: one entry per species, in FaIR order, with its properties
  (`name`, `type`, `input_mode` and the flags) and species configs

Missing values are `None` or NaN.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rscm import names
from rscm._lib.fair import (
    AerosolCloudBuilder,
    CH4LifetimeBuilder,
    EESCBuilder,
    EnergyBalanceBuilder,
    ForcingAggregationBuilder,
    GasCycleBuilder,
    GHGForcingBuilder,
    LinearForcingBuilder,
)
from rscm.fair import naming
from rscm.spec import ComponentSpec, Lag

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rscm._lib.core import ModelBuilder, RustComponent

FairConfig = dict[str, Any]
Species = dict[str, Any]

CLIMATE_CONFIGS = (
    "ocean_heat_capacity",
    "ocean_heat_transfer",
    "deep_ocean_efficacy",
    "gamma_autocorrelation",
)
GAMMA_AUTOCORRELATION = 2.0
# Species configs FaIR copies from the species CSV, apart from the gas boxes.
SPECIES_CONFIGS = (
    "tropospheric_adjustment",
    "forcing_efficacy",
    "forcing_temperature_feedback",
    "forcing_scale",
    "molecular_weight",
    "baseline_concentration",
    "forcing_reference_concentration",
    "iirf_0",
    "iirf_airborne",
    "iirf_uptake",
    "iirf_temperature",
    "baseline_emissions",
    "g0",
    "g1",
    "greenhouse_gas_radiative_efficiency",
    "contrails_radiative_efficiency",
    "erfari_radiative_efficiency",
    "h2o_stratospheric_factor",
    "lapsi_radiative_efficiency",
    "land_use_cumulative_emissions_to_forcing",
    "ozone_radiative_efficiency",
    "cl_atoms",
    "br_atoms",
    "fractional_release",
    "ch4_lifetime_chemical_sensitivity",
    "aci_shape",
)
PROPERTY_FLAGS = (
    "greenhouse_gas",
    "aerosol_chemistry_from_emissions",
    "aerosol_chemistry_from_concentration",
)

# Calculated forcing types that sum their drivers linearly.
LINEAR_TYPES = ("ari", "ozone", "contrails", "lapsi", "h2o stratospheric", "land use")


def load_config(path: str | Path) -> FairConfig:
    """Load a FaIR configuration from JSON."""
    with open(path) as f:
        config: FairConfig = json.load(f)
    return config


def _value(x: Any) -> float:
    return math.nan if x is None else float(x)


def _missing(x: Any) -> bool:
    return x is None or math.isnan(x)


def _nonzero(x: Any) -> bool:
    return not _missing(x) and x != 0


def _of_type(config: FairConfig, *types: str) -> list[Species]:
    return [s for s in config["species"] if s["type"] in types]


def _one(config: FairConfig, kind: str) -> Species | None:
    found = _of_type(config, kind)
    return found[0] if found else None


def _calculated(config: FairConfig, kind: str) -> Species | None:
    entry = _one(config, kind)
    return entry if entry is not None and entry["input_mode"] == "calculated" else None


def _scale(entry: Species) -> dict[str, float]:
    return {
        "forcing_scale": _value(entry["forcing_scale"]),
        "tropospheric_adjustment": _value(entry["tropospheric_adjustment"]),
    }


def is_ghg(entry: Species) -> bool:
    """Whether a species entry is a greenhouse gas."""
    return bool(entry.get("greenhouse_gas")) and entry["type"] in naming.GHG_TYPES


def thornhill(config: FairConfig) -> bool:
    """Whether CH4 takes its lifetime from `CH4Lifetime`."""
    return (
        config["options"]["ch4_method"] == "thornhill2021"
        and _one(config, "ch4") is not None
    )


CH4_LIFETIME_SENSITIVITY = "ch4_lifetime_chemical_sensitivity"


def _precursors(config: FairConfig, driver: str) -> list[dict[str, Any]]:
    # CH4Lifetime lists emissions and concentrations apart, so has no driver key.
    return [
        {k: v for k, v in precursor.items() if k != "driver"}
        for precursor in _chemistry_drivers(
            config, driver, CH4_LIFETIME_SENSITIVITY, key="sensitivity"
        )
    ]


def _ch4_lifetime(config: FairConfig) -> ComponentSpec:
    return ComponentSpec(
        CH4LifetimeBuilder,
        {
            "emissions": _precursors(config, "emissions"),
            "concentrations": _precursors(config, "concentration"),
            "temperature_sensitivity": config["species_configs"][
                "lifetime_temperature_sensitivity"
            ],
            "minimum_scaling": config["options"]["ch4_min"],
            "output": names.LIFETIME_SCALING_CH4,
        },
    )


def emission_drivers(config: FairConfig, entry: Species) -> list[str]:
    """Emissions variables that drive a species, none if it is not emissions-driven."""
    if entry["input_mode"] == "emissions":
        return [naming.emissions(entry["name"])]
    if entry["type"] == "co2" and entry["input_mode"] == "calculated":
        return [
            naming.emissions(s["name"])
            for s in _of_type(config, "co2 ffi", "co2 afolu")
        ]
    return []


def species_outputs(config: FairConfig) -> dict[str, dict[str, str]]:
    """Variable of each species with a value of FaIR's species attributes."""
    agents = {s["name"] for s in forcing_agents(config)}
    outputs: dict[str, dict[str, str]] = {
        "concentration": {},
        "forcing": {},
        "airborne_emissions": {},
        "cumulative_emissions": {},
    }
    for s in config["species"]:
        name = s["name"]
        if name in agents:
            outputs["forcing"][name] = naming.forcing(s)
        if is_ghg(s):
            outputs["concentration"][name] = naming.concentration(name)
            outputs["airborne_emissions"][name] = naming.airborne_emissions(name)
            outputs["cumulative_emissions"][name] = naming.cumulative_emissions(name)
        elif s["type"] == "eesc":
            outputs["concentration"][name] = (
                names.EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE
            )
    return outputs


def _gas(config: FairConfig, entry: Species) -> dict[str, Any]:
    name = entry["name"]
    if entry["input_mode"] == "concentration":
        driver = "concentration"
        emissions: list[str] = []
    else:
        driver = "emissions"
        emissions = emission_drivers(config, entry)
    gas: dict[str, Any] = {
        "driver": driver,
        "emissions": emissions,
        "inverse_emissions": naming.inverse_emissions(name),
        "emissions_unit": naming.emissions_unit(entry),
        "amount_unit": naming.amount_unit(entry),
        "concentration": naming.concentration(name),
        "concentration_unit": naming.concentration_unit(entry),
        "airborne_emissions": naming.airborne_emissions(name),
        "cumulative_emissions": naming.cumulative_emissions(name),
        "lifetime_scaling": (
            names.LIFETIME_SCALING_CH4
            if entry["type"] == "ch4" and thornhill(config)
            else None
        ),
    }
    for key in ("partition_fraction", "unperturbed_lifetime"):
        gas[key] = [_value(v) for v in entry[key]]
    for key in (
        "baseline_concentration",
        "baseline_emissions",
        "concentration_per_emission",
        "iirf_0",
        "iirf_airborne",
        "iirf_uptake",
        "iirf_temperature",
        "g0",
        "g1",
    ):
        gas[key] = _value(entry[key])
    return gas


def _gas_cycle(config: FairConfig, gases: list[Species]) -> ComponentSpec:
    return ComponentSpec(
        GasCycleBuilder,
        {
            "species": [_gas(config, s) for s in gases],
            "maximum_iirf": config["options"]["iirf_max"],
        },
    )


def _eesc(config: FairConfig) -> ComponentSpec:
    halogens = _of_type(config, *naming.HALOGEN_TYPES)
    (cfc11,) = _of_type(config, "cfc-11")
    return ComponentSpec(
        EESCBuilder,
        {
            "halogens": [
                {
                    "concentration": naming.concentration(h["name"]),
                    "unit": naming.concentration_unit(h),
                    "cl_atoms": _value(h["cl_atoms"]),
                    "br_atoms": _value(h["br_atoms"]),
                    "fractional_release": _value(h["fractional_release"]),
                }
                for h in halogens
            ],
            "cfc11_fractional_release": _value(cfc11["fractional_release"]),
            "br_cl_ratio": config["options"]["br_cl_ods_potential"],
            "output": names.EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE,
        },
    )


def _ghg_forcing(config: FairConfig, gases: list[Species]) -> ComponentSpec:
    def gas(entry: Species) -> dict[str, Any]:
        return {
            "concentration": naming.concentration(entry["name"]),
            "concentration_unit": naming.concentration_unit(entry),
            "forcing": naming.forcing(entry),
            "reference_concentration": _value(entry["forcing_reference_concentration"]),
            "baseline_concentration": _value(entry["baseline_concentration"]),
            "radiative_efficiency": _value(
                entry["greenhouse_gas_radiative_efficiency"]
            ),
            "temperature_feedback": _value(entry["forcing_temperature_feedback"]),
            **_scale(entry),
        }

    major = {
        kind: gas(next(s for s in gases if s["type"] == kind))
        for kind in ("co2", "ch4", "n2o")
    }
    minor = [gas(s) for s in gases if s["type"] not in major]
    return ComponentSpec(GHGForcingBuilder, {**major, "minor": minor})


def _driver(
    entry: Species,
    driver: str,
    coefficient: str,
    *,
    key: str = "radiative_efficiency",
    baselined: bool = True,
) -> dict[str, Any]:
    if driver == "concentration":
        variable = (
            names.EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE
            if entry["type"] == "eesc"
            else naming.concentration(entry["name"])
        )
        unit = naming.concentration_unit(entry)
        baseline = entry["baseline_concentration"]
    else:
        variable = naming.emissions(entry["name"])
        unit = naming.emissions_unit(entry)
        baseline = entry["baseline_emissions"]
    return {
        "variable": variable,
        "unit": unit,
        "driver": driver,
        "baseline": _value(baseline) if baselined else 0.0,
        key: _value(entry[coefficient]),
    }


def _chemistry_drivers(
    config: FairConfig,
    driver: str,
    coefficient: str,
    *,
    key: str = "radiative_efficiency",
) -> list[dict[str, Any]]:
    """Drivers of the species with aerosol chemistry from `driver`, in species order."""
    return [
        _driver(s, driver, coefficient, key=key)
        for s in config["species"]
        if s[f"aerosol_chemistry_from_{driver}"]
    ]


def aerosol_drivers(
    config: FairConfig, coefficient: str, key: str = "radiative_efficiency"
) -> list[tuple[Species, dict[str, Any]]]:
    """Species and drivers of an aerosol forcing, in species order.

    Species with a missing or zero `coefficient` are left out,
    as FaIR's NaN-skipping sum drops their terms.
    Calculated EESC is left out because FaIR calculates it after the aerosol forcing,
    so it is missing when the aerosol forcing reads it.
    """
    drivers = []
    for s in config["species"]:
        if not _nonzero(s.get(coefficient)):
            continue
        if s["aerosol_chemistry_from_emissions"]:
            drivers.append((s, _driver(s, "emissions", coefficient, key=key)))
        elif s["aerosol_chemistry_from_concentration"] and not (
            s["type"] == "eesc" and s["input_mode"] == "calculated"
        ):
            drivers.append((s, _driver(s, "concentration", coefficient, key=key)))
    return drivers


# Coefficient, driver and whether it is baselined, of the single-driver forcings.
_LINEAR_DRIVERS = {
    "contrails": ("contrails_radiative_efficiency", "emissions", False),
    "lapsi": ("lapsi_radiative_efficiency", "emissions", True),
    "h2o stratospheric": ("h2o_stratospheric_factor", "concentration", True),
    "land use": (
        "land_use_cumulative_emissions_to_forcing",
        "cumulative_emissions",
        False,
    ),
}


# Coefficient of each linear forcing type.
LINEAR_COEFFICIENTS = {
    "ari": "erfari_radiative_efficiency",
    "ozone": "ozone_radiative_efficiency",
    **{kind: coefficient for kind, (coefficient, _, _) in _LINEAR_DRIVERS.items()},
}


def linear_drivers(
    config: FairConfig, kind: str
) -> list[tuple[Species, dict[str, Any]]]:
    """Species and drivers of a linear forcing type, in species order."""
    species = config["species"]
    if kind == "ari":
        return aerosol_drivers(config, LINEAR_COEFFICIENTS[kind])
    if kind == "ozone":
        coefficient = LINEAR_COEFFICIENTS[kind]
        return [
            (s, _driver(s, driver, coefficient))
            for driver in ("concentration", "emissions")
            for s in species
            if s[f"aerosol_chemistry_from_{driver}"]
        ]
    coefficient, driver, baselined = _LINEAR_DRIVERS[kind]
    return [
        (s, _driver(s, driver, coefficient, baselined=baselined))
        for s in species
        if _nonzero(s.get(coefficient))
    ]


def _linear_forcing(config: FairConfig, entry: Species) -> ComponentSpec:
    return ComponentSpec(
        LinearForcingBuilder,
        {
            "forcing": naming.forcing(entry),
            "drivers": [d for _, d in linear_drivers(config, entry["type"])],
            # FaIR sums every linear forcing apart from ozone with np.nansum.
            "skip_nan_terms": entry["type"] != "ozone",
            "temperature_feedback": _value(entry["forcing_temperature_feedback"]),
            **_scale(entry),
        },
    )


def _aerosol_cloud(config: FairConfig, entry: Species) -> ComponentSpec:
    return ComponentSpec(
        AerosolCloudBuilder,
        {
            "forcing": naming.forcing(entry),
            "drivers": [
                d for _, d in aerosol_drivers(config, "aci_shape", key="shape")
            ],
            "scale": config["species_configs"]["aci_scale"],
            "temperature_feedback": _value(entry["forcing_temperature_feedback"]),
            **_scale(entry),
        },
    )


def forcing_agents(config: FairConfig) -> list[Species]:
    """Species with a forcing, in FaIR order."""
    return [
        s
        for s in config["species"]
        if is_ghg(s)
        or s["type"] in (*LINEAR_TYPES, "aci")
        or s["input_mode"] == "forcing"
    ]


def forcing_category(entry: Species) -> str:
    """ForcingAggregation category of a species entry, by MAGICC7's ERF subtotals."""
    kind = entry["type"]
    if kind in naming.GHG_TYPES:
        return "greenhouse_gas"
    if kind in ("ari", "aci"):
        return "aerosol"
    if kind in ("solar", "volcanic"):
        return "natural"
    return "anthropogenic"


SUBTOTAL_CATEGORIES = {
    names.ERF_GREENHOUSE_GASES: ("greenhouse_gas",),
    names.ERF_AEROSOLS: ("aerosol",),
    names.ERF_ANTHROPOGENIC: ("greenhouse_gas", "aerosol", "anthropogenic"),
}
"""Categories of the agents in each ForcingAggregation subtotal."""


def forcing_subtotals(
    config: FairConfig, forcing: Mapping[str, float]
) -> dict[str, float]:
    """Return the ForcingAggregation subtotals of the agents' `forcing`, NaN skipped."""
    agents = forcing_agents(config)
    return {
        name: float(
            np.nansum(
                [
                    forcing[naming.forcing(s)]
                    for s in agents
                    if forcing_category(s) in categories
                ]
            )
        )
        for name, categories in SUBTOTAL_CATEGORIES.items()
    }


def _aggregation(config: FairConfig) -> ComponentSpec:
    agents = []
    for s in forcing_agents(config):
        agent = {
            "forcing": naming.forcing(s),
            "category": forcing_category(s),
            "efficacy": _value(s["forcing_efficacy"]),
        }
        if s["input_mode"] == "forcing":
            agent["prescribed"] = naming.prescribed_forcing(s)
            agent["temperature_feedback"] = _value(s["forcing_temperature_feedback"])
        agents.append(agent)
    return ComponentSpec(ForcingAggregationBuilder, {"agents": agents})


def _energy_balance(config: FairConfig) -> ComponentSpec:
    climate = config["climate"]
    return ComponentSpec(
        EnergyBalanceBuilder,
        {
            "ocean_heat_capacity": climate["ocean_heat_capacity"],
            "ocean_heat_transfer": climate["ocean_heat_transfer"],
            "deep_ocean_efficacy": climate["deep_ocean_efficacy"],
            "gamma_autocorrelation": climate.get(
                "gamma_autocorrelation", GAMMA_AUTOCORRELATION
            ),
        },
    )


def component_specs(
    config: FairConfig, *, energy_balance: bool = True
) -> dict[str, ComponentSpec]:
    """Components reproducing a FaIR run, by name, in the order they are added.

    Each linear forcing is named by its species, such as ``LinearForcing[ozone]``.
    Without `energy_balance`, another climate component must write the surface
    temperature from the total effective radiative forcing.
    Raises an `InputValidationError` if the configuration is invalid.
    """
    from rscm.fair.validation import validate  # noqa: PLC0415 (circular import)

    validate(config)
    return _component_specs(config, energy_balance=energy_balance)


def _component_specs(
    config: FairConfig, *, energy_balance: bool
) -> dict[str, ComponentSpec]:
    gases = [s for s in config["species"] if is_ghg(s)]
    result = {}
    if thornhill(config):
        result["CH4Lifetime"] = _ch4_lifetime(config)
    if gases:
        result["GasCycle"] = _gas_cycle(config, gases)
    if _calculated(config, "eesc") is not None:
        result["EESC"] = _eesc(config)
    if gases:
        result["GHGForcing"] = _ghg_forcing(config, gases)
    for kind in LINEAR_TYPES:
        entry = _calculated(config, kind)
        if entry is not None:
            result[f"LinearForcing[{entry['name']}]"] = _linear_forcing(config, entry)
    entry = _calculated(config, "aci")
    if entry is not None:
        result["AerosolCloud"] = _aerosol_cloud(config, entry)
    result["ForcingAggregation"] = _aggregation(config)
    if energy_balance:
        result["EnergyBalance"] = _energy_balance(config)
    return result


def components(
    config: FairConfig, *, energy_balance: bool = True
) -> list[RustComponent]:
    """Build the components of `component_specs`, in the order they are added."""
    specs = component_specs(config, energy_balance=energy_balance)
    return [spec.build() for spec in specs.values()]


def lags(config: FairConfig) -> tuple[Lag, ...]:
    """FaIR's feedback lags.

    Every component reads the previous step's surface air temperature change,
    and the methane lifetime the previous step's concentrations.
    """
    result = [Lag(names.SURFACE_AIR_TEMPERATURE_CHANGE)]
    if thornhill(config):
        result += [
            Lag(driver["variable"], "CH4Lifetime")
            for driver in _chemistry_drivers(
                config, "concentration", CH4_LIFETIME_SENSITIVITY, key="sensitivity"
            )
        ]
    return tuple(result)


def with_lags(builder: ModelBuilder, config: FairConfig) -> ModelBuilder:
    """Add FaIR's feedback `lags` to a builder of `components`."""
    for lag in lags(config):
        lag.add_to(builder)
    return builder


def initial_values(
    config: FairConfig, *, energy_balance: bool = True
) -> dict[str, float]:
    """Return FaIR's values at the first time of every variable the components write.

    `energy_balance` matches `components`.
    """
    values = dict.fromkeys(
        (names.ERF, names.EFFICACY_ADJUSTED_ERF, *SUBTOTAL_CATEGORIES), 0.0
    )
    if energy_balance:
        n_layers = len(config["climate"]["ocean_heat_capacity"])
        values |= dict.fromkeys(
            (
                names.STOCHASTIC_FORCING,
                names.OCEAN_HEAT_CONTENT,
                names.NET_ENERGY_IMBALANCE,
                *map(naming.layer_temperature, range(n_layers)),
            ),
            0.0,
        )
    for s in config["species"]:
        if is_ghg(s):
            if s["input_mode"] != "concentration":
                values[naming.concentration(s["name"])] = _value(
                    s["baseline_concentration"]
                )
            values[naming.airborne_emissions(s["name"])] = 0.0
            values[naming.cumulative_emissions(s["name"])] = 0.0
        if s["type"] == "eesc" and s["input_mode"] == "calculated":
            values[names.EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE] = _value(
                s["baseline_concentration"]
            )
    for s in forcing_agents(config):
        values[naming.forcing(s)] = 0.0
    return values
