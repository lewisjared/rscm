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

from rscm._lib.fair import (
    FairAerosolCloudBuilder,
    FairCH4LifetimeBuilder,
    FairEescBuilder,
    FairEnergyBalanceBuilder,
    FairForcingAggregationBuilder,
    FairGasCycleBuilder,
    FairGhgForcingBuilder,
    FairLinearForcingBuilder,
)
from rscm.fair import naming

if TYPE_CHECKING:
    from rscm._lib.core import Component

FairConfig = dict[str, Any]
Species = dict[str, Any]

CLIMATE_CONFIGS = (
    "ocean_heat_capacity",
    "ocean_heat_transfer",
    "deep_ocean_efficacy",
    "gamma_autocorrelation",
)
GAMMA_AUTOCORRELATION = 2.0

#: Calculated forcing types that sum their drivers linearly.
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
    """Whether CH4 takes its lifetime from `FairCH4Lifetime`."""
    return (
        config["options"]["ch4_method"] == "thornhill2021"
        and _one(config, "ch4") is not None
    )


def _ch4_lifetime(config: FairConfig) -> Component:
    sensitivity = "ch4_lifetime_chemical_sensitivity"
    emissions = _chemistry_drivers(config, "emissions", sensitivity, key="sensitivity")
    concentrations = _chemistry_drivers(
        config, "concentration", sensitivity, key="sensitivity"
    )
    return FairCH4LifetimeBuilder.from_parameters(
        {
            "emissions": emissions,
            "concentrations": concentrations,
            "temperature_sensitivity": config["species_configs"][
                "lifetime_temperature_sensitivity"
            ],
            "ch4_min": config["options"]["ch4_min"],
            "output": naming.CH4_LIFETIME_SCALING,
        }
    ).build()


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
            outputs["concentration"][name] = naming.EESC
    return outputs


def _gas(config: FairConfig, entry: Species) -> dict[str, Any]:
    name = entry["name"]
    if entry["input_mode"] == "concentration":
        mode = "concentration"
        emissions: list[str] = []
    else:
        mode = "emissions"
        emissions = emission_drivers(config, entry)
    gas: dict[str, Any] = {
        "mode": mode,
        "emissions": emissions,
        "inverse_emissions": naming.emissions(name),
        "emissions_unit": naming.emissions_unit(entry),
        "amount_unit": naming.amount_unit(entry),
        "concentration": naming.concentration(name),
        "concentration_unit": naming.concentration_unit(entry),
        "airborne_emissions": naming.airborne_emissions(name),
        "cumulative_emissions": naming.cumulative_emissions(name),
        "lifetime_scaling": (
            naming.CH4_LIFETIME_SCALING
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


def _gas_cycle(config: FairConfig, gases: list[Species]) -> Component:
    return FairGasCycleBuilder.from_parameters(
        {
            "species": [_gas(config, s) for s in gases],
            "iirf_max": config["options"]["iirf_max"],
        }
    ).build()


def _eesc(config: FairConfig) -> Component:
    halogens = _of_type(config, *naming.HALOGEN_TYPES)
    (cfc11,) = _of_type(config, "cfc-11")
    return FairEescBuilder.from_parameters(
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
            "output": naming.EESC,
        }
    ).build()


def _ghg_forcing(config: FairConfig, gases: list[Species]) -> Component:
    if config["options"]["ghg_method"] != "meinshausen2020":
        msg = f"ghg_method {config['options']['ghg_method']!r} is not implemented"
        raise NotImplementedError(msg)

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

    major = {}
    for kind in ("co2", "ch4", "n2o"):
        found = [s for s in gases if s["type"] == kind]
        if not found:
            msg = "Meinshausen et al. (2020) forcing needs CO2, CH4 and N2O"
            raise ValueError(msg)
        major[kind] = gas(found[0])
    minor = [gas(s) for s in gases if s["type"] not in major]
    return FairGhgForcingBuilder.from_parameters({**major, "minor": minor}).build()


def _driver(
    entry: Species,
    kind: str,
    coefficient: str,
    *,
    key: str = "radiative_efficiency",
    baselined: bool = True,
) -> dict[str, Any]:
    if kind == "concentration":
        variable = (
            naming.EESC
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
        "kind": kind,
        "baseline": _value(baseline) if baselined else 0.0,
        key: _value(entry[coefficient]),
    }


def _chemistry_drivers(
    config: FairConfig,
    kind: str,
    coefficient: str,
    *,
    key: str = "radiative_efficiency",
) -> list[dict[str, Any]]:
    """Drivers of the species with aerosol chemistry from `kind`, in species order."""
    return [
        _driver(s, kind, coefficient, key=key)
        for s in config["species"]
        if s[f"aerosol_chemistry_from_{kind}"]
    ]


def _aerosol_drivers(
    config: FairConfig, coefficient: str, key: str = "radiative_efficiency"
) -> list[dict[str, Any]]:
    """Drivers in species order.

    Calculated EESC is left out because FaIR calculates it after the aerosol forcing,
    so it is missing when the aerosol forcing reads it.
    """
    drivers = []
    for s in config["species"]:
        if _missing(s.get(coefficient)):
            continue
        if s["aerosol_chemistry_from_emissions"]:
            drivers.append(_driver(s, "emissions", coefficient, key=key))
        elif s["aerosol_chemistry_from_concentration"] and not (
            s["type"] == "eesc" and s["input_mode"] == "calculated"
        ):
            drivers.append(_driver(s, "concentration", coefficient, key=key))
    return drivers


#: Coefficient, driver kind and whether it is baselined, of the single-driver forcings.
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


def _linear_drivers(config: FairConfig, kind: str) -> list[dict[str, Any]]:
    species = config["species"]
    if kind == "ari":
        return _aerosol_drivers(config, "erfari_radiative_efficiency")
    if kind == "ozone":
        coefficient = "ozone_radiative_efficiency"
        return _chemistry_drivers(
            config, "concentration", coefficient
        ) + _chemistry_drivers(config, "emissions", coefficient)
    coefficient, driver_kind, baselined = _LINEAR_DRIVERS[kind]
    return [
        _driver(s, driver_kind, coefficient, baselined=baselined)
        for s in species
        if _nonzero(s.get(coefficient))
    ]


def _linear_forcing(config: FairConfig, entry: Species) -> Component:
    return FairLinearForcingBuilder.from_parameters(
        {
            "forcing": naming.forcing(entry),
            "drivers": _linear_drivers(config, entry["type"]),
            "temperature_feedback": _value(entry["forcing_temperature_feedback"]),
            **_scale(entry),
        }
    ).build()


def _aerosol_cloud(config: FairConfig, entry: Species) -> Component:
    return FairAerosolCloudBuilder.from_parameters(
        {
            "forcing": naming.forcing(entry),
            "drivers": _aerosol_drivers(config, "aci_shape", key="shape"),
            "scale": config["species_configs"]["aci_scale"],
            "temperature_feedback": _value(entry["forcing_temperature_feedback"]),
            **_scale(entry),
        }
    ).build()


def forcing_agents(config: FairConfig) -> list[Species]:
    """Species with a forcing, in FaIR order."""
    return [
        s
        for s in config["species"]
        if is_ghg(s)
        or s["type"] in (*LINEAR_TYPES, "aci")
        or s["input_mode"] == "forcing"
    ]


def prescribed_temperature_feedback(entry: Species) -> float:
    """Return the forcing per kelvin the aggregation adds to a prescribed forcing."""
    if entry["input_mode"] == "forcing":
        return _value(entry["forcing_temperature_feedback"])
    return 0.0


def _aggregation(config: FairConfig) -> Component:
    agents = [
        {
            "forcing": naming.forcing(s),
            "efficacy": _value(s["forcing_efficacy"]),
            "temperature_feedback": prescribed_temperature_feedback(s),
        }
        for s in forcing_agents(config)
    ]
    return FairForcingAggregationBuilder.from_parameters({"agents": agents}).build()


def _energy_balance(config: FairConfig) -> Component:
    climate = config["climate"]
    return FairEnergyBalanceBuilder.from_parameters(
        {
            "ocean_heat_capacity": climate["ocean_heat_capacity"],
            "ocean_heat_transfer": climate["ocean_heat_transfer"],
            "deep_ocean_efficacy": climate["deep_ocean_efficacy"],
            "gamma_autocorrelation": climate.get(
                "gamma_autocorrelation", GAMMA_AUTOCORRELATION
            ),
        }
    ).build()


def build_components(
    config: FairConfig, *, energy_balance: bool = True
) -> list[Component]:
    """Components reproducing a FaIR run, in the order they must be added to a model.

    Without `energy_balance`, another climate component must write the surface
    temperature from the total effective radiative forcing.
    """
    gases = [s for s in config["species"] if is_ghg(s)]
    prescribed = [s["name"] for s in gases if s["input_mode"] == "forcing"]
    if prescribed:
        msg = f"Prescribed greenhouse gas forcing is not supported: {prescribed}"
        raise ValueError(msg)
    components = []
    if thornhill(config):
        components.append(_ch4_lifetime(config))
    if gases:
        components.append(_gas_cycle(config, gases))
    if _calculated(config, "eesc") is not None:
        components.append(_eesc(config))
    if gases:
        components.append(_ghg_forcing(config, gases))
    for kind in LINEAR_TYPES:
        entry = _calculated(config, kind)
        if entry is not None:
            components.append(_linear_forcing(config, entry))
    entry = _calculated(config, "aci")
    if entry is not None:
        components.append(_aerosol_cloud(config, entry))
    components.append(_aggregation(config))
    if energy_balance:
        components.append(_energy_balance(config))
    return components


def initial_values(
    config: FairConfig, *, energy_balance: bool = True
) -> dict[str, float]:
    """Return FaIR's values at the first time of every variable the components write.

    `energy_balance` matches `build_components`.
    """
    values = dict.fromkeys(
        (naming.TOTAL_FORCING, naming.EFFICACY_ADJUSTED_FORCING), 0.0
    )
    if energy_balance:
        n_layers = len(config["climate"]["ocean_heat_capacity"])
        values |= dict.fromkeys(
            (
                naming.STOCHASTIC_FORCING,
                naming.OCEAN_HEAT_CONTENT_CHANGE,
                naming.ENERGY_IMBALANCE,
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
            values[naming.EESC] = _value(s["baseline_concentration"])
    for s in forcing_agents(config):
        if s["input_mode"] != "forcing":
            values[naming.forcing(s)] = 0.0
    return values
