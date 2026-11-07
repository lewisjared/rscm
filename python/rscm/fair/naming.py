"""RSCM variable names and FaIR units for FaIR species."""

from __future__ import annotations

from typing import Any

from rscm import names

# RSCM names of FaIR species that do not just drop their hyphens.
SPECIES_NAMES = {
    "CO2 FFI": "CO2|Fossil",
    "CO2 AFOLU": "CO2|Biosphere",
    "Sulfur": "SOx",
    "VOC": "NMVOC",
    "NOx aviation": "NOx|Aviation",
}

# Forcing variables of calculated and prescribed agents, by FaIR species type.
FORCING_NAMES = {
    "ari": names.ERF_AEROSOLS_DIRECT_EFFECT,
    "aci": names.ERF_AEROSOLS_INDIRECT_EFFECT,
    "ozone": names.ERF_OZONE,
    "contrails": names.ERF_AVIATION_CONTRAIL,
    "lapsi": names.ERF_BLACK_CARBON_ON_SNOW,
    "h2o stratospheric": names.ERF_CH4_OXIDATION_STRATOSPHERIC_H2O,
    "land use": names.ERF_LAND_USE_CHANGE,
    "solar": names.ERF_SOLAR,
    "volcanic": names.ERF_VOLCANIC,
}

HALOGEN_TYPES = ("cfc-11", "other halogen")
GHG_TYPES = ("co2", "ch4", "n2o", "f-gas", *HALOGEN_TYPES)

# Molecule in the emissions unit, where it differs from the RSCM name.
_EMISSIONS_MOLECULE = {
    "CO2 FFI": "CO2",
    "CO2 AFOLU": "CO2",
    "Sulfur": "SO2",
    "NOx": "NO2",
    "NOx aviation": "NO2",
}


def rscm_name(species: str) -> str:
    """RSCM name of a FaIR species, as used after `Emissions|`."""
    return SPECIES_NAMES.get(species, species.replace("-", ""))


def emissions(species: str) -> str:
    """Emissions variable of a species."""
    return names.child(names.EMISSIONS, rscm_name(species))


def inverse_emissions(species: str) -> str:
    """Emissions of a species inferred from its prescribed concentration."""
    return names.child(names.INVERSE_EMISSIONS, rscm_name(species))


def concentration(species: str) -> str:
    """Concentration variable of a species."""
    return names.child(names.ATMOSPHERIC_CONCENTRATIONS, rscm_name(species))


def airborne_emissions(species: str) -> str:
    """Airborne emissions variable of a greenhouse gas."""
    return names.child(names.AIRBORNE_EMISSIONS, rscm_name(species))


def prescribed_forcing(entry: dict[str, Any]) -> str:
    """Return the prescribed forcing input, before its temperature feedback."""
    return names.prescribed(forcing(entry))


def cumulative_emissions(species: str) -> str:
    """Cumulative emissions variable of a greenhouse gas."""
    return names.child(names.CUMULATIVE_EMISSIONS, rscm_name(species))


def forcing(entry: dict[str, Any]) -> str:
    """Return the forcing variable of a species entry of a FaIR config."""
    kind = entry["type"]
    name = rscm_name(entry["name"])
    if kind == "f-gas":
        return names.child(names.ERF_F_GASES, name)
    if kind in HALOGEN_TYPES:
        return names.child(names.ERF_MONTREAL_PROTOCOL_HALOGEN_GASES, name)
    return FORCING_NAMES.get(kind, names.child(names.ERF, name))


def emissions_unit(entry: dict[str, Any]) -> str:
    """FaIR's emissions unit of a species entry, unless the entry sets one."""
    if "emissions_unit" in entry:
        return str(entry["emissions_unit"])
    name = entry["name"]
    molecule = _EMISSIONS_MOLECULE.get(name, name.replace("-", ""))
    if entry["type"] in ("co2 ffi", "co2 afolu", "co2"):
        return "Gt CO2/yr"
    if entry["type"] in ("f-gas", *HALOGEN_TYPES):
        return f"kt {molecule}/yr"
    return f"Mt {molecule}/yr"


def amount_unit(entry: dict[str, Any]) -> str:
    """Return the unit of the airborne and cumulative emissions of a species entry."""
    return emissions_unit(entry).removesuffix("/yr")


def concentration_unit(entry: dict[str, Any]) -> str:
    """FaIR's concentration unit of a species entry, unless the entry sets one."""
    if "concentration_unit" in entry:
        return str(entry["concentration_unit"])
    return {"co2": "ppm", "ch4": "ppb", "n2o": "ppb"}.get(entry["type"], "ppt")


def prescribed_variable(entry: dict[str, Any]) -> tuple[str, str] | None:
    """Return the input variable and unit of a species entry, None if calculated."""
    mode = entry["input_mode"]
    if mode == "emissions":
        return emissions(entry["name"]), emissions_unit(entry)
    if mode == "concentration":
        variable = (
            names.EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE
            if entry["type"] == "eesc"
            else concentration(entry["name"])
        )
        return variable, concentration_unit(entry)
    if mode == "forcing":
        return prescribed_forcing(entry), "W/m^2"
    return None


def layer_temperature(layer: int) -> str:
    """Temperature variable of an energy balance layer, counting the surface as 0."""
    if layer == 0:
        return names.SURFACE_AIR_TEMPERATURE_CHANGE
    return names.ocean_layer_temperature(layer + 1)
