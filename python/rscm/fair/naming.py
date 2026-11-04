"""RSCM variable names and FaIR units for FaIR species."""

from __future__ import annotations

from typing import Any

#: RSCM names of FaIR species that do not just drop their hyphens.
SPECIES_NAMES = {
    "CO2 FFI": "CO2|Fossil",
    "CO2 AFOLU": "CO2|Land Use",
    "Sulfur": "SOx",
    "VOC": "NMVOC",
    "NOx aviation": "NOx|Aviation",
    "HFC-4310mee": "HFC4310",
}

#: Forcing variables of calculated and prescribed agents, by FaIR species type.
FORCING_NAMES = {
    "ari": "Effective Radiative Forcing|Aerosol|Direct",
    "aci": "Effective Radiative Forcing|Aerosol|Indirect",
    "ozone": "Effective Radiative Forcing|Ozone",
    "contrails": "Effective Radiative Forcing|Contrails",
    "lapsi": "Effective Radiative Forcing|Black Carbon on Snow",
    "h2o stratospheric": "Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O",
    "land use": "Effective Radiative Forcing|Land Use",
    "solar": "Effective Radiative Forcing|Solar",
    "volcanic": "Effective Radiative Forcing|Volcanic",
}

HALOGEN_TYPES = ("cfc-11", "other halogen")
GHG_TYPES = ("co2", "ch4", "n2o", "f-gas", *HALOGEN_TYPES)

#: Molecule in the emissions unit, where it differs from the RSCM name.
_EMISSIONS_MOLECULE = {
    "CO2 FFI": "CO2",
    "CO2 AFOLU": "CO2",
    "Sulfur": "SO2",
    "NOx": "NO2",
    "NOx aviation": "NO2",
    "HFC-4310mee": "HFC4310mee",
}

EESC = "Equivalent Effective Stratospheric Chlorine"
CH4_LIFETIME_SCALING = "Lifetime Scaling|CH4"
SURFACE_TEMPERATURE = "Surface Temperature"
TOTAL_FORCING = "Effective Radiative Forcing"
EFFICACY_ADJUSTED_FORCING = "Effective Radiative Forcing|Efficacy Adjusted"
STOCHASTIC_FORCING = "Stochastic Forcing"
OCEAN_HEAT_CONTENT_CHANGE = "Ocean Heat Content Change"
ENERGY_IMBALANCE = "Net Energy Imbalance"


def rscm_name(specie: str) -> str:
    """RSCM name of a FaIR species, as used after `Emissions|`."""
    return SPECIES_NAMES.get(specie, specie.replace("-", ""))


def emissions(specie: str) -> str:
    """Emissions variable of a species."""
    return f"Emissions|{rscm_name(specie)}"


def concentration(specie: str) -> str:
    """Concentration variable of a species."""
    return f"Atmospheric Concentration|{rscm_name(specie)}"


def airborne_emissions(specie: str) -> str:
    """Airborne emissions variable of a greenhouse gas."""
    return f"Airborne Emissions|{rscm_name(specie)}"


def cumulative_emissions(specie: str) -> str:
    """Cumulative emissions variable of a greenhouse gas."""
    return f"Cumulative Emissions|{rscm_name(specie)}"


def forcing(entry: dict[str, Any]) -> str:
    """Return the forcing variable of a species entry of a FaIR config."""
    kind = entry["type"]
    name = rscm_name(entry["name"])
    if kind in ("co2", "ch4", "n2o"):
        return f"Effective Radiative Forcing|{name}"
    if kind == "f-gas":
        return f"Effective Radiative Forcing|F-Gases|{name}"
    if kind in HALOGEN_TYPES:
        return f"Effective Radiative Forcing|Montreal Protocol Halogen Gases|{name}"
    return FORCING_NAMES.get(kind, f"Effective Radiative Forcing|{name}")


def emissions_unit(entry: dict[str, Any]) -> str:
    """FaIR's emissions unit of a species entry, unless the entry sets one."""
    if "emissions_unit" in entry:
        return str(entry["emissions_unit"])
    specie = entry["name"]
    molecule = _EMISSIONS_MOLECULE.get(specie, specie.replace("-", ""))
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


def exogenous_variable(entry: dict[str, Any]) -> tuple[str, str] | None:
    """Return the input variable and unit of a species entry, None if calculated."""
    mode = entry["input_mode"]
    if mode == "emissions":
        return emissions(entry["name"]), emissions_unit(entry)
    if mode == "concentration":
        return concentration(entry["name"]), concentration_unit(entry)
    if mode == "forcing":
        return forcing(entry), "W/m^2"
    return None


def layer_temperature(layer: int) -> str:
    """Temperature variable of an energy balance layer, counting the surface as 0."""
    return SURFACE_TEMPERATURE if layer == 0 else f"Ocean Temperature|Layer {layer + 1}"
