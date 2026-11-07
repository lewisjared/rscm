"""
Variable names shared between components, crates and the Python package.

A name is `Quantity|Species|Sector`, then any further breakdown,
as in `Emissions|CO2|Fossil`.
A child segment is always a part of its parent.
Other qualifiers lead the quantity segment,
as in `Prescribed Atmospheric Concentrations|CO2`.
Names follow gcages and MAGICC7 where they overlap.

The constants come from `rscm_core::names` in Rust.
[convert_variable_name][rscm.names.convert_variable_name] maps names between RSCM
and the MAGICC7 and RCMIP conventions that reference data and input files use.
"""

import re
from enum import StrEnum

from rscm._lib.core import Unit
from rscm._lib.core.names import *  # noqa: F403
from rscm._lib.core.names import ERF, OCEAN_TEMPERATURE

SECTORS = ("Fossil", "Biosphere")
"""Sector labels of sectoral emissions, as in `Emissions|CO2|Fossil`."""

EMISSIONS_UNITS = {
    "CO2": "Mt CO2/yr",
    "CH4": "Mt CH4/yr",
    "N2O": "kt N2O/yr",
    "SOx": "Mt SO2/yr",
    "NOx": "Mt NO2/yr",
    "NH3": "Mt NH3/yr",
    "BC": "Mt BC/yr",
    "OC": "Mt OC/yr",
    "CO": "Mt CO/yr",
    "NMVOC": "Mt VOC/yr",
}
"""
Unit of each species' emissions, on the gcages mass basis.
Halocarbons are in kt of the gas.
"""

MAGICC7_EMISSIONS_UNITS = {
    "CO2": "GtC/yr",
    "CH4": "Mt CH4/yr",
    "N2O": "Mt N2ON/yr",
    "SOx": "Mt S/yr",
    "NOx": "Mt N/yr",
    "NH3": "Mt N/yr",
    "BC": "Mt BC/yr",
    "OC": "Mt OC/yr",
    "CO": "Mt CO/yr",
    "NMVOC": "Mt NMVOC/yr",
}
"""
Unit MAGICC7 reads and writes each species' emissions in.
Halocarbons are in kt of the gas.
"""

NH3_PER_N = 17.0 / 14.0
"""
Mt NH3 per Mt N, with the nominal masses of the unit parser.
The parser does not convert NH3 to N, because NH3 is outside the N family.
"""


class Convention(StrEnum):
    """Naming conventions that `convert_variable_name` translates between."""

    RSCM = "rscm"
    """RSCM's names, the gcages convention where they overlap."""

    MAGICC7 = "magicc7"
    """MAGICC7 output and RCMIP input names, as in `Emissions|CO2|MAGICC AFOLU`."""


_SECTORS_IN = {
    "MAGICC Fossil and Industrial": "Fossil",
    "Fossil and Industrial": "Fossil",
    "Energy and Industrial Processes": "Fossil",
    "MAGICC AFOLU": "Biosphere",
    "AFOLU": "Biosphere",
    "Land Use": "Biosphere",
}
_SECTORS_OUT = {"Fossil": "MAGICC Fossil and Industrial", "Biosphere": "MAGICC AFOLU"}

_SPECIES_IN = {
    "HFC4310": "HFC4310mee",
    "HFC43-10": "HFC4310mee",
    "HFC43-10mee": "HFC4310mee",
    "Sulfur": "SOx",
    "VOC": "NMVOC",
}
_SPECIES_OUT = {"HFC4310mee": "HFC4310"}

_FORCING_IN = {
    "Stratospheric Ozone": "Ozone|Stratospheric",
    "Tropospheric Ozone": "Ozone|Tropospheric",
    "BC on Snow": "Black Carbon on Snow",
    "Land Use": "Land-use Change",
    "Contrails": "Aviation|Contrail",
}
_FORCING_OUT = {
    "Ozone|Stratospheric": "Stratospheric Ozone",
    "Ozone|Tropospheric": "Tropospheric Ozone",
}

_NAMES_IN = {"Surface Temperature": "Surface Air Temperature Change"}
_NAMES_OUT = {v: k for k, v in _NAMES_IN.items()}

# RCMIP groups halocarbons, as in `Emissions|F-Gases|PFC|CF4`.
_HALOCARBON_GROUP = re.compile(
    r"^(Emissions|Atmospheric Concentrations)\|"
    r"(?:F-Gases|Montreal Gases)\|(?:(?:PFC|HFC|CFC)\|)?"
)
_QUANTITIES = ("Emissions", "Atmospheric Concentrations")
# Fewest segments of a name that ends in a sector, as in `Emissions|CO2|Fossil`.
_SECTORAL_SEGMENTS = 3


_HYPHENATED = re.compile(r"^(?:CFC|HCFC|HFC|Halon)-")


def _species_in(segment: str) -> str:
    if _HYPHENATED.match(segment):
        segment = segment.replace("-", "")
    return _SPECIES_IN.get(segment, segment)


def convert_variable_name(
    variable: str,
    from_convention: Convention | str = Convention.MAGICC7,
    to_convention: Convention | str = Convention.RSCM,
) -> str:
    """
    Convert a variable name between naming conventions.

    Translates sector labels (`MAGICC AFOLU` and `Biosphere`),
    species tokens (`HFC4310` and `HFC4310mee`), the temperature name
    and the forcing agents whose names differ.
    Names that need no translation come back unchanged.

    Parameters
    ----------
    variable
        Name to convert
    from_convention
        Convention of `variable`
    to_convention
        Convention to convert to

    Returns
    -------
    :
        `variable` in `to_convention`

    Examples
    --------
    >>> convert_variable_name("Emissions|CO2|MAGICC AFOLU")
    'Emissions|CO2|Biosphere'
    >>> convert_variable_name("Emissions|HFC4310mee", "rscm", "magicc7")
    'Emissions|HFC4310'
    """
    source, target = Convention(from_convention), Convention(to_convention)
    if source == target:
        return variable
    if source == Convention.MAGICC7:
        return _to_rscm(variable)
    return _to_magicc7(variable)


def _to_rscm(variable: str) -> str:
    if variable in _NAMES_IN:
        return _NAMES_IN[variable]
    variable = re.sub(
        r"^Atmospheric Concentration\|", "Atmospheric Concentrations|", variable
    )
    variable = _HALOCARBON_GROUP.sub(r"\1|", variable)
    parts = variable.split("|")
    if parts[0] in _QUANTITIES and len(parts) > 1:
        parts[1] = _species_in(parts[1])
    if parts[0] == ERF:
        rest = "|".join(parts[1:])
        rest = _FORCING_IN.get(rest, rest)
        parts = [ERF, *rest.split("|")] if rest else [ERF]
    if len(parts) >= _SECTORAL_SEGMENTS:
        parts[-1] = _SECTORS_IN.get(parts[-1], parts[-1])
    return "|".join(parts)


def _to_magicc7(variable: str) -> str:
    if variable in _NAMES_OUT:
        return _NAMES_OUT[variable]
    parts = variable.split("|")
    if parts[0] in _QUANTITIES and len(parts) > 1:
        parts[1] = _SPECIES_OUT.get(parts[1], parts[1])
    if parts[0] == ERF:
        rest = "|".join(parts[1:])
        rest = _FORCING_OUT.get(rest, rest)
        parts = [ERF, *rest.split("|")] if rest else [ERF]
    if len(parts) >= _SECTORAL_SEGMENTS and parts[0] in (*_QUANTITIES, ERF):
        parts[-1] = _SECTORS_OUT.get(parts[-1], parts[-1])
    return "|".join(parts)


def emissions_unit(
    variable: str, convention: Convention | str = Convention.RSCM
) -> str:
    """
    Return the unit of an emissions variable in a convention.

    Parameters
    ----------
    variable
        RSCM emissions name, as in `Emissions|CO2|Fossil`
    convention
        Convention of the unit

    Returns
    -------
    :
        `Mt CO2/yr` for `Emissions|CO2` in RSCM, `GtC/yr` in MAGICC7,
        and `kt <species>/yr` for a halocarbon
    """
    species = variable.split("|")[1]
    table = (
        EMISSIONS_UNITS
        if Convention(convention) == Convention.RSCM
        else MAGICC7_EMISSIONS_UNITS
    )
    return table.get(species, f"kt {species}/yr")


def emissions_conversion_factor(
    variable: str,
    from_convention: Convention | str = Convention.MAGICC7,
    to_convention: Convention | str = Convention.RSCM,
) -> float:
    """
    Factor converting emissions of `variable` from one convention's unit to another's.

    NH3 converts between Mt N and Mt NH3 with `NH3_PER_N`.

    Parameters
    ----------
    variable
        RSCM emissions name, as in `Emissions|NOx|Fossil`
    from_convention
        Convention of the unit of the values
    to_convention
        Convention of the unit to convert to

    Returns
    -------
    :
        Factor to multiply the values by
    """
    source = emissions_unit(variable, from_convention)
    target = emissions_unit(variable, to_convention)
    if source == target:
        return 1.0
    if variable.split("|")[1] == "NH3":
        from_magicc7 = Convention(from_convention) == Convention.MAGICC7
        return NH3_PER_N if from_magicc7 else 1.0 / NH3_PER_N
    return Unit(source).conversion_factor(Unit(target))


def child(parent: str, part: str) -> str:
    """Join name segments with `|`, so `child(EMISSIONS, "CO2")` is `Emissions|CO2`."""
    return f"{parent}|{part}"


def prescribed(name: str) -> str:
    """Name of the prescribed input that drives `name`."""
    return f"Prescribed {name}"


def ocean_layer_temperature(layer: int) -> str:
    """Name of the temperature of ocean `layer`, counting the surface as layer 1."""
    return f"{OCEAN_TEMPERATURE}|Layer {layer}"
