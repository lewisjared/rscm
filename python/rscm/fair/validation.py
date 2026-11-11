"""
Checks on FaIR configurations and inputs before a run.

[validate][rscm.fair.validation.validate] raises an
[InputValidationError][rscm.validation.InputValidationError] naming every bad
parameter, species and input, rather than letting them run into NaN.
`setup_model`, `component_specs`, `from_fair`, `read_calibration` and `run_ensemble`
call these checks.
Each issue's location is a configuration key such as `climate.ocean_heat_capacity`,
a species parameter such as `HFC-23.forcing_reference_concentration`,
or an input variable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from rscm import names
from rscm.fair import naming
from rscm.fair.config import (
    CH4_LIFETIME_SENSITIVITY,
    CLIMATE_CONFIGS,
    LINEAR_COEFFICIENTS,
    LINEAR_TYPES,
    PROPERTY_FLAGS,
    SPECIES_CONFIGS,
    aerosol_drivers,
    forcing_agents,
    is_ghg,
    linear_drivers,
    thornhill,
)
from rscm.validation import IssueKind, Validator

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import ArrayLike

    from rscm.core import TimeAxis
    from rscm.fair.config import FairConfig, Species

__all__ = ["check_config", "check_inputs", "validate"]

CONFIG_KEYS = ("options", "climate", "species_configs", "species")
# Metadata the fixture configurations carry.
METADATA_KEYS = ("fair_version", "time")
OPTION_KEYS = ("ch4_method", "ghg_method", "iirf_max", "br_cl_ods_potential", "ch4_min")
GLOBAL_SPECIES_CONFIGS = ("aci_scale", "lifetime_temperature_sensitivity")
REQUIRED_SPECIES_KEYS = (
    "name",
    "type",
    "input_mode",
    *PROPERTY_FLAGS,
    *SPECIES_CONFIGS,
    "partition_fraction",
    "unperturbed_lifetime",
    "concentration_per_emission",
)
SPECIES_KEYS = (*REQUIRED_SPECIES_KEYS, "emissions_unit", "concentration_unit")
CH4_METHODS = ("thornhill2021", "leach2021")
FORCING_METHODS = ("meinshausen2020",)
# FaIR's other greenhouse gas methods, which RSCM does not implement.
OTHER_FORCING_METHODS = ("leach2021", "etminan2016", "myhre1998")

_GHG_MODES = ("emissions", "concentration", "forcing")
_FORCING_MODES = ("calculated", "forcing")
# FaIR's valid input modes of each species type.
INPUT_MODES = {
    "co2 ffi": ("emissions",),
    "co2 afolu": ("emissions",),
    "co2": (*_GHG_MODES, "calculated"),
    **dict.fromkeys(("ch4", "n2o", *naming.HALOGEN_TYPES, "f-gas"), _GHG_MODES),
    **dict.fromkeys(
        (
            "sulfur",
            "black carbon",
            "organic carbon",
            "other slcf",
            "nox aviation",
        ),
        ("emissions",),
    ),
    "eesc": ("concentration", "calculated"),
    **dict.fromkeys((*LINEAR_TYPES, "aci"), _FORCING_MODES),
    **dict.fromkeys(("volcanic", "solar", "unspecified"), ("forcing",)),
}
# Gas cycle parameters whose NaN reaches the concentrations.
# FaIR sets a NaN lifetime scaling to 1, so the iIRF and g parameters may be NaN.
_GAS_PARAMETERS = (
    "baseline_concentration",
    "baseline_emissions",
    "concentration_per_emission",
)
_SCALE_PARAMETERS = ("forcing_scale", "tropospheric_adjustment")
# FaIR sets a NaN lifetime scaling to 1, so only infinity is invalid.
_LIFETIME_SCALING_PARAMETERS = (
    "iirf_0",
    "iirf_airborne",
    "iirf_uptake",
    "iirf_temperature",
    "g0",
    "g1",
)


def _number(value: Any) -> float:
    try:
        return math.nan if value is None else float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite(v: Validator, location: str, value: Any) -> bool:
    try:
        array = np.asarray(math.nan if value is None else value, dtype=np.float64)
    except (TypeError, ValueError):
        v.add(IssueKind.NON_FINITE, location, f"{value!r} is not a number")
        return False
    if array.ndim:
        return v.finite(location, array)
    return v.check(
        bool(np.isfinite(array)), IssueKind.NON_FINITE, location, f"is {value}"
    )


def _positive(v: Validator, location: str, value: Any) -> None:
    if _finite(v, location, value):
        values = np.atleast_1d(np.asarray(value, dtype=np.float64))
        v.check(
            bool((values > 0).all()),
            IssueKind.OUT_OF_RANGE,
            location,
            f"must be positive, got {value}",
        )


def _keys(
    v: Validator,
    location: str,
    given: Iterable[str],
    known: Iterable[str],
) -> None:
    known = set(known)
    for key in given:
        if key not in known:
            v.add(
                IssueKind.UNKNOWN_KEY,
                f"{location}.{key}",
                f"not a key of {location}",
            )


def _species_parameter(v: Validator, entry: Species, key: str, *, at: str) -> bool:
    return _finite(v, f"{at}{entry['name']}.{key}", entry[key])


def _check_options(v: Validator, options: Mapping[str, Any], at: str) -> None:
    _keys(v, f"{at}options", options, OPTION_KEYS)
    for key in OPTION_KEYS:
        if key not in options:
            v.add(IssueKind.MISSING, f"{at}options.{key}", "the option is missing")
    ch4_method = options.get("ch4_method")
    if ch4_method is not None and ch4_method not in CH4_METHODS:
        v.add(
            IssueKind.UNKNOWN_KEY,
            f"{at}options.ch4_method",
            f"{ch4_method!r} is not one of {CH4_METHODS}",
        )
    ghg_method = options.get("ghg_method")
    if ghg_method in OTHER_FORCING_METHODS:
        v.add(
            IssueKind.UNSUPPORTED,
            f"{at}options.ghg_method",
            f"{ghg_method!r} is not implemented, only {FORCING_METHODS}",
        )
    elif ghg_method is not None and ghg_method not in FORCING_METHODS:
        v.add(
            IssueKind.UNKNOWN_KEY,
            f"{at}options.ghg_method",
            f"{ghg_method!r} is not a FaIR greenhouse gas method",
        )
    for key in ("iirf_max", "ch4_min"):
        if key in options:
            _positive(v, f"{at}options.{key}", options[key])
    if "br_cl_ods_potential" in options:
        _finite(v, f"{at}options.br_cl_ods_potential", options["br_cl_ods_potential"])


def _check_climate(v: Validator, climate: Mapping[str, Any], at: str) -> None:
    _keys(v, f"{at}climate", climate, CLIMATE_CONFIGS)
    for key in ("ocean_heat_capacity", "ocean_heat_transfer", "deep_ocean_efficacy"):
        if key not in climate:
            v.add(IssueKind.MISSING, f"{at}climate.{key}", "the parameter is missing")
    capacity: Any = climate.get("ocean_heat_capacity")
    transfer: Any = climate.get("ocean_heat_transfer")
    # A present but null parameter is non-finite, not missing.
    has_capacity = "ocean_heat_capacity" in climate and _finite(
        v, f"{at}climate.ocean_heat_capacity", capacity
    )
    has_transfer = "ocean_heat_transfer" in climate and _finite(
        v, f"{at}climate.ocean_heat_transfer", transfer
    )
    if has_capacity:
        n_layers = len(np.atleast_1d(capacity))
        v.check(
            n_layers >= 2,  # noqa: PLR2004
            IssueKind.OUT_OF_RANGE,
            f"{at}climate.ocean_heat_capacity",
            f"the energy balance needs at least two layers, got {n_layers}",
        )
        _positive(v, f"{at}climate.ocean_heat_capacity", capacity)
    if has_transfer:
        _positive(v, f"{at}climate.ocean_heat_transfer", transfer)
    if has_capacity and has_transfer:
        v.check(
            len(np.atleast_1d(capacity)) == len(np.atleast_1d(transfer)),
            IssueKind.SHAPE,
            f"{at}climate.ocean_heat_transfer",
            f"has {len(np.atleast_1d(transfer))} layers "
            f"but ocean_heat_capacity has {len(np.atleast_1d(capacity))}",
        )
    for key in ("deep_ocean_efficacy", "gamma_autocorrelation"):
        if key in climate:
            _positive(v, f"{at}climate.{key}", climate[key])


def _modes(species: list[Species], mode: str, *types: str) -> list[Species]:
    return [s for s in species if s["type"] in types and s["input_mode"] == mode]


def _check_dependencies(v: Validator, species: list[Species], at: str) -> None:
    """Species that a calculated species needs, as FaIR requires them."""
    for s in species:
        if s["type"] == "co2" and s["input_mode"] == "calculated":
            for kind in ("co2 ffi", "co2 afolu"):
                v.check(
                    bool(_modes(species, "emissions", kind)),
                    IssueKind.MISSING,
                    f"{at}{s['name']}",
                    f"calculated CO2 needs a {kind!r} species in emissions mode",
                )
        if s["type"] == "land use" and s["input_mode"] == "calculated":
            v.check(
                bool(_modes(species, "emissions", "co2 afolu")),
                IssueKind.MISSING,
                f"{at}{s['name']}",
                "calculated land use needs a 'co2 afolu' species in emissions mode",
            )
        if s["type"] == "eesc" and s["input_mode"] == "calculated":
            cfc11 = [c for c in species if c["type"] == "cfc-11" and is_ghg(c)]
            v.check(
                bool(cfc11),
                IssueKind.MISSING,
                f"{at}{s['name']}",
                "calculated EESC needs a 'cfc-11' greenhouse gas",
            )
    gases = [s for s in species if is_ghg(s)]
    if gases:
        for kind in ("co2", "ch4", "n2o"):
            v.check(
                any(s["type"] == kind for s in gases),
                IssueKind.MISSING,
                f"{at}{kind.upper()}",
                "Meinshausen et al. (2020) forcing needs CO2, CH4 and N2O",
            )


def _check_gas(v: Validator, s: Species, at: str) -> None:
    if s["input_mode"] == "forcing":
        v.add(
            IssueKind.UNSUPPORTED,
            f"{at}{s['name']}.input_mode",
            "FaIR overwrites the prescribed forcing of a greenhouse gas",
        )
    boxes = [
        _species_parameter(v, s, key, at=at)
        for key in ("partition_fraction", "unperturbed_lifetime")
    ]
    if all(boxes) and np.size(s["partition_fraction"]) != np.size(
        s["unperturbed_lifetime"]
    ):
        v.add(
            IssueKind.SHAPE,
            f"{at}{s['name']}.unperturbed_lifetime",
            "needs one lifetime per partition fraction",
        )
    keys = [*_GAS_PARAMETERS, "forcing_reference_concentration", *_SCALE_PARAMETERS]
    if s["type"] not in ("co2", "ch4", "n2o"):
        keys.append("greenhouse_gas_radiative_efficiency")
    for key in keys:
        _species_parameter(v, s, key, at=at)
    for key in _LIFETIME_SCALING_PARAMETERS:
        v.check(
            not math.isinf(_number(s[key])),
            IssueKind.NON_FINITE,
            f"{at}{s['name']}.{key}",
            f"is {s[key]}",
        )


def _check_drivers(
    v: Validator,
    drivers: list[tuple[Species, dict[str, Any]]],
    coefficient: str,
    at: str,
) -> None:
    """Check the coefficient and baseline of each term of a calculated forcing."""
    for s, driver in drivers:
        _species_parameter(v, s, coefficient, at=at)
        baseline = (
            "baseline_concentration"
            if driver["driver"] == "concentration"
            else "baseline_emissions"
        )
        # An unbaselined driver has a zero baseline.
        _finite(v, f"{at}{s['name']}.{baseline}", driver["baseline"])


def _calculated(species: list[Species], kind: str) -> bool:
    return any(s["type"] == kind and s["input_mode"] == "calculated" for s in species)


def _check_ch4_lifetime(v: Validator, config: FairConfig, at: str) -> None:
    key = "lifetime_temperature_sensitivity"
    _finite(v, f"{at}species_configs.{key}", config["species_configs"].get(key))
    for driver in ("emissions", "concentration"):
        for s in config["species"]:
            if s[f"aerosol_chemistry_from_{driver}"]:
                _species_parameter(v, s, CH4_LIFETIME_SENSITIVITY, at=at)
                _species_parameter(v, s, f"baseline_{driver}", at=at)


def _check_species(v: Validator, config: FairConfig, at: str) -> None:
    species = config["species"]
    species_configs = config["species_configs"]
    for s in species:
        if is_ghg(s):
            _check_gas(v, s, at)
        elif s["input_mode"] == "calculated" and s["type"] in (*LINEAR_TYPES, "aci"):
            for key in _SCALE_PARAMETERS:
                _species_parameter(v, s, key, at=at)
    for s in forcing_agents(config):
        for key in ("forcing_efficacy", "forcing_temperature_feedback"):
            _species_parameter(v, s, key, at=at)
    for kind, coefficient in LINEAR_COEFFICIENTS.items():
        if _calculated(species, kind):
            _check_drivers(v, linear_drivers(config, kind), coefficient, at)
    if _calculated(species, "aci"):
        drivers = aerosol_drivers(config, "aci_shape", key="shape")
        _check_drivers(v, drivers, "aci_shape", at)
        _finite(v, f"{at}species_configs.aci_scale", species_configs.get("aci_scale"))
    if thornhill(config):
        _check_ch4_lifetime(v, config, at)
    if _calculated(species, "eesc"):
        _check_eesc(v, species, at)


def _check_eesc(v: Validator, species: list[Species], at: str) -> None:
    for s in species:
        if s["type"] == "eesc":
            _species_parameter(v, s, "baseline_concentration", at=at)
        elif s["type"] in naming.HALOGEN_TYPES and is_ghg(s):
            for key in ("cl_atoms", "br_atoms", "fractional_release"):
                _species_parameter(v, s, key, at=at)
            if s["type"] == "cfc-11" and _number(s["fractional_release"]) == 0:
                v.add(
                    IssueKind.OUT_OF_RANGE,
                    f"{at}{s['name']}.fractional_release",
                    "EESC divides by the CFC-11 fractional release, so it cannot be 0",
                )


def _check_entries(v: Validator, species: list[Species], at: str) -> bool:
    """Check each species has every key and a known type and input mode."""
    valid = True
    for s in species:
        name = s.get("name", "?")
        _keys(v, f"{at}{name}", s, SPECIES_KEYS)
        for key in REQUIRED_SPECIES_KEYS:
            if key not in s:
                v.add(IssueKind.MISSING, f"{at}{name}.{key}", "the species has no key")
                valid = False
        kind, mode = s.get("type"), s.get("input_mode")
        if kind is None or mode is None:
            continue
        if kind not in INPUT_MODES:
            v.add(
                IssueKind.UNKNOWN_KEY,
                f"{at}{name}.type",
                f"{kind!r} is not a FaIR species type",
            )
            valid = False
        elif mode not in INPUT_MODES[kind]:
            v.add(
                IssueKind.UNKNOWN_KEY,
                f"{at}{name}.input_mode",
                f"{mode!r} is not an input mode of {kind!r}, "
                f"which takes {INPUT_MODES[kind]}",
            )
            valid = False
    return valid


def check_config(config: FairConfig, v: Validator, *, at: str = "") -> bool:
    """Record the issues of a configuration.

    Returns whether its species are well formed, so its inputs can be checked.
    `at` prefixes each location, such as the name of an ensemble member.
    """
    before = len(v.issues)
    _keys(v, f"{at}config", config, (*CONFIG_KEYS, *METADATA_KEYS))
    missing = [key for key in CONFIG_KEYS if key not in config]
    for key in missing:
        v.add(IssueKind.MISSING, f"{at}{key}", "the configuration has no key")
    if missing:
        return False
    _check_options(v, config["options"], at)
    _check_climate(v, config["climate"], at)
    _keys(v, f"{at}species_configs", config["species_configs"], GLOBAL_SPECIES_CONFIGS)

    well_formed = _check_entries(v, config["species"], at)
    if well_formed:
        _check_dependencies(v, config["species"], at)
        _check_species(v, config, at)
    # Forcings that share a driver check its baseline once each.
    v.issues[before:] = list(dict.fromkeys(v.issues[before:]))
    return well_formed


def _length_ok(
    v: Validator, location: str, array: np.ndarray[Any, Any], expected: tuple[int, ...]
) -> bool:
    if array.ndim == 1 and len(array) in expected:
        return True
    lengths = " or ".join(map(str, expected))
    v.add(
        IssueKind.SHAPE,
        location,
        f"has shape {array.shape}, expected {lengths} values on the time axis",
    )
    return False


def check_inputs(
    config: FairConfig,
    inputs: Mapping[str, tuple[ArrayLike, str]],
    time_axis: TimeAxis,
    v: Validator,
    *,
    at: str = "",
) -> bool:
    """Record the issues of the inputs of a valid configuration.

    Each prescribed variable must cover the time axis and be finite over the run.
    """
    before = len(v.issues)
    n_bounds = len(time_axis)
    v.check(
        n_bounds >= 2,  # noqa: PLR2004
        IssueKind.SHAPE,
        f"{at}time_axis",
        f"needs at least two bounds, got {n_bounds}",
    )
    for s in config["species"]:
        prescribed = naming.prescribed_variable(s)
        if prescribed is None:
            continue
        variable = prescribed[0]
        location = f"{at}{variable}"
        if variable not in inputs:
            v.add(
                IssueKind.MISSING,
                location,
                f"{s['name']} is {s['input_mode']}-driven but has no input",
            )
            continue
        try:
            array = np.asarray(inputs[variable][0], dtype=np.float64)
        except (TypeError, ValueError):
            v.add(IssueKind.NON_FINITE, location, "the values are not numbers")
            continue
        if variable.startswith(f"{names.EMISSIONS}|"):
            # The last bound starts no step, so its emissions are never read.
            if _length_ok(v, location, array, (n_bounds - 1, n_bounds)):
                v.finite(location, array[: n_bounds - 1])
        elif _length_ok(v, location, array, (n_bounds,)):
            v.finite(location, array)
    return len(v.issues) == before


def check_prescribed_scales(config: FairConfig, v: Validator, *, at: str = "") -> None:
    """Record non-finite `forcing_scale` values that scale prescribed forcing."""
    for s in config["species"]:
        if s["input_mode"] == "forcing":
            _species_parameter(v, s, "forcing_scale", at=at)


def validate(
    config: FairConfig,
    inputs: Mapping[str, tuple[ArrayLike, str]] | None = None,
    time_axis: TimeAxis | None = None,
    *,
    scale_prescribed_forcing: bool = False,
) -> None:
    """Raise an `InputValidationError` with every issue of a configuration and inputs.

    The inputs are checked when given, against `time_axis`.
    `scale_prescribed_forcing` also checks the scales of prescribed forcing.
    """
    v = Validator()
    if check_config(config, v):
        if scale_prescribed_forcing:
            check_prescribed_scales(config, v)
        if inputs is not None and time_axis is not None:
            check_inputs(config, inputs, time_axis, v)
    v.raise_if_any()
