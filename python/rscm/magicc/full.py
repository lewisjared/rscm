"""Translate MAGICC7's configuration and input files into an RSCM MAGICC model.

`from_magicc7` turns MAGICC7 parameters, input files and emissions
into the configuration and inputs that [rscm.magicc.emissions_driven][]
and [rscm.magicc.concentration_driven][] build.
It covers chemistry, the carbon cycle, permafrost, GHG and ozone forcing,
the forcing MAGICC7 prescribes from its input files, forcing aggregation
with efficacies and regional patterns, and ClimateUDEB with monthly volcanic forcing.

Parameters use MAGICC7's configuration names in lower case,
such as ``core_climatesensitivity``.
`default_parameters` gives MAGICC7 v7.5.3's defaults
for every name the translation reads,
and the packaged input files are those of its default SSP245 run.
"""

from __future__ import annotations

import copy
import json
from collections import ChainMap
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from rscm._lib.magicc import ClimateUDEBBuilder, HalocarbonChemistryBuilder
from rscm.core import TimeAxis
from rscm.magicc import emissions_driven
from rscm.magicc import prescribed as pr
from rscm.names import (
    ATMOSPHERIC_BURDEN_CH4_BIOSPHERE,
    ATMOSPHERIC_BURDEN_CH4_FOSSIL,
    ATMOSPHERIC_BURDEN_CH4_NATURAL,
    ATMOSPHERIC_BURDEN_CH4_PERMAFROST,
    ATMOSPHERIC_CONCENTRATIONS,
    CARBON_POOL_DETRITUS,
    CARBON_POOL_PLANT,
    CARBON_POOL_SOIL,
    CUMULATIVE_CARBON_UPTAKE_OCEAN,
    EMISSIONS,
    EMISSIONS_UNITS,
    ERF,
    ERF_F_GASES,
    MAGICC7_EMISSIONS_UNITS,
    NITROGEN_FLUX_DEPOSITION,
    NITROGEN_FLUX_FIXATION,
    OCEAN_SURFACE_PARTIAL_PRESSURE_CO2,
    SECTORS,
    Convention,
    child,
    emissions_conversion_factor,
    emissions_unit,
)
from rscm.names import prescribed as prescribed_input

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

    from rscm.core import Model
    from rscm.magicc.model import ModelConfig

    Inputs = dict[str, tuple[NDArray[np.float64], str]]

__all__ = [
    "EFFICACY_METHODS",
    "INPUT_FILES",
    "PRESCRIBED_AGENTS",
    "REWRITTEN_PARAMETERS",
    "SECTORAL_EMISSIONS",
    "climate_parameters",
    "concentration_switch_years",
    "default_initial_concentrations",
    "default_parameters",
    "efficacy_climate",
    "forcing_agents",
    "from_magicc7",
    "halocarbon_species",
    "load_inputs",
    "magicc7_units",
    "natural_emissions",
    "nitrogen_inputs",
    "nitrogen_parameters",
    "ozone_parameters",
    "permafrost_parameters",
    "prescribed_concentrations",
    "prescribed_forcing",
]

DATA = Path(__file__).parent / "data" / "magicc7"
SECTORAL_EMISSIONS = {
    species: EMISSIONS_UNITS[species]
    for species in ("CO2", "CH4", "N2O", "SOx", "BC", "OC", "NOx", "NH3", "CO", "NMVOC")
}
"""
Emissions given by sector (``"CO2|Fossil"`` and ``"CO2|Biosphere"``) and their units,
on the gcages mass basis.
"""

INPUT_FILES = (
    *(f"file_{name}_rf" for name in ("soxi", "soxb", "bci", "bcb", "oci", "ocb")),
    "file_bcsnow_rf",
    *(
        f"file_{name}_ot"
        for name in ("soxi", "soxnb", "oci", "ocb", "ocn", "bci", "bcb", "ss")
    ),
    "file_landuse_rf",
    "file_solar_rf",
    "file_volcanic_rf",
)
"""Configuration keys of the MAGICC7 input files the chain reads."""

CONCENTRATION_UNITS = {
    "co2": "ppm",
    "ch4": "ppb",
    "n2o": "ppb",
    "fgas": "ppt",
    "mhalo": "ppt",
}
"""
Gas groups that MAGICC7 switches from prescribed concentrations to emissions,
and their concentration units.
"""

REWRITTEN_PARAMETERS = (
    "cloud_weight_bc",
    "cloud_weight_no3",
    "cloud_weight_oc",
    "cloud_weight_sox",
    "cloud_weight_ss",
    "core_delq2xco2",
    "rf_bcsnow_albedo_wm2",
    "rf_cloud_albedo_aer_wm2",
    "rf_mineraldust_dir_wm2",
    "rf_regions_no3",
)
"""
Configuration keys MAGICC7 overwrites during a run,
so its parameter dump does not hold their input values.
"""

NITROGEN_INPUT_FILES = ("file_nfix_in", "file_ndep_in")
"""Configuration keys of the nitrogen input files, read when they exist."""

DIRECT = "Aerosols|Direct Effect"
PRESCRIBED_AGENTS = {
    f"{DIRECT}|SOx|Fossil": "sox",
    f"{DIRECT}|BC|Fossil": "bc",
    f"{DIRECT}|OC|Fossil": "oc",
    f"{DIRECT}|NO3|Fossil": "no3",
    f"{DIRECT}|Mineral Dust": "aer_dir",
    "Aerosols|Indirect Effect": "cloud_albedo",
    "Black Carbon on Snow": "bcsnow",
    "Land-use Change": "landuse",
    "Solar": "solar",
}
"""
Agents calculated before the run, and the MAGICC7 efficacy key of each.
MAGICC7's total uses the direct aerosol efficacy for biomass aerosols and dust.
"""

NORMYEAR_PATTERNS = {
    "bcsnow": "Black Carbon on Snow",
    "landuse": "Land-use Change",
}
"""
Prescribed agents whose regional pattern MAGICC7 takes from their forcing
in ``rf_regions_normyear``.
"""

GREENHOUSE_GASES = ("CO2", "CH4", "N2O", "F-Gases", "Montreal Protocol Halogen Gases")
"""
Agents in MAGICC7's greenhouse gas ERF.
Agents under ``Aerosols|`` form its aerosol ERF.
"""

NATURAL_AGENTS = ("Solar",)
"""Agents left out of MAGICC7's anthropogenic ERF, along with volcanic forcing."""

HALOCARBON_GROUPS = ("F-Gases", "Montreal Protocol Halogen Gases")
"""Halocarbon agents whose efficacy MAGICC7 applies to each species."""

BIOMASS_AGENTS = tuple(
    f"{DIRECT}|{species}|Biosphere" for species in ("SOx", "NO3", "OC", "BC")
)
"""Prescribed agents MAGICC7 adds into one biomass burning aerosol agent."""
BIOMASS = f"{DIRECT}|Biomass Burning"

SUMMATION: list[Any] = [
    "CO2",
    "CH4",
    "N2O",
    "F-Gases",
    "Montreal Protocol Halogen Gases",
    [
        *(f"{DIRECT}|{species}|Fossil" for species in ("BC", "OC", "SOx", "NO3")),
        BIOMASS,
        f"{DIRECT}|Mineral Dust",
    ],
    "Aerosols|Indirect Effect",
    [
        "Ozone|Stratospheric",
        "Ozone|Tropospheric",
        "Ozone|N2O Feedback",
        "Ozone|Temperature Feedback",
    ],
    "CH4 Oxidation Stratospheric H2O",
    "Land-use Change",
    "Black Carbon on Snow",
    "Solar",
]
"""Order in which MAGICC7 adds the agents into its total, nesting its subtotals."""

EFFICACY_ERF_2XCO2 = 3.71
"""
2xCO2 effective radiative forcing behind MAGICC7's internal efficacies
until the run calculates one.
"""

CALCULATED_AGENTS = {
    "CO2": "co2",
    "CH4": "ch4",
    "N2O": "n2o",
    "F-Gases": "fgas",
    "Montreal Protocol Halogen Gases": "mhalo",
    "Ozone|Stratospheric": "stratoz",
    "CH4 Oxidation Stratospheric H2O": "ch4oxstrath2o",
    "Ozone|N2O Feedback": "ozdueton2o",
    "Ozone|Temperature Feedback": "ozduetotemperature",
}
"""Agents calculated during the run, and the MAGICC7 efficacy key of each."""

EFFICACY_METHODS = {0: "none", 1: "unit", 2: "ar6"}
"""Efficacy methods by MAGICC7's ``rf_efficacy_apply``."""

TERRESTRIAL_FEEDBACKS = {
    "npp": "npp",
    "respiration": "respiration",
    "detritus": "detritus",
    "soil": "soil",
}
"""
TerrestrialCarbon temperature sensitivities and their ``co2_feedbackfactor_*`` names.
"""

CLIMATE_PARAMETERS = {
    "ecs": "core_climatesensitivity",
    "erf_2xco2": "core_delq2xco2",
    "land_ocean_warming_ratio": "core_rlo",
    "land_ocean_heat_exchange": "core_heatxchange_landocean",
    "north_south_heat_exchange": "core_heatxchange_northsouth",
    "amplify_ocean_to_land": "core_amplify_ocn2land_heatxchng",
    "vertical_diffusivity": "core_verticaldiffusivity",
    "minimum_vertical_diffusivity": "core_verticaldiffusivity_min",
    "vertical_diffusivity_gradient_sensitivity": "core_verticaldiff_top_dkdt",
    "initial_upwelling_rate": "core_initial_upwelling_rate",
    "upwelling_variable_fraction": "core_upwelling_variable_part",
    "feedback_forcing_sensitivity": "core_feedback_qsensitivity",
    "feedback_cumulative_temperature_sensitivity": "core_feedback_cumtsensitivity",
    "feedback_cumulative_temperature_period": "core_feedback_cumtperiod",
    "temperature_adjustment_alpha": "core_tempadjust_ocn2atm_alpha",
    "temperature_adjustment_gamma": "core_tempadjust_ocn2atm_gamma",
    "polar_sinking_ratio": "core_polarsinkwater_tempratio",
    "land_ground_heat_exchange": "core_heatxchange_landground",
    "mixed_layer_depth": "core_mixedlayer_depth",
    "co2_regional_pattern": "rf_regions_co2",
    "land_heat_capacity_effective_thickness": "core_landhc_effthickness",
}
"""ClimateUDEB parameters and their MAGICC7 names."""


# CH4Chemistry defaults of the lifetimes combined into ``non_oh_lifetime``
CH4_OTHER_LIFETIMES = {
    "soil_lifetime": 150.0,
    "stratospheric_lifetime": 120.0,
    "tropospheric_cl_lifetime": 200.0,
}


def _ch4_non_oh_lifetime(lifetimes: Mapping[str, float]) -> float:
    """CH4 lifetime against soil, stratospheric and chlorine loss, as in MAGICC7."""
    tau = {**CH4_OTHER_LIFETIMES, **lifetimes}
    return 1.0 / sum(1.0 / tau[k] for k in CH4_OTHER_LIFETIMES)


def _ch4_burdens(
    concentration: float, tg_per_ppb: float, mixing_box_size: float
) -> dict[str, float]:
    """Return initial CH4Oxidation pools, with all CH4 natural as in MAGICC7."""
    return {
        ATMOSPHERIC_BURDEN_CH4_FOSSIL: 0.0,
        ATMOSPHERIC_BURDEN_CH4_BIOSPHERE: 0.0,
        ATMOSPHERIC_BURDEN_CH4_PERMAFROST: 0.0,
        ATMOSPHERIC_BURDEN_CH4_NATURAL: tg_per_ppb * concentration * mixing_box_size,
    }


@cache
def _defaults() -> dict[str, Any]:
    with open(DATA / "defaults.json") as f:
        result: dict[str, Any] = json.load(f)
    return result


def default_parameters() -> dict[str, Any]:
    """MAGICC7 v7.5.3's default configuration, for every name the chain reads."""
    return copy.deepcopy(_defaults()["parameters"])


def default_initial_concentrations() -> dict[str, float]:
    """MAGICC7's 1750 concentrations.

    CO2 is in ppm, CH4 and N2O in ppb and halocarbons in ppt.
    """
    return dict(_defaults()["initial_concentrations"])


def halocarbon_forcing(group: str) -> list[str]:
    """ERF variables of each species in a halocarbon `group`, in MAGICC7's order."""
    component = HalocarbonChemistryBuilder.from_parameters({}).build()
    prefix = f"{ERF}|{group}|"
    return [d.name for d in component.definitions() if d.name.startswith(prefix)]


def halocarbon_species() -> list[str]:
    """Names of the halocarbons the chain reads emissions of, in kt per year."""
    component = HalocarbonChemistryBuilder.from_parameters({}).build()
    prefix = f"{EMISSIONS}|"
    return [
        d.name.removeprefix(prefix)
        for d in component.definitions()
        if d.name.startswith(prefix)
    ]


def load_inputs(
    parameters: Mapping[str, Any] | None = None, directory: Path | None = None
) -> dict[str, pr.AerosolHistory]:
    """Read every input file named in ``parameters``, for any number of runs.

    Forcing files are keyed by `INPUT_FILES` and the concentration files that exist
    by ``file_<species>_conc`` in lower case, such as ``file_cfc11_conc``.
    With ``ncycle_apply`` this also reads the `NITROGEN_INPUT_FILES` that exist.
    ``directory`` defaults to the packaged files of MAGICC7's SSP245 run,
    which are also read gzipped.
    """
    p = parameters or default_parameters()
    root = directory or DATA
    out = {key: pr.read_input(_find(root, p[key])) for key in INPUT_FILES}
    for group in _concentration_files(p).values():
        for species, file in group:
            if file.strip() and (path := _find(root, file)).exists():
                out[_concentration_key(species)] = pr.read_input(path)
    if p["ncycle_apply"]:
        for key in NITROGEN_INPUT_FILES:
            if (path := _find(root, p[key])).exists():
                out[key] = pr.read_input(path)
    return out


def _find(root: Path, name: str) -> Path:
    path = root / name
    gzipped = path.with_name(path.name + ".gz")
    return gzipped if not path.exists() and gzipped.exists() else path


def _concentration_key(species: str) -> str:
    return f"file_{species.lower()}_conc"


def _concentration_input(
    inputs: Mapping[str, pr.AerosolHistory], species: str
) -> pr.AerosolHistory:
    key = _concentration_key(species)
    if key not in inputs:
        msg = f"inputs has no {key}: its file is missing or not from load_inputs"
        raise KeyError(msg)
    return inputs[key]


def _concentration_files(
    parameters: Mapping[str, Any],
) -> dict[str, list[tuple[str, str]]]:
    """Species and concentration file of each gas group."""
    p = parameters
    species = halocarbon_species()
    n_fgas = p["fgas_n"]
    return {
        "co2": [("CO2", p["file_co2_conc"])],
        "ch4": [("CH4", p["file_ch4_conc"])],
        "n2o": [("N2O", p["file_n2o_conc"])],
        # MAGICC7 pads the file lists beyond its species
        "fgas": list(zip(species[:n_fgas], p["fgas_files_conc"], strict=False)),
        "mhalo": list(zip(species[n_fgas:], p["mhalo_files_conc"], strict=False)),
    }


def concentration_switch_years(
    years: ArrayLike, parameters: Mapping[str, Any]
) -> dict[str, float | None]:
    """Last year of prescribed concentrations of each gas group after the first year.

    The groups are ``co2``, ``ch4``, ``n2o``, ``fgas`` and ``mhalo``,
    from MAGICC7's ``<group>_switchfromconc2emis_year``.
    """
    first = float(np.asarray(years, dtype=np.float64)[0])
    out: dict[str, float | None] = {}
    for group in CONCENTRATION_UNITS:
        year = float(parameters[f"{group}_switchfromconc2emis_year"])
        out[group] = year if year > first else None
    return out


def prescribed_concentrations(
    years: ArrayLike,
    parameters: Mapping[str, Any] | None = None,
    inputs: Mapping[str, pr.AerosolHistory] | None = None,
) -> dict[str, tuple[NDArray[np.float64], str]]:
    """Concentrations and units of each species prescribed before its switch year.

    Only species whose switch year is after the first year are included.
    A species without a concentration file (Halon1202 in MAGICC7) stays at zero.
    ``inputs`` defaults to `load_inputs` of ``parameters``.
    """
    p = parameters or default_parameters()
    files = inputs if inputs is not None else load_inputs(p)
    y = np.asarray(years, dtype=np.float64)
    groups = _concentration_files(p)
    out = {}
    for group, year in concentration_switch_years(y, p).items():
        if year is None:
            continue
        for name, file in groups[group]:
            values = (
                pr.on_axis(_concentration_input(files, name), y)[:, 0]
                if file.strip()
                else np.zeros(len(y))
            )
            out[name] = (values, CONCENTRATION_UNITS[group])
    return out


def _natural(natural: Mapping[str, float], gas: str) -> dict[str, float]:
    return {"natural_emissions": natural[gas]} if gas in natural else {}


def _sequential_sum(values: Iterable[float]) -> float:
    # Python 3.12+ `sum` compensates rounding, unlike MAGICC7's plain loop.
    total = 0.0
    for value in values:
        total += value
    return total


def _total(
    emissions: Mapping[str, NDArray[np.float64]], species: str
) -> NDArray[np.float64]:
    return emissions[f"{species}|Fossil"] + emissions[f"{species}|Biosphere"]


def _nitrate_inputs(
    years: NDArray[np.float64],
    emissions: Mapping[str, NDArray[np.float64]],
    parameters: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Arguments of `pr.nitrate` and `pr.nitrate_pattern`."""
    p = parameters
    args = (
        years,
        (emissions["NOx|Fossil"], emissions["NOx|Biosphere"]),
        (emissions["NH3|Fossil"], emissions["NH3|Biosphere"]),
        _total(emissions, "SOx"),
        p["rf_regions_no3"],
    )
    kwargs = {
        "harmonisation_target": p["rf_no3t_dir_wm2"],
        "harmonisation_reference_year": p["rf_no3t_dir_yr"],
        "natural_nox": p["rf_noxn_emis"],
        "natural_nh3": p["rf_nh3n_emis"],
        "natural_sox": p["sox_natural_emissions"],
        "lambda_so2": p["rf_no3_lambdaso2"],
        "burden_pi": p["rf_no3_preindburden"],
        "optical_thickness_pi": p["rf_no3_preindot"],
        "burden_to_optical_thickness_exponent": p["rf_no3_burden2ot_exp"],
    }
    return args, kwargs


def prescribed_forcing(
    years: NDArray[np.float64],
    emissions: Mapping[str, NDArray[np.float64]],
    parameters: Mapping[str, Any],
    inputs: Mapping[str, pr.AerosolHistory],
) -> dict[str, NDArray[np.float64]]:
    """Four-box ERF of every agent in `PRESCRIBED_AGENTS`, before efficacies.

    ``emissions`` are in MAGICC7's units, see `magicc7_units`.
    """
    p = parameters
    out = {}
    for species in ("SOx", "BC", "OC"):
        code = species.lower()
        fossil, biosphere = (
            f"{DIRECT}|{species}|Fossil",
            f"{DIRECT}|{species}|Biosphere",
        )
        out[fossil], out[biosphere] = pr.direct_aerosol(
            years,
            inputs[f"file_{code}i_rf"],
            inputs[f"file_{code}b_rf"],
            emissions[f"{species}|Fossil"],
            emissions[f"{species}|Biosphere"],
            harmonisation_target=p[f"rf_{code}i_dir_wm2"],
            harmonisation_reference_year=p[f"rf_{code}i_dir_yr"],
        )
    nitrate_args, nitrate_kwargs = _nitrate_inputs(years, emissions, p)
    out[f"{DIRECT}|NO3|Fossil"], out[f"{DIRECT}|NO3|Biosphere"] = pr.nitrate(
        *nitrate_args, **nitrate_kwargs
    )
    land_use = inputs["file_landuse_rf"]
    co2_biosphere = emissions["CO2|Biosphere"]
    out["Land-use Change"] = pr.land_use(
        years,
        land_use,
        co2_biosphere,
        harmonisation_target=p["rf_landuse_albedo_wm2"],
        harmonisation_reference_year=p["rf_landuse_albedo_yr"],
        constant_after=p["rf_landuse_constantafteryr"],
    )
    out[f"{DIRECT}|Mineral Dust"] = pr.mineral_dust(
        years,
        land_use,
        co2_biosphere,
        p["rf_regions_dust"],
        harmonisation_target=p["rf_mineraldust_dir_wm2"],
        harmonisation_reference_year=p["rf_mineraldust_dir_yr"],
        constant_after=p["rf_mineraldust_constantafteryr"],
    )
    out["Black Carbon on Snow"] = pr.bc_on_snow(
        years,
        inputs["file_bcsnow_rf"],
        _total(emissions, "BC"),
        harmonisation_target=p["rf_bcsnow_albedo_wm2"],
        harmonisation_reference_year=p["rf_bcsnow_albedo_yr"],
    )
    names = ("SOXI", "SOXNB", "OCI", "OCB", "OCN", "BCI", "BCB", "SS")
    out["Aerosols|Indirect Effect"] = pr.cloud_albedo(
        years,
        {name: inputs[f"file_{name.lower()}_ot"] for name in names},
        {
            "SOXI": emissions["SOx|Fossil"],
            "SOXNB": emissions["SOx|Biosphere"],
            "OCI": emissions["OC|Fossil"],
            "OCB": emissions["OC|Biosphere"],
            "BCI": emissions["BC|Fossil"],
            "BCB": emissions["BC|Biosphere"],
        },
        # The cloud code reads total nitrate, which differs from the sum of sources.
        pr._nitrate_total(*nitrate_args, **nitrate_kwargs)[0],
        emissions["NOx|Fossil"],
        {g: p[f"cloud_weight_{g}"] for g in ("no3", "bc", "oc", "sox", "ss")},
        p["rf_regions_cloud_albedo"],
        harmonisation_target=p["rf_cloud_albedo_aer_wm2"],
        harmonisation_reference_year=p["rf_cloud_albedo_aer_yr"],
        bci_soluble_ratio=p["cloud_bci2bcb_soluble_ratio"],
        natural_sox=p["sox_natural_emissions"],
    )
    solar = pr.on_axis(inputs["file_solar_rf"], years)[:, 0]
    out["Solar"] = np.repeat(pr.solar(solar, p["rf_solar_scale"])[:, None], 4, axis=1)
    return out


def climate_parameters(
    parameters: Mapping[str, Any],
    inputs: Mapping[str, pr.AerosolHistory],
    start_year: float,
    end_year: float | None = None,
) -> dict[str, Any]:
    """ClimateUDEB parameters, with regional and monthly volcanic forcing.

    With ``end_year``, months after the run ends are left out.
    """
    p = parameters
    volcanic = inputs["file_volcanic_rf"]
    values = np.asarray(volcanic.values)
    pattern = pr.volcanic_pattern(
        volcanic.years,
        values[:, 0],
        values[:, 1],
        start_year=start_year,
        last_year=p["rf_volc_forc_pattern_max_last_year"],
        scale=p["rf_volcanic_scale"],
    )
    base = {ours: p[theirs] for ours, theirs in CLIMATE_PARAMETERS.items()}
    initial = ClimateUDEBBuilder.from_parameters(
        cast("Any", {**base, "erf_2xco2": EFFICACY_ERF_2XCO2})
    ).internal_efficacy(pattern)
    method = EFFICACY_METHODS[p["rf_efficacy_apply"]]
    return {
        **base,
        "include_land_heat_capacity": bool(p["core_landheatcapacity_apply"]),
        "depth_dependent_area": float(p["core_ocn_depthdependent"]),
        "include_regional_forcing": True,
        "time_varying_erf_2xco2": True,
        "subannual_forcing": pr.volcanic_substeps(
            volcanic.years,
            values[:, 0],
            values[:, 1],
            scale=p["rf_volcanic_scale"],
            efficacy=p["rf_efficacy_volc"] if method == "ar6" else 1.0,
            internal_efficacy=initial if method != "none" else 1.0,
            regional_pattern=pattern if method != "none" else None,
            until=end_year,
        ),
    }


def _agent(
    parameters: Mapping[str, Any],
    name: str,
    key: str,
    *,
    regional: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    p = parameters
    return {
        "variable": child(ERF, name),
        "category": _category(name),
        "regional": regional,
        "efficacy": p.get(f"rf_efficacy_{key}", 1.0),
        "internal_efficacy": p[f"rf_intefficacy_{key}"],
        "regional_pattern": p.get(f"rf_regions_{key}", [1.0] * 4),
        **extra,
    }


def _category(name: str) -> str:
    if name in GREENHOUSE_GASES:
        return "greenhouse_gas"
    if name.startswith("Aerosols|"):
        return "aerosol"
    if name in NATURAL_AGENTS:
        return "natural"
    return "anthropogenic"


def forcing_agents(
    parameters: Mapping[str, Any],
    years: NDArray[np.float64],
    prescribed: Mapping[str, NDArray[np.float64]],
    emissions: Mapping[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    """ForcingAggregation agents of MAGICC7, apart from volcanic forcing.

    ``emissions`` are in MAGICC7's units, see `magicc7_units`.
    The patterns in `NORMYEAR_PATTERNS` come from `prescribed` forcing
    on `years` in ``rf_regions_normyear``, or the closest year.
    Nitrate's pattern comes from its total forcing in that year.
    """
    p = parameters
    i = int(np.abs(np.asarray(years) - p["rf_regions_normyear"]).argmin())
    patterns = {
        key: np.asarray(prescribed[name])[i].tolist()
        for key, name in NORMYEAR_PATTERNS.items()
    }
    args, kwargs = _nitrate_inputs(years, emissions, p)
    patterns["no3"] = pr.nitrate_pattern(*args, index=i, **kwargs)
    agents = [
        _agent(
            p,
            name,
            key,
            regional=name == "F-Gases",
            use_co2_internal_efficacy=name == "CO2",
        )
        for name, key in CALCULATED_AGENTS.items()
    ]
    for agent in agents:
        group = agent["variable"].removeprefix(f"{ERF}|")
        if group in HALOCARBON_GROUPS:
            agent["sum_of"] = halocarbon_forcing(group)
            agent["efficacy_per_variable"] = True
    agents.append(_agent(p, "Ozone|Tropospheric", "tropoz", regional=True))
    agents.extend(
        _agent(p, name, key, regional=True, **_pattern(patterns, key))
        for name, key in PRESCRIBED_AGENTS.items()
    )
    agents.append(
        _agent(
            p,
            BIOMASS,
            "aer_dir",
            regional=True,
            sum_of=[child(ERF, name) for name in BIOMASS_AGENTS],
        )
    )
    return agents


def _summation(terms: list[Any]) -> list[Any]:
    return [
        _summation(term) if isinstance(term, list) else child(ERF, term)
        for term in terms
    ]


def _pattern(patterns: Mapping[str, list[float]], key: str) -> dict[str, Any]:
    return {"regional_pattern": patterns[key]} if key in patterns else {}


def efficacy_climate(climate: Mapping[str, Any]) -> dict[str, Any]:
    """ForcingAggregation ``climate`` from ClimateUDEB parameters."""
    keys = (
        "ecs",
        "land_ocean_heat_exchange",
        "north_south_heat_exchange",
        "land_ocean_warming_ratio",
        "amplify_ocean_to_land",
        "co2_regional_pattern",
    )
    return {**{k: climate[k] for k in keys}, "erf_2xco2": EFFICACY_ERF_2XCO2}


def _ghg_method(parameters: Mapping[str, Any]) -> dict[str, Any]:
    method = parameters["core_co2ch4n2o_rfmethod"].strip().upper()
    if method == "OLBL":
        return {"method": method}
    # MAGICC7 applies no rapid adjustment with IPCCTAR
    return {
        "method": method,
        "erf_2xco2": parameters["core_delq2xco2"],
        **{f"adjust_{gas}": 1.0 for gas in ("co2", "ch4", "n2o")},
    }


def ozone_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """OzoneForcing parameters."""
    p = parameters
    return {
        "stratospheric_o3_scale": p["stratoz_o3scale"],
        "stratospheric_cl_exponent": p["stratoz_clexpon"],
        "stratospheric_reference_year": p["stratoz_threshold_year"],
        "tropospheric_ozone_radiative_efficiency": p["tropoz_radeff_wm2perdu"],
        "tropospheric_ozone_ch4_sensitivity": p["tropoz_ozch4"],
        "tropospheric_ozone_nox_sensitivity": p["tropoz_oznox"],
        "tropospheric_ozone_co_sensitivity": p["tropoz_ozco"],
        "tropospheric_ozone_nmvoc_sensitivity": p["tropoz_ozvoc"],
        "n2o_ozone_radiative_efficiency": p["ozdueton2o_radeff"],
        "temperature_scale": p["ozduetotemperature_scale"],
        "tropospheric_ozone_regions": {"regional_pattern": p["rf_regions_tropoz"]},
    }


def permafrost_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Permafrost parameters from the ``pf_*`` configuration."""
    p = parameters

    def soil(name: str, decomposition_ratio: float) -> dict[str, float]:
        return {
            "thaw_rate": p[f"pf_{name}_thawfreeze_percperk_rate"],
            "thaw_exponent": p[f"pf_{name}_thawfreeze_exp_temp"],
            "decomposition_ratio": decomposition_ratio,
            "q10_aerobic_alpha": p[f"pf_q10_{name}_aerob_alpha"],
            "q10_anaerobic_alpha": p[f"pf_q10_{name}_anaerob_alpha"],
            "initial_anaerobic_fraction": p[f"pf_{name}_anaerob_initial_areafraction"],
            "maximum_anaerobic_fraction": p[f"pf_{name}_anaerob_max_areafraction"],
            "anaerobic_moisture_sensitivity": p[f"pf_{name}_anaerob_moistsens"],
            "ch4_oxidation_fraction": p[f"pf_{name}_ch4oxidisation_fraction"],
        }

    return {
        "n_bands": p["pf_nbands"],
        "minimum_melting_temperature": p["pf_meltingtemp_min"],
        "maximum_melting_temperature": p["pf_meltingtemp_max"],
        "arctic_amplification": p["pf_arctic_amplification"],
        "soil_annual_cycle_amplitude": p["pf_tsoilannualcycle_ampl"],
        "aerobic_turnover_time": p["pf_ms_aerob_decomp_turnovertime"],
        "anaerobic_decomposition_ratio": p["pf_decomprate_anaerob_over_aerob_ratio"],
        "q10_temperature1": p["pf_q10_temp1"],
        "q10_temperature2": p["pf_q10_temp2"],
        "soil_water_slope": p["pf_soilwater_m"],
        "soil_water_offset": p["pf_soilwater_offset"],
        "minimum_soil_water": p["pf_soilwater_minw"],
        "total_pool": p["pf_tot_pool"],
        "mineral_soil_fraction_south": p["pf_minsoil_southern_poolfraction"],
        "mineral_soil_fraction_north": p["pf_minsoil_northern_poolfraction"],
        "zonal_pool_distribution": p["pf_zonal_pooldistr"],
        "mineral_soil": soil("ms", 1.0),
        "peat": soil("peat", p["pf_decomprate_peat_over_ms_ratio"]),
    }


def nitrogen_parameters(parameters: Mapping[str, Any]) -> dict[str, Any] | None:
    """Nitrogen limitation parameters from the ``ncycle_*`` configuration.

    Returns None when ``ncycle_apply`` is off.
    """
    p = parameters
    if not p["ncycle_apply"]:
        return None
    fixation = {
        "maximum_fixation_rate": p["ncycle_nfix_vmax"],
        "half_saturation_npp": p["ncycle_nfix_km"],
    }
    return {
        "uptake_nc_ratio": p["ncycle_ncratio_uptake"],
        "recycling_nc_ratio": p["ncycle_ncratio_recyc"],
        "recycling_cn_factor": p["ncycle_recyc2cnratio_factor"],
        "uptake_supply": p["ncycle_ncratio_uptake_supply"],
        "availability_factor_a": p["ncycle_npp2navail_factor_a"],
        "availability_factor_c": p["ncycle_npp2navail_factor_c"],
        "loss_turnover_time": p["ncycle_turnovertime_losses"],
        "prognostic_fixation": fixation if p["ncycle_apply_prognostic_nfix"] else None,
    }


def nitrogen_inputs(
    years: ArrayLike,
    emissions: Mapping[str, NDArray[np.float64]],
    parameters: Mapping[str, Any],
    inputs: Mapping[str, pr.AerosolHistory],
) -> dict[str, NDArray[np.float64]]:
    """Nitrogen deposition and fixation (Gt N/yr) driving the nitrogen limitation.

    ``emissions`` are in MAGICC7's units, see `magicc7_units`.
    A deposition file missing from ``inputs`` gives no deposition,
    as in MAGICC7 v7.5.3, which ships without its default deposition file.
    """
    p = parameters
    y = np.asarray(years, dtype=np.float64)

    def series(key: str) -> NDArray[np.float64]:
        if key not in inputs:
            return np.zeros(len(y))
        values = pr.on_axis(inputs[key], y)[:, 0]
        last = values[~np.isnan(values)][-1]
        return np.where(np.isnan(values), last, values)

    if p["ncycle_apply_ndep_from_emis"]:
        # MAGICC7 counts industrial NH3 twice and leaves out biosphere NH3
        nitrogen = (
            emissions["NOx|Fossil"]
            + emissions["NOx|Biosphere"]
            + 2.0 * emissions["NH3|Fossil"]
        )
        deposition = 1e-3 * p["ncycle_frac_ndep_land"] * nitrogen
    else:
        deposition = series("file_ndep_in")
    out = {NITROGEN_FLUX_DEPOSITION: deposition}
    if not p["ncycle_apply_prognostic_nfix"]:
        out[NITROGEN_FLUX_FIXATION] = series("file_nfix_in")
    return out


def magicc7_units(
    emissions: Mapping[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """Emissions as for `from_magicc7`, converted to the units MAGICC7 calculates in.

    NH3 becomes Mt N/yr, which the unit parser cannot convert to.
    """
    return {
        name: np.asarray(values, dtype=np.float64)
        * emissions_conversion_factor(
            child(EMISSIONS, name), Convention.RSCM, Convention.MAGICC7
        )
        for name, values in emissions.items()
    }


def _native(
    emissions: Mapping[str, NDArray[np.float64]], convention: Convention | str
) -> dict[str, NDArray[np.float64]]:
    if Convention(convention) == Convention.MAGICC7:
        return dict(emissions)
    return magicc7_units(emissions)


def _validate_emissions(
    emissions: Mapping[str, ArrayLike], n: int, halocarbons: Sequence[str]
) -> dict[str, NDArray[np.float64]]:
    required = [f"{s}|{sector}" for s in SECTORAL_EMISSIONS for sector in SECTORS]
    missing = [name for name in (*required, *halocarbons) if name not in emissions]
    if missing:
        msg = f"missing emissions: {missing}"
        raise ValueError(msg)
    unknown = set(emissions) - {*required, *halocarbons}
    if unknown:
        msg = f"unknown emissions: {sorted(unknown)}"
        raise ValueError(msg)
    out = {}
    for name, values in emissions.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (n,) or not np.all(np.isfinite(array)):
            msg = f"emissions {name!r} must be {n} finite values, one per year"
            raise ValueError(msg)
        out[name] = array
    return out


def natural_emissions(
    years: ArrayLike,
    emissions: Mapping[str, ArrayLike],
    parameters: Mapping[str, Any] | None = None,
    inputs: Mapping[str, pr.AerosolHistory] | None = None,
    convention: Convention | str = Convention.RSCM,
) -> dict[str, float]:
    """Natural CH4 and N2O emissions that close the prescribed concentrations' budget.

    The result is in MAGICC7's units, Mt CH4/yr and Mt N2ON/yr.

    As in MAGICC7, the budget covers the ``<gas>_budget_avgyears`` years
    up to ``<gas>_lastbudgetyear``, with mid-year concentrations and emissions
    and the initial lifetime.
    ``emissions`` is as for `from_magicc7`, on ``years``,
    in the units of ``convention``.
    Gases whose budget years are not all in ``years`` are left out.
    ``inputs`` defaults to `load_inputs` of ``parameters``.
    """
    p = parameters or default_parameters()
    files = inputs if inputs is not None else load_inputs(p)
    y = np.asarray(years, dtype=np.float64)
    native = _native(
        {
            f"{gas}|{sector}": np.asarray(emissions[f"{gas}|{sector}"])
            for gas in ("CH4", "N2O")
            for sector in SECTORS
        },
        convention,
    )
    non_oh_lifetime = _ch4_non_oh_lifetime(CH4_OTHER_LIFETIMES)
    oh_lifetime = 1.0 / (1.0 / p["ch4_tautot_init"] - 1.0 / non_oh_lifetime)
    budget = {
        "CH4": (
            (oh_lifetime, non_oh_lifetime),
            p["ch4_ppb2tgch4"] * p["ch4_mixboxsize"],
        ),
        "N2O": ((p["n2o_tauinit"],), p["n2o_ppb2tgn"]),
    }
    out = {}
    for gas, (lifetimes, tg_per_ppb) in budget.items():
        key = gas.lower()
        last = float(p[f"{key}_lastbudgetyear"])
        window = np.arange(last - p[f"{key}_budget_avgyears"], last + 1)
        index = np.searchsorted(y, window)
        if window[0] < y[0] or window[-1] > y[-1]:
            continue
        c = pr.on_axis(_concentration_input(files, gas), window)[:, 0].tolist()
        n = len(c) - 1
        # MAGICC7 sums each series in turn, which sets the rounding
        mean = _sequential_sum((c[k] + c[k + 1]) * 0.5 for k in range(n))
        removal = mean / lifetimes[0] + _sequential_sum(
            c[k + 1] - c[k] for k in range(n)
        )
        for lifetime in lifetimes[1:]:
            removal += mean / lifetime
        anthropogenic = 0.0
        for sector in SECTORS:
            e = native[f"{gas}|{sector}"][index]
            anthropogenic += _sequential_sum(e[1:].tolist())
            anthropogenic += _sequential_sum(e[:-1].tolist())
        out[gas] = removal * tg_per_ppb / n - anthropogenic * 0.5 / n
    return out


def _override[T](
    argument: str, defaults: dict[str, T], given: Mapping[str, T] | None
) -> dict[str, T]:
    unknown = sorted(set(given or {}) - set(defaults))
    if unknown:
        msg = f"unknown {argument}: {', '.join(unknown)}"
        raise ValueError(msg)
    return {**defaults, **(given or {})}


def _pre_run_fgas(
    config: ModelConfig, inputs: Inputs, time: NDArray[np.float64]
) -> list[float]:
    """F-gas forcing in the second year of a two-year run of ``config``."""
    model = emissions_driven.run_model(
        config,
        {name: (values[:2], unit) for name, (values, unit) in inputs.items()},
        TimeAxis.from_values(time[:2]),
    )
    return _second_year(model, ERF_F_GASES)


def _second_year(model: Model, name: str) -> list[float]:
    ts = model.timeseries()
    boxes = ts.get_four_box_timeseries_by_name(name)
    if boxes is not None:
        return [float(v) for v in boxes.values()[1]]
    scalar = ts.get_timeseries_by_name(name)
    if scalar is None:
        msg = f"missing {name}"
        raise KeyError(msg)
    return [float(scalar.values()[1])]


def from_magicc7(  # noqa: PLR0913
    time_axis: TimeAxis,
    emissions: Mapping[str, ArrayLike],
    *,
    parameters: Mapping[str, Any] | None = None,
    input_files: Mapping[str, pr.AerosolHistory] | None = None,
    initial_concentrations: Mapping[str, float] | None = None,
    convention: Convention | str = Convention.RSCM,
    concentration_driven: bool = False,
) -> tuple[ModelConfig, Inputs]:
    """Translate a MAGICC7 run into the configuration and inputs of a MAGICC model.

    ``time_axis`` has annual steps starting in the preindustrial year,
    1750 in MAGICC7's default configuration.
    ``emissions`` has one value per year for ``"<species>|Fossil"`` and
    ``"<species>|Biosphere"`` of each species in `SECTORAL_EMISSIONS`, in its unit,
    and for each name from `halocarbon_species`, in kt of the gas per year.
    ``convention`` set to MAGICC7 takes them in `MAGICC7_EMISSIONS_UNITS` instead.

    ``parameters`` overrides `default_parameters`.
    Concentrations follow MAGICC7's files until ``<gas>_switchfromconc2emis_year``,
    see `prescribed_concentrations`.
    ``concentration_driven`` prescribes them for the whole run,
    for [rscm.magicc.concentration_driven][].
    ``input_files`` replaces the files from `load_inputs`.
    Load them once and pass them to every translation of an ensemble.
    ``initial_concentrations`` overrides `default_initial_concentrations`.

    As in MAGICC7, the F-gas forcing at the first time bound is that of the second year,
    which a two-year run of the translated configuration calculates.
    """
    if len(time_axis) < 2 or np.any(np.diff(time_axis.bounds()) != 1.0):  # noqa: PLR2004
        msg = "time_axis must have at least two annual steps"
        raise ValueError(msg)
    time = np.asarray(time_axis.values(), dtype=np.float64)
    driver = (
        {f"{group}_switchfromconc2emis_year": np.inf for group in CONCENTRATION_UNITS}
        if concentration_driven
        else {}
    )
    p = ChainMap(driver, cast("dict[str, Any]", parameters or {}), default_parameters())
    files = dict(input_files) if input_files is not None else load_inputs(p)
    conc = _override(
        "initial_concentrations",
        default_initial_concentrations(),
        initial_concentrations,
    )
    species = halocarbon_species()
    data = _validate_emissions(emissions, len(time), species)
    native = _native(data, convention)
    if p["co2_plantboxresp_method"] != 2:  # noqa: PLR2004
        msg = "only co2_plantboxresp_method 2 is implemented"
        raise ValueError(msg)

    climate = climate_parameters(p, files, time[0], time[-1])
    if p["out_sealevel"]:
        climate["thermal_expansion"] = {}
    prescribed = prescribed_forcing(time, native, p, files)
    agents = forcing_agents(p, time, prescribed, native)
    permafrost = {"include_permafrost": bool(p["pf_apply"])}
    switch = concentration_switch_years(time, p)
    nitrogen = nitrogen_parameters(p)
    source = "ncycle" if nitrogen and p["ncycle_apply_cn_initial_values"] else "co2"
    feedback = bool(p["co2_tempfeedback_switch"])
    natural = natural_emissions(time, native, p, files, Convention.MAGICC7)
    components: dict[str, dict[str, Any]] = {
        "CH4Chemistry": {
            **permafrost,
            "concentration_switch_year": switch["ch4"],
            "initial_ch4_lifetime": p["ch4_tautot_init"],
            "tg_per_ppb": p["ch4_ppb2tgch4"],
            "mixing_box_size": p["ch4_mixboxsize"],
            **_natural(natural, "CH4"),
        },
        "CH4Oxidation": {
            **permafrost,
            "permafrost_fraction": p["pf_co2fromch4oxinatm_fraction"],
            "tg_per_ppb": p["ch4_ppb2tgch4"],
            "mixing_box_size": p["ch4_mixboxsize"],
            **_natural(natural, "CH4"),
        },
        "N2OChemistry": {
            "concentration_switch_year": switch["n2o"],
            "initial_n2o_lifetime": p["n2o_tauinit"],
            "tg_per_ppb": p["n2o_ppb2tgn"],
            **_natural(natural, "N2O"),
        },
        "HalocarbonChemistry": {
            "fgas_switch_year": switch["fgas"],
            "montreal_gases_switch_year": switch["mhalo"],
            "fgas_regions": {
                "regional_pattern": p["rf_regions_fgas"],
                "lower_lifetime": p["fgas_rf_regionscaling_lowertau"],
                "upper_lifetime": p["fgas_rf_regionscaling_uppertau"],
            },
        },
        "TerrestrialCarbon": {
            **{
                f"{pool}_pool_pi": p[f"{source}_{pool}pool_initial"]
                for pool in ("plant", "detritus", "soil")
            },
            **{
                f"{ours}_temperature_sensitivity": p[f"co2_feedbackfactor_{theirs}"]
                * feedback
                for ours, theirs in TERRESTRIAL_FEEDBACKS.items()
            },
            "npp_pi": p[f"{source}_npp_initial"],
            "fertilisation_method": p["co2_fertilization_method"],
            "plant_respiration_fertilisation_scale": p["co2_plantboxresp_fertscale"],
            "nitrogen": nitrogen,
        },
        "OceanCarbon": {**permafrost, "concentration_switch_year": switch["co2"]},
        "CO2Budget": {
            **permafrost,
            "concentration_switch_year": switch["co2"],
            "ocean_end_of_year": True,
        },
        "GHGForcing": {
            **{f"{gas.lower()}_pi": conc[gas] for gas in ("CO2", "CH4", "N2O")},
            **_ghg_method(p),
            "ch4_stratospheric_h2o_fraction": p["ch4_addedstrath2o_percent"],
        },
        "OzoneForcing": ozone_parameters(p),
        "ForcingAggregation": {
            "agents": agents,
            "efficacy_method": EFFICACY_METHODS[p["rf_efficacy_apply"]],
            "climate": efficacy_climate(climate),
            "summation": _summation(SUMMATION),
        },
        "ClimateUDEB": climate,
    }
    if permafrost["include_permafrost"]:
        components["Permafrost"] = permafrost_parameters(p)
    if p["out_sealevel"]:
        components["SeaLevelRise"] = {}
    terrestrial = components["TerrestrialCarbon"]
    initial = {
        **{child(ATMOSPHERIC_CONCENTRATIONS, gas): c for gas, c in conc.items()},
        **_ch4_burdens(conc["CH4"], p["ch4_ppb2tgch4"], p["ch4_mixboxsize"]),
        OCEAN_SURFACE_PARTIAL_PRESSURE_CO2: conc["CO2"],
        CUMULATIVE_CARBON_UPTAKE_OCEAN: 0.0,
        CARBON_POOL_PLANT: terrestrial["plant_pool_pi"],
        CARBON_POOL_DETRITUS: terrestrial["detritus_pool_pi"],
        CARBON_POOL_SOIL: terrestrial["soil_pool_pi"],
    }
    # In the units MAGICC's components declare, so reading them needs no conversion.
    inputs: Inputs = {
        **{
            child(EMISSIONS, gas): (_total(native, gas), MAGICC7_EMISSIONS_UNITS[gas])
            for gas in ("CH4", "N2O", "NOx", "CO", "NMVOC")
        },
        **{
            child(child(EMISSIONS, gas), sector): (
                native[f"{gas}|{sector}"],
                MAGICC7_EMISSIONS_UNITS[gas],
            )
            for gas in ("CO2", "CH4", "NOx", "CO", "NMVOC")
            for sector in SECTORS
        },
        **{
            child(EMISSIONS, s): (native[s], emissions_unit(child(EMISSIONS, s)))
            for s in species
        },
        **{
            prescribed_input(child(ATMOSPHERIC_CONCENTRATIONS, s)): values
            for s, values in prescribed_concentrations(time, p, files).items()
        },
    }
    if nitrogen:
        inputs |= {
            name: (values, "Gt N/yr")
            for name, values in nitrogen_inputs(time, native, p, files).items()
        }
    inputs |= {child(ERF, name): (v, "W/m^2") for name, v in prescribed.items()}

    config: ModelConfig = {"components": components, "initial_values": initial}
    config["initial_forcing"] = {ERF_F_GASES: _pre_run_fgas(config, inputs, time)}
    return config, inputs
