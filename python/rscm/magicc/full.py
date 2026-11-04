"""Runs of the full MAGICC7 chain from emissions.

`run_magicc7` couples every MAGICC7 process RSCM implements:
chemistry, the carbon cycle, permafrost, GHG and ozone forcing,
the forcing MAGICC7 prescribes from its input files, forcing aggregation
with efficacies and regional patterns, and ClimateUDEB with monthly volcanic forcing.

Parameters use MAGICC7's configuration names in lower case,
such as ``core_climatesensitivity``.
`default_parameters` gives MAGICC7 v7.5.3's defaults for every name the chain reads,
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

from rscm._lib.magicc import (
    CH4ChemistryBuilder,
    CH4OxidationBuilder,
    ClimateUDEBBuilder,
    CO2BudgetBuilder,
    ForcingAggregationBuilder,
    GhgForcingBuilder,
    HalocarbonChemistryBuilder,
    N2OChemistryBuilder,
    OceanCarbonBuilder,
    OzoneForcingBuilder,
    PermafrostBuilder,
    SeaLevelRiseBuilder,
    TerrestrialCarbonBuilder,
)
from rscm.core import (
    FourBoxTimeseries,
    GridType,
    InterpolationStrategy,
    ModelBuilder,
    RequirementType,
    TimeAxis,
    Timeseries,
    VariableSchema,
)
from rscm.magicc import prescribed as pr
from rscm.magicc.runner import CH4_OTHER_LIFETIMES, _ch4_burdens, _ch4_tau_other

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

    from rscm._lib.core import Component
    from rscm.core import Model

__all__ = [
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
    "halocarbon_species",
    "load_inputs",
    "natural_emissions",
    "nitrogen_inputs",
    "nitrogen_parameters",
    "ozone_parameters",
    "permafrost_parameters",
    "prescribed_concentrations",
    "prescribed_forcing",
    "run_components",
    "run_magicc7",
]

DATA = Path(__file__).parent / "data" / "magicc7"
ERF = "Effective Radiative Forcing"
REGIONAL_FORCING = f"{ERF}|Efficacy Adjusted"

#: Emissions given by sector (``"CO2|Fossil"`` and ``"CO2|AFOLU"``) and their units.
SECTORAL_EMISSIONS = {
    "CO2": "GtC/yr",
    "CH4": "Mt CH4/yr",
    "N2O": "Mt N/yr",
    "SOx": "Mt S/yr",
    "BC": "Mt BC/yr",
    "OC": "Mt OC/yr",
    "NOx": "Mt N/yr",
    "NH3": "Mt N/yr",
    "CO": "Mt CO/yr",
    "NMVOC": "Mt NMVOC/yr",
}
SECTORS = ("Fossil", "AFOLU")

#: Configuration keys of the MAGICC7 input files the chain reads.
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

#: Gas groups that MAGICC7 switches from prescribed concentrations to emissions,
#: and their concentration units.
CONCENTRATION_UNITS = {
    "co2": "ppm",
    "ch4": "ppb",
    "n2o": "ppb",
    "fgas": "ppt",
    "mhalo": "ppt",
}

#: Configuration keys MAGICC7 overwrites during a run,
#: so its parameter dump does not hold their input values.
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

#: Configuration keys of the nitrogen input files, read when they exist.
NITROGEN_INPUT_FILES = ("file_nfix_in", "file_ndep_in")

#: Agents calculated before the run, and the MAGICC7 efficacy key of each.
#: MAGICC7's total uses the direct aerosol efficacy for biomass aerosols and dust.
PRESCRIBED_AGENTS = {
    "SOx|Fossil": "sox",
    "BC|Fossil": "bc",
    "OC|Fossil": "oc",
    "NO3|Fossil": "no3",
    "Mineral Dust": "aer_dir",
    "Cloud Albedo": "cloud_albedo",
    "BC on Snow": "bcsnow",
    "Land Use": "landuse",
    "Solar": "solar",
}

#: Prescribed agents whose regional pattern MAGICC7 takes from their forcing
#: in ``rf_regions_normyear``.
NORMYEAR_PATTERNS = {
    "bcsnow": "BC on Snow",
    "landuse": "Land Use",
}

#: Halocarbon agents whose efficacy MAGICC7 applies to each species.
HALOCARBON_GROUPS = ("F-Gases", "Montreal Protocol Halogen Gases")

#: Prescribed agents MAGICC7 adds into one biomass burning aerosol agent.
BIOMASS_AGENTS = ("SOx|AFOLU", "NO3|AFOLU", "OC|AFOLU", "BC|AFOLU")
BIOMASS = "Biomass Burning Aerosols"

#: Order in which MAGICC7 adds the agents into its total, nesting its subtotals.
SUMMATION: list[Any] = [
    "CO2",
    "CH4",
    "N2O",
    "F-Gases",
    "Montreal Protocol Halogen Gases",
    ["BC|Fossil", "OC|Fossil", "SOx|Fossil", "NO3|Fossil", BIOMASS, "Mineral Dust"],
    "Cloud Albedo",
    [
        "Stratospheric Ozone",
        "Tropospheric Ozone",
        "Ozone due to N2O",
        "Ozone due to Temperature",
    ],
    "CH4 Oxidation Stratospheric H2O",
    "Land Use",
    "BC on Snow",
    "Solar",
]

#: CO2 doubling forcing behind MAGICC7's internal efficacies
#: until the run calculates one.
EFFICACY_Q2XCO2 = 3.71

#: Agents calculated during the run, and the MAGICC7 efficacy key of each.
CALCULATED_AGENTS = {
    "CO2": "co2",
    "CH4": "ch4",
    "N2O": "n2o",
    "F-Gases": "fgas",
    "Montreal Protocol Halogen Gases": "mhalo",
    "Stratospheric Ozone": "stratoz",
    "CH4 Oxidation Stratospheric H2O": "ch4oxstrath2o",
    "Ozone due to N2O": "ozdueton2o",
    "Ozone due to Temperature": "ozduetotemperature",
}

#: GhgForcing methods by MAGICC7's ``core_co2ch4n2o_rfmethod``.
GHG_METHODS = {"OLBL": "Olbl", "IPCCTAR": "Ipcctar"}

#: TerrestrialCarbon temperature sensitivities and their ``co2_feedbackfactor_*`` names.
TERRESTRIAL_FEEDBACKS = {
    "npp": "npp",
    "resp": "respiration",
    "detritus": "detritus",
    "soil": "soil",
}

#: ClimateUDEB parameters and their MAGICC7 names.
CLIMATE_PARAMETERS = {
    "ecs": "core_climatesensitivity",
    "rf_2xco2": "core_delq2xco2",
    "rlo": "core_rlo",
    "k_lo": "core_heatxchange_landocean",
    "k_ns": "core_heatxchange_northsouth",
    "amplify_ocean_to_land": "core_amplify_ocn2land_heatxchng",
    "kappa": "core_verticaldiffusivity",
    "kappa_min": "core_verticaldiffusivity_min",
    "kappa_dkdt": "core_verticaldiff_top_dkdt",
    "w_initial": "core_initial_upwelling_rate",
    "w_variable_fraction": "core_upwelling_variable_part",
    "feedback_q_sensitivity": "core_feedback_qsensitivity",
    "feedback_cumt_sensitivity": "core_feedback_cumtsensitivity",
    "feedback_cumt_period": "core_feedback_cumtperiod",
    "temp_adjust_alpha": "core_tempadjust_ocn2atm_alpha",
    "temp_adjust_gamma": "core_tempadjust_ocn2atm_gamma",
    "polar_sinking_ratio": "core_polarsinkwater_tempratio",
    "k_lg": "core_heatxchange_landground",
    "mixed_layer_depth": "core_mixedlayer_depth",
    "rf_regions_co2": "rf_regions_co2",
    "land_hc_eff_thickness": "core_landhc_effthickness",
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
    """Names of the halocarbons whose emissions the chain reads, in kt/yr."""
    component = HalocarbonChemistryBuilder.from_parameters({}).build()
    return [
        d.name.removeprefix("Emissions|")
        for d in component.definitions()
        if d.name.startswith("Emissions|")
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


def _total(
    emissions: Mapping[str, NDArray[np.float64]], species: str
) -> NDArray[np.float64]:
    return emissions[f"{species}|Fossil"] + emissions[f"{species}|AFOLU"]


def _nitrate_inputs(
    years: NDArray[np.float64],
    emissions: Mapping[str, NDArray[np.float64]],
    parameters: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Arguments of `pr.nitrate` and `pr.nitrate_pattern`."""
    p = parameters
    args = (
        years,
        (emissions["NOx|Fossil"], emissions["NOx|AFOLU"]),
        (emissions["NH3|Fossil"], emissions["NH3|AFOLU"]),
        _total(emissions, "SOx"),
        p["rf_regions_no3"],
    )
    kwargs = {
        "target": p["rf_no3t_dir_wm2"],
        "target_year": p["rf_no3t_dir_yr"],
        "natural_nox": p["rf_noxn_emis"],
        "natural_nh3": p["rf_nh3n_emis"],
        "natural_sox": p["sox_natural_emissions"],
        "lambda_so2": p["rf_no3_lambdaso2"],
        "preindustrial_burden": p["rf_no3_preindburden"],
        "preindustrial_ot": p["rf_no3_preindot"],
        "burden_to_ot_exponent": p["rf_no3_burden2ot_exp"],
    }
    return args, kwargs


def prescribed_forcing(
    years: NDArray[np.float64],
    emissions: Mapping[str, NDArray[np.float64]],
    parameters: Mapping[str, Any],
    inputs: Mapping[str, pr.AerosolHistory],
) -> dict[str, NDArray[np.float64]]:
    """Four-box ERF of every agent in `PRESCRIBED_AGENTS`, before efficacies."""
    p = parameters
    out = {}
    for species in ("SOx", "BC", "OC"):
        code = species.lower()
        out[f"{species}|Fossil"], out[f"{species}|AFOLU"] = pr.direct_aerosol(
            years,
            inputs[f"file_{code}i_rf"],
            inputs[f"file_{code}b_rf"],
            emissions[f"{species}|Fossil"],
            emissions[f"{species}|AFOLU"],
            target=p[f"rf_{code}i_dir_wm2"],
            target_year=p[f"rf_{code}i_dir_yr"],
        )
    nitrate_args, nitrate_kwargs = _nitrate_inputs(years, emissions, p)
    out["NO3|Fossil"], out["NO3|AFOLU"] = pr.nitrate(*nitrate_args, **nitrate_kwargs)
    land_use = inputs["file_landuse_rf"]
    co2_afolu = emissions["CO2|AFOLU"]
    out["Land Use"] = pr.land_use(
        years,
        land_use,
        co2_afolu,
        target=p["rf_landuse_albedo_wm2"],
        target_year=p["rf_landuse_albedo_yr"],
        constant_after=p["rf_landuse_constantafteryr"],
    )
    out["Mineral Dust"] = pr.mineral_dust(
        years,
        land_use,
        co2_afolu,
        p["rf_regions_dust"],
        target=p["rf_mineraldust_dir_wm2"],
        target_year=p["rf_mineraldust_dir_yr"],
        constant_after=p["rf_mineraldust_constantafteryr"],
    )
    out["BC on Snow"] = pr.bc_on_snow(
        years,
        inputs["file_bcsnow_rf"],
        _total(emissions, "BC"),
        target=p["rf_bcsnow_albedo_wm2"],
        target_year=p["rf_bcsnow_albedo_yr"],
    )
    names = ("SOXI", "SOXNB", "OCI", "OCB", "OCN", "BCI", "BCB", "SS")
    out["Cloud Albedo"] = pr.cloud_albedo(
        years,
        {name: inputs[f"file_{name.lower()}_ot"] for name in names},
        {
            "SOXI": emissions["SOx|Fossil"],
            "SOXNB": emissions["SOx|AFOLU"],
            "OCI": emissions["OC|Fossil"],
            "OCB": emissions["OC|AFOLU"],
            "BCI": emissions["BC|Fossil"],
            "BCB": emissions["BC|AFOLU"],
        },
        # The cloud code reads total nitrate, which differs from the sum of sources.
        pr._nitrate_total(*nitrate_args, **nitrate_kwargs)[0],
        emissions["NOx|Fossil"],
        {g: p[f"cloud_weight_{g}"] for g in ("no3", "bc", "oc", "sox", "ss")},
        p["rf_regions_cloud_albedo"],
        target=p["rf_cloud_albedo_aer_wm2"],
        target_year=p["rf_cloud_albedo_aer_yr"],
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
        cast("Any", {**base, "rf_2xco2": EFFICACY_Q2XCO2})
    ).internal_efficacy(pattern)
    apply = p["rf_efficacy_apply"]
    return {
        **base,
        "land_heat_capacity_enabled": bool(p["core_landheatcapacity_apply"]),
        "depth_dependent_area": float(p["core_ocn_depthdependent"]),
        "regional_forcing": True,
        "time_varying_q2x": True,
        "subannual_forcing": pr.volcanic_substeps(
            volcanic.years,
            values[:, 0],
            values[:, 1],
            scale=p["rf_volcanic_scale"],
            efficacy=p["rf_efficacy_volc"] if apply == 2 else 1.0,  # noqa: PLR2004
            internal_efficacy=initial if apply else 1.0,
            rf_regions=pattern if apply else None,
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
        "variable": f"{ERF}|{name}",
        "regional": regional,
        "efficacy": p.get(f"rf_efficacy_{key}", 1.0),
        "internal_efficacy": p[f"rf_intefficacy_{key}"],
        "rf_regions": p.get(f"rf_regions_{key}", [1.0] * 4),
        **extra,
    }


def forcing_agents(
    parameters: Mapping[str, Any],
    years: NDArray[np.float64],
    prescribed: Mapping[str, NDArray[np.float64]],
    emissions: Mapping[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    """ForcingAggregation agents of MAGICC7, apart from volcanic forcing.

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
        _agent(p, name, key, regional=name == "F-Gases", co2_efficacy=name == "CO2")
        for name, key in CALCULATED_AGENTS.items()
    ]
    for agent in agents:
        group = agent["variable"].removeprefix(f"{ERF}|")
        if group in HALOCARBON_GROUPS:
            agent["sum_of"] = halocarbon_forcing(group)
            agent["efficacy_per_variable"] = True
    agents.append(_agent(p, "Tropospheric Ozone", "tropoz", regional=True))
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
            sum_of=[f"{ERF}|{name}" for name in BIOMASS_AGENTS],
        )
    )
    return agents


def _summation(terms: list[Any]) -> list[Any]:
    return [
        _summation(term) if isinstance(term, list) else f"{ERF}|{term}"
        for term in terms
    ]


def _pattern(patterns: Mapping[str, list[float]], key: str) -> dict[str, Any]:
    return {"rf_regions": patterns[key]} if key in patterns else {}


def efficacy_climate(climate: Mapping[str, Any]) -> dict[str, Any]:
    """ForcingAggregation ``climate`` from ClimateUDEB parameters."""
    keys = ("ecs", "k_lo", "k_ns", "rlo", "amplify_ocean_to_land", "rf_regions_co2")
    return {**{k: climate[k] for k in keys}, "rf_2xco2": EFFICACY_Q2XCO2}


def _ghg_method(parameters: Mapping[str, Any]) -> dict[str, Any]:
    method = GHG_METHODS[parameters["core_co2ch4n2o_rfmethod"].strip().upper()]
    if method == "Olbl":
        return {"method": method}
    # MAGICC7 applies no rapid adjustment with IPCCTAR
    return {
        "method": method,
        "delq2xco2": parameters["core_delq2xco2"],
        **{f"adjust_{gas}": 1.0 for gas in ("co2", "ch4", "n2o")},
    }


def ozone_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """OzoneForcing parameters."""
    p = parameters
    return {
        "strat_o3_scale": p["stratoz_o3scale"],
        "strat_cl_exponent": p["stratoz_clexpon"],
        "strat_threshold_year": p["stratoz_threshold_year"],
        "trop_radeff": p["tropoz_radeff_wm2perdu"],
        "trop_oz_ch4": p["tropoz_ozch4"],
        "trop_oz_nox": p["tropoz_oznox"],
        "trop_oz_co": p["tropoz_ozco"],
        "trop_oz_voc": p["tropoz_ozvoc"],
        "n2o_radeff": p["ozdueton2o_radeff"],
        "temperature_scale": p["ozduetotemperature_scale"],
        "trop_regions": {"rf_regions": p["rf_regions_tropoz"]},
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
            "anaerobic_initial_fraction": p[f"pf_{name}_anaerob_initial_areafraction"],
            "anaerobic_max_fraction": p[f"pf_{name}_anaerob_max_areafraction"],
            "anaerobic_moisture_sensitivity": p[f"pf_{name}_anaerob_moistsens"],
            "ch4_oxidation_fraction": p[f"pf_{name}_ch4oxidisation_fraction"],
        }

    return {
        "n_bands": p["pf_nbands"],
        "melting_temp_min": p["pf_meltingtemp_min"],
        "melting_temp_max": p["pf_meltingtemp_max"],
        "arctic_amplification": p["pf_arctic_amplification"],
        "soil_annual_cycle_amplitude": p["pf_tsoilannualcycle_ampl"],
        "aerobic_turnover_time": p["pf_ms_aerob_decomp_turnovertime"],
        "anaerobic_decomposition_ratio": p["pf_decomprate_anaerob_over_aerob_ratio"],
        "q10_temp1": p["pf_q10_temp1"],
        "q10_temp2": p["pf_q10_temp2"],
        "soil_water_slope": p["pf_soilwater_m"],
        "soil_water_offset": p["pf_soilwater_offset"],
        "soil_water_min": p["pf_soilwater_minw"],
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
    fixation = {"vmax": p["ncycle_nfix_vmax"], "km": p["ncycle_nfix_km"]}
    return {
        "ncratio_uptake": p["ncycle_ncratio_uptake"],
        "ncratio_recycling": p["ncycle_ncratio_recyc"],
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
        # MAGICC7 counts industrial NH3 twice and leaves out AFOLU NH3
        nitrogen = (
            emissions["NOx|Fossil"]
            + emissions["NOx|AFOLU"]
            + 2.0 * emissions["NH3|Fossil"]
        )
        deposition = 1e-3 * p["ncycle_frac_ndep_land"] * nitrogen
    else:
        deposition = series("file_ndep_in")
    out = {"Nitrogen Deposition": deposition}
    if not p["ncycle_apply_prognostic_nfix"]:
        out["Nitrogen Fixation"] = series("file_nfix_in")
    return out


def _schema(
    components: Sequence[Component],
    exogenous: Mapping[str, tuple[NDArray[np.float64], str]],
    four_box: Mapping[str, tuple[NDArray[np.float64], str]],
) -> VariableSchema:
    variables = {name: (unit, GridType.Scalar) for name, (_, unit) in exogenous.items()}
    variables |= {
        name: (unit, GridType.FourBox) for name, (_, unit) in four_box.items()
    }
    for component in components:
        for d in component.definitions():
            _, grid = variables.get(d.name, (d.unit, d.grid_type))
            four = GridType.FourBox in (grid, d.grid_type)
            variables[d.name] = (d.unit, GridType.FourBox if four else GridType.Scalar)
    schema = VariableSchema()
    for name, (unit, grid) in variables.items():
        schema.add_variable(name, unit, grid)
    return schema


def run_components(  # noqa: PLR0913
    years: NDArray[np.float64],
    components: Sequence[Component],
    *,
    exogenous: Mapping[str, tuple[NDArray[np.float64], str]] = {},
    four_box: Mapping[str, tuple[NDArray[np.float64], str]] = {},
    initial: Mapping[str, float] = {},
    initial_grid: Mapping[str, Sequence[float]] = {},
) -> Model:
    """Run components on an annual axis with exogenous inputs and return the model.

    Scalar readers of a four-box variable see its area-weighted mean.
    """
    axis = TimeAxis.from_values(np.asarray(years, dtype=np.float64))
    builder = ModelBuilder().with_time_axis(axis)
    if any(
        d.grid_type == GridType.FourBox for c in components for d in c.definitions()
    ):
        builder.with_schema(_schema(components, exogenous, four_box))
        builder.with_grid_weights(GridType.FourBox, list(pr.area_fractions()))
    for component in components:
        builder.with_rust_component(component)
    for name, (values, unit) in exogenous.items():
        builder.with_exogenous_variable(
            name,
            Timeseries(
                np.asarray(values, dtype=np.float64),
                axis,
                unit,
                InterpolationStrategy.Linear,
            ),
        )
    for name, (values, unit) in four_box.items():
        builder.with_exogenous_four_box_variable(
            name,
            FourBoxTimeseries(
                np.asarray(values, dtype=np.float64),
                axis,
                unit,
                InterpolationStrategy.Linear,
            ),
        )
    builder.with_initial_values(dict(initial))
    builder.with_initial_grid_values({k: list(v) for k, v in initial_grid.items()})
    model = builder.build()
    model.run()
    return model


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
) -> dict[str, float]:
    """Natural CH4 and N2O emissions that close the prescribed concentrations' budget.

    As in MAGICC7, the budget covers the ``<gas>_budget_avgyears`` years
    up to ``<gas>_lastbudgetyear``, with mid-year concentrations and emissions
    and the initial lifetime.
    ``emissions`` is as for `run_magicc7`, on ``years``.
    Gases whose budget years are not all in ``years`` are left out.
    ``inputs`` defaults to `load_inputs` of ``parameters``.
    """
    p = parameters or default_parameters()
    files = inputs if inputs is not None else load_inputs(p)
    y = np.asarray(years, dtype=np.float64)
    tau_other = _ch4_tau_other(CH4_OTHER_LIFETIMES)
    tau_oh = 1.0 / (1.0 / p["ch4_tautot_init"] - 1.0 / tau_other)
    budget = {
        "CH4": ((tau_oh, tau_other), p["ch4_ppb2tgch4"] * p["ch4_mixboxsize"]),
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
        mean = sum((c[k] + c[k + 1]) * 0.5 for k in range(n))
        removal = mean / lifetimes[0] + sum(c[k + 1] - c[k] for k in range(n))
        for lifetime in lifetimes[1:]:
            removal += mean / lifetime
        anthropogenic = 0.0
        for sector in ("Fossil", "AFOLU"):
            e = np.asarray(emissions[f"{gas}|{sector}"], dtype=np.float64)[index]
            anthropogenic += sum(e[1:].tolist())
            anthropogenic += sum(e[:-1].tolist())
        out[gas] = removal * tg_per_ppb / n - anthropogenic * 0.5 / n
    return out


def run_magicc7(  # noqa: PLR0913
    years: ArrayLike,
    emissions: Mapping[str, ArrayLike],
    *,
    parameters: Mapping[str, Any] | None = None,
    inputs: Mapping[str, pr.AerosolHistory] | None = None,
    initial_concentrations: Mapping[str, float] | None = None,
    component_parameters: Mapping[str, Mapping[str, Any]] | None = None,
) -> Model:
    """Run the full MAGICC7 chain from emissions and return the run model.

    ``years`` are consecutive annual labels starting in the preindustrial year,
    1750 in MAGICC7's default configuration.
    ``emissions`` has one value per year for ``"<species>|Fossil"`` and
    ``"<species>|AFOLU"`` of each species in `SECTORAL_EMISSIONS`,
    and for each name from `halocarbon_species`.

    ``parameters`` overrides `default_parameters`, and ``pf_apply`` enables permafrost.
    Concentrations follow MAGICC7's files until ``<gas>_switchfromconc2emis_year``,
    see `prescribed_concentrations`.
    ``out_sealevel`` adds sea level rise with MAGICC7's default sea level parameters.
    ``inputs`` replaces the files from `load_inputs`.
    Load them once and pass them to every run of an ensemble that shares its files.
    ``initial_concentrations`` overrides `default_initial_concentrations`.
    ``component_parameters`` overrides the parameters of single components,
    keyed by component name such as ``"TerrestrialCarbon"``.
    """
    time = np.asarray(years, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) != 1.0):  # noqa: PLR2004
        msg = "years must be at least two consecutive annual labels"
        raise ValueError(msg)
    p = ChainMap(cast("dict[str, Any]", parameters or {}), default_parameters())
    files = dict(inputs) if inputs is not None else load_inputs(p)
    conc = {**default_initial_concentrations(), **(initial_concentrations or {})}
    overrides = component_parameters or {}
    species = halocarbon_species()
    data = _validate_emissions(emissions, len(time), species)

    def params(name: str, base: Mapping[str, Any]) -> Any:
        return {**base, **overrides.get(name, {})}

    sealevel = bool(p["out_sealevel"])
    climate_base = climate_parameters(p, files, time[0], time[-1])
    if sealevel:
        climate_base["thermal_expansion"] = {}
    climate = ClimateUDEBBuilder.from_parameters(params("ClimateUDEB", climate_base))
    prescribed = prescribed_forcing(time, data, p, files)
    agents = forcing_agents(p, time, prescribed, data)
    aggregation = ForcingAggregationBuilder.from_parameters(
        params(
            "ForcingAggregation",
            {
                "agents": agents,
                "efficacy_apply": p["rf_efficacy_apply"],
                "climate": efficacy_climate(climate_base),
                "summation": _summation(SUMMATION),
            },
        )
    )
    biomass = prescribed[BIOMASS_AGENTS[0]]
    for name in BIOMASS_AGENTS[1:]:
        biomass = biomass + prescribed[name]
    rf0 = {**prescribed, BIOMASS: biomass}
    initial_agents = [
        list(rf0.get(a["variable"].removeprefix(ERF + "|"), np.zeros((1, 4)))[0])
        if a["regional"]
        else [0.0]
        for a in agents
    ]
    permafrost = {"permafrost": bool(p["pf_apply"])}
    switch = concentration_switch_years(time, p)
    nitrogen = nitrogen_parameters(p)
    source = "ncycle" if nitrogen and p["ncycle_apply_cn_initial_values"] else "co2"
    if p["co2_plantboxresp_method"] != 2:  # noqa: PLR2004
        msg = "only co2_plantboxresp_method 2 is implemented"
        raise ValueError(msg)
    feedback = bool(p["co2_tempfeedback_switch"])
    terrestrial = params(
        "TerrestrialCarbon",
        {
            **{
                f"{pool}_pool_pi": p[f"{source}_{pool}pool_initial"]
                for pool in ("plant", "detritus", "soil")
            },
            **{
                f"{ours}_temp_sensitivity": p[f"co2_feedbackfactor_{theirs}"] * feedback
                for ours, theirs in TERRESTRIAL_FEEDBACKS.items()
            },
            "npp_pi": p[f"{source}_npp_initial"],
            "fertilization_method": p["co2_fertilization_method"],
            "plantbox_resp_fertscale": p["co2_plantboxresp_fertscale"],
            "nitrogen": nitrogen,
        },
    )
    natural = natural_emissions(time, data, p, files)
    components = [
        CH4ChemistryBuilder.from_parameters(
            params(
                "CH4Chemistry",
                {
                    **permafrost,
                    "concentration_switch_year": switch["ch4"],
                    "tau_total_init": p["ch4_tautot_init"],
                    "ppb_to_tg": p["ch4_ppb2tgch4"],
                    "mixbox_size": p["ch4_mixboxsize"],
                    **_natural(natural, "CH4"),
                },
            )
        ).build(),
        CH4OxidationBuilder.from_parameters(
            params(
                "CH4Oxidation",
                {
                    **permafrost,
                    "permafrost_fraction": p["pf_co2fromch4oxinatm_fraction"],
                    "ppb_to_tg": p["ch4_ppb2tgch4"],
                    "mixbox_size": p["ch4_mixboxsize"],
                    **_natural(natural, "CH4"),
                },
            )
        ).build(),
        N2OChemistryBuilder.from_parameters(
            params(
                "N2OChemistry",
                {
                    "concentration_switch_year": switch["n2o"],
                    "tau_n2o": p["n2o_tauinit"],
                    "ppb_to_tg": p["n2o_ppb2tgn"],
                    **_natural(natural, "N2O"),
                },
            )
        ).build(),
        HalocarbonChemistryBuilder.from_parameters(
            params(
                "HalocarbonChemistry",
                {
                    "fgas_switch_year": switch["fgas"],
                    "mhalo_switch_year": switch["mhalo"],
                    "fgas_regions": {
                        "rf_regions": p["rf_regions_fgas"],
                        "lower_tau": p["fgas_rf_regionscaling_lowertau"],
                        "upper_tau": p["fgas_rf_regionscaling_uppertau"],
                    },
                },
            )
        ).build(),
        TerrestrialCarbonBuilder.from_parameters(cast("Any", terrestrial)).build(),
        OceanCarbonBuilder.from_parameters(
            params(
                "OceanCarbon",
                {**permafrost, "concentration_switch_year": switch["co2"]},
            )
        ).build(),
        CO2BudgetBuilder.from_parameters(
            params(
                "CO2Budget",
                {
                    **permafrost,
                    "concentration_switch_year": switch["co2"],
                    "ocean_end_of_year": True,
                },
            )
        ).build(),
        GhgForcingBuilder.from_parameters(
            params(
                "GhgForcing",
                {
                    **{f"{gas.lower()}_pi": conc[gas] for gas in ("CO2", "CH4", "N2O")},
                    **_ghg_method(p),
                    "ch4_strat_h2o_fraction": p["ch4_addedstrath2o_percent"],
                },
            )
        ).build(),
        OzoneForcingBuilder.from_parameters(
            params("OzoneForcing", ozone_parameters(p))
        ).build(),
        aggregation.build(),
        climate.build(),
    ]
    sea_level = (
        [SeaLevelRiseBuilder.from_parameters(params("SeaLevelRise", {})).build()]
        if sealevel
        else []
    )
    components.extend(sea_level)
    if permafrost["permafrost"]:
        components.insert(
            0,
            PermafrostBuilder.from_parameters(
                params("Permafrost", permafrost_parameters(p))
            ).build(),
        )
    initial = {
        **{f"Atmospheric Concentration|{gas}": c for gas, c in conc.items()},
        **_ch4_burdens(conc["CH4"], p["ch4_ppb2tgch4"], p["ch4_mixboxsize"]),
        "Ocean Surface pCO2": conc["CO2"],
        "Cumulative Ocean Uptake": 0.0,
        **{
            f"Carbon Pool|{pool.title()}": terrestrial[f"{pool}_pool_pi"]
            for pool in ("plant", "detritus", "soil")
        },
        **{
            d.name: 0.0
            for c in sea_level
            for d in c.definitions()
            if d.requirement_type == RequirementType.Output
        },
    }
    exogenous = {
        "Emissions|CO2|Fossil": (data["CO2|Fossil"], "GtC/yr"),
        "Emissions|CO2|Land Use": (data["CO2|AFOLU"], "GtC/yr"),
        "Emissions|CH4|Fossil": (data["CH4|Fossil"], "Mt CH4/yr"),
        "Emissions|CH4|AFOLU": (data["CH4|AFOLU"], "Mt CH4/yr"),
        **{
            f"Emissions|{gas}": (_total(data, gas), SECTORAL_EMISSIONS[gas])
            for gas in ("CH4", "N2O", "NOx", "CO", "NMVOC")
        },
        **{
            f"Emissions|{gas}|{sector}": (
                data[f"{gas}|{sector}"],
                SECTORAL_EMISSIONS[gas],
            )
            for gas in ("NOx", "CO", "NMVOC")
            for sector in SECTORS
        },
        **{f"Emissions|{s}": (data[s], "kt/yr") for s in species},
        **{
            f"Atmospheric Concentration|{s}|Prescribed": values
            for s, values in prescribed_concentrations(time, p, files).items()
        },
    }
    if nitrogen:
        exogenous |= {
            name: (values, "Gt N/yr")
            for name, values in nitrogen_inputs(time, data, p, files).items()
        }
    four_box = {f"{ERF}|{name}": (v, "W/m^2") for name, v in prescribed.items()}

    def run(n: int, agents: list[list[float]]) -> Model:
        total, boxes = aggregation.aggregate(agents)
        return run_components(
            time[:n],
            components,
            exogenous={k: (v[:n], unit) for k, (v, unit) in exogenous.items()},
            four_box={k: (v[:n], unit) for k, (v, unit) in four_box.items()},
            initial={**initial, **climate.initial_values(total)},
            initial_grid={REGIONAL_FORCING: boxes},
        )

    # MAGICC7 starts from its second year's F-gas forcing.
    fgas = [a["variable"] for a in agents].index(f"{ERF}|F-Gases")
    initial_agents[fgas] = _second_year_fgas(run(2, initial_agents))
    return run(len(time), initial_agents)


def _second_year_fgas(model: Model) -> list[float]:
    ts = model.timeseries()
    name = f"{ERF}|F-Gases"
    boxes = ts.get_fourbox_timeseries_by_name(name)
    if boxes is not None:
        return [float(v) for v in boxes.values()[1]]
    scalar = ts.get_timeseries_by_name(name)
    if scalar is None:
        msg = f"missing {name}"
        raise KeyError(msg)
    return [float(scalar.values()[1])]
