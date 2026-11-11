"""
Checks on a MAGICC7 configuration and its inputs before `from_magicc7` builds a model.

Every MAGICC7 configuration key is one of:

- read by `from_magicc7`, the keys of `default_parameters`,
- ignorable, in `IGNORED_PARAMETERS`: it has no effect on any output RSCM produces,
- unsupported, in `UNSUPPORTED_PARAMETERS`: RSCM lacks what it switches on,
  so a value that would change the run is rejected.

Any other key is rejected as unknown, which catches misspellings.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rscm.validation import IssueKind, Validator

__all__ = [
    "IGNORED_PARAMETERS",
    "UNSUPPORTED_PARAMETERS",
    "Context",
    "Unsupported",
    "check_parameters",
    "same_value",
]


@dataclass(frozen=True)
class Context:
    """What a rule sees: the merged configuration and the run's first and last years."""

    parameters: Mapping[str, Any]
    start: float
    end: float


@dataclass(frozen=True)
class Unsupported:
    """A MAGICC7 key whose effect RSCM lacks."""

    gap: str
    """What RSCM does not implement."""
    rejects: str
    """Which values are rejected."""
    check: Callable[[Any, Any, Context], bool]
    """Whether ``(value, default, context)`` is rejected."""


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().upper()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Sequence | np.ndarray):
        items = [_normalise(v) for v in value]
        # MAGICC7 pads species tables with zeros or blanks to their maximum length.
        while items and items[-1] in (0, ""):
            items.pop()
        return items
    return value


def same_value(value: Any, default: Any) -> bool:
    """Whether a configuration value equals a default, as MAGICC7 reads them.

    Strings compare without case or surrounding blanks,
    and lists without trailing padding.
    """
    a, b = _normalise(value), _normalise(default)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            same_value(x, y) for x, y in zip(a, b, strict=True)
        )
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-12, atol=0.0))
    except (TypeError, ValueError):
        return bool(a == b)


def _not_default(value: Any, default: Any, context: Context) -> bool:
    return not same_value(value, default)


def differs(gap: str) -> Unsupported:
    """Reject any value other than the default."""
    return Unsupported(gap, "a value other than the default", _not_default)


def _sea_level(value: Any, default: Any, context: Context) -> bool:
    return bool(context.parameters.get("out_sealevel")) and _not_default(
        value, default, context
    )


def sea_level(gap: str) -> Unsupported:
    """Reject a value other than the default when sea level is calculated."""
    return Unsupported(
        gap, "a value other than the default with out_sealevel on", _sea_level
    )


def _before_end(value: Any, default: Any, context: Context) -> bool:
    return float(value) < context.end


def held_constant(agent: str) -> Unsupported:
    """Reject holding forcing constant from a year before the run ends."""
    return Unsupported(
        f"holding {agent} forcing constant", "a year before the run ends", _before_end
    )


def allowed(gap: str, *values: Any) -> Unsupported:
    """Reject any value but ``values``."""

    def check(value: Any, default: Any, context: Context) -> bool:
        return not any(same_value(value, v) for v in values)

    shown = ", ".join(repr(v) for v in values)
    return Unsupported(gap, f"a value other than {shown}", check)


def with_switch(gap: str, switch: str, condition: Callable[[Any], bool]) -> Unsupported:
    """Reject a value other than the default when ``switch`` meets ``condition``."""

    def check(value: Any, default: Any, context: Context) -> bool:
        return condition(context.parameters.get(switch)) and _not_default(
            value, default, context
        )

    return Unsupported(gap, f"a value other than the default with this {switch}", check)


def matches(other: str) -> Unsupported:
    """Reject a value that differs from the key ``other``, which RSCM shares with it."""

    def check(value: Any, default: Any, context: Context) -> bool:
        return not same_value(value, context.parameters.get(other, default))

    return Unsupported(
        f"separate F-gas and Montreal gas values, RSCM reads {other} for both",
        f"a value other than {other}",
        check,
    )


def scaled_by(scale: str) -> Unsupported:
    """Reject a table other than the default, or the default times ``scale``."""

    def check(value: Any, default: Any, context: Context) -> bool:
        factor = float(context.parameters.get(scale, 1.0))
        scaled = [v * factor for v in default]
        return not (same_value(value, default) or same_value(value, scaled))

    return Unsupported(
        "halocarbon species other than MAGICC7's",
        f"a value other than the default, or the default times {scale}",
        check,
    )


def _run_years(context: Context) -> dict[str, float]:
    return {"startyear": context.start, "endyear": context.end, "stepsperyear": 12.0}


def run_year(key: str) -> Unsupported:
    """Reject run years that disagree with the time axis."""

    def check(value: Any, default: Any, context: Context) -> bool:
        return not same_value(value, _run_years(context)[key])

    return Unsupported(
        "setting the run years apart from the time axis",
        "a value other than the time axis gives",
        check,
    )


IGNORED = {
    "selects MAGICC7 output files": (
        *(
            f"out_{name}"
            for name in (
                "allowanydynamicvars aogcmtuning aogcmtuning_vars ascii_binary"
                " binary_format carboncycle ccycletuning ccycletuning_vars"
                " climsenseff_baseperiod climsenseff_futureperiod co2eq_validrange"
                " compact concentrations dynamic_vars emissions forcing"
                " forcing_subannual gwpemissions heatuptake internal_interp_length"
                " inverseemis keydata_1 keydata_1_vars keydata_2 keydata_2_vars"
                " lifetimes misc observationaltuning observationaltuning_vars oceanarea"
                " parameters permafrost precipinput rcpdata sum_gwpemissions summaryidx"
                " surfaceforcing temperature temperature_subannual tempoceanlayers"
                " timeseriesmix warnings zero_heat_period zero_slr_period"
                " zero_temp_period"
            ).split()
        ),
        "output_oceanvolume",
        "core_heat_depths",
        "core_heat_trendperiod",
    ),
    "run metadata": (
        "bulkrunmodus",
        "compact_fname",
        "compact_params",
        "compact_vars",
        "magiccversion",
        "pathname_infiles",
        "pathname_outfiles",
        "run_id",
        "run_weight",
        "rundate",
        "runname",
    ),
    "reads emissions files, which from_magicc7 takes as arrays": (
        *(
            f"file_{species}{sector}_emis"
            for species in (
                "bc",
                "ch4",
                "co2",
                "co",
                "n2o",
                "nh3",
                "nmvoc",
                "nox",
                "oc",
                "sox",
            )
            for sector in ("b", "i")
        ),
        "file_emisscen",
        *(f"file_emisscen_{n}" for n in range(2, 9)),
        "fgas_adjstfutremis2past_0no1scale",
        "fgas_files_emis",
        "fgas_scen_sequence",
        "mhalo_adjstfutremis2past_0no1scale",
        "mhalo_files_emis",
        "gen_rcpplusregions2nh",
        "gen_rcpplussplitregions2nh",
        "gen_rcpregions2nh",
        "gen_scen_forceglobal",
        "gen_sresregions2nh",
        "ghgaer_names",
        "ghgaer_scen_scenadjust2hist_0no1scale",
        "ghgaer_scen_sequence",
        "scen_co2bhistadjust_0no1scale2shift",
        "scen_histadjust_0no1scale2shift",
    ),
    "scales surface forcing, which RSCM does not output": tuple(
        f"srf_factor_{agent}"
        for agent in (
            "airh2o bc bcsnow ch4 ch4oxstrath2o cirrus cloud_albedo cloud_cover co2"
            " contrail dust fgas landuse mhalo n2o no3 oc qxtra solar sox stratoz"
            " tropoz volc"
        ).split()
    ),
    "sets precipitation, which RSCM does not output": (
        *(
            f"precip_efficacy_{agent}"
            for agent in ("ch4", "co2", "fgassum", "mhalosum", "n2o", "tropoz")
        ),
        "precip_regemis_bc",
        "precip_regemis_sox",
    ),
    "sets semi-empirical sea level, which RSCM does not output": (
        "slr_semiempi_rate_sens",
        "slr_semiempi_switchfromobs2calc",
        "slr_semiempi_tempbaseperiod",
        "slr_semiempi_zeroratetemp",
    ),
    "sets an emissions metric or label only": (
        "ch4_gwp",
        "co2_gwp",
        "n2o_gwp",
        "fgas_formula",
        "fgas_metricvals",
        "mhalo_formula",
        "mhalo_metricvals",
    ),
    "is declared but never used by MAGICC7": (
        "bcoc_switchfromrf2emis_year",
        "core_use_default_delq2xco2",
        "file_nloss_in",
        "file_nrecyc_in",
        "file_soxt_rf",
        "ratiofactor_bcoc_co_ratio",
        "ratiofactor_bcoc_co_year",
        "rf_regions_seasalt",
        "rfsub_regions_cloud_albedo",
        "rfsub_regions_cloud_cover",
        "rfsub_regions_dust",
        "rfsub_regions_tropoz",
        "slr_ais_sid_region",
        "slr_landwater_start",
    ),
    "is overwritten during the run": (
        *(
            f"rf_intefficacy_{agent}"
            for agent in (
                "airh2o",
                "cirrus",
                "cloud_cover",
                "contrail",
                "dust",
                "qxtra",
                "volc",
            )
        ),
        *(f"rf_{s}b_dir_{v}" for s in ("bc", "oc", "sox") for v in ("factor", "wm2")),
        "file_no3t_rf",
    ),
    "is overwritten from a file that RSCM requires to be the default": (
        *(
            f"slr_ais_sid_{name}_{region}"
            for name in ("dt", "tempscaling")
            for region in ("amundsen", "eastantarctica", "ross", "weddell")
        ),
        *(
            f"slr_ais_sid_irf_{r}"
            for r in ("amundsen", "eastantarctica", "ross", "weddell")
        ),
        "slr_ais_smb_coef1",
        "slr_ais_smb_coef2",
        "slr_ais_smb_sens_exponent",
        "slr_gis_smb_coef1",
        "slr_gis_smb_coef2",
        "slr_gis_smb_sens_exponent",
        "slr_gl_equislr",
        "slr_gl_equitemp",
        "slr_gl_sens_mmpyrdeg",
        "slr_gl_temp_exponent",
        "ocn_areas_raw_depth_prescr",
        "ocn_areas_raw_relative_prescr_nh",
        "ocn_areas_raw_relative_prescr_sh",
        "ocn_nraw_prescr",
        "temp_initial_ocean_cmip5_profile_nh",
        "temp_initial_ocean_cmip5_profile_sh",
    ),
    "is only read with a switch that RSCM requires off": (
        *(
            f"air_{name}"
            for name in ("bc_scale", "ch4_noxair_scale", "ch4_refyr", "sox_scale")
        ),
        "air_tropoz_noxair_scale",
        "air_tropoz_refyr",
        "file_sector_emis",
        *(f"rf_air_{n}" for n in ("cirrus_applysaturation", "cirrus_scaleyr2max")),
        *(
            f"rf_air_{a}_{v}"
            for a in ("cirrus", "contrail", "h2o")
            for v in ("wm2", "yr")
        ),
        *(
            f"rf_{k}_{a}"
            for k in ("efficacy", "regions")
            for a in ("airh2o", "cirrus", "contrail")
        ),
        "file_extra_rf",
        "rf_efficacy_qxtra",
        "rf_extra_factor",
        "rf_extra_offset",
        "rf_regions_qxtra",
        "ch4_clathratefeed_alpha",
        "ch4_clathratefeed_inipool",
        "ch4_clathratefeed_yrstart",
        "core_amv_scale",
        "core_amv_shfraction",
        "file_amv_index",
        "core_elnino_relaxfactor",
        "core_elnino_scale",
        "file_elnino_n34",
        "co2_capconc_ppm",
        "co2_preindco2conc",
        "co2_zeroemis_after_pgc",
        "core_switchfrompresc2temp_year",
        "file_prescr_surfacetemp",
        "rf_bcb_dir_yr",
        "rf_ocb_dir_yr",
        "rf_soxb_dir_yr",
        "cloud_limit_max",
        "file_slr_historic",
        "file_upwelling_rate",
        "n2o_scaleemis",
        *(
            f"slr_ais_{name}"
            for name in (
                "discharge_startyear sid_dschrg_sens sid_fastrate sid_tempsens_exponent"
                " sid_thresholdtemp sid_totalvol sid_zerotemp"
            ).split()
        ),
        *(f"slr_gis_smb_coef_fw{n}" for n in (1, 2, 3)),
        *(
            f"slr_landwater_{name}"
            for name in (
                "case maxvolume_mm startyear switchyear volume_exponent"
            ).split()
        ),
        "file_slr_landwater_high",
        "file_slr_landwater_low",
        "ch4_fossfuelfraction",
    ),
    "only matters with nonzero cloud cover forcing, which RSCM rejects": (
        "rf_cloud_cover_aer_apply",
        "rf_cloud_cover_aer_yr",
        "rf_efficacy_cloud_cover",
        "rf_regions_cloud_cover",
    ),
    "is only read with the IPCCTAR greenhouse gas forcing method": (
        "ch4_radeff_wm2perppb",
        "n2o_radeff_wm2perppb",
    ),
    "has no effect on the outputs RSCM produces": (
        "cloud_apply_limit_max",
        "co2_feedtemp_landonly_apply",
        "core_dependence_rlo_on_dt2x",
        "core_maximal_temperature",
        "core_upwelling_scaling_method",
        "n2o_apply_scaleemis",
        "oceancc_convltn2int_yrsago",
        "rf_solar_future_meanlastxyears",
    ),
}
"""Ignorable MAGICC7 keys, by why they are ignorable."""

IGNORED_PARAMETERS = {key: reason for reason, keys in IGNORED.items() for key in keys}
"""Ignorable MAGICC7 keys and why each has no effect on RSCM's outputs."""

_SPECIES_TABLES = (
    "br_atoms c_atoms cl_atoms eff_mixboxsize f_atoms fract_releasefactor"
    " h_atoms n_atoms names o_atoms radiat_eff s_atoms"
    " tau_oh tau_strat tau_tot"  # naming: allow
).split()

_SEA_LEVEL_FILES = (
    *(
        f"file_slr_ais_sid_irf_{r}"
        for r in ("amundsen", "eastantarctica", "ross", "weddell")
    ),
    *(f"file_slr_{n}" for n in ("aissid_params", "aissmb_params", "gissmb_params")),
    "file_slr_gl_params",
    "file_slr_gl_xtraparams",
)

_SEA_LEVEL_PARAMETERS = (
    *(f"slr_ais_sid_{n}" for n in ("basalmelt", "irf_yrspan", "scaling", "startyear")),
    "slr_ais_smb_startyear",
    *(
        f"slr_gis_sid_{n}"
        for n in (
            "case dschrg_sens_high dschrg_sens_low scaling startyear"
            " tempsens_exponent_high tempsens_exponent_low totalvol_high totalvol_low"
        ).split()
    ),
    *(
        f"slr_gis_smb_{n}"
        for n in ("initial_volume_mm", "startyear", "volume_exponent")
    ),
    *(f"slr_gl_{n}" for n in ("normpara_temp", "normpara_vol", "startyear")),
)


def _nonzero(value: Any) -> bool:
    return value is not None and float(value) != 0.0


UNSUPPORTED_PARAMETERS: dict[str, Unsupported] = {
    "startyear": run_year("startyear"),
    "endyear": run_year("endyear"),
    "stepsperyear": run_year("stepsperyear"),
    "ch4_applyfossfuelfraction": differs("a constant fossil fraction of CH4"),
    "ch4_clathratefeed_apply": differs("the methane clathrate feedback"),
    "ch4_incl_ch4ox": differs("leaving CO2 from fossil CH4 oxidation out"),
    "co2_capconc_apply": differs("capping CO2 concentrations"),
    "co2_preindco2conc_apply": differs("a prescribed preindustrial CO2 concentration"),
    "co2_zeroemis_afterxpgc_apply": differs(
        "zero CO2 emissions after a cumulative total"
    ),
    "core_amv_apply": differs("Atlantic multidecadal variability"),
    "core_elnino_apply": differs("El Nino variability"),
    "core_hemisfraction_nh_land": differs(
        "hemispheric land fractions other than MAGICC7's"
    ),
    "core_hemisfraction_sh_land": differs(
        "hemispheric land fractions other than MAGICC7's"
    ),
    "core_ocn_nlevels": differs("an ocean other than 50 layers"),
    "core_prescrtemp_apply": differs("prescribed temperatures"),
    "core_switch_ocn_areafactor": differs("ocean area profiles other than CMIP5's"),
    "core_switch_ocn_tempprofile": differs(
        "initial ocean temperature profiles other than CMIP5's"
    ),
    "core_switch_tempadjust_ocn2atm": differs(
        "switching the ocean to atmosphere temperature adjustment off"
    ),
    **{
        f"{group}_{t}": differs("halocarbon species other than MAGICC7's")
        for group in ("fgas", "mhalo")
        for t in _SPECIES_TABLES
    },
    # MAGICC7 multiplies the efficiencies by the sum scale and writes them back.
    "fgas_radiat_eff": scaled_by("rf_fgassum_scale"),
    "mhalo_radiat_eff": scaled_by("rf_mhalosum_scale"),
    "mhalo_n": differs("halocarbon species other than MAGICC7's"),
    "fgas_use_tauoh_var": matches("mhalo_use_tauoh_var"),
    "fgas_use_taustrat_var": matches("mhalo_use_taustrat_var"),
    "fgas_taustrat_sens2meridflux": matches("mhalo_taustrat_sens2meridflux"),
    "file_ch4n_emis": differs("natural CH4 emissions files"),
    "file_n2on_emis": differs("natural N2O emissions files"),
    "file_soxn_emis": differs("natural SOx emissions other than the default file"),
    "file_mineraldust_rf": differs("mineral dust forcing apart from the land use file"),
    "file_ocn_areafactor": differs("ocean area profiles other than CMIP5's"),
    "file_ocn_iniprofile": differs(
        "initial ocean temperature profiles other than CMIP5's"
    ),
    **{
        key: sea_level("sea level tuning files other than the defaults")
        for key in _SEA_LEVEL_FILES
    },
    **{
        key: sea_level("sea level parameters other than the defaults")
        for key in _SEA_LEVEL_PARAMETERS
    },
    "slr_ais_sid_parameterisation": sea_level(
        "Antarctic discharge other than LEVERMANN"
    ),
    "slr_gis_smb_parameterisation": sea_level(
        "Greenland surface mass balance other than DEFAULT"
    ),
    "slr_landwater_switch": sea_level("the land water contribution to sea level"),
    "slr_switch_calc2hist": Unsupported(
        "historical sea level from a file",
        "a year after the run starts with out_sealevel on",
        lambda value, default, context: (
            bool(context.parameters.get("out_sealevel"))
            and float(value) > context.start
        ),
    ),
    **{
        f"file_tuningmodel_{n}": allowed(
            "MAGICC7 tuning model files", "", "USER", "PYMAGICC"
        )
        for n in range(1, 11)
    },
    "gen_n_grammpromol": differs("a nitrogen molar mass other than MAGICC7's"),
    "gen_s_grammpromol": differs("a sulphur molar mass other than MAGICC7's"),
    "mhalo_co2ipattern_apply": differs(
        "the emissions-based Montreal gas forcing pattern"
    ),
    "mhalo_rf_boxscale_lowertau": with_switch(
        "the emissions-based Montreal gas forcing pattern",
        "mhalo_co2ipattern_apply",
        _nonzero,
    ),
    "mhalo_rf_boxscale_uppertau": with_switch(
        "the emissions-based Montreal gas forcing pattern",
        "mhalo_co2ipattern_apply",
        _nonzero,
    ),
    "oceancc_rad_setting": differs("ocean carbon radiative settings other than 0"),
    **{
        f"rf_{agent}_constantafteryr": held_constant(name)
        for agent, name in (
            ("aer", "aerosol"),
            ("ch4", "CH4"),
            ("co2", "CO2"),
            ("fgas", "F-gas"),
            ("mhalo", "Montreal gas"),
            ("n2o", "N2O"),
            ("stratoz", "stratospheric ozone"),
            ("total", "total"),
            ("tropoz", "tropospheric ozone"),
        )
    },
    **{
        f"rf_{s}b_dir_apply": differs("biomass aerosol scaling apart from industrial")
        for s in ("bc", "oc", "sox")
    },
    "rf_cloud_cover_aer_wm2": with_switch(
        "aerosol cloud cover forcing", "rf_cloud_cover_aer_apply", lambda v: v == 1
    ),
    "rf_cloud_cover_aer_factor": with_switch(
        "aerosol cloud cover forcing", "rf_cloud_cover_aer_apply", lambda v: v in (2, 3)
    ),
    "rf_extra_read": differs("extra forcing files"),
    "rf_initialization_method": differs("forcing initialisation other than JUMPSTART"),
    "rf_preind_referenceyr": differs(
        "a preindustrial forcing reference year other than 1750"
    ),
    "rf_total_runmodus": differs("forcing run modes other than ALL"),
    "rf_volc_futr_meanlastxyears": differs(
        "future volcanic forcing from a recent mean"
    ),
    "rf_volcanic_historic_meanzero": differs("zero-mean historical volcanic forcing"),
    "sector_include": with_switch(
        "sectoral emissions and aviation forcing",
        "file_sector_emis",
        lambda v: bool(str(v or "").strip()),
    ),
}
"""Unsupported MAGICC7 keys, and the gap and values each rejects."""


def check_parameters(
    validator: Validator,
    given: Mapping[str, Any],
    read: Mapping[str, Any],
    unsupported_defaults: Mapping[str, Any],
    run: tuple[float, float],
) -> None:
    """Check configuration keys: unknown ones, and unsupported values.

    ``given`` holds the caller's parameters, ``read`` the defaults of the keys
    `from_magicc7` reads and ``unsupported_defaults`` MAGICC7's defaults
    of `UNSUPPORTED_PARAMETERS`.
    ``run`` holds the first and last run years.
    """
    known = {*read, *IGNORED_PARAMETERS, *UNSUPPORTED_PARAMETERS}
    merged = {**unsupported_defaults, **read, **given}
    context = Context(merged, *run)
    for key in sorted(given):
        if key in UNSUPPORTED_PARAMETERS:
            rule = UNSUPPORTED_PARAMETERS[key]
            if rule.check(given[key], unsupported_defaults.get(key), context):
                validator.add(
                    IssueKind.UNSUPPORTED,
                    key,
                    f"{given[key]!r} needs {rule.gap}, which RSCM does not implement"
                    f" (rejects {rule.rejects})",
                )
        elif key not in known:
            close = difflib.get_close_matches(key, known, n=1)
            hint = f", did you mean {close[0]!r}?" if close else ""
            validator.add(
                IssueKind.UNKNOWN_KEY, key, f"not a MAGICC7 configuration key{hint}"
            )
