"""MAGICC parameter status tracking and registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

__all__ = [
    "MAGICC_PARAMETERS",
    "ParameterInfo",
    "ParameterStatus",
    "get_coverage_report",
    "get_coverage_stats",
]


class ParameterStatus(Enum):
    """Status of a MAGICC parameter in RSCM."""

    SUPPORTED = auto()  # Mapped to RSCM config
    NOT_IMPLEMENTED = auto()  # Feature not yet in RSCM
    NOT_NEEDED = auto()  # Output/file control handled differently
    DEPRECATED = auto()  # Superseded in MAGICC7


@dataclass
class ParameterInfo:
    """Metadata about a MAGICC parameter."""

    name: str
    status: ParameterStatus
    rscm_path: str | None = None  # Path in RSCM config (if SUPPORTED)
    unit: str | None = None
    description: str | None = None
    category: str | None = None  # e.g., "climate", "carbon_cycle", "forcing"

    def __post_init__(self) -> None:
        """Validate parameter information."""
        # SUPPORTED parameters must have rscm_path
        if self.status == ParameterStatus.SUPPORTED and self.rscm_path is None:
            msg = f"SUPPORTED parameter '{self.name}' must have rscm_path"
            raise ValueError(msg)


# Registry of MAGICC parameters
MAGICC_PARAMETERS: dict[str, ParameterInfo] = {
    # Time configuration
    "startyear": ParameterInfo(
        name="startyear",
        status=ParameterStatus.SUPPORTED,
        rscm_path="time.start",
        unit="year",
        description="Simulation start year",
        category="time",
    ),
    "endyear": ParameterInfo(
        name="endyear",
        status=ParameterStatus.SUPPORTED,
        rscm_path="time.end",
        unit="year",
        description="Simulation end year",
        category="time",
    ),
    # Climate sensitivity and forcing
    "core_climatesensitivity": ParameterInfo(
        name="core_climatesensitivity",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.climate_sensitivity",
        unit="K",
        description="Equilibrium climate sensitivity for 2xCO2",
        category="climate",
    ),
    "core_delq2xco2": ParameterInfo(
        name="core_delq2xco2",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.forcing_2xco2",
        unit="W/m^2",
        description="Radiative forcing from doubling CO2",
        category="climate",
    ),
    # Radiative forcing method
    "core_co2ch4n2o_rfmethod": ParameterInfo(
        name="core_co2ch4n2o_rfmethod",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit=None,
        description="Method for calculating CO2/CH4/N2O forcing (IPCCTAR/OLBL)",
        category="forcing",
    ),
    # Rapid adjustments (OLBL method)
    "core_rfrapidadjust_co2": ParameterInfo(
        name="core_rfrapidadjust_co2",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit=None,
        description="Rapid adjustment factor for CO2 forcing",
        category="forcing",
    ),
    "core_rfrapidadjust_ch4": ParameterInfo(
        name="core_rfrapidadjust_ch4",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit=None,
        description="Rapid adjustment factor for CH4 forcing",
        category="forcing",
    ),
    "core_rfrapidadjust_n2o": ParameterInfo(
        name="core_rfrapidadjust_n2o",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit=None,
        description="Rapid adjustment factor for N2O forcing",
        category="forcing",
    ),
    # Forcing scaling factors
    "rf_solar_scale": ParameterInfo(
        name="rf_solar_scale",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.forcing.parameters.solar_scale",
        unit=None,
        description="Scaling factor for solar forcing",
        category="forcing",
    ),
    "rf_volcanic_scale": ParameterInfo(
        name="rf_volcanic_scale",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.forcing.parameters.volcanic_scale",
        unit=None,
        description="Scaling factor for volcanic forcing",
        category="forcing",
    ),
    # Forcing mode control
    "rf_total_runmodus": ParameterInfo(
        name="rf_total_runmodus",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit=None,
        description="Forcing run mode (e.g., CO2-only)",
        category="forcing",
    ),
    "rf_efficacy_apply": ParameterInfo(
        name="rf_efficacy_apply",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.efficacy_apply",
        unit=None,
        description="Efficacy application mode (0=disabled, 1=prescribed, 2=AR6 mode)",
        category="forcing",
    ),
    "rf_efficacy_co2": ParameterInfo(
        name="rf_efficacy_co2",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.prescribed_efficacy_co2",
        unit=None,
        description="Prescribed CO2 efficacy factor (default 1.0)",
        category="forcing",
    ),
    # Concentration/emissions switching
    "co2_switchfromconc2emis_year": ParameterInfo(
        name="co2_switchfromconc2emis_year",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit="year",
        description="Year to switch from concentration to emissions driven for CO2",
        category="carbon_cycle",
    ),
    "ch4_switchfromconc2emis_year": ParameterInfo(
        name="ch4_switchfromconc2emis_year",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit="year",
        description="Year to switch from concentration to emissions driven for CH4",
        category="carbon_cycle",
    ),
    "n2o_switchfromconc2emis_year": ParameterInfo(
        name="n2o_switchfromconc2emis_year",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit="year",
        description="Year to switch from concentration to emissions driven for N2O",
        category="carbon_cycle",
    ),
    "mhalo_switchfromconc2emis_year": ParameterInfo(
        name="mhalo_switchfromconc2emis_year",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit="year",
        description="Year to switch Montreal Protocol gases to emissions driven",
        category="carbon_cycle",
    ),
    "fgas_switchfromconc2emis_year": ParameterInfo(
        name="fgas_switchfromconc2emis_year",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit="year",
        description="Year to switch F-gases to emissions driven",
        category="carbon_cycle",
    ),
    # File paths (handled in inputs section)
    "file_co2_conc": ParameterInfo(
        name="file_co2_conc",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="CO2 concentration input file (use inputs section)",
        category="file",
    ),
    "file_ch4_conc": ParameterInfo(
        name="file_ch4_conc",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="CH4 concentration input file (use inputs section)",
        category="file",
    ),
    "file_n2o_conc": ParameterInfo(
        name="file_n2o_conc",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="N2O concentration input file (use inputs section)",
        category="file",
    ),
    "file_emisscen": ParameterInfo(
        name="file_emisscen",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="Emissions scenario file (use inputs section)",
        category="file",
    ),
    # Output control flags
    "out_dynamic_vars": ParameterInfo(
        name="out_dynamic_vars",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="Internal variables written to output",
        category="output",
    ),
    "out_forcing": ParameterInfo(
        name="out_forcing",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="Output forcing flag",
        category="output",
    ),
    "out_concentrations": ParameterInfo(
        name="out_concentrations",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="Output concentrations flag",
        category="output",
    ),
    "out_emissions": ParameterInfo(
        name="out_emissions",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="Output emissions flag",
        category="output",
    ),
    "out_temperature": ParameterInfo(
        name="out_temperature",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="Output temperature flag",
        category="output",
    ),
    "out_carboncycle": ParameterInfo(
        name="out_carboncycle",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="Output carbon cycle flag",
        category="output",
    ),
    "out_ascii_binary": ParameterInfo(
        name="out_ascii_binary",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="Output format (ASCII/BINARY)",
        category="output",
    ),
    # Ocean UDEB parameters
    "core_initial_upwelling_rate": ParameterInfo(
        name="core_initial_upwelling_rate",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.w_initial",
        unit="m/yr",
        description="Initial upwelling rate",
        category="climate",
    ),
    "core_upwelling_variable_part": ParameterInfo(
        name="core_upwelling_variable_part",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.w_variable_fraction",
        unit="1",
        description="Fraction of upwelling that varies with temperature",
        category="climate",
    ),
    "core_ocn_depthdependent": ParameterInfo(
        name="core_ocn_depthdependent",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.depth_dependent_area",
        unit="1",
        description="Depth-dependent ocean area scale factor (0=cylindrical, 1=full)",
        category="climate",
    ),
    "core_verticaldiff_top_dkdt": ParameterInfo(
        name="core_verticaldiff_top_dkdt",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.kappa_dkdt",
        unit="cm^2/s/K",
        description="Temperature gradient coefficient for vertical diffusivity",
        category="climate",
    ),
    "core_landheatcapacity_apply": ParameterInfo(
        name="core_landheatcapacity_apply",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.land_heat_capacity_enabled",
        unit=None,
        description="Enable ground heat reservoir (0/1)",
        category="climate",
    ),
    "core_landhc_effthickness": ParameterInfo(
        name="core_landhc_effthickness",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.land_hc_eff_thickness",
        unit="m",
        description="Effective thickness of ground heat reservoir",
        category="climate",
    ),
    "core_heatxchange_landground": ParameterInfo(
        name="core_heatxchange_landground",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.k_lg",
        unit="W/m^2/K",
        description="Land-ground heat exchange coefficient",
        category="climate",
    ),
    "core_heatxchange_northsouth": ParameterInfo(
        name="core_heatxchange_northsouth",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.k_ns",
        unit="W/m^2/K",
        description="Inter-hemispheric heat exchange coefficient",
        category="climate",
    ),
    "core_feedback_cumtsensitivity": ParameterInfo(
        name="core_feedback_cumtsensitivity",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.feedback_cumt_sensitivity",
        unit="1",
        description="Sensitivity of ECS to cumulative temperature",
        category="climate",
    ),
    "core_feedback_qsensitivity": ParameterInfo(
        name="core_feedback_qsensitivity",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.climate.parameters.feedback_q_sensitivity",
        unit="1",
        description="Sensitivity of ECS to forcing level",
        category="climate",
    ),
    # Variability control parameters
    "core_amv_apply": ParameterInfo(
        name="core_amv_apply",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit=None,
        description="Enable Atlantic Multidecadal Variability (0/1)",
        category="climate",
    ),
    "core_elnino_apply": ParameterInfo(
        name="core_elnino_apply",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit=None,
        description="Enable El Nino parameterisation (0/1)",
        category="climate",
    ),
    "ch4_incl_ch4ox": ParameterInfo(
        name="ch4_incl_ch4ox",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit=None,
        description="Include CH4 oxidation to CO2 feedback (0/1)",
        category="carbon_cycle",
    ),
    "co2_tempfeedback_switch": ParameterInfo(
        name="co2_tempfeedback_switch",
        status=ParameterStatus.NOT_IMPLEMENTED,
        unit=None,
        description="Global CO2 temperature feedback switch (0/1); rejected on import",
        category="carbon_cycle",
    ),
    "co2_fertilization_method": ParameterInfo(
        name="co2_fertilization_method",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.terrestrial_carbon.parameters.fertilization_method",
        unit=None,
        description="CO2 fertilization method (1=log, 2=Gifford, 3=sigmoid)",
        category="carbon_cycle",
    ),
    "co2_plantboxresp_method": ParameterInfo(
        name="co2_plantboxresp_method",
        status=ParameterStatus.NOT_NEEDED,
        unit=None,
        description="Plant box respiration method",
        category="carbon_cycle",
    ),
    "co2_plantboxresp_fertscale": ParameterInfo(
        name="co2_plantboxresp_fertscale",
        status=ParameterStatus.SUPPORTED,
        rscm_path="components.terrestrial_carbon.parameters.plantbox_resp_fertscale",
        unit=None,
        description="Fertilization effect on respiration",
        category="carbon_cycle",
    ),
}


#: MAGICC permafrost parameters and their path below ``components``.
_PERMAFROST = {
    "pf_apply": (
        "permafrost.enabled",
        "Include the Permafrost component and its feedbacks",
    ),
    "pf_nbands": ("permafrost.parameters.n_bands", "Number of zonal bands"),
    "pf_meltingtemp_min": (
        "permafrost.parameters.melting_temp_min",
        "Southern melting threshold",
    ),
    "pf_meltingtemp_max": (
        "permafrost.parameters.melting_temp_max",
        "Northern melting threshold",
    ),
    "pf_arctic_amplification": (
        "permafrost.parameters.arctic_amplification",
        "Arctic to global warming ratio",
    ),
    "pf_tsoilannualcycle_ampl": (
        "permafrost.parameters.soil_annual_cycle_amplitude",
        "Seasonal soil temperature amplitude",
    ),
    "pf_ms_aerob_decomp_turnovertime": (
        "permafrost.parameters.aerobic_turnover_time",
        "Mineral soil aerobic turnover time",
    ),
    "pf_decomprate_anaerob_over_aerob_ratio": (
        "permafrost.parameters.anaerobic_decomposition_ratio",
        "Anaerobic to aerobic decomposition ratio",
    ),
    "pf_decomprate_peat_over_ms_ratio": (
        "permafrost.parameters.peat.decomposition_ratio",
        "Peat to mineral soil decomposition ratio",
    ),
    "pf_q10_temp1": ("permafrost.parameters.q10_temp1", "Q10 reference temperature"),
    "pf_q10_temp2": ("permafrost.parameters.q10_temp2", "Q10 offset temperature"),
    "pf_soilwater_m": ("permafrost.parameters.soil_water_slope", "Soil moisture slope"),
    "pf_soilwater_offset": (
        "permafrost.parameters.soil_water_offset",
        "Soil moisture offset",
    ),
    "pf_soilwater_minw": (
        "permafrost.parameters.soil_water_min",
        "Minimum soil moisture",
    ),
    "pf_tot_pool": ("permafrost.parameters.total_pool", "Total permafrost carbon"),
    "pf_minsoil_southern_poolfraction": (
        "permafrost.parameters.mineral_soil_fraction_south",
        "Southern mineral soil carbon fraction",
    ),
    "pf_minsoil_northern_poolfraction": (
        "permafrost.parameters.mineral_soil_fraction_north",
        "Northern mineral soil carbon fraction",
    ),
    "pf_zonal_pooldistr": (
        "permafrost.parameters.zonal_pool_distribution",
        "Zonal carbon distribution shape",
    ),
    "pf_co2fromch4oxinatm_fraction": (
        "ch4_oxidation.parameters.permafrost_fraction",
        "Fraction of permafrost CH4 oxidised to fossil CO2",
    ),
    **{
        f"pf_{magicc}_{key}": (
            f"permafrost.parameters.{rscm}.{field}",
            f"{rscm} {description}",
        )
        for magicc, rscm in (("ms", "mineral_soil"), ("peat", "peat"))
        for key, field, description in (
            ("thawfreeze_percperk_rate", "thaw_rate", "thaw rate"),
            ("thawfreeze_exp_temp", "thaw_exponent", "thaw exponent"),
            (
                "anaerob_initial_areafraction",
                "anaerobic_initial_fraction",
                "anaerobic fraction",
            ),
            (
                "anaerob_max_areafraction",
                "anaerobic_max_fraction",
                "maximum anaerobic fraction",
            ),
            (
                "anaerob_moistsens",
                "anaerobic_moisture_sensitivity",
                "anaerobic moisture sensitivity",
            ),
            ("ch4oxidisation_fraction", "ch4_oxidation_fraction", "soil CH4 oxidation"),
        )
    },
    **{
        f"pf_q10_{magicc}_{kind}_alpha": (
            f"permafrost.parameters.{rscm}.q10_{name}_alpha",
            f"{rscm} {name} Q10 exponent",
        )
        for magicc, rscm in (("ms", "mineral_soil"), ("peat", "peat"))
        for kind, name in (("aerob", "aerobic"), ("anaerob", "anaerobic"))
    },
}

MAGICC_PARAMETERS.update(
    {
        name: ParameterInfo(
            name=name,
            status=ParameterStatus.SUPPORTED,
            rscm_path=f"components.{path}",
            description=description,
            category="carbon_cycle",
        )
        for name, (path, description) in _PERMAFROST.items()
    }
)

#: MAGICC nitrogen limitation switches and their path below ``nitrogen``.
_NITROGEN_SWITCHES = {
    "ncycle_apply": ("enabled", "Limit NPP by nitrogen"),
    "ncycle_apply_ndep_from_emis": (
        "deposition_from_emissions",
        "Calculate nitrogen deposition from NOx and NH3 emissions",
    ),
    "ncycle_apply_prognostic_nfix": (
        "prognostic_fixation",
        "Calculate nitrogen fixation from NPP",
    ),
}

#: MAGICC nitrogen limitation parameters and their path below ``nitrogen``.
_NITROGEN = {
    "ncycle_nfix_vmax": ("prognostic_fixation.vmax", "Maximum fixation rate"),
    "ncycle_nfix_km": ("prognostic_fixation.km", "NPP at half the maximum fixation"),
    "ncycle_ncratio_uptake": ("ncratio_uptake", "Plant uptake per unit NPP"),
    "ncycle_ncratio_recyc": ("ncratio_recycling", "Recycling per unit respiration"),
    "ncycle_recyc2cnratio_factor": (
        "recycling_cn_factor",
        "Damping of the recycled N:C ratio",
    ),
    "ncycle_ncratio_uptake_supply": ("uptake_supply", "Half saturation of supply"),
    "ncycle_npp2navail_factor_a": ("availability_factor_a", "Availability factor A"),
    "ncycle_npp2navail_factor_c": ("availability_factor_c", "Availability factor C"),
    "ncycle_turnovertime_losses": ("loss_turnover_time", "Pool loss turnover time"),
}

MAGICC_PARAMETERS.update(
    {
        name: ParameterInfo(
            name=name,
            status=ParameterStatus.SUPPORTED,
            rscm_path=f"components.terrestrial_carbon.{prefix}.{path}",
            description=description,
            category="carbon_cycle",
        )
        for prefix, table in (
            ("nitrogen", _NITROGEN_SWITCHES),
            ("parameters.nitrogen", _NITROGEN),
        )
        for name, (path, description) in table.items()
    }
)

MAGICC_PARAMETERS["out_sealevel"] = ParameterInfo(
    name="out_sealevel",
    status=ParameterStatus.SUPPORTED,
    rscm_path="components.sea_level_rise.enabled",
    description="Include the SeaLevelRise component and thermal expansion",
    category="climate",
)


def get_coverage_report() -> str:
    """Generate a markdown coverage report of MAGICC parameter support.

    Returns
    -------
    str
        Markdown-formatted report showing parameters by status.
    """
    lines = [
        "# MAGICC Parameter Support Report",
        "",
        "This report shows the support status of MAGICC parameters in RSCM.",
        "",
    ]

    # Group by status
    by_status: dict[ParameterStatus, list[ParameterInfo]] = {
        status: [] for status in ParameterStatus
    }
    for param in MAGICC_PARAMETERS.values():
        by_status[param.status].append(param)

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for status in ParameterStatus:
        count = len(by_status[status])
        lines.append(f"| {status.name} | {count} |")
    lines.append(f"| **Total** | **{len(MAGICC_PARAMETERS)}** |")
    lines.append("")

    # Details by status
    for status in ParameterStatus:
        params = by_status[status]
        if not params:
            continue

        lines.append(f"## {status.name} ({len(params)} parameters)")
        lines.append("")

        if status == ParameterStatus.SUPPORTED:
            lines.append("| Parameter | RSCM Path | Unit |")
            lines.append("|-----------|-----------|------|")
            for p in sorted(params, key=lambda x: x.name):
                unit = p.unit or "-"
                lines.append(f"| `{p.name}` | `{p.rscm_path}` | {unit} |")
        else:
            # Group by category
            by_cat: dict[str, list[ParameterInfo]] = {}
            for p in params:
                cat = p.category or "other"
                by_cat.setdefault(cat, []).append(p)

            for cat in sorted(by_cat.keys()):
                lines.append(f"### {cat.title()}")
                lines.append("")
                for p in sorted(by_cat[cat], key=lambda x: x.name):
                    desc = f" - {p.description}" if p.description else ""
                    lines.append(f"- `{p.name}`{desc}")
                lines.append("")

        lines.append("")

    return "\n".join(lines)


def get_coverage_stats() -> dict[str, int]:
    """Get parameter support statistics.

    Returns
    -------
    dict
        Dict with status names as keys and counts as values.
    """
    stats: dict[str, int] = {status.name: 0 for status in ParameterStatus}
    stats["total"] = len(MAGICC_PARAMETERS)

    for param in MAGICC_PARAMETERS.values():
        stats[param.status.name] += 1

    return stats
