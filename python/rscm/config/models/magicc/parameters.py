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
}


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
