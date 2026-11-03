r"""
Generate the full emissions-driven MAGICC reference run.

Runs SSP245 with MAGICC7's default configuration, emissions-driven from 1750
for CO2, CH4, N2O and all halocarbons. Permafrost is disabled. Every agent's
forcing is written for all four boxes, so each RSCM component can be checked
against its own MAGICC counterpart as well as the coupled model.

A second run enables permafrost and a third nitrogen limitation with deposition
from emissions. A fourth writes sea level rise,
and the variants each change one climate, efficacy or method switch.
Their emissions and inputs are unchanged,
so only the variables compared against the coupled model are kept.

Usage:
    export MAGICC_ROOT=/path/to/magicc-v7.5.3

    uv run --with pymagicc --with scmdata --with "pandas<3" --with "xarray<2025" \
        scripts/regression/generate_full_run.py

Output:
    tests/regression/data/full_run/01_ssp245_emissions.csv.gz
    tests/regression/data/full_run/01_ssp245_emissions_config.json
    tests/regression/data/full_run/01_ssp245_emissions_parameters.json
    tests/regression/data/full_run/02_ssp245_emissions_permafrost*
    tests/regression/data/full_run/03_ssp245_emissions_nitrogen*
    tests/regression/data/full_run/04_ssp245_emissions_sealevel*
    tests/regression/data/full_run/05_ssp245_default*
    tests/regression/data/full_run/06_ssp245_emissions_<variant>*
"""

import json
from fnmatch import fnmatchcase
from pathlib import Path

import pandas as pd
from utils import make_config, output_dir, run_magicc_ctx

OUT = output_dir("full_run")

#: Variables kept for every box, rather than only for the World.
BOX_VARIABLES = (
    "Surface Temperature",
    "TOTAL_EFFRF",
    "Effective Radiative Forcing|Aerosols|Direct Effect|*|MAGICC *",
    "Effective Radiative Forcing|Aerosols|Direct Effect|Mineral Dust",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Black Carbon on Snow",
    "Effective Radiative Forcing|Land-use Change",
    "Effective Radiative Forcing|Solar",
    "Effective Radiative Forcing|Tropospheric Ozone",
)

#: Variables kept from the reference run, which drives the component tests.
REFERENCE_VARIABLES = (
    *BOX_VARIABLES,
    "Atmospheric Concentrations|*",
    "Emissions|*",
    "Effective Radiative Forcing",
    *(
        f"Effective Radiative Forcing|{agent}"
        for agent in (
            "CH4 Oxidation Stratospheric H2O",
            "F-Gases",
            "Montreal Protocol Halogen Gases",
            "Stratospheric Ozone",
            "Volcanic",
        )
    ),
    "CH4_TAUOH",
    "EESC_CONC",
    "OZDUETOEffective Radiative Forcing|N2O",
    "OZDUETOTEMPERATURE_ERF",
)

CONFIG = make_config(
    startyear=1750,
    endyear=2105,
    core_climatesensitivity=3.0,
    pf_apply=0,
    co2_switchfromconc2emis_year=1750,
    ch4_switchfromconc2emis_year=1750,
    n2o_switchfromconc2emis_year=1750,
    mhalo_switchfromconc2emis_year=1750,
    fgas_switchfromconc2emis_year=1750,
)

PERMAFROST = make_config(
    CONFIG,
    pf_apply=1,
    out_dynamic_vars=["DAT_CO2PF_EMIS", "DAT_CH4PF_EMIS", "DAT_PFTHAWED_AREA"],
)

#: Variables kept from the permafrost run.
PERMAFROST_VARIABLES = (
    "Surface Temperature",
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Volcanic",
    "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
    "PFTHAWED_AREA",
)

NITROGEN = make_config(CONFIG, ncycle_apply=1, ncycle_apply_ndep_from_emis=1)

#: Variables kept from the nitrogen run.
NITROGEN_VARIABLES = (
    "Surface Temperature",
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
)

#: CARBONCYCLE.OUT columns kept from the nitrogen run, which pymagicc does not read.
NITROGEN_CARBONCYCLE = ("NCYCLE_LIMIT_FACTOR", "NCYCLE_NPOOL", "CURRENT_NPP")

SEALEVEL = make_config(CONFIG, out_sealevel=1)

#: Variables kept from the sea level run.
SEALEVEL_VARIABLES = (
    "Surface Temperature",
    "Sea Level Rise",
    *(
        f"SLR_{name}"
        for name in ("EXPANSION", "GL", "GIS_SMB", "GIS_SID", "AIS_SMB", "AIS_SID")
    ),
)

#: MAGICC7's own default: concentrations prescribed until 2015, and permafrost on.
DEFAULT = make_config(startyear=1750, endyear=2105, core_climatesensitivity=3.0)

#: Variables kept from the default run.
DEFAULT_VARIABLES = (
    "Surface Temperature",
    "Atmospheric Concentrations|*",
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Volcanic",
)

#: Climate, efficacy and method switches of the emissions run, each changed alone.
VARIANTS = {
    "ecs_1p5": {"core_climatesensitivity": 1.5},
    "ecs_4p5": {"core_climatesensitivity": 4.5},
    "no_upwelling": {"core_initial_upwelling_rate": 0},
    "constant_upwelling": {"core_upwelling_variable_part": 0},
    "cylindrical_ocean": {"core_ocn_depthdependent": 0},
    "constant_diffusivity": {"core_verticaldiff_top_dkdt": 0},
    "no_ground_heat": {"core_landheatcapacity_apply": 0},
    "no_interhemispheric_exchange": {"core_heatxchange_northsouth": 0},
    "constant_feedback": {
        "core_feedback_qsensitivity": 0,
        "core_feedback_cumtsensitivity": 0,
    },
    "no_efficacy": {"rf_efficacy_apply": 0},
    "prescribed_efficacy": {"rf_efficacy_apply": 1},
    "ipcctar": {"core_co2ch4n2o_rfmethod": "IPCCTAR"},
    "gifford_fertilisation": {"co2_fertilization_method": 2.0},
    "respiration_fertscale": {"co2_plantboxresp_fertscale": 0.5},
    "no_carbon_temperature_feedback": {"co2_tempfeedback_switch": 0},
}

#: Variables kept from each variant.
VARIANT_VARIABLES = (
    "Surface Temperature",
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Volcanic",
)

#: Name, configuration, kept variables, CARBONCYCLE.OUT columns
#: and whether to write the run's parameters.
#: A variant's parameters are the reference run's with its switches.
CASES = (
    ("01_ssp245_emissions", CONFIG, REFERENCE_VARIABLES, (), True),
    ("02_ssp245_emissions_permafrost", PERMAFROST, PERMAFROST_VARIABLES, (), True),
    (
        "03_ssp245_emissions_nitrogen",
        NITROGEN,
        NITROGEN_VARIABLES,
        NITROGEN_CARBONCYCLE,
        True,
    ),
    ("04_ssp245_emissions_sealevel", SEALEVEL, SEALEVEL_VARIABLES, (), True),
    ("05_ssp245_default", DEFAULT, DEFAULT_VARIABLES, (), True),
    *(
        (
            f"06_ssp245_emissions_{name}",
            make_config(CONFIG, **switches),
            VARIANT_VARIABLES,
            (),
            False,
        )
        for name, switches in VARIANTS.items()
    ),
)


def read_carboncycle(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read columns of CARBONCYCLE.OUT in the layout of a pymagicc timeseries."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("YEARS"))
    header = lines[start].split()
    rows = [[float(v) for v in line.split()] for line in lines[start + 1 :] if line]
    data = pd.DataFrame(rows, columns=header).set_index("YEARS")
    data = data.loc[1750:2105, list(columns)].T
    data.columns = data.columns.astype(int)
    return data.rename_axis("variable").reset_index().assign(region="World")


def _kept(variable: str, region: str, variables: tuple[str, ...]) -> bool:
    def matches(patterns: tuple[str, ...]) -> bool:
        return any(fnmatchcase(variable, p) for p in patterns)

    return matches(variables) and (region == "World" or matches(BOX_VARIABLES))


def run_case(
    name: str,
    config: dict,
    variables: tuple[str, ...],
    carboncycle: tuple[str, ...],
    parameters: bool,
) -> None:
    """Run MAGICC with one configuration and write its fixture."""
    with run_magicc_ctx() as m:
        # CARBONCYCLE.OUT is only written as ASCII, the rest at full precision
        if carboncycle:
            m.set_output_variables(
                write_ascii=True, write_binary=False, carboncycle=True
            )
            m.run(**config)
            extra = read_carboncycle(Path(m.out_dir) / "CARBONCYCLE.OUT", carboncycle)
        m.set_output_variables(
            write_ascii=False,
            write_binary=True,
            forcing=True,
            concentrations=True,
            emissions=True,
            temperature=True,
            lifetimes=True,
            carboncycle=True,
            parameters=True,
        )
        res = m.run(**config)

    table = res.filter(year=range(1750, 2106)).timeseries().reset_index()
    rows = zip(table.variable, table.region)
    table = table[[_kept(v, r, variables) for v, r in rows]]
    if carboncycle:
        times = {t.year: t for t in table.columns if hasattr(t, "year")}
        table = pd.concat([table, extra.rename(columns=times)])
    table.to_csv(OUT / f"{name}.csv.gz", index=False)
    with open(OUT / f"{name}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    if parameters:
        with open(OUT / f"{name}_parameters.json", "w") as f:
            json.dump(
                res.metadata["parameters"], f, indent=1, sort_keys=True, default=str
            )
    print(f"Wrote {name} ({len(res.get_unique_meta('variable'))} variables)")


def main() -> None:
    """Run MAGICC for every case."""
    for case in CASES:
        run_case(*case)


if __name__ == "__main__":
    main()
