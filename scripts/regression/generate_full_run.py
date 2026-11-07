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

Usage (`utils.DEFAULT_MAGICC_ROOT` unless `MAGICC_ROOT` is set):
    uv run --group regression scripts/regression/generate_full_run.py

Output:
    tests/regression/data/full_run/01_ssp245_emissions.csv.gz
    tests/regression/data/full_run/01_ssp245_emissions_config.json
    tests/regression/data/full_run/01_ssp245_emissions_parameters.json
    tests/regression/data/full_run/02_ssp245_emissions_permafrost*
    tests/regression/data/full_run/03_ssp245_emissions_nitrogen*
    tests/regression/data/full_run/04_ssp245_emissions_sealevel*
    tests/regression/data/full_run/05_ssp245_default*
    tests/regression/data/full_run/06_ssp245_emissions_<variant>*
    python/rscm/magicc/data/magicc7/*.gz (the run's forcing input files)
    python/rscm/magicc/data/magicc7/defaults.json (the configuration the chain reads)
"""

import gzip
import json
import shutil
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pandas as pd
from utils import MAGICC_ROOT, make_config, output_dir, run_magicc_ctx

OUT = output_dir("full_run")

# Variables kept for every box, rather than only for the World.
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

# Variables kept from the reference run, which drives the component tests.
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

# Variables kept from the permafrost run.
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

# Variables kept from the nitrogen run.
NITROGEN_VARIABLES = (
    "Surface Temperature",
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
)

# CARBONCYCLE.OUT columns kept from the nitrogen run, which pymagicc does not read.
NITROGEN_CARBONCYCLE = ("NCYCLE_LIMIT_FACTOR", "NCYCLE_NPOOL", "CURRENT_NPP")

SEALEVEL = make_config(CONFIG, out_sealevel=1)

# Variables kept from the sea level run.
SEALEVEL_VARIABLES = (
    "Surface Temperature",
    "Sea Level Rise",
    *(
        f"SLR_{name}"
        for name in ("EXPANSION", "GL", "GIS_SMB", "GIS_SID", "AIS_SMB", "AIS_SID")
    ),
)

# MAGICC7's own default: concentrations prescribed until 2015, and permafrost on.
DEFAULT = make_config(startyear=1750, endyear=2105, core_climatesensitivity=3.0)

# Variables kept from the default run.
DEFAULT_VARIABLES = (
    "Surface Temperature",
    "Atmospheric Concentrations|*",
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Volcanic",
)

# Climate, efficacy and method switches of the emissions run, each changed alone.
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

# Variables kept from each variant.
VARIANT_VARIABLES = (
    "Surface Temperature",
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Volcanic",
)

# Name, configuration, kept variables, CARBONCYCLE.OUT columns
# and whether to write the run's parameters.
# A variant's parameters are the reference run's with its switches.
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

# MAGICC7 input files behind the forcing that RSCM prescribes.
INPUTS = (
    *(f"HISTSSP_{name}_RF.IN" for name in ("SOXI", "SOXB", "BCI", "BCB", "OCI", "OCB")),
    "HISTSSP_BCSNOW_RF.IN",
    *(
        f"HISTSSP_{name}_OT.IN"
        for name in ("SOXI", "SOXNB", "OCI", "OCB", "OCN", "BCI", "BCB", "SS")
    ),
    "GISS_LANDUSE_RF.IN",
    "SSP245_SOLAR_RF.IN",
    "SSP245_VOLCANIC_RF.MON",
    "NCYCLE_NFIX3.IN",
)


PACKAGE_DATA = Path(__file__).parents[2] / "python/rscm/magicc/data/magicc7"


def default_config() -> dict:
    """MAGICC7's default configuration, as recorded by the default run.

    Keys MAGICC7 overwrites during a run come from its input configuration.
    """
    import f90nml  # noqa: PLC0415

    from rscm.magicc import full  # noqa: PLC0415

    with open(OUT / "05_ssp245_default_parameters.json") as f:
        raw = json.load(f)
    cfg = f90nml.read(MAGICC_ROOT / "run" / "MAGCFG_DEFAULTALL.CFG")["nml_allcfgs"]
    inputs = {key: cfg[key] for key in full.REWRITTEN_PARAMETERS}
    return {**raw["allcfgs"], **raw["outputcfgs"], **inputs}


def concentration_inputs() -> list[str]:
    """Concentration files the default run prescribes before switching to emissions."""
    config = default_config()
    names = [config[f"file_{gas}_conc"] for gas in ("co2", "ch4", "n2o")]
    names += config["fgas_files_conc"] + config["mhalo_files_conc"]
    return [name for name in names if name.strip()]


def copy_inputs() -> None:
    """Copy the forcing and concentration input files, compressed, into the package."""
    for name in (*INPUTS, *concentration_inputs()):
        with (
            open(MAGICC_ROOT / "run" / name, "rb") as src,
            open(PACKAGE_DATA / f"{name}.gz", "wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as dst,
        ):
            shutil.copyfileobj(src, dst)


class _Recorder(dict):
    """A configuration that records the keys read from it."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.used: set[str] = set()

    def __getitem__(self, key: str):
        self.used.add(key)
        return super().__getitem__(key)


def write_defaults() -> None:
    """Write the configuration values and 1750 concentrations the full chain reads."""
    from rscm.core import TimeAxis  # noqa: PLC0415
    from rscm.magicc import full  # noqa: PLC0415

    config = default_config()
    data = pd.read_csv(OUT / f"{CASES[0][0]}.csv.gz")
    data = data[data.region == "World"].set_index("variable")
    data = data[[c for c in data.columns if c[:4].isdigit()]]
    species = ("CO2", "CH4", "N2O", *full.halocarbon_species())
    concentrations = {
        s: float(data.loc[f"Atmospheric Concentrations|{s}"].iloc[0]) for s in species
    }
    # MAGICC7 reports an F-gas's second-year concentration in its first year.
    files = full.prescribed_concentrations([1750.0, 1751.0], config)
    for s in full.halocarbon_species()[: config["fgas_n"]]:
        concentrations[s] = float(files[s][0][0])
    sectors = {"Fossil": "MAGICC Fossil and Industrial", "AFOLU": "MAGICC AFOLU"}
    emissions = {
        **{
            f"{s}|{sector}": data.loc[f"Emissions|{s}|{theirs}"].to_numpy()
            for s in full.SECTORAL_EMISSIONS
            for sector, theirs in sectors.items()
        },
        **{s: data.loc[f"Emissions|{s}"].to_numpy() for s in full.halocarbon_species()},
    }
    time_axis = TimeAxis.from_values(np.array([float(c[:4]) for c in data.columns]))
    defaults = {"parameters": {}, "initial_concentrations": concentrations}
    full._defaults = lambda: defaults
    recorder = _Recorder(config)
    for pf_apply, ncycle_apply in ((0, 0), (1, 0), (0, 1)):
        recorder["pf_apply"] = pf_apply
        recorder["ncycle_apply"] = ncycle_apply
        for flag in ("ndep_from_emis", "prognostic_nfix", "cn_initial_values"):
            recorder[f"ncycle_apply_{flag}"] = ncycle_apply
        full.run_magicc7(time_axis, emissions, parameters=recorder)
    parameters = {key: config[key] for key in sorted(recorder.used)}
    with open(PACKAGE_DATA / "defaults.json", "w") as f:
        json.dump(
            {"parameters": parameters, "initial_concentrations": concentrations},
            f,
            indent=1,
        )
        f.write("\n")


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
    """Run MAGICC for every case and copy the inputs."""
    for case in CASES:
        run_case(*case)
    copy_inputs()
    write_defaults()


if __name__ == "__main__":
    main()
