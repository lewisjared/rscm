r"""
Generate FaIR v2.2.5 reference runs for the rscm-fair parity tests.

Each case writes three files to ``tests/regression/data/fair``:

- ``<case>.json``: the FaIR configuration of the single config and scenario
- ``<case>_timebounds.csv.gz``: concentrations, forcing, temperature and the rest
- ``<case>_timepoints.csv.gz``: emissions

``alpha_lifetime`` sits on timebounds,
but its value at bound ``i`` is the one used for step ``i``.

Columns are ``<FAIR attribute>|<specie or layer>``.
Inputs are part of the output.
Emissions-driven species keep their emissions,
concentration-driven species their concentrations
and forcing-driven species their forcing.
FaIR adds the temperature feedback to prescribed forcing in place,
so a case with that feedback also writes the input as ``prescribed_forcing|<specie>``.

Usage:
    uv run --no-project --with fair==2.2.5 --with numpy==2.4.2 \
        python -I scripts/fair/generate_fixtures.py

NumPy's AVX-512 ``exp`` differs from the C library's by an ulp,
which FaIR's longest CO2 lifetime box turns into a 5e-9 relative difference.
The script disables it so the runs are the same on every x86 machine.

The RCMIP scenarios and fair-calibrate files are downloaded through pooch on first use.
The calibrated cases also copy the species configs and their posterior members
to ``tests/regression/data/fair/calibration``, for the loader tests.
"""

import json
import math
import os
from pathlib import Path

os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4 AVX512_ICL"

import numpy as np
import pandas as pd
import pooch
from fair import FAIR, __version__
from fair.interface import fill, initialise
from fair.io import read_properties

OUT = Path(__file__).parents[2] / "tests" / "regression" / "data" / "fair"

#: fair-calibrate v1.6.1 (RCMIP constraint set) files and their MD5 checksums.
CALIBRATION_URL = "https://zenodo.org/api/records/23208444/files/{}/content"
CALIBRATION_FILES = {
    "species_configs_properties.csv": "7174759997d890824f6ec498dbb6b64b",
    "calibrated_constrained_parameters.csv": "e0b6d80a3ff1496dd0e2b1705818803b",
    "historical_emissions_1750-2023_cmip7.csv": "ae0fecc9f7cd8335682e980a27a6a3b8",
    "solar_forcing_timebounds_cmip7.csv": "a1052f5e8323b20707558c8211d16cb5",
    "volcanic_forcing_timebounds_cmip7.csv": "46144fb7d00c6191d7c3ac3a0cdd2379",
    "land_use_forcing_timebounds_cmip7.csv": "cb8dd7d69c6f232f85a2afde538cfd3e",
    "irrigation_forcing_timebounds_cmip7.csv": "e8abb5e5dfaefdec803c2bd96ba592c7",
}
#: Posterior members with a parity case, the first three in the file.
CALIBRATION_MEMBERS = (1939, 2939, 3503)

#: Three-layer climate configuration close to the AR6 ensemble median.
CLIMATE_3 = {
    "ocean_heat_transfer": [1.1, 1.6, 0.9],
    "ocean_heat_capacity": [8.0, 14.0, 100.0],
    "deep_ocean_efficacy": 1.1,
}
CLIMATE_2 = {
    "ocean_heat_transfer": [1.2, 0.7],
    "ocean_heat_capacity": [8.0, 120.0],
    "deep_ocean_efficacy": 1.3,
}

EESC = "Equivalent effective stratospheric chlorine"

#: Eruptions in the energy balance cases: first and end year after 1850, size (W/m2).
ERUPTIONS = ((40, 42, 3.0), (120, 121, 2.0))

TIMEBOUND_ATTRIBUTES = (
    "concentration",
    "forcing",
    "temperature",
    "airborne_emissions",
    "cumulative_emissions",
    "ocean_heat_content_change",
    "toa_imbalance",
    "forcing_sum",
    "stochastic_forcing",
    "alpha_lifetime",
)
TIMEPOINT_ATTRIBUTES = ("emissions",)


def _value(x):
    """JSON value with NaN as None."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    x = float(x)
    return None if math.isnan(x) else x


def _make(species, properties, time, ch4_method, n_layers):
    f = FAIR(ch4_method=ch4_method, n_layers=n_layers)
    f.define_time(*time)
    f.define_scenarios(["scenario"])
    f.define_configs(["config"])
    f.define_species(species, properties)
    f.allocate()
    return f


def _climate(f, climate):
    for name, value in climate.items():
        fill(f.climate_configs[name], value, config="config")


def _initialise(f):
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)


def _frame(f, attributes, time_dim, config):
    columns = {}
    for attribute in attributes:
        data = getattr(f, attribute).sel(scenario="scenario", config=config)
        other = [d for d in data.dims if d != time_dim]
        if not other:
            columns[attribute] = data.values
            continue
        if len(other) != 1:
            raise ValueError(attribute)
        (dim,) = other
        for label in data[dim].values:
            values = data.sel({dim: label}).values
            if not np.all(np.isnan(values)):
                columns[f"{attribute}|{label}"] = values
    return pd.DataFrame(columns, index=pd.Index(f.__dict__[time_dim], name=time_dim))


def _config(f, climate, config):
    configs = f.species_configs.sel(config=config)
    species = []
    for name in f.species:
        entry = {
            "name": name,
            **{
                k: _value(v) if k not in ("type", "input_mode") else v
                for k, v in f.properties[name].items()
            },
        }
        for var in configs.data_vars:
            data = configs[var]
            if "specie" not in data.dims:
                continue
            values = data.sel(specie=name).values
            entry[var] = [_value(v) for v in values] if values.ndim else _value(values)
        species.append(entry)
    climate_configs = f.climate_configs.sel(config=config)
    climate_out = {
        name: (
            [_value(v) for v in climate_configs[name].values]
            if climate_configs[name].ndim
            else _value(climate_configs[name].values)
        )
        for name in (*climate, "gamma_autocorrelation")
    }
    species_configs = {
        var: _value(configs[var].values)
        for var in configs.data_vars
        if "specie" not in configs[var].dims
    }
    return {
        "fair_version": __version__,
        "time": {
            "timebounds": [float(t) for t in f.timebounds],
            "timepoints": [float(t) for t in f.timepoints],
        },
        "options": {
            "ch4_method": f.ch4_method,
            "ghg_method": f.ghg_method,
            "iirf_max": f.iirf_max,
            "br_cl_ods_potential": f.br_cl_ods_potential,
            "ch4_min": f._ch4_min,
        },
        "climate": climate_out,
        "species_configs": species_configs,
        "species": species,
    }


def _write(name, f, climate, config="config", extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(
        json.dumps(_config(f, climate, config), indent=1) + "\n"
    )
    for attributes, dim, suffix in (
        (TIMEBOUND_ATTRIBUTES, "timebounds", "timebounds"),
        (TIMEPOINT_ATTRIBUTES, "timepoints", "timepoints"),
    ):
        frame = _frame(f, attributes, dim, config)
        if dim == "timebounds" and extra:
            frame = frame.assign(**extra)
        frame.to_csv(
            OUT / f"{name}_{suffix}.csv.gz",
            float_format="%.17g",
            compression={"method": "gzip", "mtime": 0},
        )
    print(f"wrote {name}")


def energy_balance(name, climate, step):
    """Write a run driven by a CO2-like forcing ramp with two eruptions."""
    species = ["Prescribed"]
    properties = {
        "Prescribed": {
            "type": "unspecified",
            "input_mode": "forcing",
            "greenhouse_gas": False,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        }
    }
    n_layers = len(climate["ocean_heat_transfer"])
    f = _make(species, properties, (1850, 2150, step), "leach2021", n_layers)
    _climate(f, climate)
    t = f.timebounds - 1850
    forcing = 5.35 * np.log(np.minimum(1.01**t, 4.0))
    for start, end, size in ERUPTIONS:
        forcing[(t >= start) & (t < end)] -= size
    fill(f.forcing, forcing[:, None, None, None])
    _initialise(f)
    f.run(progress=False)
    _write(name, f, climate)


def _scenario(  # noqa: PLR0913
    scenario_name,
    *,
    step=1,
    ch4_method="thornhill2021",
    only=None,
    concentration=(),
    eesc=None,
    feedback=None,
):
    """Run the default AR6 species on an RCMIP scenario.

    `eesc` prescribes the EESC.
    `feedback` sets `forcing_temperature_feedback` by specie.
    """
    species, properties = read_properties()
    if only is not None:
        species = [s for s in species if s in only]
        properties = {s: properties[s] for s in species}
    for specie in concentration:
        properties[specie]["input_mode"] = "concentration"
    f = FAIR(ch4_method=ch4_method)
    f.define_time(1750, 2100, step)
    f.define_scenarios([scenario_name])
    f.define_configs(["config"])
    f.define_species(species, properties)
    f.allocate()
    f.fill_species_configs()
    if eesc is None:
        f.fill_from_rcmip()
    else:
        # RCMIP has no EESC concentrations, so fill_from_rcmip must skip it.
        f.properties_df.loc[EESC, "input_mode"] = "calculated"
        f.fill_from_rcmip()
        f.properties_df.loc[EESC, "input_mode"] = "concentration"
    for specie, value in (feedback or {}).items():
        fill(f.species_configs["forcing_temperature_feedback"], value, specie=specie)
    _climate(f, CLIMATE_3)
    _initialise(f)
    if eesc is not None:
        fill(f.concentration, eesc, specie=EESC)
    f.run(progress=False)
    # Rename the scenario so the frames select it like the other cases
    for attribute in (*TIMEBOUND_ATTRIBUTES, *TIMEPOINT_ATTRIBUTES):
        data = getattr(f, attribute)
        setattr(f, attribute, data.assign_coords(scenario=["scenario"]))
    return f


def scenario(name, scenario_name, **kwargs):
    """Write a run of the default AR6 species driven by an RCMIP scenario."""
    _write(name, _scenario(scenario_name, **kwargs), CLIMATE_3)


def prescribed():
    """Write SSP2-4.5 with prescribed EESC and feedback on the prescribed forcing.

    The EESC is that of the calculated run. Prescribed, it also reaches ERFari.
    """
    feedback = {"Solar": -0.2, "Volcanic": 0.1}
    calculated = _scenario("ssp245")
    eesc = calculated.concentration.sel(specie=EESC).values
    inputs = {
        f"prescribed_forcing|{specie}": calculated.forcing.sel(
            specie=specie, scenario="scenario", config="config"
        ).values
        for specie in feedback
    }
    f = _scenario("ssp245", concentration=(EESC,), eesc=eesc, feedback=feedback)
    _write("ssp245_prescribed", f, CLIMATE_3, extra=inputs)


def _calibration_file(name):
    return pooch.retrieve(
        CALIBRATION_URL.format(name), known_hash=f"md5:{CALIBRATION_FILES[name]}"
    )


def _forcing_file(name, column):
    df = pd.read_csv(_calibration_file(name), index_col=0)
    return df[column] if column else df.iloc[:, 0]


def calibrated():
    """Write deterministic CMIP7 historical runs of fair-calibrate v1.6.1 members.

    fair-calibrate scales the prescribed forcing by its `forcing_scale` outside FaIR,
    so this does the same.
    """
    species_file = _calibration_file("species_configs_properties.csv")
    parameters_file = _calibration_file("calibrated_constrained_parameters.csv")
    emissions = pd.read_csv(
        _calibration_file("historical_emissions_1750-2023_cmip7.csv"),
        index_col="variable",
    )
    years = [str(y) for y in range(1750, 2024)]
    prescribed = {
        "Solar": _forcing_file("solar_forcing_timebounds_cmip7.csv", None),
        "Volcanic": _forcing_file("volcanic_forcing_timebounds_cmip7.csv", None),
        "Land use": _forcing_file("land_use_forcing_timebounds_cmip7.csv", "M"),
        "Irrigation": _forcing_file("irrigation_forcing_timebounds_cmip7.csv", "M"),
    }
    species, properties = read_properties(species_file)
    parameters = pd.read_csv(parameters_file, index_col=0)
    emissions_by_specie = {
        s: emissions.loc[s, years].to_numpy(dtype=float)[:, None, None]
        for s in species
        if properties[s]["input_mode"] == "emissions"
    }
    forcing_by_specie = {
        s: forcing.loc[1750:2024].to_numpy()[:, None, None]
        for s, forcing in prescribed.items()
    }

    out = OUT / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "species_configs_properties.csv").write_bytes(
        Path(species_file).read_bytes()
    )
    lines = Path(parameters_file).read_text().splitlines(keepends=True)
    members = {str(m) for m in CALIBRATION_MEMBERS}
    (out / "calibrated_constrained_parameters.csv").write_text(
        lines[0] + "".join(line for line in lines[1:] if line.split(",")[0] in members)
    )

    for member in CALIBRATION_MEMBERS:
        f = FAIR(ch4_method="thornhill2021")
        f.define_time(1750, 2024, 1)
        f.define_scenarios(["scenario"])
        f.define_configs([member])
        f.define_species(species, properties)
        f.allocate()
        f.fill_species_configs(species_file)
        f.override_defaults(parameters_file)
        fill(f.climate_configs["stochastic_run"], False)
        for specie, values in emissions_by_specie.items():
            fill(f.emissions, values, specie=specie)
        for specie, values in forcing_by_specie.items():
            scale = parameters.loc[member, f"forcing_scale[{specie}]"]
            fill(f.forcing, values * scale, specie=specie)
        _initialise(f)
        f.run(progress=False)
        _write(f"calibrated_{member}", f, CLIMATE_3, member)


def main():
    """Write every case."""
    energy_balance("ebm_3layer", CLIMATE_3, 1)
    energy_balance("ebm_2layer", CLIMATE_2, 1)
    energy_balance("ebm_3layer_5yr", CLIMATE_3, 5)
    scenario(
        "major_ghg_ssp245",
        "ssp245",
        ch4_method="leach2021",
        only=("CO2 FFI", "CO2 AFOLU", "CO2", "CH4", "N2O"),
    )
    scenario("ssp245", "ssp245")
    scenario("ssp119", "ssp119")
    scenario("ssp245_leach", "ssp245", ch4_method="leach2021")
    scenario("ssp245_5yr", "ssp245", step=5)
    halogens = [
        s
        for s, p in read_properties()[1].items()
        if p["type"] in ("f-gas", "other halogen", "cfc-11") and s != "Halon-1202"
    ]
    scenario("ssp245_concentration", "ssp245", concentration=("CH4", *halogens))
    prescribed()
    calibrated()


if __name__ == "__main__":
    main()
