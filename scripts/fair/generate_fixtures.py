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

Usage:
    uv run --no-project --with fair==2.2.5 --with numpy==2.4.2 \
        python -I scripts/fair/generate_fixtures.py

NumPy's AVX-512 ``exp`` differs from the C library's by an ulp,
which FaIR's longest CO2 lifetime box turns into a 1e-8 relative difference.
The script disables it so the runs are the same on every x86 machine.

The RCMIP scenarios are downloaded through pooch on first use.
"""

import json
import math
import os
from pathlib import Path

os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4 AVX512_ICL"

import numpy as np
import pandas as pd
from fair import FAIR, __version__
from fair.interface import fill, initialise
from fair.io import read_properties

OUT = Path(__file__).parents[2] / "tests" / "regression" / "data" / "fair"

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


def _frame(f, attributes, time_dim):
    columns = {}
    for attribute in attributes:
        data = getattr(f, attribute).sel(scenario="scenario", config="config")
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


def _config(f, climate):
    configs = f.species_configs.sel(config="config")
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
    climate_out = {
        name: (
            [_value(v) for v in f.climate_configs[name].sel(config="config").values]
            if f.climate_configs[name].ndim > 1
            else _value(f.climate_configs[name].sel(config="config").values)
        )
        for name in climate
    }
    climate_out["gamma_autocorrelation"] = _value(
        f.climate_configs["gamma_autocorrelation"].sel(config="config").values
    )
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


def _write(name, f, climate):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(_config(f, climate), indent=1) + "\n")
    for attributes, dim, suffix in (
        (TIMEBOUND_ATTRIBUTES, "timebounds", "timebounds"),
        (TIMEPOINT_ATTRIBUTES, "timepoints", "timepoints"),
    ):
        _frame(f, attributes, dim).to_csv(
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


def scenario(  # noqa: PLR0913
    name,
    scenario_name,
    *,
    step=1,
    ch4_method="thornhill2021",
    only=None,
    concentration=(),
):
    """Write a run of the default AR6 species driven by an RCMIP scenario."""
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
    f.fill_from_rcmip()
    _climate(f, CLIMATE_3)
    _initialise(f)
    f.run(progress=False)
    # Rename the scenario so the frames select it like the other cases
    for attribute in (*TIMEBOUND_ATTRIBUTES, *TIMEPOINT_ATTRIBUTES):
        data = getattr(f, attribute)
        setattr(f, attribute, data.assign_coords(scenario=["scenario"]))
    _write(name, f, CLIMATE_3)


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


if __name__ == "__main__":
    main()
