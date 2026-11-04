# FaIR runs

`rscm.fair` is a port of [FaIR](https://github.com/OMS-NetZero/FAIR) v2.2.5 built from RSCM components.
It reproduces FaIR's default methods to a relative tolerance of 1e-9:

- Meinshausen et al. (2020) greenhouse gas forcing.
- The Leach et al. (2021) or Thornhill et al. (2021) CH4 lifetime.
- The n-layer energy balance model, run deterministically.

Variables use RSCM names (see `rscm.fair.naming`) and keep FaIR's units.
FaIR's emissions are on timepoints, one per step.
Its concentrations, forcing and temperatures are on timebounds, one more than the steps.

## Components

`build_components` turns a configuration into components, in the order they must be added to a model:

| Component                | FaIR step                                                    |
| ------------------------ | ------------------------------------------------------------ |
| `FairCH4Lifetime`        | CH4 lifetime scaling (Thornhill 2021 only)                   |
| `FairGasCycle`           | Gas boxes, concentrations, airborne and cumulative emissions |
| `FairEesc`               | Equivalent effective stratospheric chlorine                  |
| `FairGhgForcing`         | Meinshausen 2020 greenhouse gas forcing                      |
| `FairLinearForcing`      | Aerosol-radiation, ozone, contrails, LAPSI, H2O and land use |
| `FairAerosolCloud`       | Aerosol-cloud interactions                                   |
| `FairForcingAggregation` | Total and efficacy-adjusted forcing                          |
| `FairEnergyBalance`      | Layer temperatures, TOA imbalance and ocean heat content     |

A configuration is a plain dict with FaIR's options, climate configs and one entry per species.
`rscm.fair.config` documents its keys.
`naming.exogenous_variable` names the input of each species.
Prescribed forcing is an input such as `Prescribed Effective Radiative Forcing|Solar`.
`FairForcingAggregation` adds its temperature feedback and writes `Effective Radiative Forcing|Solar`.

## Running a calibrated member

`read_calibration` reads fair-calibrate's species CSV and posterior CSV, as FaIR's `override_defaults` does.
fair-calibrate scales prescribed forcing by its `forcing_scale` outside FaIR,
so `scale_prescribed_forcing=True` does the same.

```python
from rscm.fair import naming, read_calibration, run_fair

configs = read_calibration(
    "species_configs_properties.csv",
    "calibrated_constrained_parameters.csv",
)
config = configs[1939]

# emissions[name] has one value per step and forcing[name] one per bound
inputs = {}
for entry in config["species"]:
    variable = naming.exogenous_variable(entry)
    if variable is not None:
        name, unit = variable
        source = emissions if entry["input_mode"] == "emissions" else forcing
        inputs[name] = (source[entry["name"]], unit)

model = run_fair(config, inputs, timebounds, scale_prescribed_forcing=True)
temperature = model.timeseries().get_timeseries_by_name("Surface Temperature")
```

`read_config` reads a species CSV without a posterior, with the climate configs given directly.

## Ensembles

`run_ensemble` runs every configuration on every scenario and returns an `xarray.Dataset` in FaIR's layout,
with `timebounds`, `scenario`, `config`, `specie` and `layer` dimensions:

```python
from rscm.fair import run_ensemble

result = run_ensemble(configs, {"historical": inputs}, timebounds, scale_prescribed_forcing=True)
result.temperature.sel(layer=0, scenario="historical")
```

Each member is its own model, run in a Python loop.
It needs xarray, from the `fair` extra (`pip install 'rscm[fair]'`).

## Converting from FaIR

`from_fair` takes a configured `fair.FAIR` and returns the configuration, inputs and timebounds
of one of its configs and scenarios:

```python
from rscm.fair import from_fair, run_fair

config, inputs, timebounds = from_fair(f, config="member", scenario="ssp245")
model = run_fair(config, inputs, timebounds)
```

## Other climate components

`build_components(config, energy_balance=False)` leaves out `FairEnergyBalance`,
so another component can write `Surface Temperature` from `Effective Radiative Forcing`.
The regression tests couple FaIR's gas cycle and forcing with MAGICC's `ClimateUDEB` this way.
`initial_values` takes the same flag.

## Differences from FaIR

- Stochastic runs are not supported. `read_calibration` ignores the stochastic parameters
  and `from_fair` rejects a stochastic config.
- Only the Meinshausen 2020 greenhouse gas method is implemented.
- `build_components` rejects greenhouse gases with prescribed forcing.
  FaIR 2.2.5 overwrites it with the forcing of the unset concentration, which is NaN,
  so it drops out of the total.
- FaIR calculates EESC after the aerosol forcing, so calculated EESC never reaches ERFari or ERFaci.
  `rscm.fair` leaves it out of those drivers to match. Prescribed EESC does reach them.
- Restarts from a previous run are not supported.

## Parity tests

`tests/regression/test_fair.py` compares every output with FaIR reference runs:
the energy balance alone, emissions-driven, concentration-driven and 5-yearly SSP runs,
an SSP run with prescribed EESC and temperature feedback on the prescribed forcing,
and three fair-calibrate v1.6.1 posterior members over the CMIP7 history.
`scripts/fair/generate_fixtures.py` writes the reference runs:

```bash
uv run --no-project --with fair==2.2.5 --with numpy==2.4.2 \
    python -I scripts/fair/generate_fixtures.py
```

The script disables NumPy's AVX-512 `exp`.
It differs from the C library's by an ulp,
which CO2's 1e9-year gas box turns into a relative difference of about 5e-9.
