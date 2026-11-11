# FaIR runs

`rscm.fair` is a port of [FaIR](https://github.com/OMS-NetZero/FAIR) v2.2.5 built from RSCM components.
It reproduces FaIR's default methods to a relative tolerance of 1e-9:

- Meinshausen et al. (2020) greenhouse gas forcing.
- The Leach et al. (2021) or Thornhill et al. (2021) CH4 lifetime.
- The n-layer energy balance model, run deterministically.

Variables use RSCM names (see `rscm.fair.naming`) and keep FaIR's units.
FaIR's emissions are on timepoints, one per step.
Its concentrations, forcing and temperatures are on timebounds, one more than the steps.
The runners take a time axis with a step starting at each timebound, `TimeAxis.from_values(timebounds)`.

## Components

`component_specs` gives the components of a configuration by name, in the order they are added to a model,
and `components` builds them.
Each `LinearForcing` is named by its species, such as `LinearForcing[Ozone]`.

| Component            | FaIR step                                                    |
| -------------------- | ------------------------------------------------------------ |
| `CH4Lifetime`        | CH4 lifetime scaling (Thornhill 2021 only)                   |
| `GasCycle`           | Gas boxes, concentrations, airborne and cumulative emissions |
| `EESC`               | Equivalent effective stratospheric chlorine                  |
| `GHGForcing`         | Meinshausen 2020 greenhouse gas forcing                      |
| `LinearForcing`      | Aerosol-radiation, ozone, contrails, LAPSI, H2O and land use |
| `AerosolCloud`       | Aerosol-cloud interactions                                   |
| `ForcingAggregation` | Total and efficacy-adjusted forcing                          |
| `EnergyBalance`      | Layer temperatures, TOA imbalance and ocean heat content     |

A configuration is a plain dict with FaIR's options, climate configs and one entry per species.
`rscm.fair.config` documents its keys.
`naming.prescribed_variable` names the input of each species.
Prescribed forcing is an input such as `Prescribed Effective Radiative Forcing|Solar`.
`ForcingAggregation` adds its temperature feedback and writes `Effective Radiative Forcing|Solar`.
It also writes the greenhouse gas, aerosol and anthropogenic ERF with MAGICC7's definitions,
from each agent's `category`
(see [forcing aggregation](modules/module_07b_rf_aggregation.md#reported-subtotals)).

## Running a calibrated member

`read_calibration` reads fair-calibrate's species CSV and posterior CSV, as FaIR's `override_defaults` does.
fair-calibrate scales prescribed forcing by its `forcing_scale` outside FaIR,
so `scale_prescribed_forcing=True` does the same.

```python
from rscm.core import TimeAxis
from rscm.fair import naming, read_calibration, setup_model

configs = read_calibration(
    "species_configs_properties.csv",
    "calibrated_constrained_parameters.csv",
)
config = configs[1939]

# emissions[name] has one value per step and forcing[name] one per bound
inputs = {}
for entry in config["species"]:
    variable = naming.prescribed_variable(entry)
    if variable is not None:
        name, unit = variable
        source = emissions if entry["input_mode"] == "emissions" else forcing
        inputs[name] = (source[entry["name"]], unit)

model = setup_model(
    config, inputs, TimeAxis.from_values(timebounds), scale_prescribed_forcing=True
).run()
temperature = model.timeseries().get_timeseries_by_name("Surface Air Temperature Change")
```

`read_config` reads a species CSV without a posterior, with the climate configs given directly.

## Ensembles

`run_ensemble` runs every configuration on every scenario and returns an `xarray.Dataset` in FaIR's layout,
with `timebounds`, `scenario`, `config`, `specie` and `layer` dimensions:

```python
from rscm.fair import run_ensemble

result = run_ensemble(
    configs,
    {"historical": inputs},
    TimeAxis.from_values(timebounds),
    scale_prescribed_forcing=True,
)
result.temperature.sel(layer=0, scenario="historical")
```

Each member is its own model.
The members of each scenario run in parallel, on `n_threads` threads or one per core.
It needs xarray, from the `fair` extra (`pip install 'rscm[fair]'`).

`rscm.spec.run_all` runs any list of setups in parallel and returns the models,
for results in another layout:

```python
from rscm.spec import run_all

setups = [setup_model(c, inputs, time_axis) for c in configs.values()]
models = run_all(setups)
```

## Converting from FaIR

`from_fair` takes a configured `fair.FAIR` and returns the configuration, inputs and time axis
of one of its configs and scenarios:

```python
from rscm.fair import from_fair, setup_model

config, inputs, time_axis = from_fair(f, config="member", scenario="ssp245")
model = setup_model(config, inputs, time_axis).run()
```

## Inspecting a model before building

`setup_model` returns a `rscm.spec.ModelSetup`:
the components with their parameters, FaIR's lags, the initial values and the prescribed data.

```python
from rscm.fair import setup_model
from rscm.spec import diff

setup = setup_model(config, inputs, time_axis)
print(setup)
setup.spec.components["EnergyBalance"].parameters
setup.prescribed_inputs()  # inputs no component produces, and whether each is supplied
model = setup.run()
```

`diff(setup, other)` shows what changes between two setups,
such as the species parameters and inputs that switch when a gas becomes concentration-driven.
`setup.build()` returns the unrun model and `setup.run()` the run one.

## Other climate components

`components(config, energy_balance=False)` leaves out `EnergyBalance`,
so another component can write `Surface Air Temperature Change` from `Effective Radiative Forcing`.
The regression tests couple FaIR's gas cycle and forcing with MAGICC's `ClimateUDEB` this way.
`initial_values` takes the same flag.

## Validation

`setup_model`, `components`, `run_ensemble`, `read_calibration` and `from_fair`
check the configuration and inputs before building a model.
They collect every problem and raise one `rscm.validation.InputValidationError`,
whose issues each name a kind and a location:
a configuration key such as `climate.ocean_heat_capacity`,
a species parameter such as `HFC-23.forcing_reference_concentration`,
or an input variable.
`rscm.fair.validation.validate(config, inputs, time_axis)` runs the same checks on their own.

They reject:

- Unknown configuration keys and calibration columns (`unknown-key`).
- Fewer than two ocean layers, non-positive heat capacities or mismatched layer lengths.
- NaN or infinite climate parameters, and species parameters that a calculation reads,
  such as the CH4 lifetime sensitivity of each precursor
  or the `forcing_scale` of prescribed forcing when `scale_prescribed_forcing` applies it.
  Terms a calculation skips, such as a zero coefficient, are not checked.
- Species a calculation needs but the configuration lacks (`missing`):
  calculated CO2 without CO2 FFI and CO2 AFOLU in emissions mode,
  land use without CO2 AFOLU, calculated EESC without CFC-11,
  and Meinshausen 2020 forcing without CO2, CH4 and N2O.
- Missing inputs, inputs of the wrong length and a time axis with fewer than two bounds.
- NaN or infinite inputs, apart from emissions at the last bound, which no step reads.
- Options RSCM lacks (`unsupported`), listed under the differences below.

`error.exit_code` follows `sysexits.h`, so a script can `sys.exit(error.exit_code)`:

| Kind | Exit code |
| --- | --- |
| `non-finite`, `out-of-range`, `shape` | 65 (`EX_DATAERR`) |
| `missing`, `unknown-key` | 78 (`EX_CONFIG`) |
| `unsupported` | 69 (`EX_UNAVAILABLE`) |

The error takes the exit code of its first issue.

## Differences from FaIR

- Only the Meinshausen 2020 greenhouse gas method is implemented.
  `leach2021`, `etminan2016` and `myhre1998` are rejected as unsupported.
- Greenhouse gases with prescribed forcing are rejected as unsupported.
  FaIR 2.2.5 overwrites it with the forcing of the unset concentration, which is NaN,
  so it drops out of the total.
- Like FaIR's `np.nansum`, ERFari, ERFaci, the minor linear forcings and the total skip a NaN term.
  Ozone forcing is a plain sum, as in FaIR, so a NaN term makes it NaN.
  Validation rejects NaN parameters and inputs before the run, so this only matters
  for terms a calculation drops, such as a species without a coefficient.
- FaIR calculates EESC after the aerosol forcing, so calculated EESC never reaches ERFari or ERFaci.
  `rscm.fair` leaves it out of those drivers to match. Prescribed EESC does reach them.
- `from_fair` reads species properties from `properties_df`, as FaIR runs from it,
  and copies FaIR's arrays.
  It rejects as unsupported what RSCM cannot reproduce:
    - A stochastic config.
    - `temperature_prescribed`.
    - A `ghg_forcing_offset` other than FaIR's default.
    - An initial state other than FaIR's default, such as a nonzero temperature,
      cumulative emissions, gas partitions or a concentration away from the baseline.
      An uninitialised (NaN) initial state is rejected as non-finite.
    - A FAIR that has already run with a temperature feedback on prescribed forcing,
      because FaIR adds the feedback to that forcing in place.
- `run_ensemble` starts cumulative emissions from 0, the only initial value `from_fair` accepts.
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
which CO2's 1e9-year gas box turns into a relative difference
that reaches 4.8e-7 over 2500-year runs.
