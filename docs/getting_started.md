# Getting Started

This guide walks you through installing RSCM and running your first climate model simulation.
See the [Python API](api/rscm/index.md) for full class documentation.

## Installation

Install RSCM from PyPI:

```bash
pip install rscm
```

The wheels include the compiled Rust extension, so no Rust toolchain is required.
To work on RSCM itself, build from source as described in the [Developer Guide](developers/index.md).

Optional extras:

- `pip install 'rscm[fair]'` adds xarray, for FaIR ensembles.
- The tutorial notebooks draw component graphs with [Graphviz](https://graphviz.org/download/).
  Install it with your system package manager, for example `apt install graphviz` or `brew install graphviz`.
  Without it the notebooks still run and skip the graphs.

## Quickstart: a temperature projection

This runs the two-layer energy balance model with its default parameters
under a forcing ramp from 0 to 4 W/m^2 over 1850 to 2100, and plots the warming.
The plot uses matplotlib, which is not an RSCM dependency.

```python
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
from rscm.config.models.two_layer import TwoLayerParameters
from rscm.core import InterpolationStrategy, ModelBuilder, TimeAxis, Timeseries
from rscm.two_layer import TwoLayerBuilder

time_axis = TimeAxis.from_values(np.arange(1850.0, 2101.0))
forcing = Timeseries(
    np.linspace(0.0, 4.0, len(time_axis)), time_axis, "W/m^2", InterpolationStrategy.Linear
)
two_layer = TwoLayerBuilder.from_parameters(asdict(TwoLayerParameters())).build()

model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_component(two_layer)
    .with_prescribed_variable("Effective Radiative Forcing", forcing)
    .with_initial_values({"Surface Air Temperature Change": 0.0, "Ocean Temperature|Layer 2": 0.0})
    .build()
)
model.run()

temperature = model.timeseries().get_timeseries_by_name("Surface Air Temperature Change")
print(f"{temperature.values()[-1]:.2f} K in 2100")
plt.plot(time_axis.values(), temperature.values())
```

```text
2.89 K in 2100
```

What each step does:

- `TimeAxis` sets the years the model steps through, here annual from 1850 to 2100.
- `Timeseries` holds the prescribed forcing.
  `InterpolationStrategy` controls how it is read between its points.
- [`TwoLayerParameters`][rscm.config.models.two_layer.TwoLayerParameters] holds the default parameters.
  Override any of them, for example `TwoLayerParameters(lambda0=1.2)`.
  `TwoLayerBuilder.from_parameters` needs all six, so start from the defaults rather than a hand-written dict.
- `ModelBuilder` couples the components, the prescribed data and an initial value for every state variable.
  `build()` raises an exception if they are inconsistent.
- `run()` raises `RuntimeError` if a component fails during a step.
  Rebuild the model before retrying a failed run, because component state may already have changed.
- `model.timeseries()` returns every variable in the model as a `TimeseriesCollection`.

## Available Models and Components

RSCM ships with pre-built components for common climate modelling tasks:

| Component                       | Module            | Description                                       |
| ------------------------------- | ----------------- | ------------------------------------------------- |
| `TwoLayerBuilder`               | `rscm.two_layer`  | Two-layer energy balance model for temperature    |
| `CarbonCycleBuilder`            | `rscm.components` | One-box carbon cycle (emissions to concentration) |
| `CO2ERFBuilder`                 | `rscm.components` | CO2 concentration to effective radiative forcing  |
| `FourBoxOceanHeatUptakeBuilder` | `rscm.components` | Example four-box disaggregation of scalar forcing |

`rscm.magicc` reimplements MAGICC7's modules,
and `rscm.magicc.emissions_driven` and `rscm.magicc.concentration_driven` build the full chain.
See [MAGICC Runs](magicc_runs.md) to run it, and the [MAGICC module specs](modules/README.md) for each module.

The [Components](components/index.md) reference lists the inputs, outputs and parameters of every component.

### Coupling Components

Components can be coupled together.
This pipeline runs from CO2 emissions to temperature,
and reuses `time_axis` and the `two_layer` component from the Quickstart:

```python
from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import VariableSchema

emissions = Timeseries(
    values=np.array([0.5, 10.0, 2.0]),
    time_axis=TimeAxis.from_bounds(np.array([1850.0, 2020.0, 2100.0, 2101.0])),
    unit="GtC/yr",
    interpolation_strategy=InterpolationStrategy.Linear,
)

carbon_cycle = CarbonCycleBuilder.from_parameters(
    {"tau": 30.0, "co2_pi": 278.0, "alpha_temperature": 0.0}
).build()
co2_erf = CO2ERFBuilder.from_parameters({"erf_2xco2": 3.7, "co2_pi": 278.0}).build()
```

The `CO2ERFBuilder` component writes `Effective Radiative Forcing|CO2`,
but the `TwoLayerBuilder` component reads `Effective Radiative Forcing`.
A [`VariableSchema`][rscm.core.VariableSchema] declares every variable the components use,
plus an aggregate that connects the two:

```python
schema = VariableSchema()
schema.add_variable("Emissions|CO2", "GtC/yr")
schema.add_variable("Atmospheric Concentrations|CO2", "ppm")
schema.add_variable("Cumulative Emissions|CO2", "Gt C")
schema.add_variable("Cumulative Carbon Uptake|Land", "Gt C")
schema.add_variable("Effective Radiative Forcing|CO2", "W/m^2")
schema.add_variable("Surface Air Temperature Change", "K")
schema.add_variable("Ocean Temperature|Layer 2", "K")
schema.add_aggregate(
    "Effective Radiative Forcing", "W/m^2", "sum", ["Effective Radiative Forcing|CO2"]
)
```

The builder chain takes the schema, the components, the emissions and an initial value for every state variable.
Temperature feeds back on the carbon cycle, so the carbon cycle reads the previous step's temperature:

```python
model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_schema(schema)
    .with_component(carbon_cycle)
    .with_component(co2_erf)
    .with_component(two_layer)
    .with_lag_for("Surface Air Temperature Change", "CarbonCycle")
    .with_prescribed_variable("Emissions|CO2", emissions)
    .with_initial_values({
        "Atmospheric Concentrations|CO2": 278.0,
        "Cumulative Emissions|CO2": 0.0,
        "Cumulative Carbon Uptake|Land": 0.0,
        "Surface Air Temperature Change": 0.0,
        "Ocean Temperature|Layer 2": 0.0,
    })
).build()
model.run()
```

RSCM builds a dependency graph from the components,
and solves producers before their consumers within each step.
The order you add components in does not matter.
A feedback loop must lag one of its edges with `with_lag` or `with_lag_for`, or `build()` fails.
See [Key Concepts](key_concepts.md#dependency-graph) for details.

## Common errors

**Missing time axis.**
Always call `with_time_axis` before `build()`.
Without one, the model does not run on the years of your data.

**Misspelt variable names.**
A misspelt prescribed variable leaves the input it was meant for unsupplied,
so `build()` raises a `ValueError` that names the variable.
Compare the name in the message with the one you passed.
`ModelBuilder.prescribed_inputs()` lists every input no component produces, with its unit and consumers,
and the [Components](components/index.md) reference lists the names each component reads and writes.

**Missing parameters.**
`from_parameters` needs every parameter, and its `ValueError` names what is missing.
Start from the defaults in `TwoLayerParameters`, as the quickstart does,
or from the component's entry in the [Components](components/index.md) reference.

**Integer time arrays.**
Pass float arrays to `TimeAxis.from_values` and `TimeAxis.from_bounds`,
such as `np.arange(1850.0, 2101.0)` rather than `np.arange(1850, 2101)`.

**Graph rendering in the notebooks.**
`model.as_dot()` returns plain DOT text and needs nothing extra.
Rendering it to an image needs the Graphviz `dot` binary, see [Installation](#installation).

## Next Steps

- **[Tutorials](tutorials.md)**: Step-by-step guides for common tasks
- **[Key Concepts](key_concepts.md)**: Understand Components, Models, and Timeseries
- **[Coupled Models](notebooks/coupled_model.py)**: Build a feedback-coupled carbon cycle and climate model
- **[Variable Schemas](notebooks/variable_schema.py)**: Aggregate outputs from multiple components
  (e.g. sum forcing agents into totals)
- **[MAGICC Runs](magicc_runs.md)**: Run the emissions-driven or concentration-driven MAGICC7 model
  from your own emissions
- **[MAGICC Coupled Model](notebooks/magicc_coupled_model.py)**: Couple the MAGICC components yourself
- **[Spatial Grids](grids.md)**: Work with regional climate data
- **[Configuration Layer](model_configuration.md)**: Define models in TOML files
- **[Python API](api/rscm/index.md)**: Full API documentation

### Want to Extend RSCM?

If you want to write components in Python or contribute Rust components,
see the [Developer Guide](developers/index.md).
