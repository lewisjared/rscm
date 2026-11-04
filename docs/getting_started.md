# Getting Started

This guide walks you through installing RSCM and running your first climate model simulation.
See the [Python API](api/rscm/) for full class documentation.

## Installation

Install RSCM from PyPI:

```bash
pip install rscm
```

The wheels include the compiled Rust extension, so no Rust toolchain is required.

Some features described in these docs are newer than the latest PyPI release.
These include the MAGICC components and runners, such as `run_magicc7`, and the configuration layer.
To use them, build from source as described in the [Developer Guide](developers/index.md).

## Quickstart: Running a Climate Model

Let's run the two-layer climate model to calculate the temperature response to radiative forcing.
The plotting step uses matplotlib, which is not an RSCM dependency.

### Step 1: Import and Set Up

```python
import numpy as np
from rscm.core import (
    ModelBuilder,
    TimeAxis,
    Timeseries,
    InterpolationStrategy,
)
from rscm.two_layer import TwoLayerBuilder
```

### Step 2: Define the Time Axis

The time axis specifies when the model runs:

```python
# Annual timesteps from 1850 to 2100
time_axis = TimeAxis.from_values(np.arange(1850.0, 2101.0, 1.0))
```

### Step 3: Create Input Data

Provide radiative forcing as an exogenous (external) timeseries:

```python
# Simple forcing scenario: 0 before 1950, then ramping to 4 W/m^2
forcing = Timeseries(
    values=np.array([0.0, 0.0, 2.0, 4.0]),
    time_axis=TimeAxis.from_bounds(np.array([1850.0, 1950.0, 2000.0, 2050.0, 2100.0])),
    units="W/m^2",
    interpolation_strategy=InterpolationStrategy.Linear,
)
```

### Step 4: Build and Run the Model

```python
# Create a two-layer climate component. All six parameters are required.
two_layer = TwoLayerBuilder.from_parameters({
    "lambda0": 1.2,                # Climate feedback parameter (W/m^2/K)
    "a": 0.0,                      # Nonlinear feedback coefficient (W/m^2/K^2)
    "efficacy": 1.0,               # Ocean heat uptake efficacy
    "eta": 0.8,                    # Heat exchange coefficient (W/m^2/K)
    "heat_capacity_surface": 7.0,  # Surface layer heat capacity (W yr/m^2/K)
    "heat_capacity_deep": 100.0,   # Deep ocean heat capacity (W yr/m^2/K)
}).build()

# Build and run the model
model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_rust_component(two_layer)
    .with_exogenous_variable("Effective Radiative Forcing", forcing)
    .with_initial_values({
        "Surface Temperature": 0.0,
        "Deep Ocean Temperature": 0.0,
    })
).build()

model.run()
```

`build()` raises an exception if the components and inputs are inconsistent.
`run()` raises `RuntimeError` if a component fails during a timestep.
Rebuild the model before retrying a failed run, because component state may already have changed.

### Step 5: Analyse Results

```python
# Extract results
results = model.timeseries()
temperature = results.get_timeseries_by_name("Surface Temperature")

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(time_axis.values(), temperature.values())
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (K)")
plt.title("Surface Temperature Response to Forcing")
plt.grid(True)
plt.show()
```

## Available Models and Components

RSCM ships with pre-built components for common climate modelling tasks:

| Component                       | Module            | Description                                      |
| ------------------------------- | ----------------- | ------------------------------------------------ |
| `TwoLayerBuilder`               | `rscm.two_layer`  | Two-layer energy balance model for temperature   |
| `CarbonCycleBuilder`            | `rscm.components` | One-box carbon cycle (emissions to concentration) |
| `CO2ERFBuilder`                 | `rscm.components` | CO2 concentration to effective radiative forcing |
| `FourBoxOceanHeatUptakeBuilder` | `rscm.components` | Example four-box disaggregation of scalar forcing |

`rscm.magicc` provides builders for the MAGICC7 components:

- Chemistry: `CH4Chemistry`, `CH4Oxidation`, `N2OChemistry` and `HalocarbonChemistry`.
- Carbon cycle: `TerrestrialCarbon` (with optional nitrogen limitation), `OceanCarbon`, `CO2Budget`
  and `Permafrost`.
- Forcing: `GhgForcing`, `OzoneForcing`, `AerosolDirect`, `AerosolIndirect` and `ForcingAggregation`.
- Climate: the `ClimateUDEB` upwelling-diffusion energy balance model and `SeaLevelRise`.

`rscm.magicc.full.run_magicc7` couples all of them into the full MAGICC7 chain.
It uses MAGICC7 v7.5.3's packaged defaults and SSP245 input files,
and matches MAGICC7's output on its default SSP245 run.
`rscm.magicc.prescribed` reproduces the aerosol, land-use, solar and volcanic forcing
that MAGICC7 derives from its input files.
`rscm.magicc.run_magicc` and `rscm.magicc.run_magicc_emissions` are smaller runners
for concentration-driven and emissions-driven experiments.

See [MAGICC Runs](magicc_runs.md#full-magicc7-chain) for usage and agreement with MAGICC7.
The [MAGICC Coupled Model](notebooks/magicc_coupled_model.py) tutorial wires the components together by hand.
See the [MAGICC module specs](modules/README.md) for the implementation status of each module.

The [Components](components/index.md) reference lists the inputs, outputs and parameters of every component.

### Coupling Components

Components can be coupled together. Here's an emissions-to-temperature pipeline.
`CO2ERF` writes `Effective Radiative Forcing|CO2`, but `TwoLayer` reads `Effective Radiative Forcing`.
A [`VariableSchema`][rscm.core.VariableSchema] aggregate connects the two:

```python
from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import VariableSchema

emissions = Timeseries(
    values=np.array([0.5, 10.0, 2.0]),
    time_axis=TimeAxis.from_bounds(np.array([1850.0, 2020.0, 2100.0, 2101.0])),
    units="GtC / yr",
    interpolation_strategy=InterpolationStrategy.Linear,
)

carbon_cycle = CarbonCycleBuilder.from_parameters(
    {"tau": 30.0, "conc_pi": 278.0, "alpha_temperature": 0.0}
).build()
co2_erf = CO2ERFBuilder.from_parameters({"erf_2xco2": 3.7, "conc_pi": 278.0}).build()

schema = VariableSchema()
schema.add_variable("Emissions|CO2|Anthropogenic", "GtC / yr")
schema.add_variable("Atmospheric Concentration|CO2", "ppm")
schema.add_variable("Cumulative Emissions|CO2", "Gt C")
schema.add_variable("Cumulative Land Uptake", "Gt C")
schema.add_variable("Effective Radiative Forcing|CO2", "W/m^2")
schema.add_variable("Surface Temperature", "K")
schema.add_variable("Deep Ocean Temperature", "K")
schema.add_aggregate(
    "Effective Radiative Forcing", "W/m^2", "Sum", ["Effective Radiative Forcing|CO2"]
)

# Build the coupled model, adding producers before their consumers
model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_schema(schema)
    .with_rust_component(carbon_cycle)
    .with_rust_component(co2_erf)
    .with_rust_component(two_layer)
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
    .with_initial_values({
        "Atmospheric Concentration|CO2": 278.0,
        "Cumulative Emissions|CO2": 0.0,
        "Cumulative Land Uptake": 0.0,
        "Surface Temperature": 0.0,
        "Deep Ocean Temperature": 0.0,
    })
).build()
model.run()
```

RSCM builds a dependency graph from the components,
and solves producers before their consumers within each timestep.
A component only depends on producers that were added before it.
Add components in producer-to-consumer order so that each one reads values from the same timestep.
See [Key Concepts](key_concepts.md#dependency-graph) for details.

## Next Steps

- **[Tutorials](tutorials.md)**: Step-by-step guides for common tasks
- **[Key Concepts](key_concepts.md)**: Understand Components, Models, and Timeseries
- **[Coupled Models](notebooks/coupled_model.py)**: Build a feedback-coupled carbon cycle and climate model
- **[Variable Schemas](notebooks/variable_schema.py)**: Aggregate outputs from multiple components (e.g. sum forcing agents into totals)
- **[MAGICC Runs](magicc_runs.md)**: Run the full MAGICC7 chain or smaller MAGICC runners on SSP scenarios
- **[MAGICC Coupled Model](notebooks/magicc_coupled_model.py)**: Couple the MAGICC components yourself
- **[Spatial Grids](grids.md)**: Work with regional climate data
- **[Configuration Layer](model_configuration.md)**: Define models in TOML files
- **[Python API](api/rscm/)**: Full API documentation

### Want to Extend RSCM?

If you want to create custom components in Python or contribute Rust components, see the [Developer Guide](developers/index.md).
