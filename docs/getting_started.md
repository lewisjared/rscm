# Getting Started

This guide walks you through installing RSCM and running your first climate model simulation.

## Installation

Install RSCM from PyPI:

```bash
pip install rscm
```

That's it. No Rust toolchain required.

## Quickstart: Running a Climate Model

Let's run the two-layer climate model to calculate temperature response to radiative forcing.

### Step 1: Import and Set Up

```python
import numpy as np
from rscm.core import (
    ModelBuilder,
    TimeAxis,
    Timeseries,
    InterpolationStrategy,
)
from rscm.two_layer import TwoLayerComponentBuilder
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
# Create a two-layer climate component with default parameters
two_layer = TwoLayerComponentBuilder.from_parameters({
    "lambda_0": 1.2,      # Climate feedback parameter (W/m^2/K)
    "efficacy": 1.0,      # Ocean heat uptake efficacy
    "eta": 0.8,           # Ocean heat transfer coefficient (W/m^2/K)
    "C_upper": 7.0,       # Upper ocean heat capacity (W yr/m^2/K)
    "C_lower": 100.0,     # Deep ocean heat capacity (W yr/m^2/K)
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

## Available Components

RSCM ships with pre-built components for common climate modelling tasks:

| Component | Module | Description |
|-----------|--------|-------------|
| `TwoLayerComponentBuilder` | `rscm.two_layer` | Two-layer energy balance model for temperature |
| `CarbonCycleBuilder` | `rscm.components` | Simple carbon cycle (emissions to concentration) |
| `CO2ERFBuilder` | `rscm.components` | CO2 concentration to effective radiative forcing |

### Coupling Components

Components can be coupled together. Here's an emissions-to-temperature pipeline:

```python
from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.two_layer import TwoLayerComponentBuilder

# Create components
carbon_cycle = CarbonCycleBuilder.from_parameters({...}).build()
co2_erf = CO2ERFBuilder.from_parameters({...}).build()
two_layer = TwoLayerComponentBuilder.from_parameters({...}).build()

# Build coupled model
model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_rust_component(carbon_cycle)
    .with_rust_component(co2_erf)
    .with_rust_component(two_layer)
    .with_exogenous_variable("Emissions|CO2", emissions)
    .with_initial_values({...})
).build()
```

RSCM automatically resolves component dependencies and executes them in the correct order.

## Next Steps

- **[Tutorials](tutorials.md)**: Step-by-step guides for common tasks
- **[Key Concepts](key_concepts.md)**: Understand Components, Models, and Timeseries
- **[Spatial Grids](grids.md)**: Work with regional climate data
- **[API Reference](api/)**: Full Python API documentation

### Want to Extend RSCM?

If you want to create custom components in Python or contribute Rust components, see the [Developer Guide](developers/index.md).
