# Spatial Grids

RSCM supports spatially-resolved variables using a grid system that balances physical realism with computational simplicity. This page explains the available grid types and how to use them.

See the [Python API](api/rscm/) for full class documentation.

## Overview

Climate variables often have spatial structure - temperatures differ between land and ocean, and between hemispheres. RSCM provides three grid types to capture this:

| Grid Type       | Regions         | Use Case                                      |
| --------------- | --------------- | --------------------------------------------- |
| **Scalar**      | 1 (global)      | Well-mixed quantities, global averages        |
| **Hemispheric** | 2 (N/S)         | Latitudinal gradients                         |
| **FourBox**     | 4 (NO/NL/SO/SL) | MAGICC-style ocean-land-hemisphere resolution |

## Grid Types

### Scalar (Global Average)

The default grid type. Represents a single global value.

**Use for:**

- Well-mixed atmospheric gases (CO2, CH4)
- Global mean quantities (total radiative forcing)
- Components that work with global averages

**Python:**

```python
from rscm.component import Input, Output

emissions = Input("Emissions|CO2", unit="GtCO2")  # Scalar by default
forcing = Output("ERF|CO2", unit="W/m^2")  # Scalar by default
```

**Rust:**

```rust
#[inputs(
    emissions { name = "Emissions|CO2", unit = "GtCO2" },  // Scalar by default
)]
#[outputs(
    forcing { name = "ERF|CO2", unit = "W/m^2" },  // Scalar by default
)]
```

### Hemispheric (Northern/Southern)

Two-region grid splitting the globe into Northern and Southern hemispheres.

**Regions:**

- `Northern` - Northern Hemisphere (typically 0-90N)
- `Southern` - Southern Hemisphere (typically 0-90S)

**Use for:**

- Variables with strong latitudinal gradients
- Inter-hemispheric transport
- When ocean-land distinction is less important

**Python:**

```python
from rscm.component import Input, Output
from rscm.core import HemisphericSlice

temp = Input("Temperature|Hemispheric", unit="K", grid="Hemispheric")
response = Output("Response|Hemispheric", unit="W/m^2", grid="Hemispheric")

def solve(self, t_current, t_next, inputs):
    # Access all regions at start of timestep
    temp_slice = inputs.temp.at_start_all()
    temp_n = temp_slice.northern
    temp_s = temp_slice.southern

    return self.Outputs(
        response=HemisphericSlice(
            northern=temp_n * 0.5,
            southern=temp_s * 0.4,
        )
    )
```

**Rust:**

```rust
#[inputs(
    temp { name = "Temperature|Hemispheric", unit = "K", grid = "Hemispheric" },
)]
#[outputs(
    response { name = "Response|Hemispheric", unit = "W/m^2", grid = "Hemispheric" },
)]
```

### FourBox (MAGICC Standard)

Four-region grid used by MAGICC and similar reduced-complexity models.

**Regions:**

| Region         | Code | Description                        |
| -------------- | ---- | ---------------------------------- |
| Northern Ocean | `NO` | Ocean areas in Northern Hemisphere |
| Northern Land  | `NL` | Land areas in Northern Hemisphere  |
| Southern Ocean | `SO` | Ocean areas in Southern Hemisphere |
| Southern Land  | `SL` | Land areas in Southern Hemisphere  |

**Use for:**

- MAGICC-equivalent models
- Capturing ocean-land thermal differences
- Regional climate pattern emulation

**Python:**

```python
from rscm.component import Input, Output
from rscm.core import FourBoxSlice

erf = Input("ERF", unit="W/m^2")
heat_uptake = Output("Heat Uptake", unit="W/m^2", grid="FourBox")

def solve(self, t_current, t_next, inputs):
    erf_val = inputs.erf.at_start()

    return self.Outputs(
        heat_uptake=FourBoxSlice(
            northern_ocean=erf_val * 1.2,
            northern_land=erf_val * 0.6,
            southern_ocean=erf_val * 1.6,
            southern_land=erf_val * 0.6,
        )
    )
```

**Rust:**

```rust
#[inputs(
    erf { name = "ERF", unit = "W/m^2" },
)]
#[outputs(
    heat_uptake { name = "Heat Uptake", unit = "W/m^2", grid = "FourBox" },
)]
pub struct FourBoxHeatUptake {
    pub ocean_ratio: f64,
    pub land_ratio: f64,
}

fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState)
    -> RSCMResult<OutputState>
{
    let inputs = FourBoxHeatUptakeInputs::from_input_state(input_state);
    let erf = inputs.erf.at_start();

    let outputs = FourBoxHeatUptakeOutputs {
        heat_uptake: FourBoxSlice::from_array([
            erf * self.ocean_ratio * 1.0,  // Northern Ocean
            erf * self.land_ratio,          // Northern Land
            erf * self.ocean_ratio * 1.33,  // Southern Ocean
            erf * self.land_ratio,          // Southern Land
        ]),
    };
    Ok(outputs.into())
}
```

## Grid Operations

### Accessing Regional Values

**Python:**

```python
def solve(self, t_current, t_next, inputs):
    # Scalar input - single value at start of timestep
    scalar_val = inputs.emissions.at_start()

    # FourBox input - access all regions at start of timestep
    fb_slice = inputs.regional_temp.at_start_all()  # Returns FourBoxSlice
    no = fb_slice.northern_ocean
    nl = fb_slice.northern_land
    so = fb_slice.southern_ocean
    sl = fb_slice.southern_land

    # Or access individual regions by index
    no = inputs.regional_temp.at_start(region=0)  # northern_ocean

    # Hemispheric input
    hemi_slice = inputs.hemi_temp.at_start_all()  # Returns HemisphericSlice
    northern = hemi_slice.northern
    southern = hemi_slice.southern
```

**Rust:**

```rust
fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState)
    -> RSCMResult<OutputState>
{
    let inputs = MyComponentInputs::from_input_state(input_state);

    // Scalar input
    let scalar_val = inputs.emissions.at_start();

    // FourBox input - access individual regions
    let no = inputs.regional_temp.at_start(FourBoxRegion::NorthernOcean);
    let nl = inputs.regional_temp.at_start(FourBoxRegion::NorthernLand);

    // Or get all at once
    let all_regions = inputs.regional_temp.at_start_all();  // Vec<f64>

    // Global aggregate (weighted average)
    let global_mean = inputs.regional_temp.current_global();
}
```

### Aggregation

Grid values can be aggregated to coarser resolutions:

```text
FourBox -> Hemispheric -> Scalar
```

**Python:**

```python
from rscm.core import FourBoxSlice, StateValue

# StateValue wraps grid or scalar values
grid_val = StateValue.four_box(FourBoxSlice.from_array([15.0, 14.0, 10.0, 9.0]))

# Aggregate to scalar
global_val = grid_val.to_scalar()  # 12.0, the unweighted mean of the regions
```

`StateValue.to_scalar()` ignores grid weights.
Aggregation inside a model uses the configured [grid weights](#configuring-aggregation-weights).

**Rust:**

```rust
use rscm_core::spatial::{FourBoxGrid, SpatialGrid};

let grid = FourBoxGrid::magicc_standard();
let regional_values = vec![15.0, 14.0, 10.0, 9.0];  // [NO, NL, SO, SL]

// Aggregate to global
let global = grid.aggregate_global(&regional_values);  // 12.0 with equal weights
```

### Transformation Between Grids

`SpatialGrid::transform_to` in Rust supports these transformations:

| From        | To          | Method                                  |
| ----------- | ----------- | --------------------------------------- |
| FourBox     | Hemispheric | Aggregate (ocean + land per hemisphere) |
| FourBox     | Scalar      | Aggregate (weighted average)            |
| Hemispheric | Scalar      | Aggregate (weighted average)            |
| Scalar      | FourBox     | Broadcast (same value to all regions)   |
| Scalar      | Hemispheric | Broadcast                               |

**Not supported:** Hemispheric -> FourBox (requires physical assumptions about ocean-land split)

The model only applies the aggregations automatically.
It rejects broadcasts when coupling components,
as described in [Schema-Driven Auto-Aggregation](#schema-driven-auto-aggregation).

**Rust:**

```rust
use rscm_core::spatial::{FourBoxGrid, HemisphericGrid, SpatialGrid};

let four_box = FourBoxGrid::magicc_standard();
let hemispheric = HemisphericGrid::equal_weights();

let fb_values = vec![15.0, 14.0, 10.0, 9.0];

// Transform FourBox to Hemispheric
let hemi_values = four_box.transform_to(&fb_values, &hemispheric)?;
// hemi_values = [14.5, 9.5]  (averages of ocean+land per hemisphere)
```

## Schema-Driven Auto-Aggregation

When using a `VariableSchema`, RSCM can automatically aggregate grid data when components use different resolutions. The schema declares the "native" grid type for each variable, and the model handles transformations transparently.

### How It Works

**Read-side aggregation:** When a component declares an input at a coarser resolution than the schema, the data is automatically aggregated when the component reads it.
This currently only works for Rust components.
Python components receive the window at the schema's resolution.

**Write-side aggregation:** When a component produces output at a finer resolution than the schema declares, the data is automatically aggregated when writing to the collection.

```text
Supported transformations (aggregation only):

FourBox     -> Hemispheric  (aggregate ocean+land per hemisphere)
FourBox     -> Scalar       (weighted average of all regions)
Hemispheric -> Scalar       (weighted average of hemispheres)
```

Disaggregation (broadcasting from coarser to finer grids) is **not supported** because it would require inventing spatial structure.

### Example: FourBox to Scalar Auto-Aggregation

A component producing FourBox output can feed directly into a component expecting Scalar input.
In Python, declare the variable as Scalar in the schema so the producer's output is aggregated on write:

**Python:**

```python
import numpy as np
from rscm.component import Component, Input, Output
from rscm.core import (
    FourBoxSlice,
    GridType,
    ModelBuilder,
    PythonComponent,
    TimeAxis,
    VariableSchema,
)


class RegionalProducer(Component):
    """Produces FourBox temperature data."""

    temp = Output("Surface Temperature", unit="K", grid="FourBox")

    def solve(self, t_current, t_next, inputs):
        # Regional temperatures: [NO, NL, SO, SL]
        return self.Outputs(
            temp=FourBoxSlice.from_array([288.0, 292.0, 285.0, 289.0])
        )


class GlobalConsumer(Component):
    """Consumes scalar (global average) temperature."""

    temp = Input("Surface Temperature", unit="K")  # Scalar input
    response = Output("Climate Response", unit="W/m^2")

    def solve(self, t_current, t_next, inputs):
        # Receives the aggregated global average written in the previous step
        global_temp = inputs.temp.at_start()
        return self.Outputs(response=global_temp * 0.1)


# Schema stores Surface Temperature as Scalar, so FourBox output is aggregated on write
schema = VariableSchema()
schema.add_variable("Surface Temperature", "K", GridType.Scalar)
schema.add_variable("Climate Response", "W/m^2", GridType.Scalar)

model = (
    ModelBuilder()
    .with_time_axis(TimeAxis.from_values(np.arange(2020.0, 2026.0)))
    .with_schema(schema)
    .with_py_component(PythonComponent.build(RegionalProducer()))
    .with_py_component(PythonComponent.build(GlobalConsumer()))
    .build()
)
model.run()

# With the default equal weights: (288 + 292 + 285 + 289) / 4 = 288.5 K
print(model.timeseries().get_timeseries_by_name("Surface Temperature").values())
```

**Rust:**

Rust components also support read-side aggregation.
Here the schema stores the FourBox values and the Scalar consumer reads the aggregate:

```rust
let schema = VariableSchema::new()
    .variable_with_grid("Surface Temperature", "K", GridType::FourBox)
    .variable("Climate Response", "W/m^2");

let model = ModelBuilder::new()
    .with_time_axis(time_axis)
    .with_schema(schema)
    .with_component(Arc::new(regional_producer))
    .with_component(Arc::new(global_consumer))
    .build()?;
```

### Configuring Aggregation Weights

By default, aggregation uses equal weights. For physically meaningful results, configure area-based weights via `ModelBuilder`:

**Python:**

```python
from rscm.core import GridType, ModelBuilder

# Area weights from MAGICC's land fractions (42% of the NH, 21% of the SH)
model = (
    ModelBuilder()
    .with_schema(schema)
    .with_grid_weights(GridType.FourBox, [0.29, 0.21, 0.395, 0.105])
    .with_grid_weights(GridType.Hemispheric, [0.5, 0.5])
    # ... add components
    .build()
)
```

The MAGICC runners in `rscm.magicc` derive the FourBox weights in the same way,
from the ClimateUDEB `nh_land_fraction` and `sh_land_fraction` parameters.
In emissions-driven runs the CH4, N2O and carbon-cycle feedbacks read the FourBox surface temperature as a Scalar.
They therefore see the area-weighted mean.

**Rust:**

```rust
let model = ModelBuilder::new()
    .with_schema(schema)
    .with_grid_weights(GridType::FourBox, vec![0.29, 0.21, 0.395, 0.105])
    .with_grid_weights(GridType::Hemispheric, vec![0.5, 0.5])
    .with_component(...)
    .build()?;
```

**Weight order:**

| Grid Type   | Weight Order                                         |
| ----------- | ---------------------------------------------------- |
| FourBox     | [NorthernOcean, NorthernLand, SouthernOcean, SouthernLand] |
| Hemispheric | [Northern, Southern]                                 |

Weights must sum to 1.0.
Python raises `ValueError` for invalid weights, while the Rust builder panics.

### Introspection

To see which transformations the model will perform, print its debug information:

```python
print(model.debug_info("plain"))
```

The `Grid Transforms` section lists each transformed variable,
for example `Surface Temperature FourBox -> Scalar (write)`.
In Rust, `Model::required_transformations()` returns the same information.

## Gridded Exogenous Data and Initial Values

`FourBoxTimeseries` holds exogenous FourBox data.
Its values have shape `(n_times, 4)` in the order NO, NL, SO, SL.
Pass it to `ModelBuilder.with_exogenous_four_box_variable()`.
It is interpolated onto the model time axis like a scalar `Timeseries`.

`ModelBuilder.with_initial_grid_values()` sets the initial value of a grid state variable,
with one value per region in grid order.
A FourBox state variable can also take a single value from `with_initial_values()`,
which is used for every region.

**Python:**

```python
import numpy as np
from rscm.component import Component, Input, State
from rscm.core import (
    FourBoxSlice,
    FourBoxTimeseries,
    InterpolationStrategy,
    ModelBuilder,
    PythonComponent,
    TimeAxis,
)

time_axis = TimeAxis.from_values(np.arange(2000.0, 2006.0))

# Regional warming that ramps linearly from zero in 2000
temperature = FourBoxTimeseries(
    np.array([[0.0, 0.0, 0.0, 0.0], [1.2, 1.8, 0.8, 1.4]]),
    TimeAxis.from_values(np.array([2000.0, 2005.0])),
    "K",
    InterpolationStrategy.Linear,
)


class OceanHeat(Component):
    temperature = Input("Surface Temperature", unit="K", grid="FourBox")
    heat = State("Ocean Heat", unit="J/m^2", grid="FourBox")

    def solve(self, t_current, t_next, inputs):
        t = inputs.temperature.at_start_all()
        h = inputs.heat.at_start_all()
        return self.Outputs(
            heat=FourBoxSlice.from_array([h[i] + 1e8 * t[i] for i in range(4)])
        )


model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_py_component(PythonComponent.build(OceanHeat()))
    .with_exogenous_four_box_variable("Surface Temperature", temperature)
    .with_initial_grid_values({"Ocean Heat": [0.0, 0.0, 0.0, 0.0]})
).build()
model.run()

heat = model.timeseries().get_fourbox_timeseries_by_name("Ocean Heat")
print(heat.values()[-1])  # [2.4e8, 3.6e8, 1.6e8, 2.8e8]
```

FourBox exogenous data also feeds Rust components that read a Scalar input.
Declare the variable as FourBox in the schema, and the input is aggregated with the grid weights on read.
Here `CarbonCycle` reads the area-weighted mean of the regional temperatures:

```python
from rscm.components import CarbonCycleBuilder
from rscm.core import GridType, Timeseries, VariableSchema

schema = VariableSchema()
schema.add_variable("Surface Temperature", "K", GridType.FourBox)
schema.add_variable("Emissions|CO2|Anthropogenic", "GtC / yr")
schema.add_variable("Atmospheric Concentration|CO2", "ppm")
schema.add_variable("Cumulative Emissions|CO2", "Gt C")
schema.add_variable("Cumulative Land Uptake", "Gt C")

carbon_cycle = CarbonCycleBuilder.from_parameters(
    {"tau": 30.0, "conc_pi": 278.0, "alpha_temperature": 0.1}
).build()
emissions = Timeseries(np.full(6, 10.0), time_axis, "GtC / yr", InterpolationStrategy.Previous)

model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_schema(schema)
    .with_grid_weights(GridType.FourBox, [0.29, 0.21, 0.395, 0.105])
    .with_rust_component(carbon_cycle)
    .with_exogenous_four_box_variable("Surface Temperature", temperature)
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
    .with_initial_values({
        "Atmospheric Concentration|CO2": 278.0,
        "Cumulative Emissions|CO2": 0.0,
        "Cumulative Land Uptake": 0.0,
    })
).build()
model.run()
print(model.debug_info("plain"))  # Lists "Surface Temperature FourBox -> Scalar (read)"
```

There is no Hemispheric equivalent of `FourBoxTimeseries` in Python yet.

## Grid Coupling Validation

When building a model **without a schema**, RSCM validates that connected components use compatible grid types:

```python
# Without schema - this will fail at build time:
class Producer(Component):
    output = Output("Temperature", unit="K", grid="FourBox")

class Consumer(Component):
    input = Input("Temperature", unit="K")  # Scalar - MISMATCH!
```

Error message:

```text
ValueError: Grid type mismatch for variable 'Temperature': producer component 'PythonComponent'
outputs FourBox but consumer component 'PythonComponent' expects Scalar.
```

**Fixes:**

1. **Add a schema** to enable auto-aggregation (recommended)
2. Ensure producer and consumer use the same grid type
3. Add an explicit aggregation component

## Best Practices

### When to Use Each Grid Type

| Scenario                                  | Recommended Grid                        |
| ----------------------------------------- | --------------------------------------- |
| Atmospheric CO2 concentration             | Scalar (well-mixed)                     |
| Regional temperature response             | FourBox                                 |
| Hemispheric temperature gradient          | Hemispheric                             |
| Ocean heat uptake                         | FourBox (ocean-land differences matter) |
| Global radiative forcing                  | Scalar                                  |
| Aerosol forcing (spatially heterogeneous) | FourBox                                 |

### Aggregation Order Matters

For nonlinear processes, the order of aggregation matters:

```python
# Approach 1: Aggregate inputs, compute once
global_temp = aggregate(regional_temps)
response = nonlinear_function(global_temp)

# Approach 2: Compute regionally, aggregate outputs
regional_responses = [nonlinear_function(t) for t in regional_temps]
response = aggregate(regional_responses)

# These may give different results!
```

Use **Approach 2** when regional differences in the nonlinear response are important.

### Custom Disaggregation

If you need to disaggregate from a coarser to finer grid, create an explicit component:

```python
from rscm.component import Component, Input, Output
from rscm.core import FourBoxSlice


class HemisphericToFourBox(Component):
    """Disaggregate hemispheric temperature to four-box."""

    hemi_temp = Input("Temperature|Hemispheric", unit="K", grid="Hemispheric")
    fb_temp = Output("Temperature|FourBox", unit="K", grid="FourBox")

    def __init__(self, ocean_land_ratio: float = 1.1):
        self.ocean_land_ratio = ocean_land_ratio

    def solve(self, t_current, t_next, inputs):
        hemi_slice = inputs.hemi_temp.at_start_all()
        n = hemi_slice.northern
        s = hemi_slice.southern

        return self.Outputs(
            fb_temp=FourBoxSlice(
                northern_ocean=n * self.ocean_land_ratio,
                northern_land=n / self.ocean_land_ratio,
                southern_ocean=s * self.ocean_land_ratio,
                southern_land=s / self.ocean_land_ratio,
            )
        )
```

This makes the disaggregation assumptions explicit and configurable.

## Quick Reference

### Python

| Class              | Description                          |
| ------------------ | ------------------------------------ |
| `FourBoxSlice`     | Container for four regional values   |
| `HemisphericSlice` | Container for two hemispheric values |
| `StateValue`       | Wrap scalar or grid values           |

### Rust

| Type                | Description                            |
| ------------------- | -------------------------------------- |
| `FourBoxSlice`      | `[f64; 4]` wrapper with named access   |
| `HemisphericSlice`  | `[f64; 2]` wrapper with named access   |
| `FourBoxGrid`       | Grid configuration with weights        |
| `HemisphericGrid`   | Grid configuration with weights        |
| `SpatialGrid` trait | `aggregate_global()`, `transform_to()` |
