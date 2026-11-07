# Spatial Grids

RSCM supports spatially-resolved variables,
using a grid system that balances physical realism with computational simplicity.
This page explains the available grid types and how to use them.

See the [Python API](api/rscm/index.md) for full class documentation.

## Overview

Climate variables often have spatial structure.
Temperatures differ between land and ocean, and between hemispheres.
RSCM provides three grid types to capture this:

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

=== "Python"

    ```python
    from rscm.component import Input, Output

    emissions = Input("Emissions|CO2", unit="GtCO2")  # Scalar by default
    forcing = Output("ERF|CO2", unit="W/m^2")  # Scalar by default
    ```

=== "Rust"

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

| Region   | Index | Python attribute | Rust enum                     |
| -------- | ----- | ---------------- | ----------------------------- |
| Northern | 0     | `northern`       | `HemisphericRegion::Northern` |
| Southern | 1     | `southern`       | `HemisphericRegion::Southern` |

**Use for:**

- Variables with strong latitudinal gradients
- Inter-hemispheric transport
- When ocean-land distinction is less important

=== "Python"

    ```python
    from rscm.component import Component, Input, Output
    from rscm.core import HemisphericSlice


    class HemisphericResponse(Component):
        temp = Input("Temperature", unit="K", grid="hemispheric")
        response = Output("Response", unit="W/m^2", grid="hemispheric")

        def solve(self, t_current, t_next, inputs):
            temp = inputs.temp.get()  # HemisphericSlice
            return self.Outputs(
                response=HemisphericSlice(northern=temp.northern * 0.5, southern=temp.southern * 0.4)
            )
    ```

=== "Rust"

    ```rust
    #[inputs(
        temp { name = "Temperature", unit = "K", grid = "hemispheric" },
    )]
    #[outputs(
        response { name = "Response", unit = "W/m^2", grid = "hemispheric" },
    )]
    ```

### FourBox (MAGICC Standard)

Four-region grid used by MAGICC and similar reduced-complexity models.

| Region         | Code | Index | Python attribute | Rust enum                      |
| -------------- | ---- | ----- | ---------------- | ------------------------------ |
| Northern Ocean | `NO` | 0     | `northern_ocean` | `FourBoxRegion::NorthernOcean` |
| Northern Land  | `NL` | 1     | `northern_land`  | `FourBoxRegion::NorthernLand`  |
| Southern Ocean | `SO` | 2     | `southern_ocean` | `FourBoxRegion::SouthernOcean` |
| Southern Land  | `SL` | 3     | `southern_land`  | `FourBoxRegion::SouthernLand`  |

**Use for:**

- MAGICC-equivalent models
- Capturing ocean-land thermal differences
- Regional climate pattern emulation

=== "Python"

    ```python
    from rscm.component import Component, Input, Output
    from rscm.core import FourBoxSlice


    class FourBoxHeatUptake(Component):
        erf = Input("ERF", unit="W/m^2")
        heat_uptake = Output("Heat Uptake", unit="W/m^2", grid="four_box")

        def solve(self, t_current, t_next, inputs):
            erf = inputs.erf.get()
            return self.Outputs(
                heat_uptake=FourBoxSlice(
                    northern_ocean=erf * 1.2,
                    northern_land=erf * 0.6,
                    southern_ocean=erf * 1.6,
                    southern_land=erf * 0.6,
                )
            )
    ```

=== "Rust"

    ```rust
    #[inputs(
        erf { name = "ERF", unit = "W/m^2" },
    )]
    #[outputs(
        heat_uptake { name = "Heat Uptake", unit = "W/m^2", grid = "four_box" },
    )]

    // In solve(), build the output from an array in region order
    let outputs = FourBoxHeatUptakeOutputs {
        heat_uptake: FourBoxSlice::from_array([erf * 1.2, erf * 0.6, erf * 1.6, erf * 0.6]),
    };
    ```

    [`FourBoxOceanHeatUptake`](components/fourboxoceanheatuptake.md) is a complete Rust four-box component.

## Grid Operations

### Accessing Regional Values

A grid input returns every region at once, or one region by index:

=== "Python"

    ```python
    temp = inputs.regional_temp.get()  # FourBoxSlice
    no = temp.northern_ocean
    no = temp.get(0)  # the same value, by index
    ```

=== "Rust"

    ```rust
    let no = inputs.regional_temp.get(FourBoxRegion::NorthernOcean);
    let all_regions = inputs.regional_temp.get_all();  // Vec<f64> in region order
    let global_mean = inputs.regional_temp.grid().aggregate_global(&all_regions);  // weighted by the grid weights
    ```

Hemispheric inputs work the same way, with `HemisphericSlice` and `HemisphericRegion`.

### Aggregation

Grid values can be aggregated to coarser resolutions:

```text
FourBox -> Hemispheric -> Scalar
```

=== "Python"

    ```python
    from rscm.core import FourBoxSlice, StateValue

    grid_val = StateValue.four_box(FourBoxSlice.from_array([15.0, 14.0, 10.0, 9.0]))
    global_val = grid_val.to_scalar()  # 12.0, the unweighted mean of the regions
    ```

    `StateValue.to_scalar()` ignores grid weights.
    Aggregation inside a model uses the configured [grid weights](#configuring-aggregation-weights).

=== "Rust"

    ```rust
    use rscm_core::spatial::{FourBoxGrid, SpatialGrid};

    let grid = FourBoxGrid::magicc_standard();  // equal weights
    let global = grid.aggregate_global(&[15.0, 14.0, 10.0, 9.0]);  // 12.0
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

```rust
use rscm_core::spatial::{FourBoxGrid, HemisphericGrid, SpatialGrid};

let four_box = FourBoxGrid::magicc_standard();
let hemispheric = HemisphericGrid::equal_weights();

let hemi_values = four_box.transform_to(&[15.0, 14.0, 10.0, 9.0], &hemispheric)?;
// [14.5, 9.5], the mean of ocean and land in each hemisphere
```

The model only applies the aggregations automatically.
It rejects broadcasts when coupling components,
as described in [Schema-Driven Auto-Aggregation](#schema-driven-auto-aggregation).

## Schema-Driven Auto-Aggregation

When using a `VariableSchema`,
RSCM can automatically aggregate grid data when components use different resolutions.
The schema declares the "native" grid type for each variable,
and the model handles transformations transparently.

### How It Works

**Read-side aggregation:** When a component declares an input at a coarser resolution than the schema,
the data is automatically aggregated when the component reads it.
This currently only works for Rust components.
Python components receive the window at the schema's resolution.

**Write-side aggregation:** When a component produces output at a finer resolution than the schema declares,
the data is automatically aggregated when writing to the collection.

Only the aggregations in the [transformation table](#transformation-between-grids) apply.
Disaggregation (broadcasting from coarser to finer grids) is **not supported**,
because it would require inventing spatial structure.

### Example: FourBox to Scalar Auto-Aggregation

A component producing four-box output can feed directly into a component expecting Scalar input.

=== "Python"

    In Python, the producer writes four-box values and the consumer reads a Scalar:

    ```python
    import numpy as np
    from rscm.component import Component, Input, Output
    from rscm.core import FourBoxSlice, GridType, ModelBuilder, PythonComponent, TimeAxis, VariableSchema


    class RegionalProducer(Component):
        temp = Output("Surface Temperature", unit="K", grid="four_box")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(temp=FourBoxSlice.from_array([288.0, 292.0, 285.0, 289.0]))


    class GlobalConsumer(Component):
        temp = Input("Surface Temperature", unit="K")
        response = Output("Climate Response", unit="W/m^2")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(response=inputs.temp.get() * 0.1)
    ```

    Declare the variable as Scalar in the schema, so the four-box output is aggregated on write:

    ```python
    schema = VariableSchema()
    schema.add_variable("Surface Temperature", "K", GridType.Scalar)
    schema.add_variable("Climate Response", "W/m^2", GridType.Scalar)

    model = (
        ModelBuilder()
        .with_time_axis(TimeAxis.from_values(np.arange(2020.0, 2026.0)))
        .with_schema(schema)
        .with_component(RegionalProducer())
        .with_component(GlobalConsumer())
        .build()
    )
    model.run()
    ```

    `Surface Temperature` is then 288.5 K from the second step on,
    the equal-weight mean of the four regions.
    The first value is NaN, because components write their outputs to the next index.

=== "Rust"

    Rust components also support read-side aggregation.
    Here the schema stores the four-box values and the Scalar consumer reads the aggregate:

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

By default, aggregation uses equal weights.
For physically meaningful results, configure area-based weights via `ModelBuilder`.
These weights come from MAGICC's land fractions (42% of the NH, 21% of the SH):

=== "Python"

    ```python
    model = (
        ModelBuilder()
        .with_schema(schema)
        .with_grid_weights(GridType.FourBox, [0.29, 0.21, 0.395, 0.105])
        .with_grid_weights(GridType.Hemispheric, [0.5, 0.5])
        # ... add components
        .build()
    )
    ```

=== "Rust"

    ```rust
    let model = ModelBuilder::new()
        .with_schema(schema)
        .with_grid_weights(GridType::FourBox, vec![0.29, 0.21, 0.395, 0.105])
        .with_grid_weights(GridType::Hemispheric, vec![0.5, 0.5])
        .with_component(...)
        .build()?;
    ```

**Weight order:**

| Grid Type   | Weight Order                                               |
| ----------- | ---------------------------------------------------------- |
| FourBox     | [NorthernOcean, NorthernLand, SouthernOcean, SouthernLand] |
| Hemispheric | [Northern, Southern]                                       |

Weights must sum to 1.0.
Python raises `ValueError` for invalid weights, while the Rust builder panics.

`run_magicc` and `run_magicc_emissions` derive their four-box weights from the ClimateUdeb land fractions,
while `run_magicc7` always uses the 42% and 21% defaults.
See [MAGICC Runs](magicc_runs.md).

### Introspection

To see which transformations the model will perform, print its coupling:

```python
print(model.coupling("plain"))
```

The `Grid Transforms` section lists each transformed variable,
for example `Surface Temperature four_box -> scalar (write)`.
In Rust, `Model::required_transformations()` returns the same information.

## Gridded Prescribed Data and Initial Values

`FourBoxTimeseries` holds prescribed four-box data.
Its values have shape `(n_times, 4)` in region order.
Pass it to `ModelBuilder.with_prescribed_variable()`.
It is interpolated onto the model time axis like a scalar `Timeseries`.

`ModelBuilder.with_initial_grid_values()` sets the initial value of a grid state variable,
with one value per region in grid order.
A four-box state variable can also take a single value from `with_initial_values()`,
which is used for every region.

```python
import numpy as np
from rscm.component import Component, Input, State
from rscm.core import FourBoxSlice, FourBoxTimeseries, InterpolationStrategy
from rscm.core import ModelBuilder, PythonComponent, TimeAxis

time_axis = TimeAxis.from_values(np.arange(2000.0, 2006.0))

# Regional warming that ramps linearly from zero in 2000
temperature = FourBoxTimeseries(
    np.array([[0.0, 0.0, 0.0, 0.0], [1.2, 1.8, 0.8, 1.4]]),
    TimeAxis.from_values(np.array([2000.0, 2005.0])),
    "K",
    InterpolationStrategy.Linear,
)


class OceanHeat(Component):
    temperature = Input("Surface Temperature", unit="K", grid="four_box")
    heat = State("Ocean Heat", unit="J/m^2", grid="four_box")

    def solve(self, t_current, t_next, inputs):
        t = inputs.temperature.get()
        h = inputs.heat.get()
        return self.Outputs(heat=FourBoxSlice.from_array([h[i] + 1e8 * t[i] for i in range(4)]))


model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_component(OceanHeat())
    .with_prescribed_variable("Surface Temperature", temperature)
    .with_initial_grid_values({"Ocean Heat": [0.0, 0.0, 0.0, 0.0]})
).build()
model.run()

heat = model.timeseries().get_four_box_timeseries_by_name("Ocean Heat")
print(heat.values()[-1])
```

```text
[2.4e+08 3.6e+08 1.6e+08 2.8e+08]
```

Four-box prescribed data also feeds Rust components that read a Scalar input.
Declare the variable as FourBox in the schema, and the input is aggregated with the grid weights on read.
Here the carbon cycle from [Getting Started](getting_started.md#coupling-components),
built with a nonzero `alpha_temperature`, reads the area-weighted mean of the regional temperatures:

```python
schema = VariableSchema()
schema.add_variable("Surface Temperature", "K", GridType.FourBox)
# ... the carbon-cycle variables, as in Getting Started

model = (
    ModelBuilder()
    .with_schema(schema)
    .with_grid_weights(GridType.FourBox, [0.29, 0.21, 0.395, 0.105])
    .with_component(carbon_cycle)
    .with_prescribed_variable("Surface Temperature", temperature)
    # ... time axis, emissions and initial values, as in Getting Started
).build()
```

`model.coupling("plain")` then lists `Surface Temperature four_box -> scalar (read)`.

There is no Hemispheric equivalent of `FourBoxTimeseries` in Python yet.

## Grid Coupling Validation

When building a model **without a schema**,
RSCM validates that connected components use compatible grid types:

```python
# Without schema - this will fail at build time:
class Producer(Component):
    output = Output("Temperature", unit="K", grid="four_box")

class Consumer(Component):
    input = Input("Temperature", unit="K")  # Scalar - MISMATCH!
```

Error message:

```text
ValueError: Grid type mismatch for variable 'Temperature': producer component 'Producer'
outputs four_box but consumer component 'Consumer' expects scalar.
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

    hemi_temp = Input("Temperature", unit="K", grid="hemispheric")
    fb_temp = Output("Regional Temperature", unit="K", grid="four_box")

    def __init__(self, ocean_land_ratio: float = 1.1):
        self.ocean_land_ratio = ocean_land_ratio

    def solve(self, t_current, t_next, inputs):
        hemi = inputs.hemi_temp.get()
        return self.Outputs(
            fb_temp=FourBoxSlice(
                northern_ocean=hemi.northern * self.ocean_land_ratio,
                northern_land=hemi.northern / self.ocean_land_ratio,
                southern_ocean=hemi.southern * self.ocean_land_ratio,
                southern_land=hemi.southern / self.ocean_land_ratio,
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
