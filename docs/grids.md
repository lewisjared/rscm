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
    temp_slice = inputs.temp.current_all_at_start()
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
    fb_slice = inputs.regional_temp.current_all_at_start()  # Returns FourBoxSlice
    no = fb_slice.northern_ocean
    nl = fb_slice.northern_land
    so = fb_slice.southern_ocean
    sl = fb_slice.southern_land

    # Or access individual regions by index
    no = inputs.regional_temp.at_start(region=0)  # northern_ocean

    # Hemispheric input
    hemi_slice = inputs.hemi_temp.current_all_at_start()  # Returns HemisphericSlice
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
    let all_regions = inputs.regional_temp.current_all_at_start();  // Vec<f64>

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
from rscm.core import FourBoxGrid, StateValue

# StateValue wraps grid or scalar values
grid_val = StateValue.four_box(FourBoxSlice.uniform(10.0))

# Aggregate to scalar
global_val = grid_val.to_scalar()  # Weighted average
```

**Rust:**

```rust
use rscm_core::spatial::{FourBoxGrid, SpatialGrid};

let grid = FourBoxGrid::magicc_standard();
let regional_values = vec![15.0, 14.0, 10.0, 9.0];  // [NO, NL, SO, SL]

// Aggregate to global
let global = grid.aggregate_global(&regional_values);  // 12.0 with equal weights
```

### Transformation Between Grids

Some transformations are supported automatically:

| From        | To          | Method                                  |
| ----------- | ----------- | --------------------------------------- |
| FourBox     | Hemispheric | Aggregate (ocean + land per hemisphere) |
| FourBox     | Scalar      | Aggregate (weighted average)            |
| Hemispheric | Scalar      | Aggregate (weighted average)            |
| Scalar      | FourBox     | Broadcast (same value to all regions)   |
| Scalar      | Hemispheric | Broadcast                               |

**Not supported:** Hemispheric -> FourBox (requires physical assumptions about ocean-land split)

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

## Grid Coupling Validation

When building a model, RSCM validates that connected components use compatible grid types:

```python
# This will fail at build time:
class Producer(Component):
    output = Output("Temperature", unit="K", grid="FourBox")

class Consumer(Component):
    input = Input("Temperature", unit="K")  # Scalar - MISMATCH!
```

Error message:

```text
GridTypeMismatch: Variable 'Temperature' has grid type 'FourBox' from Producer
but Consumer expects 'Scalar'
```

**Fix:** Ensure producer and consumer use the same grid type, or add an explicit aggregation component.

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
class HemisphericToFourBox(Component):
    """Disaggregate hemispheric temperature to four-box."""

    hemi_temp = Input("Temperature|Hemispheric", unit="K", grid="Hemispheric")
    fb_temp = Output("Temperature|FourBox", unit="K", grid="FourBox")

    def __init__(self, ocean_land_ratio: float = 1.1):
        self.ocean_land_ratio = ocean_land_ratio

    def solve(self, t_current, t_next, inputs):
        hemi_slice = inputs.hemi_temp.current_all_at_start()
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
