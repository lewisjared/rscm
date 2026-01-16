# Components in Rust

This guide demonstrates how to implement climate model components in Rust using the `ComponentIO` derive macro.

## Overview

Components are the building blocks of RSCM models. Each component:

- Declares its inputs, outputs, and state variables via the `ComponentIO` derive macro
- Implements the `Component` trait with `definitions()` and `solve()` methods
- Can be coupled with other components in a model

## Related Resources

- [Components in Python](component_python.py): Creating components in Python
- [Coupled Models](coupled_model.py): Combining multiple components
- [Key Concepts](../key_concepts.md): Core RSCM architecture
- [Python API](../api/rscm/): Complete API documentation

## Creating a Component

Use the `ComponentIO` derive macro with struct-level attributes to declare inputs and outputs:

```rust
use rscm_core::component::{Component, InputState, OutputState, RequirementDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Parameters for the CO2 ERF component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CO2ERFParameters {
    /// ERF due to a doubling of atmospheric CO2 concentrations (W/m^2)
    pub erf_2xco2: FloatValue,
    /// Pre-industrial atmospheric CO2 concentration (ppm)
    pub conc_pi: FloatValue,
}

/// CO2 effective radiative forcing component
///
/// Computes ERF using the standard logarithmic relationship:
/// $$ ERF = \frac{ERF_{2xCO2}}{\log(2)} \cdot \log\left(\frac{C}{C_0}\right) $$
#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[inputs(
    concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
)]
#[outputs(
    erf { name = "Effective Radiative Forcing|CO2", unit = "W / m^2" },
)]
pub struct CO2ERF {
    parameters: CO2ERFParameters,
}

impl CO2ERF {
    pub fn from_parameters(parameters: CO2ERFParameters) -> Self {
        Self { parameters }
    }

    /// Core physics calculation - extracted for testability
    pub fn calculate_erf(&self, concentration: FloatValue) -> FloatValue {
        self.parameters.erf_2xco2 / 2.0_f64.ln()
            * (1.0 + (concentration - self.parameters.conc_pi) / self.parameters.conc_pi).ln()
    }
}

#[typetag::serde]
impl Component for CO2ERF {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        // Use the macro-generated definitions
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        // Use the generated typed inputs struct
        let inputs = CO2ERFInputs::from_input_state(input_state);

        // Access values via typed fields
        let concentration = inputs.concentration.current();
        let erf = self.calculate_erf(concentration);

        // Return using the generated outputs struct
        let outputs = CO2ERFOutputs { erf };
        Ok(outputs.into())
    }
}
```

## What the Macro Generates

The `ComponentIO` macro generates:

### Input Struct

```rust
pub struct CO2ERFInputs<'a> {
    pub concentration: TimeseriesWindow<'a>,
}

impl<'a> CO2ERFInputs<'a> {
    pub fn from_input_state(input_state: &'a InputState<'_>) -> Self { ... }
}
```

### Output Struct

```rust
pub struct CO2ERFOutputs {
    pub erf: FloatValue,
}

impl From<CO2ERFOutputs> for OutputState { ... }
```

### Definitions Method

```rust
impl CO2ERF {
    pub fn generated_definitions() -> Vec<RequirementDefinition> { ... }
}
```

## Input Access Patterns

The `TimeseriesWindow` type provides several methods:

| Method | Description |
|--------|-------------|
| `current()` | Value at current timestep |
| `previous()` | Value at previous timestep |
| `at_offset(n)` | Value at relative offset |
| `last_n(n)` | Array of last n values |

## State Variables

For variables that are both read and written each timestep, use `#[states(...)]`:

```rust
#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[inputs(
    emissions { name = "Emissions|CO2", unit = "GtCO2" },
)]
#[outputs(
    uptake { name = "Land Uptake", unit = "GtCO2" },
)]
#[states(
    concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
    cumulative { name = "Cumulative Emissions|CO2", unit = "Gt C" },
)]
pub struct CarbonCycle {
    pub tau: FloatValue,
    pub conc_pi: FloatValue,
}
```

State variables appear in both the `Inputs` and `Outputs` structs, allowing you to read the previous value and write the new value.

## Grid-Based Variables

RSCM supports spatially-resolved variables:

- **FourBox**: Northern Ocean, Northern Land, Southern Ocean, Southern Land
- **Hemispheric**: Northern, Southern

Specify the grid type in your declaration:

```rust
use rscm_core::state::FourBoxSlice;

#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[inputs(
    forcing { name = "Effective Radiative Forcing", unit = "W/m^2" },
)]
#[outputs(
    regional_temp { name = "Regional Temperature", unit = "K", grid = "FourBox" },
)]
pub struct RegionalClimate {
    pub sensitivity: FloatValue,
}

#[typetag::serde]
impl Component for RegionalClimate {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = RegionalClimateInputs::from_input_state(input_state);
        let forcing = inputs.forcing.current();

        // FourBox outputs use FourBoxSlice
        let outputs = RegionalClimateOutputs {
            regional_temp: FourBoxSlice::from_array([
                forcing * self.sensitivity * 0.8,   // Northern Ocean
                forcing * self.sensitivity * 1.2,   // Northern Land
                forcing * self.sensitivity * 0.7,   // Southern Ocean
                forcing * self.sensitivity * 1.1,   // Southern Land
            ]),
        };
        Ok(outputs.into())
    }
}
```

### Grid Type Mapping

| Grid Type        | Input Type                                  | Output Type        |
| ---------------- | ------------------------------------------- | ------------------ |
| Scalar (default) | `TimeseriesWindow<'a>`                      | `FloatValue`       |
| FourBox          | `GridTimeseriesWindow<'a, FourBoxGrid>`     | `FourBoxSlice`     |
| Hemispheric      | `GridTimeseriesWindow<'a, HemisphericGrid>` | `HemisphericSlice` |

## Exposing to Python

To expose a Rust component to Python, use the `create_component_builder!` macro:

```rust
use rscm_core::create_component_builder;

create_component_builder!(CO2ERFBuilder, CO2ERF, CO2ERFParameters);
```

This generates a Python-callable `CO2ERFBuilder` class with a `from_parameters()` method.

Add the builder to a Python module in `rscm-components/src/python/mod.rs`:

```rust
m.add_class::<CO2ERFBuilder>()?;
```

Update the `.pyi` stub file to provide type hints:

```python
class CO2ERFBuilder(ComponentBuilder): ...
```

## Example Usage

See `crates/rscm-components/src/components/` for complete component implementations:

- `co2_erf.rs`: Simple scalar component
- `carbon_cycle.rs`: IVP-based component with state variables
- `four_box_ocean_heat_uptake.rs`: Grid-based outputs

## Testing Components

Extract core physics into separate methods for unit testing:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_erf_at_preindustrial() {
        let component = CO2ERF::from_parameters(CO2ERFParameters {
            erf_2xco2: 3.7,
            conc_pi: 278.0,
        });
        let erf = component.calculate_erf(278.0);
        assert!((erf - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_erf_at_2x_co2() {
        let params = CO2ERFParameters {
            erf_2xco2: 3.7,
            conc_pi: 278.0,
        };
        let component = CO2ERF::from_parameters(params.clone());
        let erf = component.calculate_erf(params.conc_pi * 2.0);
        assert!((erf - params.erf_2xco2).abs() < 1e-10);
    }
}
```

Integration tests can use `ModelBuilder` to verify component behaviour in context - see `crates/rscm-core/src/example_components.rs` for patterns.
