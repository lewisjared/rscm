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
- [Documenting Components](../developers/documenting_components.md): How component pages are generated
- [Python API](../api/rscm/): Complete API documentation

## Creating a Component

Use the `ComponentIO` derive macro with struct-level attributes to declare inputs and outputs:

```rust
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
    ScalarWindow,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::StateValue;
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
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["radiative-forcing", "co2", "simple", "stable"], category = "Radiative Forcing")]
#[inputs(
    concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
)]
#[outputs(
    erf { name = "Effective Radiative Forcing|CO2", unit = "W/m^2" },
)]
pub struct CO2ERF {
    parameters: CO2ERFParameters,
}

impl CO2ERF {
    pub fn from_parameters(parameters: CO2ERFParameters) -> Self {
        Self { parameters }
    }

    /// Core physics calculation, extracted for testability
    pub fn calculate_erf(&self, concentration: FloatValue) -> FloatValue {
        self.parameters.erf_2xco2 / 2.0_f64.ln()
            * (1.0 + (concentration - self.parameters.conc_pi) / self.parameters.conc_pi).ln()
    }
}

#[typetag::serde]
impl Component for CO2ERF {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = CO2ERFInputs::from_input_state(input_state);

        let concentration = inputs.concentration.get();
        let erf = self.calculate_erf(concentration);

        Ok(CO2ERFOutputs { erf }.into())
    }
}
```

The generated code refers to `GridType`, `RequirementType`, `ScalarWindow`, `StateValue`
and the other types above by their short names.
Import them in the module that derives `ComponentIO`, even where your own code does not use them.

### Attributes

| Attribute                                       | Purpose                                                    |
| ----------------------------------------------- | ---------------------------------------------------------- |
| `#[inputs(field { name, unit, grid }, ...)]`    | Variables read from other components or exogenous data     |
| `#[outputs(field { name, unit, grid }, ...)]`   | Variables written each timestep                            |
| `#[states(field { name, unit, grid }, ...)]`    | Variables read at the start and written at the end         |
| `#[component(tags = [...], category = "...")]`  | Optional metadata for the generated component docs         |

`grid` is optional and is one of `"Scalar"` (the default), `"FourBox"` or `"Hemispheric"`.

## What the Macro Generates

For a struct `CO2ERF`, the `ComponentIO` macro generates the following.

### Input Struct

```rust
pub struct CO2ERFInputs<'a> {
    pub concentration: ScalarWindow<'a>,
}

impl<'a> CO2ERFInputs<'a> {
    pub fn from_input_state(input_state: &'a InputState<'_>) -> Self { ... }
}
```

`from_input_state` panics if a declared variable is missing from the input state.
Inside a model every declared variable is present.
A variable with no producer and no exogenous data reads as NaN rather than failing.

### Output Struct

```rust
pub struct CO2ERFOutputs {
    pub erf: FloatValue,
}

impl From<CO2ERFOutputs> for OutputState { ... }
```

### Associated Functions

```rust
impl CO2ERF {
    pub fn generated_definitions() -> Vec<RequirementDefinition> { ... }
    pub fn component_metadata() -> ComponentMetadata { ... }
}
```

## Input Access Patterns

A scalar input is a `ScalarWindow`.
It reads scalar data directly and aggregates FourBox or Hemispheric data to a global value.

| Method         | Description                                                                  |
| -------------- | ---------------------------------------------------------------------------- |
| `at_start()`   | Value at the start of the timestep (index N)                                 |
| `at_end()`     | Value at the end of the timestep (index N+1), or `None`                      |
| `get()`        | `at_start()` for exogenous and own state, `at_end()` for upstream outputs   |
| `previous()`   | Value at the previous timestep (index N-1), or `None`                        |
| `at_offset(n)` | Value at a relative offset, or `None` if out of bounds                       |

Prefer `get()` unless the physics needs a specific end of the timestep.
See "Timestep Semantics" in [Key Concepts](../key_concepts.md) for the details.

## State Variables

For variables that are both read and written each timestep, use `#[states(...)]`:

```rust
#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[inputs(
    emissions { name = "Emissions|CO2", unit = "GtC / yr" },
)]
#[outputs(
    uptake { name = "Land Uptake", unit = "GtC / yr" },
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

State variables appear in both the `Inputs` and `Outputs` structs.
This lets `solve` read the value at the start of the timestep and write the new value.
State variables need an initial value, set with `ModelBuilder::with_initial_values`.

### Internal State

Some components carry history that is not a model variable, such as ocean layer temperatures.
Define a state type that implements `ComponentState`,
then override `create_initial_state` and `solve_with_state` on the `Component` trait:

```rust
use rscm_core::component::ComponentState;
use std::any::Any;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LaggedState {
    pub history: Vec<FloatValue>,
}

#[typetag::serde]
impl ComponentState for LaggedState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[inputs(
    forcing { name = "Effective Radiative Forcing", unit = "W/m^2" },
)]
#[outputs(
    lagged { name = "Lagged Forcing", unit = "W/m^2" },
)]
pub struct Lagged {
    pub window: usize,
}

impl Lagged {
    fn solve_impl(&self, input_state: &InputState, state: &mut LaggedState) -> RSCMResult<OutputState> {
        let inputs = LaggedInputs::from_input_state(input_state);
        state.history.push(inputs.forcing.get());
        let start = state.history.len().saturating_sub(self.window);
        let recent = &state.history[start..];
        let lagged = recent.iter().sum::<FloatValue>() / recent.len() as FloatValue;
        Ok(LaggedOutputs { lagged }.into())
    }
}

#[typetag::serde]
impl Component for Lagged {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(&self, _t_current: Time, _t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        self.solve_impl(input_state, &mut LaggedState::default())
    }

    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(LaggedState::default())
    }

    fn solve_with_state(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
        internal_state: &mut dyn ComponentState,
    ) -> RSCMResult<OutputState> {
        let state = internal_state
            .as_any_mut()
            .downcast_mut::<LaggedState>()
            .expect("Lagged: expected LaggedState");
        self.solve_impl(input_state, state)
    }
}
```

The model calls `solve_with_state` and keeps one state per component.
The state is serialised with the model, so `Model::to_toml` checkpoints it and `from_toml` resumes from it.
`solve` is only used when the component is solved on its own, so it starts from a fresh state.
See `OceanCarbon` in `crates/rscm-magicc/src/carbon/ocean.rs` for a full example.

## ODE-Based Components

Components defined by differential equations implement the `IVP` trait and solve with `IVPBuilder`.
`calculate_dy_dt` reads inputs from the input state and the current solution from `y`:

```rust
use ode_solvers::Vector1;
use rscm_core::errors::RSCMError;
use rscm_core::ivp::{get_last_step, IVPBuilder, IVP};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[inputs(
    forcing { name = "Effective Radiative Forcing", unit = "W/m^2" },
)]
#[states(
    temperature { name = "Surface Temperature", unit = "K" },
)]
pub struct OneBox {
    pub heat_capacity: FloatValue,
    pub feedback: FloatValue,
}

impl IVP<Time, Vector1<FloatValue>> for OneBox {
    fn calculate_dy_dt(
        &self,
        _t: Time,
        input_state: &InputState,
        y: &Vector1<FloatValue>,
        dy_dt: &mut Vector1<FloatValue>,
    ) {
        let forcing = OneBoxInputs::from_input_state(input_state).forcing.get();
        dy_dt[0] = (forcing - self.feedback * y[0]) / self.heat_capacity;
    }
}

#[typetag::serde]
impl Component for OneBox {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let inputs = OneBoxInputs::from_input_state(input_state);
        let y0 = Vector1::new(inputs.temperature.at_start());

        let mut solver = IVPBuilder::new(Arc::new(self.clone()), input_state, y0)
            .to_rk4(t_current, t_next, 0.1);
        solver
            .integrate()
            .map_err(|e| RSCMError::Error(format!("OneBox integration failed: {e}")))?;
        let y = get_last_step(solver.results(), t_next);

        Ok(OneBoxOutputs { temperature: y[0] }.into())
    }
}
```

The component must be `Clone`, because `IVPBuilder` takes it in an `Arc`.
Add `ode_solvers` to the crate's dependencies for the vector types.
`CarbonCycle` in `crates/rscm-components/src/components/carbon_cycle.rs` is a three-variable example.

## Grid-Based Variables

RSCM supports spatially-resolved variables:

- **FourBox**: Northern Ocean, Northern Land, Southern Ocean, Southern Land
- **Hemispheric**: Northern, Southern

Specify the grid type in your declaration:

```rust
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
    ScalarWindow,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{FourBoxSlice, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

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
        let forcing = inputs.forcing.get();

        // Order: Northern Ocean, Northern Land, Southern Ocean, Southern Land
        let outputs = RegionalClimateOutputs {
            regional_temp: FourBoxSlice::from_array([
                forcing * self.sensitivity * 0.8,
                forcing * self.sensitivity * 1.2,
                forcing * self.sensitivity * 0.7,
                forcing * self.sensitivity * 1.1,
            ]),
        };
        Ok(outputs.into())
    }
}
```

### Grid Type Mapping

| Grid Type        | Input Type                              | Output Type        |
| ---------------- | --------------------------------------- | ------------------ |
| Scalar (default) | `ScalarWindow<'a>`                      | `FloatValue`       |
| FourBox          | `GridTimeseriesWindow<'a, FourBoxGrid>` | `FourBoxSlice`     |
| Hemispheric      | `HemisphericWindow<'a>`                 | `HemisphericSlice` |

A FourBox input also needs `FourBoxGrid` (from `rscm_core::spatial`) and `GridTimeseriesWindow` in scope.
A Hemispheric input needs `HemisphericWindow`.
Grid windows take a region, for example `at_start(FourBoxRegion::NorthernOcean)`.
They also provide `at_start_all()`, `get_all()` and `current_global()`.

## Error Handling

`solve` returns `RSCMResult<OutputState>`.
Return an `RSCMError` rather than panicking when inputs or configuration are invalid:

```rust
use rscm_core::errors::RSCMError;

if t_next - t_current != 1.0 {
    return Err(RSCMError::Error(format!(
        "CO2ERF requires annual timesteps, got {}",
        t_next - t_current
    )));
}
```

The model stops and adds the component name and timestep to the error.
From Python it is raised as a `RuntimeError`.

From Rust, `Model::run` and `Model::step` panic on a component error.
Use `try_run` or `try_step` to get the `RSCMResult` instead:

```rust
let mut model = ModelBuilder::new()
    .with_component(Arc::new(component))
    .with_time_axis(time_axis)
    .build()?;
model.try_run()?;
```

The time index does not advance past a failed step,
but components that already ran may have updated their state.
Rebuild the model before retrying.

## Exposing to Python

Python builders call `from_parameters(parameters)` on the component,
which deserialises the parameters from a dict.
Generate a builder with the `create_component_builder!` macro:

```rust
use rscm_core::create_component_builder;

create_component_builder!(CO2ERFBuilder, CO2ERF, CO2ERFParameters);
```

This generates a Python-callable `CO2ERFBuilder` class with `from_parameters()` and `build()` methods.

Register the builder in the crate's Python module, for example `crates/rscm-components/src/python/mod.rs`:

```rust
m.add_class::<CO2ERFBuilder>()?;
```

Add the class to the matching stub file, for example `python/rscm/_lib/components.pyi`:

```python
class CO2ERFBuilder(ComponentBuilder): ...
```

Run `make validate-pyi` to check the stubs match the compiled module.

## Example Usage

See these files for complete component implementations:

- `crates/rscm-components/src/components/co2_erf.rs`: Simple scalar component
- `crates/rscm-components/src/components/carbon_cycle.rs`: IVP-based component with state variables
- `crates/rscm-components/src/components/four_box_ocean_heat_uptake.rs`: Grid-based outputs
- `crates/rscm-magicc/src/`: MAGICC components, including internal state and fallible construction

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

Integration tests can use `ModelBuilder` to check component behaviour in context.
See `crates/rscm-core/src/example_components.rs` for patterns.
