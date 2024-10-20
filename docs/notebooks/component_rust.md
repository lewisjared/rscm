# Component in Rust

This notebook demonstrates how to implement a component in Rust.
The component is a simple carbon cycle model that calculates the atmospheric CO2 concentration, cumulative land uptake, and cumulative emissions of CO2.

The model is based on the following ordinary differential equations:

...

The model is implemented as a struct that implements the `Component` trait.
The `Component` trait requires the implementation of the `definitions` and `solve` methods that make the component work with the RSCM framework.

Some boilerplate code has been omitted for brevity.
The boilerplate code should be replaced with a macro that generates the code.

## Parameters

We define a struct `CarbonCycleParameters` to hold the parameters of the model. These parameters are used when initialising the component.

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct CarbonCycleParameters {
    /// Timescale of the box's response
    /// unit: yr
    pub tau: FloatValue,
    /// Pre-industrial atmospheric CO_2 concentration
    /// unit: ppm
    pub conc_pi: FloatValue,
    /// Sensitivity of lifetime to changes in global-mean temperature
    /// unit: 1 / K
    pub alpha_temperature: FloatValue,
}
```

## CarbonCycleComponent

The `CarbonCycleComponent` struct holds the parameters and solver options for the component.

```rust
#[derive(Debug, Clone)]
pub struct CarbonCycleComponent {
    parameters: CarbonCycleParameters,
    // Parameters for the solver (e.g. step size)
    // TODO: Move SolverOptions to rscm-core
    solver_options: SolverOptions,
}
```

Since this component is an initial value problem we can implement the IVP trait for the component.
This trait requires the implementation of the `calculate_dy_dt` method that calculates the derivative of the y vector at a given time.
```rust
use ode_solvers::Vector3;
use rscm_component::constants::GTC_PER_PPM;
use rscm_core::component::{InputState, State};
use rscm_core::ivp::IVP;
use rscm_core::timeseries::{FloatValue, Time};

type ModelState = Vector3<FloatValue>;


impl IVP<Time, ModelState> for CarbonCycleComponent {
    fn calculate_dy_dt(
        &self,
        _t: Time,
        input_state: &InputState,
        _y: &Vector3<FloatValue>,
        dy_dt: &mut Vector3<FloatValue>,
    ) {
        let emissions = input_state.get("Emissions|CO2|Anthropogenic");
        let temperature = input_state.get("Surface Temperature");
        let conc = input_state.get("Atmospheric Concentration|CO2");

        // dC / dt = E - (C - C_0) / (\tau \exp(alpha_temperature * temperature))
        let lifetime =
            self.parameters.tau * (self.parameters.alpha_temperature * temperature).exp();
        let uptake = (conc - self.parameters.conc_pi) / lifetime; // ppm / yr

        dy_dt[0] = emissions / GTC_PER_PPM - uptake; // ppm / yr
        dy_dt[1] = uptake * GTC_PER_PPM; // GtC / yr
        dy_dt[2] = *emissions // GtC / yr
    }
}
```

In order to make the component work with the RSCM framework,
we need to implement the `Component` trait for the `CarbonCycleComponent` struct.
This describes the common interface for all components in the RSCM framework
and is used by the `Model` struct to interact with the component.

The `definitions` method returns a list of `RequirementDefinition` objects that define the input and output requirements of the component.
These requirements are used to validate the input and output states of the component and are used by the model to determine the order in which components are solved.

The `solve` method takes the current time, the next time, and the input state as arguments and returns the output state at the end of the time step.

```rust
impl Component for CarbonCycleComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::new(
                "Emissions|CO2|Anthropogenic",
                "GtC / yr",
                RequirementType::Input,
            ),
            RequirementDefinition::new("Surface Temperature", "K", RequirementType::Input),
            RequirementDefinition::new(
                "Atmospheric Concentration|CO2",
                "ppm",
                RequirementType::InputAndOutput,
            ),
            RequirementDefinition::new(
                "Cumulative Emissions|CO2",
                "Gt C",
                RequirementType::InputAndOutput,
            ),
            RequirementDefinition::new(
                "Cumulative Land Uptake",
                "Gt C",
                RequirementType::InputAndOutput,
            ),
        ]
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let y0 = ModelState::new(
            *input_state.get("Atmospheric Concentration|CO2"),
            *input_state.get("Cumulative Land Uptake"),
            *input_state.get("Cumulative Emissions|CO2"),
        );

        let solver = IVPBuilder::new(Arc::new(self.to_owned()), input_state.clone(), y0);

        let mut solver = solver.to_rk4(t_current, t_next, self.solver_options.step_size);
        solver.integrate().expect("Failed solving");

        let results = get_last_step(solver.results(), t_next);

        let mut output = HashMap::new();
        output.insert("Atmospheric Concentration|CO2".to_string(), results[0]);
        output.insert("Cumulative Land Uptake".to_string(), results[1]);
        output.insert("Cumulative Emissions|CO2".to_string(), results[2]);

        Ok(OutputState::from_hashmap_and_verify(
            output,
            self.output_names(),
        ))
    }
}
```

## Exposing to Python

We use the `create_component_builder!` macro to create a builder for the component.
This macro generates a struct that implements the `ComponentBuilder` trait and provides a method to create a new instance of the component.
The component builder returns a `PyRustComponent` struct
which is a wrapper around the Rust component that exposes the component to Python.

```rust
create_component_builder!(CO2ERFBuilder, CO2ERF, CO2ERFParameters);
```

`CO2ERFBuilder` can then be added to a python module (e.g. see `rscm-components/src/python/mod.rs`).

Additionally, the `.pyi` files should be updated to include the new component
to enable useful type hints in Python.
`ComponentBuilder` is a protocol which describes the functionality of a component builder.

```python
class CO2ERFBuilder(ComponentBuilder): ...
```

## Example
See `rscm-components/src/components/carbon_cycle.rs` for the full implementation of the component.
