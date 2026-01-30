//! Basic model tests: step, dot, serialisation.

use crate::component::{Component, ComponentState, InputState, OutputState, RequirementDefinition};
use crate::errors::RSCMResult;
use crate::example_components::{TestComponent, TestComponentParameters};
use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
use crate::model::{Model, ModelBuilder};
use crate::spatial::{ScalarGrid, ScalarRegion};
use crate::timeseries::{FloatValue, Time, TimeAxis, Timeseries};
use is_close::is_close;
use numpy::array;
use numpy::ndarray::{Array, Axis};
use serde::{Deserialize, Serialize};
use std::iter::zip;
use std::sync::Arc;

fn get_emissions() -> Timeseries<FloatValue> {
    let values = array![0.0, 10.0].insert_axis(Axis(1));
    Timeseries::new(
        values,
        Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
        ScalarGrid,
        "GtC / yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    )
}

#[test]
fn step() {
    let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
    let mut model = ModelBuilder::new()
        .with_time_axis(time_axis)
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build()
        .unwrap();

    assert_eq!(model.current_time(), 2020.0);
    model.step();
    model.step();
    assert_eq!(model.current_time(), 2022.0);
    model.run();
    assert!(model.finished());

    let concentrations = model
        .timeseries()
        .get_data("Concentrations|CO2")
        .and_then(|data| data.as_scalar())
        .unwrap();

    // The first value for an endogenous timeseries without a y0 value is NaN.
    // This is because the values in the timeseries represents the state at the start
    // of a time step.
    // Since the values from t-1 aren't known we can't solve for y0
    assert!(concentrations.at(0, ScalarRegion::Global).unwrap().is_nan());
    let mut iter = concentrations.values().into_iter();
    iter.next(); // Skip the first value
    assert!(iter.all(|x| !x.is_nan()));
}

#[test]
fn dot() {
    let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
    let model = ModelBuilder::new()
        .with_time_axis(time_axis)
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build()
        .unwrap();

    let exp = r#"digraph {
    0 [ label = "NullComponent"]
    1 [ label = "TestComponent { parameters: TestComponentParameters { conversion_factor: 0.5 } }"]
    0 -> 1 [ label = ""]
}
"#;

    let res = format!("{:?}", model.as_dot());
    assert_eq!(res, exp);
}

#[test]
fn serialise_and_deserialise_model() {
    use crate::model::Model;

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build()
        .unwrap();

    model.step();

    let serialised = serde_json::to_string_pretty(&model).unwrap();
    println!("Pretty JSON");
    println!("{}", serialised);
    let serialised = toml::to_string(&model).unwrap();
    println!("TOML");
    println!("{}", serialised);

    let expected = r#"initial_node = 0
time_index = 1

[components]
node_holes = []
edge_property = "directed"
edges = [[0, 1, { name = "", unit = "", requirement_type = "EmptyLink", grid_type = "Scalar" }]]

[[components.nodes]]
type = "NullComponent"

[[components.nodes]]
type = "TestComponent"

[components.nodes.parameters]
conversion_factor = 0.5

[[collection.timeseries]]
name = "Concentrations|CO2"
variable_type = "Endogenous"

[collection.timeseries.data.Scalar]
units = "ppm"
latest = 1
interpolation_strategy = "Linear"

[collection.timeseries.data.Scalar.values]
v = 1
dim = [5, 1]
data = [nan, 5.0, nan, nan, nan]

[collection.timeseries.data.Scalar.time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]

[[collection.timeseries]]
name = "Emissions|CO2"
variable_type = "Exogenous"

[collection.timeseries.data.Scalar]
units = "GtC / yr"
latest = 4
interpolation_strategy = "Previous"

[collection.timeseries.data.Scalar.values]
v = 1
dim = [5, 1]
data = [10.0, 10.0, 10.0, 10.0, 10.0]

[collection.timeseries.data.Scalar.time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]

[time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]

[component_states.0]
state_type = "unit"

[component_states.1]
state_type = "unit"
"#;

    assert_eq!(serialised, expected);

    let deserialised = toml::from_str::<Model>(&serialised).unwrap();

    assert!(zip(
        model
            .timeseries()
            .get_data("Emissions|CO2")
            .and_then(|data| data.as_scalar())
            .unwrap()
            .values(),
        deserialised
            .timeseries()
            .get_data("Emissions|CO2")
            .and_then(|data| data.as_scalar())
            .unwrap()
            .values()
    )
    .all(|(x0, x1)| { is_close!(*x0, *x1) || (x0.is_nan() && x0.is_nan()) }));

    assert_eq!(model.current_time_bounds(), (2021.0, 2022.0));
    assert_eq!(deserialised.current_time_bounds(), (2021.0, 2022.0));
}

// Module-level types for checkpoint test to enable proper serialization/deserialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatefulTestComponentState {
    call_count: usize,
}

#[typetag::serde]
impl ComponentState for StatefulTestComponentState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatefulTestComponent {}

#[typetag::serde]
impl Component for StatefulTestComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![RequirementDefinition::scalar_input("input", "m")]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        _input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        Ok(OutputState::new())
    }

    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(StatefulTestComponentState { call_count: 0 })
    }

    fn solve_with_state(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        internal_state: &mut dyn ComponentState,
    ) -> RSCMResult<OutputState> {
        let state = internal_state
            .as_any_mut()
            .downcast_mut::<StatefulTestComponentState>()
            .expect("Wrong state type");
        state.call_count += 1;
        self.solve(t_current, t_next, input_state)
    }
}

#[test]
fn test_checkpoint_preserves_component_state() {
    // Build a model with a stateful component
    let time_axis = TimeAxis::from_values(Array::range(2020.0, 2030.0, 1.0));
    let input_ts = Timeseries::new(
        array![1.0, 2.0].insert_axis(Axis(1)),
        Arc::new(TimeAxis::from_bounds(array![2020.0, 2025.0, 2030.0])),
        ScalarGrid,
        "m".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );

    let mut model = ModelBuilder::new()
        .with_time_axis(time_axis)
        .with_component(Arc::new(StatefulTestComponent {}))
        .with_exogenous_variable("input", input_ts)
        .build()
        .unwrap();

    // Run a few steps
    model.step();
    model.step();
    model.step();

    // Find the stateful component node by scanning all component nodes
    let stateful_node = model.components.node_indices().find(|i| {
        if let Some(state) = model.get_component_state(*i) {
            state.as_any().is::<StatefulTestComponentState>()
        } else {
            false
        }
    });

    let node_idx = stateful_node.expect("Should find stateful component");
    let state = model.get_component_state(node_idx).unwrap();
    let typed_state = state
        .as_any()
        .downcast_ref::<StatefulTestComponentState>()
        .unwrap();
    assert_eq!(typed_state.call_count, 3, "Should have been called 3 times");

    // Checkpoint and restore
    let checkpoint = model.checkpoint().unwrap();
    let restored = Model::from_checkpoint(&checkpoint).unwrap();

    // Verify state was preserved
    let restored_state = restored.get_component_state(node_idx).unwrap();
    let restored_typed = restored_state
        .as_any()
        .downcast_ref::<StatefulTestComponentState>()
        .unwrap();
    assert_eq!(
        restored_typed.call_count, 3,
        "Restored state should have same call_count"
    );
    assert_eq!(
        restored.time_index(),
        model.time_index(),
        "Time index should be preserved"
    );
}
