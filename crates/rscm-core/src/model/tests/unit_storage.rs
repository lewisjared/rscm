//! Tests that the store holds each variable in one unit and converts on both sides.
//!
//! With a schema, a variable is stored in the schema's unit.
//! A producer's output is converted into it on write, and a consumer's read out of it.

use crate::component::{Component, InputState, OutputState, VariableDefinition, VariableRole};
use crate::errors::RSCMResult;
use crate::model::{Model, ModelBuilder};
use crate::schema::VariableSchema;
use crate::state::{FourBoxSlice, StateValue};
use crate::timeseries::{FloatValue, Time, TimeAxis};
use is_close::is_close;
use numpy::ndarray::Array;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Writes `value` to `Forcing` in `unit` every step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Producer {
    unit: String,
    value: FloatValue,
}

#[typetag::serde]
impl Component for Producer {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::new(
            "Forcing",
            &self.unit,
            VariableRole::Output,
        )]
    }

    fn solve(&self, _: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        let mut output = OutputState::new();
        output.insert("Forcing".to_string(), StateValue::Scalar(self.value));
        Ok(output)
    }
}

/// Copies `Forcing`, read in `unit`, to `Forcing Copy` in the same unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Consumer {
    unit: String,
}

#[typetag::serde]
impl Component for Consumer {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::new("Forcing", &self.unit, VariableRole::Input),
            VariableDefinition::new("Forcing Copy", &self.unit, VariableRole::Output),
        ]
    }

    fn solve(&self, _: Time, _: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let value = input_state.get_scalar_window("Forcing").get();
        let mut output = OutputState::new();
        output.insert("Forcing Copy".to_string(), StateValue::Scalar(value));
        Ok(output)
    }
}

/// Adds one to its `Heat` state, declared in `unit`, every step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Accumulator {
    unit: String,
}

#[typetag::serde]
impl Component for Accumulator {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::new(
            "Heat",
            &self.unit,
            VariableRole::State,
        )]
    }

    fn solve(&self, _: Time, _: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let value = input_state.get_scalar_window("Heat").get() + 1.0;
        let mut output = OutputState::new();
        output.insert("Heat".to_string(), StateValue::Scalar(value));
        Ok(output)
    }
}

/// Writes `[1, 2, 3, 4]` to the four-box `Regional Forcing` in `unit`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FourBoxProducer {
    unit: String,
}

#[typetag::serde]
impl Component for FourBoxProducer {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::four_box_output(
            "Regional Forcing",
            &self.unit,
        )]
    }

    fn solve(&self, _: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        let mut output = OutputState::new();
        output.insert(
            "Regional Forcing".to_string(),
            StateValue::FourBox(FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0])),
        );
        Ok(output)
    }
}

fn run(builder: &mut ModelBuilder) -> Model {
    let mut model = builder
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .build()
        .expect("model builds");
    model.run();
    model
}

/// The stored value of `name` at the first step's end and the unit it is labelled with.
fn stored(model: &Model, name: &str) -> (FloatValue, String) {
    let item = model
        .timeseries()
        .get_by_name(name)
        .expect("variable is stored");
    let ts = item.data.as_scalar().expect("scalar variable");
    (
        ts.at_scalar(1).expect("value after one step"),
        ts.unit().to_string(),
    )
}

fn assert_stored(model: &Model, name: &str, value: FloatValue, unit: &str) {
    let (actual, actual_unit) = stored(model, name);
    assert!(
        is_close!(actual, value),
        "{name}: expected {value}, got {actual}"
    );
    assert_eq!(actual_unit, unit, "{name} is labelled with the wrong unit");
}

fn schema() -> VariableSchema {
    VariableSchema::new()
        .variable("Forcing", "W/m^2")
        .variable("Forcing Copy", "W/m^2")
}

#[test]
fn producer_output_is_converted_into_the_schema_unit() {
    let model = run(ModelBuilder::new()
        .with_schema(schema())
        .with_component(Arc::new(Producer {
            unit: "kW/m^2".to_string(),
            value: 1.0,
        }))
        .with_component(Arc::new(Consumer {
            unit: "kW/m^2".to_string(),
        })));

    assert_stored(&model, "Forcing", 1000.0, "W/m^2");
    assert_stored(&model, "Forcing Copy", 1000.0, "W/m^2");
}

#[test]
fn consumer_in_the_schema_unit_reads_without_conversion() {
    let model = run(ModelBuilder::new()
        .with_schema(schema())
        .with_component(Arc::new(Producer {
            unit: "kW/m^2".to_string(),
            value: 1.0,
        }))
        .with_component(Arc::new(Consumer {
            unit: "W/m^2".to_string(),
        })));

    assert_stored(&model, "Forcing", 1000.0, "W/m^2");
    assert_stored(&model, "Forcing Copy", 1000.0, "W/m^2");
}

#[test]
fn without_a_schema_the_producer_unit_is_stored() {
    let model = run(ModelBuilder::new()
        .with_component(Arc::new(Producer {
            unit: "kW/m^2".to_string(),
            value: 1.0,
        }))
        .with_component(Arc::new(Consumer {
            unit: "W/m^2".to_string(),
        })));

    assert_stored(&model, "Forcing", 1.0, "kW/m^2");
    assert_stored(&model, "Forcing Copy", 1000.0, "W/m^2");
}

#[test]
fn state_round_trips_through_the_schema_unit() {
    let model = run(ModelBuilder::new()
        .with_schema(VariableSchema::new().variable("Heat", "J"))
        .with_component(Arc::new(Accumulator {
            unit: "kJ".to_string(),
        }))
        .with_initial_values([("Heat".to_string(), 0.0)].into()));

    // One kJ a step, stored in J.
    assert_stored(&model, "Heat", 1000.0, "J");
    let ts = model.timeseries().get_by_name("Heat").unwrap();
    let after_two = ts.data.as_scalar().unwrap().at_scalar(2).unwrap();
    assert!(is_close!(after_two, 2000.0), "got {after_two}");
}

#[test]
fn grid_outputs_are_converted_in_every_region() {
    let model = run(ModelBuilder::new()
        .with_schema(VariableSchema::new().variable_with_grid(
            "Regional Forcing",
            "mW/m^2",
            crate::component::GridType::FourBox,
        ))
        .with_component(Arc::new(FourBoxProducer {
            unit: "W/m^2".to_string(),
        })));

    let item = model.timeseries().get_by_name("Regional Forcing").unwrap();
    let ts = item.data.as_four_box().expect("four-box variable");
    let values: Vec<FloatValue> = ts.at_time_index(1).unwrap().to_vec();
    assert_eq!(values, vec![1000.0, 2000.0, 3000.0, 4000.0]);
}
