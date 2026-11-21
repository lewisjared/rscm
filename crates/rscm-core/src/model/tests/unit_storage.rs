//! Tests that the store holds each variable in one unit and converts on both sides.
//!
//! With a schema, a variable is stored in the schema's unit.
//! A producer's output is converted into it on write, and a consumer's read out of it.

use crate::component::{Component, InputState, OutputState, VariableDefinition, VariableRole};
use crate::errors::RSCMResult;
use crate::model::TransformDirection::{self, Read, Write};
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

#[test]
fn initial_values_with_a_unit_are_converted_into_the_stored_unit() {
    let model = run(ModelBuilder::new()
        .with_schema(VariableSchema::new().variable("Heat", "J"))
        .with_component(Arc::new(Accumulator {
            unit: "kJ".to_string(),
        }))
        .with_initial_values_in_units([("Heat".to_string(), (2.0, "kJ".to_string()))].into()));

    let ts = model.timeseries().get_by_name("Heat").unwrap();
    let ts = ts.data.as_scalar().unwrap();
    assert!(is_close!(ts.at_scalar(0).unwrap(), 2000.0));
    assert_stored(&model, "Heat", 3000.0, "J");
}

#[test]
fn initial_grid_values_with_a_unit_are_converted_in_every_region() {
    let model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_schema(VariableSchema::new().variable_with_grid(
            "Regional Forcing",
            "mW/m^2",
            crate::component::GridType::FourBox,
        ))
        .with_component(Arc::new(FourBoxProducer {
            unit: "W/m^2".to_string(),
        }))
        .with_initial_grid_values_in_units(
            [(
                "Regional Forcing".to_string(),
                (vec![1.0, 2.0, 3.0, 4.0], "W/m^2".to_string()),
            )]
            .into(),
        )
        .build()
        .expect("model builds");

    let item = model.timeseries().get_by_name("Regional Forcing").unwrap();
    let values = item.data.as_four_box().unwrap().at_time_index(0).unwrap();
    assert_eq!(values, vec![1000.0, 2000.0, 3000.0, 4000.0]);
}

#[test]
fn an_incompatible_initial_value_unit_fails_the_build() {
    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_component(Arc::new(Accumulator {
            unit: "J".to_string(),
        }))
        .with_initial_values_in_units([("Heat".to_string(), (0.0, "GtC".to_string()))].into())
        .build();

    let Err(err) = result else {
        panic!("build should fail");
    };
    let err = err.to_string();
    assert!(err.contains("Incompatible units"), "{err}");
}

#[test]
fn a_bare_initial_value_replaces_one_given_in_a_unit() {
    let model = run(ModelBuilder::new()
        .with_schema(VariableSchema::new().variable("Heat", "J"))
        .with_component(Arc::new(Accumulator {
            unit: "kJ".to_string(),
        }))
        .with_initial_values_in_units([("Heat".to_string(), (2.0, "kJ".to_string()))].into())
        .with_initial_values([("Heat".to_string(), 3.0)].into()));

    let ts = model.timeseries().get_by_name("Heat").unwrap();
    assert!(is_close!(
        ts.data.as_scalar().unwrap().at_scalar(0).unwrap(),
        3.0
    ));
}

/// Writes `value` to `output` in `output_unit`, after reading `input` if it has one.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Relay {
    input: Option<(String, String)>,
    output: (String, String),
    value: FloatValue,
}

#[typetag::serde]
impl Component for Relay {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut defs = vec![VariableDefinition::new(
            &self.output.0,
            &self.output.1,
            VariableRole::Output,
        )];
        if let Some((name, unit)) = &self.input {
            defs.push(VariableDefinition::new(name, unit, VariableRole::Input));
        }
        defs
    }

    fn solve(&self, _: Time, _: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let value = match &self.input {
            Some((name, _)) => input_state.get_scalar_window(name).get(),
            None => self.value,
        };
        let mut output = OutputState::new();
        output.insert(self.output.0.clone(), StateValue::Scalar(value));
        Ok(output)
    }
}

#[test]
fn a_reading_instance_does_not_convert_another_instances_output() {
    let model = run(ModelBuilder::new()
        .with_component_as(
            "Source",
            Arc::new(Relay {
                input: None,
                output: ("Energy".to_string(), "kJ".to_string()),
                value: 1.0,
            }),
        )
        .with_component(Arc::new(Relay {
            input: Some(("Energy".to_string(), "J".to_string())),
            output: ("Energy Copy".to_string(), "J".to_string()),
            value: 0.0,
        })));

    assert_stored(&model, "Energy", 1.0, "kJ");
    assert_stored(&model, "Energy Copy", 1000.0, "J");
}

#[test]
fn coupling_reports_the_direction_and_units_of_each_conversion() {
    let model = run(ModelBuilder::new()
        .with_schema(schema())
        .with_component(Arc::new(Producer {
            unit: "kW/m^2".to_string(),
            value: 1.0,
        }))
        .with_component(Arc::new(Consumer {
            unit: "kW/m^2".to_string(),
        })));

    let conversions: Vec<_> = model
        .coupling()
        .unwrap()
        .conversions
        .into_iter()
        .map(|c| {
            let unit = c.unit.unwrap();
            (
                c.variable,
                c.component,
                c.direction,
                unit.from,
                unit.to,
                unit.factor,
            )
        })
        .collect();
    let row = |v: &str, c: &str, d: TransformDirection, from: &str, to: &str, f: f64| {
        (v.into(), c.into(), d, from.into(), to.into(), f)
    };
    assert_eq!(
        conversions,
        vec![
            row("Forcing", "Consumer", Read, "W/m^2", "kW/m^2", 0.001),
            row("Forcing", "Producer", Write, "kW/m^2", "W/m^2", 1000.0),
            row("Forcing Copy", "Consumer", Write, "kW/m^2", "W/m^2", 1000.0),
        ]
    );
}

#[test]
fn the_latest_initial_value_for_a_name_wins() {
    let build = |grid_last: bool| {
        let mut builder = ModelBuilder::new();
        builder
            .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
            .with_component(Arc::new(FourBoxProducer {
                unit: "W/m^2".to_string(),
            }));
        let scalar = [("Regional Forcing".to_string(), (1.0, "kW/m^2".to_string()))];
        let grid = [("Regional Forcing".to_string(), vec![1.0, 2.0, 3.0, 4.0])];
        if grid_last {
            builder.with_initial_values_in_units(scalar.into());
            builder.with_initial_grid_values(grid.into());
        } else {
            builder.with_initial_grid_values(grid.into());
            builder.with_initial_values_in_units(scalar.into());
        }
        let model = builder.build().unwrap();
        let item = model.timeseries().get_by_name("Regional Forcing").unwrap();
        item.data.as_four_box().unwrap().at_time_index(0).unwrap()
    };

    assert_eq!(build(true), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(build(false), vec![1000.0; 4]);
}
