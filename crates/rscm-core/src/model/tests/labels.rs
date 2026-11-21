//! Component labels key coupling, so instances of one type couple independently.

use crate::component::{Component, InputState, OutputState, VariableDefinition, VariableRole};
use crate::errors::{RSCMError, RSCMResult};
use crate::model::{Model, ModelBuilder};
use crate::schema::{AggregateOp, VariableSchema};
use crate::state::{StateValue, VariableSource};
use crate::timeseries::{FloatValue, Time, TimeAxis};
use numpy::ndarray::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Adds one joule to `Energy` every step.
#[derive(Debug, Serialize, Deserialize)]
struct Counter;

#[typetag::serde]
impl Component for Counter {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::new("Energy", "J", VariableRole::State)]
    }

    fn solve(&self, _: Time, _: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let energy = input_state.get_scalar_window("Energy").get();
        Ok(HashMap::from([(
            "Energy".to_string(),
            StateValue::Scalar(energy + 1.0),
        )]))
    }
}

/// Copies `Energy`, read in `unit`, to `output` in the same unit.
#[derive(Debug, Serialize, Deserialize)]
struct Reader {
    unit: String,
    output: String,
}

#[typetag::serde]
impl Component for Reader {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::new("Energy", &self.unit, VariableRole::Input),
            VariableDefinition::new(&self.output, &self.unit, VariableRole::Output),
        ]
    }

    fn solve(&self, _: Time, _: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let energy = input_state.get_scalar_window("Energy").get();
        Ok(HashMap::from([(
            self.output.clone(),
            StateValue::Scalar(energy),
        )]))
    }
}

fn reader(unit: &str, output: &str) -> Arc<Reader> {
    Arc::new(Reader {
        unit: unit.to_string(),
        output: output.to_string(),
    })
}

fn builder() -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2000.0, 2004.0, 1.0)))
        .with_initial_values(HashMap::from([("Energy".to_string(), 0.0)]))
        .with_component(Arc::new(Counter));
    builder
}

fn values(model: &Model, variable: &str) -> Vec<FloatValue> {
    let data = model.timeseries().get_data(variable).unwrap();
    data.as_scalar().unwrap().values().iter().copied().collect()
}

fn key(variable: &str, label: &str) -> (String, String) {
    (variable.to_string(), label.to_string())
}

#[test]
fn instances_of_one_type_couple_independently() {
    let mut model = builder()
        .with_component_as("Now", reader("J", "Now Energy"))
        .with_component_as("Before", reader("kJ", "Before Energy"))
        .with_lag_for("Energy", "Before")
        .build()
        .unwrap();
    model.run();

    let sources = model.variable_sources();
    assert_eq!(sources[&key("Energy", "Now")], VariableSource::Upstream);
    assert_eq!(sources[&key("Energy", "Before")], VariableSource::Lagged);
    assert!(!model.unit_conversions().contains_key(&key("Energy", "Now")));
    assert_eq!(model.unit_conversions()[&key("Energy", "Before")], 1e-3);

    // Each step stores a reader's output at the step's end.
    assert_eq!(values(&model, "Energy"), [0.0, 1.0, 2.0, 3.0]);
    assert_eq!(values(&model, "Now Energy")[1..], [1.0, 2.0, 3.0]);
    assert_eq!(values(&model, "Before Energy")[1..], [0.0, 1e-3, 2e-3]);
}

#[test]
fn coupling_and_errors_name_labels() {
    let model = builder()
        .with_component_as("Now", reader("J", "Now Energy"))
        .with_component_as("Before", reader("kJ", "Before Energy"))
        .with_lag_for("Energy", "Before")
        .build()
        .unwrap();
    let coupling = model.coupling().unwrap();
    let names: Vec<&str> = coupling
        .components
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(names, ["Before", "Counter", "Now"]);
    let lagged: Vec<_> = coupling
        .edges
        .iter()
        .map(|e| (e.consumer.as_str(), e.lagged))
        .collect();
    assert_eq!(lagged, [("Before", true), ("Now", false)]);
    let conversions: Vec<&str> = coupling
        .conversions
        .iter()
        .map(|c| c.component.as_str())
        .collect();
    assert_eq!(conversions, ["Before"]);

    let error = builder()
        .with_component_as("Now", reader("J", "Energy"))
        .build()
        .expect_err("two producers must not build");
    assert!(
        matches!(&error, RSCMError::MultipleProducers { components, .. } if components == &["Counter", "Now"]),
        "{error}"
    );
}

#[test]
fn duplicate_labels_fail_to_build() {
    let error = builder()
        .with_component(reader("J", "First"))
        .with_component(reader("J", "Second"))
        .build()
        .expect_err("two components labelled Reader must not build");
    assert!(
        matches!(&error, RSCMError::DuplicateLabel { label } if label == "Reader"),
        "{error}"
    );
    assert!(error.to_string().contains("with_component_as"), "{error}");

    let schema = VariableSchema::new()
        .variable("Energy", "J")
        .aggregate("Total", "J", AggregateOp::Sum)
        .from("Energy")
        .build();
    let error = builder()
        .with_schema(schema)
        .with_component_as("Total", reader("J", "Copy"))
        .build()
        .expect_err("a component labelled like an aggregate must not build");
    assert!(
        matches!(&error, RSCMError::DuplicateLabel { label } if label == "Total"),
        "{error}"
    );
}

#[test]
fn a_lag_into_an_aggregate_reads_the_previous_step() {
    let run = |lag: bool| {
        let schema = VariableSchema::new()
            .variable("Energy", "J")
            .aggregate("Total", "J", AggregateOp::Sum)
            .from("Energy")
            .build();
        let mut builder = builder();
        builder.with_schema(schema);
        if lag {
            builder.with_lag_for("Energy", "Total");
        }
        let mut model = builder.build().unwrap();
        model.run();
        model
    };

    let model = run(true);
    assert_eq!(
        model.variable_sources()[&key("Energy", "Total")],
        VariableSource::Lagged
    );
    let coupling = model.coupling().unwrap();
    assert!(coupling.components.iter().any(|c| c.name == "Total"));
    assert_eq!(values(&model, "Total")[1..], [0.0, 1.0, 2.0]);
    assert_eq!(values(&run(false), "Total")[1..], [1.0, 2.0, 3.0]);

    let error = builder()
        .with_lag_for("Energy", "Total")
        .build()
        .expect_err("a lag into an unknown label must not build");
    assert!(
        error
            .to_string()
            .contains("no component labelled \"Total\""),
        "{error}"
    );
}
