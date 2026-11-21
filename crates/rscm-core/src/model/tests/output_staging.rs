//! A component's outputs are all validated before any of them is written.

use crate::component::{Component, GridType, InputState, OutputState, VariableDefinition};
use crate::errors::RSCMResult;
use crate::model::{GridConversion, Model, ModelBuilder};
use crate::schema::VariableSchema;
use crate::state::{FourBoxSlice, HemisphericSlice, StateValue};
use crate::timeseries::{Time, TimeAxis};
use numpy::ndarray::Array;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Declares `declared` and returns `emitted` every step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Emitter {
    declared: Vec<VariableDefinition>,
    emitted: Vec<(String, StateValue)>,
}

#[typetag::serde]
impl Component for Emitter {
    fn definitions(&self) -> Vec<VariableDefinition> {
        self.declared.clone()
    }

    fn solve(&self, _: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        Ok(self.emitted.iter().cloned().collect())
    }
}

/// Producer of `B`, so a timeseries with that name is stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BProducer;

#[typetag::serde]
impl Component for BProducer {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::scalar_output("B", "K")]
    }

    fn solve(&self, _: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        Ok([("B".to_string(), StateValue::Scalar(7.0))].into())
    }
}

fn builder() -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder.with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)));
    builder
}

fn first_step_error(model: &mut Model) -> String {
    let error = model.try_step().expect_err("step should fail").to_string();
    assert!(error.contains("Emitter"), "{error}");
    assert!(error.contains("2020..2021"), "{error}");
    error
}

fn assert_a_unwritten(model: &Model) {
    let a = model
        .timeseries()
        .get_data("A")
        .unwrap()
        .as_scalar()
        .unwrap();
    assert!(a.at_scalar(1).unwrap().is_nan());
}

#[test]
fn undeclared_output_fails_even_when_stored() {
    let mut model = builder()
        .with_component(Arc::new(BProducer))
        .with_component(Arc::new(Emitter {
            declared: vec![VariableDefinition::scalar_output("A", "K")],
            emitted: vec![
                ("A".into(), StateValue::Scalar(1.0)),
                ("B".into(), StateValue::Scalar(2.0)),
            ],
        }))
        .build()
        .unwrap();

    let error = first_step_error(&mut model);
    assert!(error.contains("undeclared output B"), "{error}");
    assert_a_unwritten(&model);
}

/// A model where `A` is written before `T`, and `T` goes through a `source` to `target` write aggregation.
fn model_with_write_transform(
    t_output: VariableDefinition,
    emitted_t: StateValue,
    source: GridType,
    target: GridType,
) -> Model {
    let mut model = builder()
        .with_component(Arc::new(Emitter {
            declared: vec![VariableDefinition::scalar_output("A", "K"), t_output],
            emitted: vec![
                ("A".into(), StateValue::Scalar(1.0)),
                ("T".into(), emitted_t),
            ],
        }))
        .build()
        .unwrap();
    model.step_plan.steps[0].writes[1].grid = Some(GridConversion {
        from: source,
        to: target,
    });
    model
}

#[test]
fn failing_aggregation_commits_nothing() {
    let mut model = model_with_write_transform(
        VariableDefinition::scalar_output("T", "K"),
        StateValue::Scalar(2.0),
        GridType::Scalar,
        GridType::FourBox,
    );

    let error = first_step_error(&mut model);
    assert!(error.contains("could not aggregate T"), "{error}");
    assert_a_unwritten(&model);
}

#[test]
fn aggregated_value_off_the_stored_grid_commits_nothing() {
    let mut model = model_with_write_transform(
        VariableDefinition::four_box_output("T", "K"),
        StateValue::FourBox(FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0])),
        GridType::FourBox,
        GridType::Scalar,
    );

    let error = first_step_error(&mut model);
    assert!(error.contains("could not write T"), "{error}");
    assert_a_unwritten(&model);
}

#[test]
fn value_off_the_transform_source_grid_commits_nothing() {
    let schema = VariableSchema::new()
        .variable_with_grid("A", "K", GridType::Scalar)
        .variable_with_grid("T", "K", GridType::Scalar);
    let mut model = builder()
        .with_schema(schema)
        .with_component(Arc::new(Emitter {
            declared: vec![
                VariableDefinition::scalar_output("A", "K"),
                VariableDefinition::four_box_output("T", "K"),
            ],
            emitted: vec![
                ("A".into(), StateValue::Scalar(1.0)),
                (
                    "T".into(),
                    StateValue::Hemispheric(HemisphericSlice::from_array([1.0, 2.0])),
                ),
            ],
        }))
        .build()
        .unwrap();

    let error = first_step_error(&mut model);
    assert!(error.contains("could not write T"), "{error}");
    assert_a_unwritten(&model);
}

#[test]
fn grid_mismatch_commits_nothing() {
    let mut model = builder()
        .with_component(Arc::new(Emitter {
            declared: vec![
                VariableDefinition::scalar_output("A", "K"),
                VariableDefinition::scalar_output("C", "K"),
            ],
            emitted: vec![
                ("A".into(), StateValue::Scalar(1.0)),
                (
                    "C".into(),
                    StateValue::FourBox(FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0])),
                ),
            ],
        }))
        .build()
        .unwrap();

    let error = first_step_error(&mut model);
    assert!(error.contains("could not write C"), "{error}");
    assert_a_unwritten(&model);
}
