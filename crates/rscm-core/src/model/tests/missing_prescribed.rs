//! Inputs that no component produces must be supplied.

use crate::component::{Component, InputState, OutputState, VariableDefinition, VariableRole};
use crate::errors::{RSCMError, RSCMResult};
use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
use crate::model::ModelBuilder;
use crate::state::StateValue;
use crate::timeseries::{Time, TimeAxis, Timeseries};
use numpy::array;
use numpy::ndarray::{Array, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Writes `forcing + emissions` to `temperature`.
#[derive(Debug, Serialize, Deserialize)]
struct Warming;

#[typetag::serde]
impl Component for Warming {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::new("forcing", "W/m^2", VariableRole::Input),
            VariableDefinition::new("emissions", "GtC/yr", VariableRole::Input),
            VariableDefinition::new("temperature", "K", VariableRole::Output),
        ]
    }

    fn solve(&self, _: Time, _: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let value = input_state.get_scalar_window("forcing").get()
            + input_state.get_scalar_window("emissions").get();
        Ok(HashMap::from([(
            "temperature".to_string(),
            StateValue::Scalar(value),
        )]))
    }
}

/// Writes `forcing` to `ocean_heat`.
#[derive(Debug, Serialize, Deserialize)]
struct Ocean;

#[typetag::serde]
impl Component for Ocean {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::new("forcing", "W/m^2", VariableRole::Input),
            VariableDefinition::new("ocean_heat", "J", VariableRole::Output),
        ]
    }

    fn solve(&self, _: Time, _: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let value = input_state.get_scalar_window("forcing").get();
        Ok(HashMap::from([(
            "ocean_heat".to_string(),
            StateValue::Scalar(value),
        )]))
    }
}

fn builder() -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder.with_time_axis(TimeAxis::from_values(Array::range(2000.0, 2004.0, 1.0)));
    builder
}

fn series(unit: &str) -> Timeseries<f64> {
    Timeseries::new(
        array![1.0, 1.0].insert_axis(Axis(1)),
        Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
        crate::spatial::ScalarGrid,
        unit.to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    )
}

#[test]
fn missing_inputs_fail_the_build() {
    let error = builder()
        .with_component(Arc::new(Warming))
        .with_component(Arc::new(Ocean))
        .build()
        .expect_err("unsupplied inputs must not build");
    let RSCMError::MissingPrescribedInput { inputs } = &error else {
        panic!("expected MissingPrescribedInput, got {error}");
    };
    assert_eq!(
        inputs,
        &[
            ("emissions".to_string(), vec!["Warming".to_string()]),
            (
                "forcing".to_string(),
                vec!["Ocean".to_string(), "Warming".to_string()]
            ),
        ]
    );
    let message = error.to_string();
    assert!(
        message.contains("'forcing' (read by Ocean, Warming)"),
        "{message}"
    );
    assert!(
        message.contains("with_missing_prescribed_allowed()"),
        "{message}"
    );
}

#[test]
fn prescribed_data_and_initial_values_supply_inputs() {
    builder()
        .with_component(Arc::new(Warming))
        .with_prescribed_variable("forcing", series("W/m^2"))
        .with_initial_values(HashMap::from([("emissions".to_string(), 0.0)]))
        .build()
        .expect("supplied inputs build");
}

#[test]
fn lagged_input_without_producer_must_be_supplied() {
    let error = builder()
        .with_component(Arc::new(Ocean))
        .with_lag("forcing")
        .build()
        .expect_err("a lagged input with no producer is prescribed");
    assert!(
        matches!(&error, RSCMError::MissingPrescribedInput { inputs } if inputs[0].0 == "forcing"),
        "{error}"
    );
}

#[test]
fn flag_fills_missing_inputs_with_nan() {
    let mut model = builder()
        .with_component(Arc::new(Warming))
        .with_missing_prescribed_allowed()
        .build()
        .expect("the flag allows missing inputs");
    let forcing = model
        .timeseries()
        .get_data("forcing")
        .and_then(|data| data.as_scalar())
        .unwrap();
    assert!(forcing.values().iter().all(|v| v.is_nan()));
    model.run();
}

#[test]
fn prescribed_inputs_lists_unproduced_inputs() {
    let mut builder = builder();
    builder
        .with_component(Arc::new(Warming))
        .with_component(Arc::new(Ocean))
        .with_prescribed_variable("forcing", series("W/m^2"));
    let inputs: Vec<_> = builder
        .prescribed_inputs()
        .into_iter()
        .map(|input| (input.definition.name, input.consumers, input.supplied))
        .collect();
    assert_eq!(
        inputs,
        [
            ("emissions".to_string(), vec!["Warming".to_string()], false),
            (
                "forcing".to_string(),
                vec!["Ocean".to_string(), "Warming".to_string()],
                true
            ),
        ]
    );
}

fn constant(value: f64, start: f64, end: f64) -> Timeseries<f64> {
    let values = Array::range(start, end, 1.0);
    Timeseries::new(
        Array::from_elem(values.len(), value).insert_axis(Axis(1)),
        Arc::new(TimeAxis::from_values(values)),
        crate::spatial::ScalarGrid,
        "W/m^2".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    )
}

fn ocean_builder() -> ModelBuilder {
    let mut builder = builder();
    builder.with_component(Arc::new(Ocean));
    builder
}

#[test]
fn build_without_time_axis_fails() {
    let error = ModelBuilder::new()
        .with_component(Arc::new(Ocean))
        .with_prescribed_variable("forcing", series("W/m^2"))
        .build()
        .expect_err("a model needs a time axis");
    assert!(matches!(error, RSCMError::MissingTimeAxis), "{error}");
}

#[test]
fn build_without_components_fails() {
    let error = builder().build().expect_err("a model needs a component");
    assert!(matches!(error, RSCMError::NoComponents), "{error}");
}

#[test]
fn unknown_initial_value_suggests_close_match() {
    let error = ocean_builder()
        .with_prescribed_variable("forcing", series("W/m^2"))
        .with_initial_values(HashMap::from([("ocean_haet".to_string(), 0.0)]))
        .build()
        .expect_err("an initial value nothing reads must not build");
    assert_eq!(
        error.to_string(),
        "Initial value for 'ocean_haet' is not used: no component reads or produces it. \
         Did you mean 'ocean_heat'?"
    );
}

#[test]
fn unknown_initial_grid_value_fails() {
    let error = ocean_builder()
        .with_prescribed_variable("forcing", series("W/m^2"))
        .with_initial_grid_values(HashMap::from([("nothing".to_string(), vec![0.0])]))
        .build()
        .expect_err("an initial value nothing reads must not build");
    assert!(
        matches!(&error, RSCMError::UnknownInitialValue { variable, suggestions } if variable == "nothing" && suggestions.is_empty()),
        "{error}"
    );
}

#[test]
fn unread_prescribed_variable_suggests_close_match() {
    let error = ocean_builder()
        .with_prescribed_variable("forcing", series("W/m^2"))
        .with_prescribed_variable("forcng", series("W/m^2"))
        .build()
        .expect_err("a prescribed variable nothing reads must not build");
    assert_eq!(
        error.to_string(),
        "Prescribed variable 'forcng' is not used: no component reads it. Did you mean 'forcing'?"
    );
}

#[test]
fn unread_variable_in_a_collection_is_ignored() {
    let mut collection = crate::timeseries_collection::TimeseriesCollection::new();
    collection.add_timeseries("forcing".to_string(), series("W/m^2"));
    collection.add_timeseries("unrelated".to_string(), series("W/m^2"));
    ocean_builder()
        .with_prescribed_collection(collection)
        .build()
        .expect("collections may hold extra variables");
}

#[test]
fn prescribing_an_output_fails() {
    let error = ocean_builder()
        .with_prescribed_variable("forcing", series("W/m^2"))
        .with_prescribed_variable("ocean_heat", series("J"))
        .build()
        .expect_err("prescribing a produced variable must not build");
    assert!(
        matches!(&error, RSCMError::PrescribedOutput { variable, component } if variable == "ocean_heat" && component == "Ocean"),
        "{error}"
    );
}

#[test]
fn later_prescribed_data_replaces_earlier() {
    let mut collection = crate::timeseries_collection::TimeseriesCollection::new();
    collection.add_timeseries("forcing".to_string(), constant(1.0, 2000.0, 2004.0));
    let mut builder = ocean_builder();
    builder
        .with_prescribed_collection(collection.clone())
        .with_prescribed_variable("forcing", constant(2.0, 2000.0, 2004.0))
        .with_prescribed_variable("forcing", constant(3.0, 2000.0, 2004.0));
    let forcing = |builder: &ModelBuilder| {
        builder
            .build()
            .unwrap()
            .timeseries()
            .get_data("forcing")
            .unwrap()
            .as_scalar()
            .unwrap()
            .values()[[0, 0]]
    };
    assert_eq!(forcing(&builder), 3.0);
    builder.with_prescribed_collection(collection);
    assert_eq!(forcing(&builder), 1.0);
}

#[test]
fn extrapolated_prescribed_data_warns() {
    let (_, warnings) = ocean_builder()
        .with_prescribed_variable("forcing", constant(1.0, 2001.0, 2003.0))
        .build_with_warnings()
        .unwrap();
    assert_eq!(
        warnings,
        ["Prescribed data for 'forcing' covers 2001 to 2003, so it is extrapolated back to 2000"]
    );
    let mut builder = ocean_builder();
    builder.with_time_axis(TimeAxis::from_values(Array::range(2000.0, 2010.0, 1.0)));
    let (_, warnings) = builder
        .with_prescribed_variable("forcing", constant(1.0, 2001.0, 2003.0))
        .build_with_warnings()
        .unwrap();
    assert_eq!(
        warnings,
        ["Prescribed data for 'forcing' covers 2001 to 2003, \
          so it is extrapolated back to 2000 and forward to 2009"]
    );
}

#[test]
fn covering_prescribed_data_does_not_warn() {
    let (_, warnings) = ocean_builder()
        .with_prescribed_variable("forcing", constant(1.0, 2000.0, 2004.0))
        .build_with_warnings()
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
}
