//! Inputs that no component produces must be supplied.

use crate::component::{
    Component, InputState, OutputState, RequirementDefinition, RequirementType,
};
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
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::new("forcing", "W/m^2", RequirementType::Input),
            RequirementDefinition::new("emissions", "GtC/yr", RequirementType::Input),
            RequirementDefinition::new("temperature", "K", RequirementType::Output),
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
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::new("forcing", "W/m^2", RequirementType::Input),
            RequirementDefinition::new("ocean_heat", "J", RequirementType::Output),
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
    let RSCMError::MissingExogenousInput { inputs } = &error else {
        panic!("expected MissingExogenousInput, got {error}");
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
        message.contains("with_missing_exogenous_allowed()"),
        "{message}"
    );
}

#[test]
fn exogenous_data_and_initial_values_supply_inputs() {
    builder()
        .with_component(Arc::new(Warming))
        .with_exogenous_variable("forcing", series("W/m^2"))
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
        .expect_err("a lagged input with no producer is exogenous");
    assert!(
        matches!(&error, RSCMError::MissingExogenousInput { inputs } if inputs[0].0 == "forcing"),
        "{error}"
    );
}

#[test]
fn flag_fills_missing_inputs_with_nan() {
    let mut model = builder()
        .with_component(Arc::new(Warming))
        .with_missing_exogenous_allowed()
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
