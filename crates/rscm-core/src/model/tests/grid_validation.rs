//! Grid validation tests: grid type mismatch detection.

use crate::component::{Component, InputState, OutputState, RequirementDefinition};
use crate::errors::RSCMResult;
use crate::model::ModelBuilder;
use crate::state::{FourBoxSlice, StateValue};
use crate::timeseries::{Time, TimeAxis};
use numpy::ndarray::Array;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A component that produces a FourBox output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourBoxProducer;

#[typetag::serde]
impl Component for FourBoxProducer {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![RequirementDefinition::four_box_output("Temperature", "K")]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        _input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let mut output = OutputState::new();
        output.insert(
            "Temperature".to_string(),
            StateValue::FourBox(FourBoxSlice::from_array([288.0, 290.0, 287.0, 285.0])),
        );
        Ok(output)
    }
}

/// A component that expects a Scalar input
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalarConsumer;

#[typetag::serde]
impl Component for ScalarConsumer {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::scalar_input("Temperature", "K"),
            RequirementDefinition::scalar_output("Result", "W / m^2"),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let temp = input_state.get_scalar_window("Temperature").at_start();
        let mut output = OutputState::new();
        output.insert("Result".to_string(), StateValue::Scalar(temp * 2.0));
        Ok(output)
    }
}

/// A component that expects a FourBox input (compatible with FourBoxProducer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourBoxConsumer;

#[typetag::serde]
impl Component for FourBoxConsumer {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::four_box_input("Temperature", "K"),
            RequirementDefinition::scalar_output("GlobalTemperature", "K"),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let temp = input_state
            .get_four_box_window("Temperature")
            .current_global();
        let mut output = OutputState::new();
        output.insert("GlobalTemperature".to_string(), StateValue::Scalar(temp));
        Ok(output)
    }
}

#[test]
fn test_grid_type_mismatch_returns_error() {
    // This should return an error because FourBoxProducer outputs FourBox
    // but ScalarConsumer expects Scalar
    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(FourBoxProducer))
        .with_component(Arc::new(ScalarConsumer))
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(err_msg.contains("Grid type mismatch for variable 'Temperature'"));
    assert!(err_msg.contains("FourBoxProducer"));
    assert!(err_msg.contains("ScalarConsumer"));
    assert!(err_msg.contains("FourBox"));
    assert!(err_msg.contains("Scalar"));
}

#[test]
fn test_matching_grid_types_ok() {
    // This should work because both use FourBox for Temperature
    let _model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(FourBoxProducer))
        .with_component(Arc::new(FourBoxConsumer))
        .build()
        .unwrap();
}
