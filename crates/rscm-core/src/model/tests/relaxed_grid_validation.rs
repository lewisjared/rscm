//! Relaxed grid validation tests with schema-based aggregation.

use crate::component::{Component, GridType, InputState, OutputState, RequirementDefinition};
use crate::errors::RSCMResult;
use crate::model::ModelBuilder;
use crate::schema::VariableSchema;
use crate::state::{FourBoxSlice, HemisphericSlice, StateValue};
use crate::timeseries::{Time, TimeAxis};
use numpy::ndarray::Array;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A component that produces FourBox output
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FourBoxProducer {
    var_name: String,
}

#[typetag::serde]
impl Component for FourBoxProducer {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![RequirementDefinition::four_box_output(&self.var_name, "K")]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        _input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let mut output = OutputState::new();
        output.insert(
            self.var_name.clone(),
            StateValue::FourBox(FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0])),
        );
        Ok(output)
    }
}

/// A component that produces Hemispheric output
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HemisphericProducer {
    var_name: String,
}

#[typetag::serde]
impl Component for HemisphericProducer {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![RequirementDefinition::hemispheric_output(
            &self.var_name,
            "K",
        )]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        _input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let mut output = OutputState::new();
        output.insert(
            self.var_name.clone(),
            StateValue::Hemispheric(HemisphericSlice::from_array([1.0, 2.0])),
        );
        Ok(output)
    }
}

/// A component that produces Scalar output
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalarProducer {
    var_name: String,
}

#[typetag::serde]
impl Component for ScalarProducer {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![RequirementDefinition::scalar_output(&self.var_name, "K")]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        _input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let mut output = OutputState::new();
        output.insert(self.var_name.clone(), StateValue::Scalar(1.5));
        Ok(output)
    }
}

/// A component that consumes Scalar input
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalarConsumer {
    input_var: String,
    output_var: String,
}

#[typetag::serde]
impl Component for ScalarConsumer {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::scalar_input(&self.input_var, "K"),
            RequirementDefinition::scalar_output(&self.output_var, "K"),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let value = input_state.get_scalar_window(&self.input_var).at_start();
        let mut output = OutputState::new();
        output.insert(self.output_var.clone(), StateValue::Scalar(value * 2.0));
        Ok(output)
    }
}

/// A component that consumes FourBox input
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FourBoxConsumer {
    input_var: String,
    output_var: String,
}

#[typetag::serde]
impl Component for FourBoxConsumer {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::four_box_input(&self.input_var, "K"),
            RequirementDefinition::scalar_output(&self.output_var, "K"),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let value = input_state
            .get_four_box_window(&self.input_var)
            .current_global();
        let mut output = OutputState::new();
        output.insert(self.output_var.clone(), StateValue::Scalar(value));
        Ok(output)
    }
}

// Write-side aggregation tests

#[test]
fn test_write_side_fourbox_to_scalar_allowed() {
    // Schema declares Scalar, component produces FourBox
    // Should be allowed (write-side aggregation)
    let schema = VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(FourBoxProducer {
            var_name: "Temperature".to_string(),
        }))
        .build();

    assert!(
        result.is_ok(),
        "Write-side FourBox->Scalar aggregation should be allowed: {:?}",
        result.err()
    );
}

#[test]
fn test_write_side_fourbox_to_hemispheric_allowed() {
    // Schema declares Hemispheric, component produces FourBox
    // Should be allowed (write-side aggregation)
    let schema =
        VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Hemispheric);

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(FourBoxProducer {
            var_name: "Temperature".to_string(),
        }))
        .build();

    assert!(
        result.is_ok(),
        "Write-side FourBox->Hemispheric aggregation should be allowed: {:?}",
        result.err()
    );
}

#[test]
fn test_write_side_hemispheric_to_scalar_allowed() {
    // Schema declares Scalar, component produces Hemispheric
    // Should be allowed (write-side aggregation)
    let schema = VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(HemisphericProducer {
            var_name: "Temperature".to_string(),
        }))
        .build();

    assert!(
        result.is_ok(),
        "Write-side Hemispheric->Scalar aggregation should be allowed: {:?}",
        result.err()
    );
}

#[test]
fn test_write_side_scalar_to_fourbox_rejected() {
    // Schema declares FourBox, component produces Scalar
    // Should be rejected (cannot broadcast/disaggregate)
    let schema = VariableSchema::new().variable_with_grid("Temperature", "K", GridType::FourBox);

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(ScalarProducer {
            var_name: "Temperature".to_string(),
        }))
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Grid transformation not supported"),
        "Should indicate transformation not supported: {}",
        err
    );
}

#[test]
fn test_write_side_scalar_to_hemispheric_rejected() {
    // Schema declares Hemispheric, component produces Scalar
    // Should be rejected (cannot broadcast/disaggregate)
    let schema =
        VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Hemispheric);

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(ScalarProducer {
            var_name: "Temperature".to_string(),
        }))
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Grid transformation not supported"),
        "Should indicate transformation not supported: {}",
        err
    );
}

#[test]
fn test_write_side_hemispheric_to_fourbox_rejected() {
    // Schema declares FourBox, component produces Hemispheric
    // Should be rejected (cannot disaggregate)
    let schema = VariableSchema::new().variable_with_grid("Temperature", "K", GridType::FourBox);

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(HemisphericProducer {
            var_name: "Temperature".to_string(),
        }))
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Grid transformation not supported"),
        "Should indicate transformation not supported: {}",
        err
    );
}

// Read-side aggregation tests

#[test]
fn test_read_side_fourbox_schema_scalar_consumer_allowed() {
    // Schema declares FourBox, component consumes Scalar
    // Should be allowed (read-side aggregation)
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::FourBox)
        .variable("Output", "K");

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(FourBoxProducer {
            var_name: "Temperature".to_string(),
        }))
        .with_component(Arc::new(ScalarConsumer {
            input_var: "Temperature".to_string(),
            output_var: "Output".to_string(),
        }))
        .build();

    assert!(
        result.is_ok(),
        "Read-side FourBox->Scalar aggregation should be allowed: {:?}",
        result.err()
    );
}

#[test]
fn test_read_side_hemispheric_schema_scalar_consumer_allowed() {
    // Schema declares Hemispheric, component consumes Scalar
    // Should be allowed (read-side aggregation)
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::Hemispheric)
        .variable("Output", "K");

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(HemisphericProducer {
            var_name: "Temperature".to_string(),
        }))
        .with_component(Arc::new(ScalarConsumer {
            input_var: "Temperature".to_string(),
            output_var: "Output".to_string(),
        }))
        .build();

    assert!(
        result.is_ok(),
        "Read-side Hemispheric->Scalar aggregation should be allowed: {:?}",
        result.err()
    );
}

#[test]
fn test_read_side_scalar_schema_fourbox_consumer_rejected() {
    // Schema declares Scalar, component consumes FourBox
    // Should be rejected (cannot disaggregate/broadcast)
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::Scalar)
        .variable("Output", "K");

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(ScalarProducer {
            var_name: "Temperature".to_string(),
        }))
        .with_component(Arc::new(FourBoxConsumer {
            input_var: "Temperature".to_string(),
            output_var: "Output".to_string(),
        }))
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Grid transformation not supported"),
        "Should indicate transformation not supported: {}",
        err
    );
}

#[test]
fn test_same_grid_always_allowed() {
    // Same grid types should always be allowed
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::FourBox)
        .variable("Output", "K");

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(FourBoxProducer {
            var_name: "Temperature".to_string(),
        }))
        .with_component(Arc::new(FourBoxConsumer {
            input_var: "Temperature".to_string(),
            output_var: "Output".to_string(),
        }))
        .build();

    assert!(
        result.is_ok(),
        "Same grid types should always be allowed: {:?}",
        result.err()
    );
}
