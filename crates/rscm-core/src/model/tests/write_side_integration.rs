//! Integration tests for write-side grid aggregation during model execution.

use crate::component::{Component, GridType, InputState, OutputState, RequirementDefinition};
use crate::errors::RSCMResult;
use crate::model::ModelBuilder;
use crate::schema::VariableSchema;
use crate::state::{FourBoxSlice, HemisphericSlice, StateValue};
use crate::timeseries::{FloatValue, Time, TimeAxis};
use is_close::is_close;
use numpy::ndarray::Array;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Component that produces FourBox output with known values
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FourBoxWriter {
    var_name: String,
    /// Values to produce: [NO, NL, SO, SL]
    values: [FloatValue; 4],
}

#[typetag::serde]
impl Component for FourBoxWriter {
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
            StateValue::FourBox(FourBoxSlice::from_array(self.values)),
        );
        Ok(output)
    }
}

/// Component that produces Hemispheric output with known values
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HemisphericWriter {
    var_name: String,
    /// Values to produce: [Northern, Southern]
    values: [FloatValue; 2],
}

#[typetag::serde]
impl Component for HemisphericWriter {
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
            StateValue::Hemispheric(HemisphericSlice::from(self.values)),
        );
        Ok(output)
    }
}

#[test]
fn test_write_aggregation_fourbox_to_scalar_execution() {
    // Schema declares Scalar, component produces FourBox
    // The model should aggregate FourBox values to Scalar on write
    let schema = VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(FourBoxWriter {
            var_name: "Temperature".to_string(),
            values: [10.0, 20.0, 30.0, 40.0], // [NO, NL, SO, SL]
        }))
        .build()
        .expect("Model should build");

    // Verify write transform is registered
    assert!(
        model.write_transforms().contains_key("Temperature"),
        "Write transform should be registered"
    );

    // Run one step
    model.step();

    // Check the collection has scalar data (not FourBox)
    let data = model.timeseries().get_data("Temperature").unwrap();
    let ts = data.as_scalar().expect("Should be stored as Scalar");

    // Get the value at index 1 (after first step)
    let value = ts.at_scalar(1).expect("Should have value at index 1");

    // With default equal weights [0.25, 0.25, 0.25, 0.25]:
    // 10*0.25 + 20*0.25 + 30*0.25 + 40*0.25 = 25.0
    assert!(
        is_close!(value, 25.0),
        "Expected aggregated value 25.0, got {}",
        value
    );
}

#[test]
fn test_write_aggregation_fourbox_to_scalar_custom_weights() {
    // Schema declares Scalar, component produces FourBox
    // Use custom weights for aggregation
    let schema = VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_schema(schema)
        .with_grid_weights(GridType::FourBox, vec![0.36, 0.14, 0.36, 0.14])
        .with_component(Arc::new(FourBoxWriter {
            var_name: "Temperature".to_string(),
            values: [10.0, 20.0, 30.0, 40.0],
        }))
        .build()
        .expect("Model should build");

    model.step();

    let data = model.timeseries().get_data("Temperature").unwrap();
    let ts = data.as_scalar().expect("Should be stored as Scalar");
    let value = ts.at_scalar(1).expect("Should have value at index 1");

    // With custom weights [0.36, 0.14, 0.36, 0.14]:
    // 10*0.36 + 20*0.14 + 30*0.36 + 40*0.14 = 3.6 + 2.8 + 10.8 + 5.6 = 22.8
    assert!(
        is_close!(value, 22.8),
        "Expected aggregated value 22.8, got {}",
        value
    );
}

#[test]
fn test_write_aggregation_fourbox_to_hemispheric_execution() {
    // Schema declares Hemispheric, component produces FourBox
    let schema =
        VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Hemispheric);

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(FourBoxWriter {
            var_name: "Temperature".to_string(),
            values: [10.0, 20.0, 30.0, 40.0],
        }))
        .build()
        .expect("Model should build");

    model.step();

    let data = model.timeseries().get_data("Temperature").unwrap();
    let ts = data
        .as_hemispheric()
        .expect("Should be stored as Hemispheric");

    let northern = ts.at_index(1, 0).expect("Should have northern value");
    let southern = ts.at_index(1, 1).expect("Should have southern value");

    // With equal weights [0.25, 0.25, 0.25, 0.25]:
    // Northern = (10*0.25 + 20*0.25) / (0.25 + 0.25) = 7.5 / 0.5 = 15.0
    // Southern = (30*0.25 + 40*0.25) / (0.25 + 0.25) = 17.5 / 0.5 = 35.0
    assert!(
        is_close!(northern, 15.0),
        "Expected northern 15.0, got {}",
        northern
    );
    assert!(
        is_close!(southern, 35.0),
        "Expected southern 35.0, got {}",
        southern
    );
}

#[test]
fn test_write_aggregation_hemispheric_to_scalar_execution() {
    // Schema declares Scalar, component produces Hemispheric
    let schema = VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(HemisphericWriter {
            var_name: "Temperature".to_string(),
            values: [15.0, 35.0], // [Northern, Southern]
        }))
        .build()
        .expect("Model should build");

    model.step();

    let data = model.timeseries().get_data("Temperature").unwrap();
    let ts = data.as_scalar().expect("Should be stored as Scalar");
    let value = ts.at_scalar(1).expect("Should have value at index 1");

    // With default equal weights [0.5, 0.5]:
    // 15*0.5 + 35*0.5 = 25.0
    assert!(
        is_close!(value, 25.0),
        "Expected aggregated value 25.0, got {}",
        value
    );
}

#[test]
fn test_write_aggregation_multiple_steps() {
    // Run multiple steps to ensure aggregation works consistently
    let schema = VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(FourBoxWriter {
            var_name: "Temperature".to_string(),
            values: [10.0, 20.0, 30.0, 40.0],
        }))
        .build()
        .expect("Model should build");

    // Run all steps
    model.run();

    let data = model.timeseries().get_data("Temperature").unwrap();
    let ts = data.as_scalar().expect("Should be stored as Scalar");

    // Check values at all indices after initial
    for i in 1..5 {
        let value = ts
            .at_scalar(i)
            .unwrap_or_else(|| panic!("Should have value at index {}", i));
        assert!(
            is_close!(value, 25.0),
            "Expected aggregated value 25.0 at index {}, got {}",
            i,
            value
        );
    }
}

#[test]
fn test_no_schema_no_aggregation() {
    // Without a schema, no aggregation should happen
    // Component produces FourBox, it should stay as FourBox
    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_component(Arc::new(FourBoxWriter {
            var_name: "Temperature".to_string(),
            values: [10.0, 20.0, 30.0, 40.0],
        }))
        .build()
        .expect("Model should build without schema");

    // Verify no write transforms
    assert!(
        model.write_transforms().is_empty(),
        "Should have no write transforms without schema"
    );

    model.step();

    // Data should remain as FourBox
    let data = model.timeseries().get_data("Temperature").unwrap();
    assert!(
        data.as_four_box().is_some(),
        "Should be stored as FourBox without schema"
    );
}
