//! Tests for read-side grid auto-aggregation during model execution.
//!
//! These tests verify that when a component reads a variable at a coarser resolution
//! than the schema declares, the model automatically aggregates the data on read.

use crate::component::{Component, GridType, InputState, OutputState, RequirementDefinition};
use crate::errors::RSCMResult;
use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
use crate::model::ModelBuilder;
use crate::schema::VariableSchema;
use crate::spatial::{FourBoxGrid, HemisphericGrid, HemisphericRegion};
use crate::state::{FourBoxSlice, StateValue};
use crate::timeseries::{FloatValue, GridTimeseries, Time, TimeAxis};
use crate::timeseries_collection::VariableType;
use is_close::is_close;
use numpy::ndarray::{Array, Array2};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Component that reads a scalar input and outputs a scalar result.
/// Used to test read-side aggregation when schema has FourBox/Hemispheric data.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalarReader {
    input_var: String,
    output_var: String,
}

#[typetag::serde]
impl Component for ScalarReader {
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
        // Read scalar value (model should auto-aggregate if source is grid)
        let value = input_state.get_scalar_window(&self.input_var).at_start();
        let mut output = OutputState::new();
        output.insert(self.output_var.clone(), StateValue::Scalar(value));
        Ok(output)
    }
}

/// Component that reads hemispheric input and outputs a scalar result.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HemisphericReader {
    input_var: String,
    output_var: String,
}

#[typetag::serde]
impl Component for HemisphericReader {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::hemispheric_input(&self.input_var, "K"),
            RequirementDefinition::scalar_output(&self.output_var, "K"),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let window = input_state.get_hemispheric_window(&self.input_var);
        let northern = window.at_start(HemisphericRegion::Northern);
        let southern = window.at_start(HemisphericRegion::Southern);
        // Return mean as output
        let mean = (northern + southern) / 2.0;
        let mut output = OutputState::new();
        output.insert(self.output_var.clone(), StateValue::Scalar(mean));
        Ok(output)
    }
}

/// Component that reads FourBox input (used for error case testing).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FourBoxReader {
    input_var: String,
    output_var: String,
}

#[typetag::serde]
impl Component for FourBoxReader {
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
        let window = input_state.get_four_box_window(&self.input_var);
        let global = window.current_global();
        let mut output = OutputState::new();
        output.insert(self.output_var.clone(), StateValue::Scalar(global));
        Ok(output)
    }
}

/// Component that writes FourBox output (producer for chained tests).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FourBoxWriter {
    var_name: String,
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

/// Helper to create FourBox exogenous data.
fn create_four_box_exogenous(
    name: &str,
    values: [[f64; 4]; 3], // 3 timesteps, 4 regions each
) -> (String, GridTimeseries<FloatValue, FourBoxGrid>) {
    let grid = FourBoxGrid::magicc_standard();
    let time_axis = Arc::new(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)));
    let flat_values: Vec<f64> = values.iter().flat_map(|row| row.iter().copied()).collect();
    let data = Array2::from_shape_vec((3, 4), flat_values).unwrap();

    let ts = GridTimeseries::new(
        data,
        time_axis,
        grid,
        "K".to_string(),
        InterpolationStrategy::from(LinearSplineStrategy::new(true)),
    );
    (name.to_string(), ts)
}

/// Helper to create Hemispheric exogenous data.
fn create_hemispheric_exogenous(
    name: &str,
    values: [[f64; 2]; 3], // 3 timesteps, 2 hemispheres each
) -> (String, GridTimeseries<FloatValue, HemisphericGrid>) {
    let grid = HemisphericGrid::equal_weights();
    let time_axis = Arc::new(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)));
    let flat_values: Vec<f64> = values.iter().flat_map(|row| row.iter().copied()).collect();
    let data = Array2::from_shape_vec((3, 2), flat_values).unwrap();

    let ts = GridTimeseries::new(
        data,
        time_axis,
        grid,
        "K".to_string(),
        InterpolationStrategy::from(LinearSplineStrategy::new(true)),
    );
    (name.to_string(), ts)
}

#[test]
fn test_read_aggregation_fourbox_to_scalar() {
    // Schema declares FourBox variable, component reads as Scalar
    // Model should auto-aggregate FourBox to Scalar on read
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::FourBox)
        .variable("GlobalTemp", "K");

    // FourBox data: [NO, NL, SO, SL] for each timestep
    let (name, ts) = create_four_box_exogenous(
        "Temperature",
        [
            [10.0, 20.0, 30.0, 40.0], // t=2020
            [11.0, 21.0, 31.0, 41.0], // t=2021
            [12.0, 22.0, 32.0, 42.0], // t=2022
        ],
    );

    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(ScalarReader {
            input_var: "Temperature".to_string(),
            output_var: "GlobalTemp".to_string(),
        }));
    builder
        .exogenous_variables
        .add_four_box_timeseries(name, ts, VariableType::Exogenous);

    let mut model = builder.build().expect("Model should build");

    // Verify read transform is registered
    assert!(
        model.read_transforms().contains_key("Temperature"),
        "Read transform should be registered for Temperature"
    );

    // Run one step
    model.step();

    // Check the output - should be the aggregated (mean) of the FourBox values
    let data = model.timeseries().get_data("GlobalTemp").unwrap();
    let ts = data.as_scalar().expect("Should be Scalar");
    let value = ts.at_scalar(1).expect("Should have value at index 1");

    // With equal weights [0.25, 0.25, 0.25, 0.25] at t=2020:
    // (10 + 20 + 30 + 40) / 4 = 25.0
    assert!(
        is_close!(value, 25.0),
        "Expected aggregated value 25.0, got {}",
        value
    );
}

#[test]
fn test_read_aggregation_fourbox_to_hemispheric() {
    // Schema declares FourBox variable, component reads as Hemispheric
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::FourBox)
        .variable("MeanTemp", "K");

    let (name, ts) = create_four_box_exogenous(
        "Temperature",
        [
            [10.0, 20.0, 30.0, 40.0], // [NO, NL, SO, SL]
            [10.0, 20.0, 30.0, 40.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
    );

    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(HemisphericReader {
            input_var: "Temperature".to_string(),
            output_var: "MeanTemp".to_string(),
        }));
    builder
        .exogenous_variables
        .add_four_box_timeseries(name, ts, VariableType::Exogenous);

    let mut model = builder.build().expect("Model should build");

    // Verify read transform is registered
    assert!(
        model.read_transforms().contains_key("Temperature"),
        "Read transform should be registered"
    );

    model.step();

    let data = model.timeseries().get_data("MeanTemp").unwrap();
    let ts = data.as_scalar().expect("Should be Scalar");
    let value = ts.at_scalar(1).expect("Should have value at index 1");

    // FourBox [10, 20, 30, 40] -> Hemispheric [15, 35] (mean of each hemisphere)
    // Then component computes mean: (15 + 35) / 2 = 25.0
    assert!(
        is_close!(value, 25.0),
        "Expected mean of hemispheres 25.0, got {}",
        value
    );
}

#[test]
fn test_read_aggregation_hemispheric_to_scalar() {
    // Schema declares Hemispheric variable, component reads as Scalar
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::Hemispheric)
        .variable("GlobalTemp", "K");

    let (name, ts) = create_hemispheric_exogenous(
        "Temperature",
        [
            [15.0, 35.0], // [Northern, Southern]
            [16.0, 36.0],
            [17.0, 37.0],
        ],
    );

    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(ScalarReader {
            input_var: "Temperature".to_string(),
            output_var: "GlobalTemp".to_string(),
        }));
    builder
        .exogenous_variables
        .add_hemispheric_timeseries(name, ts, VariableType::Exogenous);

    let mut model = builder.build().expect("Model should build");

    assert!(
        model.read_transforms().contains_key("Temperature"),
        "Read transform should be registered"
    );

    model.step();

    let data = model.timeseries().get_data("GlobalTemp").unwrap();
    let ts = data.as_scalar().expect("Should be Scalar");
    let value = ts.at_scalar(1).expect("Should have value at index 1");

    // With equal weights [0.5, 0.5]: (15 + 35) / 2 = 25.0
    assert!(
        is_close!(value, 25.0),
        "Expected aggregated value 25.0, got {}",
        value
    );
}

#[test]
fn test_read_aggregation_multiple_consumers() {
    // Two components reading same FourBox variable at different resolutions
    // One reads as Scalar, other reads as Hemispheric
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::FourBox)
        .variable("GlobalTemp", "K")
        .variable("HemisphericMean", "K");

    let (name, ts) = create_four_box_exogenous(
        "Temperature",
        [
            [10.0, 20.0, 30.0, 40.0],
            [10.0, 20.0, 30.0, 40.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
    );

    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(ScalarReader {
            input_var: "Temperature".to_string(),
            output_var: "GlobalTemp".to_string(),
        }))
        .with_component(Arc::new(HemisphericReader {
            input_var: "Temperature".to_string(),
            output_var: "HemisphericMean".to_string(),
        }));
    builder
        .exogenous_variables
        .add_four_box_timeseries(name, ts, VariableType::Exogenous);

    let mut model = builder.build().expect("Model should build");

    model.step();

    // ScalarReader should get mean of all 4: 25.0
    let global = model
        .timeseries()
        .get_data("GlobalTemp")
        .unwrap()
        .as_scalar()
        .unwrap()
        .at_scalar(1)
        .unwrap();
    assert!(
        is_close!(global, 25.0),
        "Expected global 25.0, got {}",
        global
    );

    // HemisphericReader gets hemispheric aggregation [15, 35], then computes mean: 25.0
    let hemi_mean = model
        .timeseries()
        .get_data("HemisphericMean")
        .unwrap()
        .as_scalar()
        .unwrap()
        .at_scalar(1)
        .unwrap();
    assert!(
        is_close!(hemi_mean, 25.0),
        "Expected hemispheric mean 25.0, got {}",
        hemi_mean
    );
}

#[test]
fn test_read_disaggregation_scalar_to_fourbox_rejected() {
    // Schema declares Scalar, but component wants FourBox input
    // This is disaggregation (broadcast) and should be rejected at build time
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::Scalar)
        .variable("Result", "K");

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(FourBoxReader {
            input_var: "Temperature".to_string(),
            output_var: "Result".to_string(),
        }))
        .build();

    assert!(result.is_err(), "Disaggregation should be rejected");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Grid transformation not supported"),
        "Error should mention unsupported transformation: {}",
        msg
    );
}

#[test]
fn test_read_aggregation_chain_write_then_read() {
    // Test: FourBoxWriter writes FourBox -> schema is FourBox
    //       -> ScalarReader reads at_start (from previous timestep) as Scalar (read aggregation)
    //
    // ScalarReader uses at_start() which reads from the previous timestep.
    // We need to run at least 2 steps: first step populates, second step reads.
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::FourBox)
        .variable("GlobalTemp", "K");

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2024.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(FourBoxWriter {
            var_name: "Temperature".to_string(),
            values: [10.0, 20.0, 30.0, 40.0],
        }))
        .with_component(Arc::new(ScalarReader {
            input_var: "Temperature".to_string(),
            output_var: "GlobalTemp".to_string(),
        }))
        .build()
        .expect("Model should build");

    // FourBoxWriter writes FourBox, ScalarReader reads as Scalar via read aggregation
    assert!(
        model.read_transforms().contains_key("Temperature"),
        "Read transform should be registered"
    );

    // First step: FourBoxWriter writes to index 1, ScalarReader reads from index 0 (NaN)
    model.step();

    // Second step: FourBoxWriter writes to index 2, ScalarReader reads from index 1 (the FourBox value)
    model.step();

    // Check the value at index 2 - ScalarReader read the aggregated value from step 1
    let value = model
        .timeseries()
        .get_data("GlobalTemp")
        .unwrap()
        .as_scalar()
        .unwrap()
        .at_scalar(2)
        .unwrap();

    // FourBox [10, 20, 30, 40] aggregated to Scalar: 25.0
    assert!(
        is_close!(value, 25.0),
        "Expected aggregated value 25.0, got {}",
        value
    );
}
