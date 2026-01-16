//! Grid transformation components for automatic grid conversion
//!
//! These components are automatically inserted by the model builder when
//! connecting components with different grid types. They perform the
//! necessary aggregation to transform values from finer to coarser grids.

use crate::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use crate::errors::RSCMResult;
use crate::spatial::{FourBoxGrid, HemisphericGrid, SpatialGrid};
use crate::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Component that transforms grid values from FourBox to Scalar (global aggregation)
///
/// This component is automatically inserted by the model builder when a FourBox
/// producer connects to a Scalar consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourBoxToScalarTransform {
    /// Name of the variable being transformed
    variable_name: String,
    /// Unit of the variable
    unit: String,
    /// Grid weights for aggregation
    grid: FourBoxGrid,
}

impl FourBoxToScalarTransform {
    /// Create a new FourBox to Scalar transformation
    pub fn new(variable_name: &str, unit: &str, grid: FourBoxGrid) -> Self {
        Self {
            variable_name: variable_name.to_string(),
            unit: unit.to_string(),
            grid,
        }
    }

    /// Create with default MAGICC standard weights
    pub fn with_standard_weights(variable_name: &str, unit: &str) -> Self {
        Self::new(variable_name, unit, FourBoxGrid::magicc_standard())
    }

    /// Get the input variable name (FourBox suffixed)
    fn input_name(&self) -> String {
        format!("{}|FourBox", self.variable_name)
    }
}

#[typetag::serde]
impl Component for FourBoxToScalarTransform {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::with_grid(
                &self.input_name(),
                &self.unit,
                RequirementType::Input,
                GridType::FourBox,
            ),
            RequirementDefinition::with_grid(
                &self.variable_name,
                &self.unit,
                RequirementType::Output,
                GridType::Scalar,
            ),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let input_value = input_state.get_latest_value(&self.input_name());

        let global = match input_value {
            Some(crate::state::StateValue::FourBox(slice)) => {
                self.grid.aggregate_global(&slice.as_array().to_vec())
            }
            Some(crate::state::StateValue::Hemispheric(slice)) => {
                let values = slice.as_array().to_vec();
                let hemispheric_grid = crate::spatial::HemisphericGrid::equal_weights();
                hemispheric_grid.aggregate_global(&values)
            }
            Some(crate::state::StateValue::Scalar(v)) => v, // Already scalar
            None => FloatValue::NAN,
        };

        let mut output = HashMap::new();
        output.insert(
            self.variable_name.clone(),
            crate::state::StateValue::Scalar(global),
        );
        Ok(output)
    }
}

/// Component that transforms grid values from FourBox to Hemispheric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourBoxToHemisphericTransform {
    /// Name of the variable being transformed
    variable_name: String,
    /// Unit of the variable
    unit: String,
    /// Grid weights for aggregation
    grid: FourBoxGrid,
}

impl FourBoxToHemisphericTransform {
    /// Create a new FourBox to Hemispheric transformation
    pub fn new(variable_name: &str, unit: &str, grid: FourBoxGrid) -> Self {
        Self {
            variable_name: variable_name.to_string(),
            unit: unit.to_string(),
            grid,
        }
    }

    /// Create with default MAGICC standard weights
    pub fn with_standard_weights(variable_name: &str, unit: &str) -> Self {
        Self::new(variable_name, unit, FourBoxGrid::magicc_standard())
    }

    /// Get the input variable name (FourBox suffixed)
    fn input_name(&self) -> String {
        format!("{}|FourBox", self.variable_name)
    }

    /// Get the output variable name (Hemispheric suffixed)
    fn output_name(&self) -> String {
        format!("{}|Hemispheric", self.variable_name)
    }
}

#[typetag::serde]
impl Component for FourBoxToHemisphericTransform {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::with_grid(
                &self.input_name(),
                &self.unit,
                RequirementType::Input,
                GridType::FourBox,
            ),
            RequirementDefinition::with_grid(
                &self.output_name(),
                &self.unit,
                RequirementType::Output,
                GridType::Hemispheric,
            ),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let input_value = input_state.get_latest_value(&self.input_name());

        let hemispheric = match input_value {
            Some(crate::state::StateValue::FourBox(slice)) => {
                let target = HemisphericGrid::equal_weights();
                self.grid
                    .transform_to(&slice.as_array().to_vec(), &target)?
            }
            Some(crate::state::StateValue::Hemispheric(slice)) => slice.as_array().to_vec(),
            Some(crate::state::StateValue::Scalar(v)) => vec![v, v], // Broadcast scalar
            None => vec![FloatValue::NAN, FloatValue::NAN],
        };

        let mut output = HashMap::new();
        output.insert(
            self.output_name(),
            crate::state::StateValue::Hemispheric(crate::state::HemisphericSlice::from_array([
                hemispheric[0],
                hemispheric[1],
            ])),
        );
        Ok(output)
    }
}

/// Component that transforms grid values from Hemispheric to Scalar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HemisphericToScalarTransform {
    /// Name of the variable being transformed
    variable_name: String,
    /// Unit of the variable
    unit: String,
    /// Grid weights for aggregation
    grid: HemisphericGrid,
}

impl HemisphericToScalarTransform {
    /// Create a new Hemispheric to Scalar transformation
    pub fn new(variable_name: &str, unit: &str, grid: HemisphericGrid) -> Self {
        Self {
            variable_name: variable_name.to_string(),
            unit: unit.to_string(),
            grid,
        }
    }

    /// Create with equal hemisphere weights
    pub fn with_equal_weights(variable_name: &str, unit: &str) -> Self {
        Self::new(variable_name, unit, HemisphericGrid::equal_weights())
    }

    /// Get the input variable name (Hemispheric suffixed)
    fn input_name(&self) -> String {
        format!("{}|Hemispheric", self.variable_name)
    }
}

#[typetag::serde]
impl Component for HemisphericToScalarTransform {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::with_grid(
                &self.input_name(),
                &self.unit,
                RequirementType::Input,
                GridType::Hemispheric,
            ),
            RequirementDefinition::with_grid(
                &self.variable_name,
                &self.unit,
                RequirementType::Output,
                GridType::Scalar,
            ),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let input_value = input_state.get_latest_value(&self.input_name());

        let global = match input_value {
            Some(crate::state::StateValue::Hemispheric(slice)) => {
                self.grid.aggregate_global(&slice.as_array().to_vec())
            }
            Some(crate::state::StateValue::Scalar(v)) => v, // Already scalar
            _ => FloatValue::NAN,
        };

        let mut output = HashMap::new();
        output.insert(
            self.variable_name.clone(),
            crate::state::StateValue::Scalar(global),
        );
        Ok(output)
    }
}

/// Helper function to check if two grid types are compatible for connection
///
/// Returns true if:
/// - Grid types are identical
/// - Producer has finer grid than consumer (can aggregate)
pub fn grids_compatible(producer: GridType, consumer: GridType) -> bool {
    match (producer, consumer) {
        // Same type always compatible
        (GridType::Scalar, GridType::Scalar) => true,
        (GridType::FourBox, GridType::FourBox) => true,
        (GridType::Hemispheric, GridType::Hemispheric) => true,

        // Finer to coarser is OK (can aggregate)
        (GridType::FourBox, GridType::Scalar) => true,
        (GridType::FourBox, GridType::Hemispheric) => true,
        (GridType::Hemispheric, GridType::Scalar) => true,

        // Coarser to finer is NOT OK (cannot disaggregate without assumptions)
        (GridType::Scalar, GridType::FourBox) => false,
        (GridType::Scalar, GridType::Hemispheric) => false,
        (GridType::Hemispheric, GridType::FourBox) => false,
    }
}

/// Check if auto-transformation is needed between two grid types
pub fn needs_transform(producer: GridType, consumer: GridType) -> bool {
    producer != consumer && grids_compatible(producer, consumer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grids_compatible_same_type() {
        assert!(grids_compatible(GridType::Scalar, GridType::Scalar));
        assert!(grids_compatible(GridType::FourBox, GridType::FourBox));
        assert!(grids_compatible(
            GridType::Hemispheric,
            GridType::Hemispheric
        ));
    }

    #[test]
    fn test_grids_compatible_finer_to_coarser() {
        assert!(grids_compatible(GridType::FourBox, GridType::Scalar));
        assert!(grids_compatible(GridType::FourBox, GridType::Hemispheric));
        assert!(grids_compatible(GridType::Hemispheric, GridType::Scalar));
    }

    #[test]
    fn test_grids_incompatible_coarser_to_finer() {
        assert!(!grids_compatible(GridType::Scalar, GridType::FourBox));
        assert!(!grids_compatible(GridType::Scalar, GridType::Hemispheric));
        assert!(!grids_compatible(GridType::Hemispheric, GridType::FourBox));
    }

    #[test]
    fn test_needs_transform() {
        // Same type: no transform needed
        assert!(!needs_transform(GridType::Scalar, GridType::Scalar));
        assert!(!needs_transform(GridType::FourBox, GridType::FourBox));

        // Different but compatible: transform needed
        assert!(needs_transform(GridType::FourBox, GridType::Scalar));
        assert!(needs_transform(GridType::FourBox, GridType::Hemispheric));
        assert!(needs_transform(GridType::Hemispheric, GridType::Scalar));

        // Incompatible: no transform (would error)
        assert!(!needs_transform(GridType::Scalar, GridType::FourBox));
    }

    #[test]
    fn test_four_box_to_scalar_transform_definitions() {
        let transform = FourBoxToScalarTransform::with_standard_weights("Temperature", "K");
        let defs = transform.definitions();

        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "Temperature|FourBox");
        assert_eq!(defs[0].grid_type, GridType::FourBox);
        assert_eq!(defs[1].name, "Temperature");
        assert_eq!(defs[1].grid_type, GridType::Scalar);
    }

    #[test]
    fn test_hemispheric_to_scalar_transform_definitions() {
        let transform = HemisphericToScalarTransform::with_equal_weights("Precipitation", "mm/yr");
        let defs = transform.definitions();

        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "Precipitation|Hemispheric");
        assert_eq!(defs[0].grid_type, GridType::Hemispheric);
        assert_eq!(defs[1].name, "Precipitation");
        assert_eq!(defs[1].grid_type, GridType::Scalar);
    }

    // Integration tests for grid auto-transform
    mod integration_tests {
        use super::*;
        use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use crate::spatial::FourBoxGrid;
        use crate::timeseries::{GridTimeseries, TimeAxis};
        use crate::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};
        use numpy::array;
        use numpy::ndarray::Array2;
        use std::sync::Arc;

        /// Helper to create a FourBox timeseries item for testing
        fn create_four_box_item(name: &str, values: [FloatValue; 4]) -> TimeseriesItem {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
            let data = Array2::from_shape_vec(
                (2, 4),
                vec![
                    values[0], values[1], values[2], values[3], // Time 0
                    values[0], values[1], values[2], values[3], // Time 1
                ],
            )
            .unwrap();

            let ts = GridTimeseries::new(
                data,
                time_axis,
                grid,
                "K".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            TimeseriesItem {
                data: TimeseriesData::FourBox(ts),
                name: name.to_string(),
                variable_type: VariableType::Endogenous,
            }
        }

        /// Helper to create a Hemispheric timeseries item for testing
        fn create_hemispheric_item(name: &str, values: [FloatValue; 2]) -> TimeseriesItem {
            use crate::spatial::HemisphericGrid;

            let grid = HemisphericGrid::equal_weights();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
            let data = Array2::from_shape_vec(
                (2, 2),
                vec![
                    values[0], values[1], // Time 0
                    values[0], values[1], // Time 1
                ],
            )
            .unwrap();

            let ts = GridTimeseries::new(
                data,
                time_axis,
                grid,
                "mm/yr".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            TimeseriesItem {
                data: TimeseriesData::Hemispheric(ts),
                name: name.to_string(),
                variable_type: VariableType::Endogenous,
            }
        }

        #[test]
        fn test_four_box_to_scalar_transform_solve() {
            // Create transform component
            let transform = FourBoxToScalarTransform::with_standard_weights("Temperature", "K");

            // Create input state with FourBox values
            // Values: NO=16.0, NL=14.0, SO=12.0, SL=10.0
            // With equal weights, global average = (16+14+12+10)/4 = 13.0
            let item = create_four_box_item("Temperature|FourBox", [16.0, 14.0, 12.0, 10.0]);
            let input_state = InputState::build(vec![&item], 2000.0);

            // Run the transform
            let output = transform.solve(2000.0, 2001.0, &input_state).unwrap();

            // Check the output is correctly aggregated
            assert_eq!(
                output.get("Temperature"),
                Some(&crate::state::StateValue::Scalar(13.0))
            );
        }

        #[test]
        fn test_four_box_to_scalar_transform_with_nan_input() {
            let transform = FourBoxToScalarTransform::with_standard_weights("Temperature", "K");

            // Create empty input state (missing variable)
            let input_state = InputState::empty();

            // Run the transform - should produce NaN when variable is missing
            let output = transform.solve(2000.0, 2001.0, &input_state).unwrap();

            let value = output.get("Temperature").unwrap();
            if let crate::state::StateValue::Scalar(v) = value {
                assert!(v.is_nan());
            } else {
                panic!("Expected scalar output");
            }
        }

        #[test]
        fn test_hemispheric_to_scalar_transform_solve() {
            // Create transform component
            let transform =
                HemisphericToScalarTransform::with_equal_weights("Precipitation", "mm/yr");

            // Create input state with Hemispheric values
            // Values: N=1000.0, S=500.0
            // With equal weights, global average = (1000+500)/2 = 750.0
            let item = create_hemispheric_item("Precipitation|Hemispheric", [1000.0, 500.0]);
            let input_state = InputState::build(vec![&item], 2000.0);

            // Run the transform
            let output = transform.solve(2000.0, 2001.0, &input_state).unwrap();

            // Check the output is correctly aggregated
            assert_eq!(
                output.get("Precipitation"),
                Some(&crate::state::StateValue::Scalar(750.0))
            );
        }

        #[test]
        fn test_hemispheric_to_scalar_with_weighted_grid() {
            // Create transform with custom weights (70% NH, 30% SH)
            let grid = HemisphericGrid::with_weights([0.7, 0.3]);
            let transform = HemisphericToScalarTransform::new("Precipitation", "mm/yr", grid);

            // Create input state with Hemispheric values
            // Values: N=1000.0, S=500.0
            // With weights 0.7/0.3, global average = 1000*0.7 + 500*0.3 = 700 + 150 = 850.0
            let item = create_hemispheric_item("Precipitation|Hemispheric", [1000.0, 500.0]);
            let input_state = InputState::build(vec![&item], 2000.0);

            // Run the transform
            let output = transform.solve(2000.0, 2001.0, &input_state).unwrap();

            // Check the output is correctly aggregated
            assert_eq!(
                output.get("Precipitation"),
                Some(&crate::state::StateValue::Scalar(850.0))
            );
        }

        #[test]
        fn test_four_box_to_scalar_with_weighted_grid() {
            // Create a grid with custom weights (ocean-heavy: 40% NO, 10% NL, 40% SO, 10% SL)
            let grid = FourBoxGrid::with_weights([0.4, 0.1, 0.4, 0.1]);
            let transform = FourBoxToScalarTransform::new("Temperature", "K", grid);

            // Values: NO=16.0, NL=14.0, SO=12.0, SL=10.0
            // Weighted average = 16*0.4 + 14*0.1 + 12*0.4 + 10*0.1 = 6.4 + 1.4 + 4.8 + 1.0 = 13.6
            let item = create_four_box_item("Temperature|FourBox", [16.0, 14.0, 12.0, 10.0]);
            let input_state = InputState::build(vec![&item], 2000.0);

            let output = transform.solve(2000.0, 2001.0, &input_state).unwrap();

            let expected = 16.0 * 0.4 + 14.0 * 0.1 + 12.0 * 0.4 + 10.0 * 0.1;
            let value = output.get("Temperature").unwrap();
            if let crate::state::StateValue::Scalar(v) = value {
                assert!((v - expected).abs() < 1e-10);
            } else {
                panic!("Expected scalar output");
            }
        }
    }
}
