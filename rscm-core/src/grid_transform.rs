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
            Some(crate::state::StateValue::Grid(values)) => self.grid.aggregate_global(&values),
            Some(crate::state::StateValue::Scalar(v)) => v, // Already scalar
            None => FloatValue::NAN,
        };

        let mut output = HashMap::new();
        output.insert(self.variable_name.clone(), global);
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
            Some(crate::state::StateValue::Grid(values)) => {
                let target = HemisphericGrid::equal_weights();
                self.grid.transform_to(&values, &target)?
            }
            Some(crate::state::StateValue::Scalar(v)) => vec![v, v], // Broadcast scalar
            None => vec![FloatValue::NAN, FloatValue::NAN],
        };

        // Note: For hemispheric output we need grid timeseries support
        // For now, just output the northern value as a simple case
        let mut output = HashMap::new();
        // TODO: This should output a grid timeseries, not a scalar
        // For now, we output the average as a placeholder
        output.insert(self.output_name(), (hemispheric[0] + hemispheric[1]) / 2.0);
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
            Some(crate::state::StateValue::Grid(values)) => self.grid.aggregate_global(&values),
            Some(crate::state::StateValue::Scalar(v)) => v, // Already scalar
            None => FloatValue::NAN,
        };

        let mut output = HashMap::new();
        output.insert(self.variable_name.clone(), global);
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
}
