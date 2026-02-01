//! Type definitions for the model module.

use crate::component::{Component, GridType, RequirementDefinition};
use crate::units::Unit;
use petgraph::Graph;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Type alias for a component wrapped in an Arc for shared ownership.
pub type C = Arc<dyn Component>;

/// Type alias for the component dependency graph.
pub type CGraph = Graph<C, RequirementDefinition>;

/// Internal definition tracking for a variable during model building.
#[derive(Debug)]
pub(crate) struct VariableDefinition {
    pub name: String,
    /// The unit string (original, preserved for display/error messages).
    pub unit: String,
    /// The parsed unit (if parsing succeeded), used for compatibility checks.
    pub parsed_unit: Option<Unit>,
    pub grid_type: GridType,
}

impl VariableDefinition {
    pub fn from_requirement_definition(definition: &RequirementDefinition) -> Self {
        // Attempt to parse the unit; store None if parsing fails
        // (the error will be reported separately during validation)
        let parsed_unit = Unit::parse(&definition.unit).ok();

        Self {
            name: definition.name.clone(),
            unit: definition.unit.clone(),
            parsed_unit,
            grid_type: definition.grid_type,
        }
    }
}

/// Information about a unit conversion needed at runtime.
///
/// When two components use different (but compatible) units for the same variable,
/// a conversion factor must be applied when passing data between them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitConversionInfo {
    /// The variable name.
    pub variable: String,
    /// The component name that needs the conversion applied on input.
    pub component: String,
    /// The conversion factor to multiply values by.
    /// `converted = original * factor`
    pub factor: f64,
    /// The source unit (what the producer outputs).
    pub source_unit: String,
    /// The target unit (what the consumer expects).
    pub target_unit: String,
}

/// Direction of a grid transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformDirection {
    /// Read-side: aggregating schema data before component reads it
    /// (e.g., schema has FourBox, component wants Scalar)
    Read,
    /// Write-side: aggregating component output before storing in schema
    /// (e.g., component produces FourBox, schema wants Scalar)
    Write,
}

/// A required grid transformation identified during validation.
///
/// These are collected during component validation against the schema
/// and used to configure runtime grid aggregation (transform-on-read/write).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequiredTransformation {
    /// The variable name being transformed
    pub variable: String,
    /// The unit of the variable
    pub unit: String,
    /// The source grid type (finer resolution)
    pub source_grid: GridType,
    /// The target grid type (coarser resolution)
    pub target_grid: GridType,
    /// Direction of the transformation
    pub direction: TransformDirection,
}
