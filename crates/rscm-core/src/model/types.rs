//! Type definitions for the model module.

use crate::component::{Component, GridType, RequirementDefinition};
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
    pub unit: String,
    pub grid_type: GridType,
}

impl VariableDefinition {
    pub fn from_requirement_definition(definition: &RequirementDefinition) -> Self {
        Self {
            name: definition.name.clone(),
            unit: definition.unit.clone(),
            grid_type: definition.grid_type,
        }
    }
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
