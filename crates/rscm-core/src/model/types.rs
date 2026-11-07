//! Type definitions for the model module.

use crate::component::{Component, GridType, TimeseriesType, VariableDefinition, VariableRole};
use crate::units::Unit;
use petgraph::Graph;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::warn;

/// Type alias for a component wrapped in an Arc for shared ownership.
pub type C = Arc<dyn Component>;

/// Type alias for the component dependency graph.
pub type CGraph = Graph<C, GraphEdge>;

/// An edge of the component graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdge {
    /// Links a component with no producer to the initial node, so the graph stays connected.
    Link,
    /// A variable passed from its producer to a consumer.
    Variable(VariableDefinition),
}

impl GraphEdge {
    /// The variable's name, empty for a link.
    pub fn name(&self) -> &str {
        match self {
            GraphEdge::Link => "",
            GraphEdge::Variable(definition) => &definition.name,
        }
    }
}

/// Internal definition tracking for a variable during model building.
#[derive(Debug)]
pub(crate) struct ResolvedVariable {
    pub name: String,
    /// The unit string (original, preserved for display/error messages).
    pub unit: String,
    /// The parsed unit (if parsing succeeded), used for compatibility checks.
    #[allow(dead_code)]
    pub parsed_unit: Option<Unit>,
    pub grid: GridType,
    pub role: VariableRole,
    pub timeseries_type: TimeseriesType,
}

impl ResolvedVariable {
    pub fn from_definition(definition: &VariableDefinition) -> Self {
        let parsed = Unit::parse(&definition.unit);
        if let Err(e) = &parsed {
            // A unit we cannot parse still works for string comparison, without conversion.
            warn!(
                variable = %definition.name,
                unit = %definition.unit,
                error = %e,
                "Could not parse unit string; unit conversion will not be available"
            );
        }
        let parsed_unit = parsed.ok();

        Self {
            name: definition.name.clone(),
            unit: definition.unit.clone(),
            parsed_unit,
            grid: definition.grid,
            role: definition.role,
            timeseries_type: definition.resolved_timeseries_type(),
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
#[serde(rename_all = "snake_case")]
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
