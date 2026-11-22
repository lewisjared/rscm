//! Type definitions for the model module.

use crate::component::{Component, GridType, TimeseriesType, VariableDefinition, VariableRole};
use crate::units::Unit;
use petgraph::Graph;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::warn;

/// Type alias for a component wrapped in an Arc for shared ownership.
pub type C = Arc<dyn Component>;

/// The component dependency graph, with each edge the variable its consumer reads.
pub type CGraph = Graph<C, VariableDefinition>;

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

/// Direction of a grid transformation or unit conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformDirection {
    /// From the stored unit and grid to the reader's declared ones.
    Read,
    /// From the writer's declared unit and grid to the stored ones.
    Write,
}

impl std::fmt::Display for TransformDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Read => "read",
            Self::Write => "write",
        })
    }
}
