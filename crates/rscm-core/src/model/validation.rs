//! Validation functions for model building.

use crate::component::RequirementDefinition;
use crate::errors::{RSCMError, RSCMResult};
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers, Visitable};
use std::collections::HashMap;

use super::types::VariableDefinition;

/// Checks if the new definition is valid.
///
/// If any definitions share a name then the units and grid types must be equivalent.
/// When `has_schema` is true, grid type checking is skipped because the schema validation
/// will handle grid compatibility with relaxed rules (allowing aggregation).
///
/// Returns an error if the parameter definition is inconsistent with any existing definitions.
pub(crate) fn verify_definition(
    definitions: &mut HashMap<String, VariableDefinition>,
    definition: &RequirementDefinition,
    component_name: &str,
    existing_component_name: Option<&str>,
    has_schema: bool,
) -> RSCMResult<()> {
    let existing = definitions.get(&definition.name);
    match existing {
        Some(existing) => {
            if existing.unit != definition.unit {
                return Err(RSCMError::Error(format!(
                    "Unit mismatch for variable '{}': component '{}' uses '{}' but component '{}' uses '{}'. \
                     All producers and consumers of a variable must use the same unit.",
                    definition.name,
                    existing_component_name.unwrap_or("unknown"),
                    existing.unit,
                    component_name,
                    definition.unit
                )));
            }

            // Skip grid type check when schema is present - schema validation handles it
            // with relaxed rules that allow aggregation
            if !has_schema && existing.grid_type != definition.grid_type {
                return Err(RSCMError::GridTypeMismatch {
                    variable: definition.name.clone(),
                    producer_component: existing_component_name.unwrap_or("unknown").to_string(),
                    consumer_component: component_name.to_string(),
                    producer_grid: existing.grid_type.to_string(),
                    consumer_grid: definition.grid_type.to_string(),
                });
            }
        }
        None => {
            definitions.insert(
                definition.name.clone(),
                VariableDefinition::from_requirement_definition(definition),
            );
        }
    }
    Ok(())
}

/// Check that a component graph is valid.
///
/// We require a directed acyclic graph which doesn't contain any cycles
/// (other than a self-referential node).
/// This avoids the case where component `A` depends on a component `B`,
/// but component `B` also depends on component `A`.
pub(crate) fn is_valid_graph<G>(g: G) -> bool
where
    G: IntoNodeIdentifiers + IntoNeighbors + Visitable,
{
    use petgraph::visit::{depth_first_search, DfsEvent};

    depth_first_search(g, g.node_identifiers(), |event| match event {
        DfsEvent::BackEdge(a, b) => {
            // If the cycle is self-referential then that is fine
            match a == b {
                true => Ok(()),
                false => Err(()),
            }
        }
        _ => Ok(()),
    })
    .is_err()
}
