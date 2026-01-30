//! Validation functions for model building.

use crate::component::RequirementDefinition;
use crate::errors::{RSCMError, RSCMResult};
use crate::units::Unit;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers, Visitable};
use std::collections::HashMap;
use tracing::warn;

use super::types::{UnitConversionInfo, VariableDefinition};

/// Result of unit validation between two definitions.
#[derive(Debug)]
pub(crate) enum UnitValidationResult {
    /// Units are identical (string match), no conversion needed.
    Identical,
    /// Units are different but dimensionally compatible, conversion factor provided.
    Compatible { factor: f64 },
}

/// Checks if the new definition is valid.
///
/// If any definitions share a name then the units must be dimensionally compatible
/// and grid types must be equivalent (unless a schema is present).
/// When `has_schema` is true, grid type checking is skipped because the schema validation
/// will handle grid compatibility with relaxed rules (allowing aggregation).
///
/// Returns an error if the parameter definition is inconsistent with any existing definitions.
/// Returns an optional `UnitConversionInfo` if the units differ but are compatible.
pub(crate) fn verify_definition(
    definitions: &mut HashMap<String, VariableDefinition>,
    definition: &RequirementDefinition,
    component_name: &str,
    existing_component_name: Option<&str>,
    has_schema: bool,
) -> RSCMResult<Option<UnitConversionInfo>> {
    let existing = definitions.get(&definition.name);
    match existing {
        Some(existing) => {
            // First check for exact string match (fast path)
            if existing.unit != definition.unit {
                // Units differ - check dimensional compatibility
                let validation_result = validate_unit_compatibility(
                    &existing.unit,
                    &definition.unit,
                    &definition.name,
                )?;

                // Skip grid type check when schema is present
                if !has_schema && existing.grid_type != definition.grid_type {
                    return Err(RSCMError::GridTypeMismatch {
                        variable: definition.name.clone(),
                        producer_component: existing_component_name
                            .unwrap_or("unknown")
                            .to_string(),
                        consumer_component: component_name.to_string(),
                        producer_grid: existing.grid_type.to_string(),
                        consumer_grid: definition.grid_type.to_string(),
                    });
                }

                // Return conversion info if units are compatible but different
                if let UnitValidationResult::Compatible { factor } = validation_result {
                    return Ok(Some(UnitConversionInfo {
                        variable: definition.name.clone(),
                        component: component_name.to_string(),
                        factor,
                        source_unit: existing.unit.clone(),
                        target_unit: definition.unit.clone(),
                    }));
                }
            } else {
                // Units match exactly - still check grid type
                if !has_schema && existing.grid_type != definition.grid_type {
                    return Err(RSCMError::GridTypeMismatch {
                        variable: definition.name.clone(),
                        producer_component: existing_component_name
                            .unwrap_or("unknown")
                            .to_string(),
                        consumer_component: component_name.to_string(),
                        producer_grid: existing.grid_type.to_string(),
                        consumer_grid: definition.grid_type.to_string(),
                    });
                }
            }
        }
        None => {
            // Validate that the unit can be parsed before adding the definition
            if let Err(e) = Unit::parse(&definition.unit) {
                // Log a warning but don't fail - the unit might be a custom/unknown unit
                // that we can't parse but is still valid for string comparison.
                // The error will be caught later if incompatible units are encountered.
                warn!(
                    variable = %definition.name,
                    unit = %definition.unit,
                    error = %e,
                    "Could not parse unit string; unit conversion will not be available"
                );
            }
            definitions.insert(
                definition.name.clone(),
                VariableDefinition::from_requirement_definition(definition),
            );
        }
    }
    Ok(None)
}

/// Validates that two unit strings are dimensionally compatible.
///
/// Returns `UnitValidationResult::Identical` if the normalized units are the same,
/// `UnitValidationResult::Compatible` with the conversion factor if they have the same dimension,
/// or an error if they are incompatible.
fn validate_unit_compatibility(
    existing_unit: &str,
    new_unit: &str,
    variable_name: &str,
) -> RSCMResult<UnitValidationResult> {
    // Parse both units
    let parsed_existing = Unit::parse(existing_unit).map_err(|e| RSCMError::UnitParseError {
        variable: variable_name.to_string(),
        unit_string: existing_unit.to_string(),
        details: e.to_string(),
    })?;

    let parsed_new = Unit::parse(new_unit).map_err(|e| RSCMError::UnitParseError {
        variable: variable_name.to_string(),
        unit_string: new_unit.to_string(),
        details: e.to_string(),
    })?;

    // Check if they're the same after normalization
    if parsed_existing == parsed_new {
        return Ok(UnitValidationResult::Identical);
    }

    // Check dimensional compatibility and get conversion factor
    if parsed_existing.is_compatible(&parsed_new) {
        // Calculate conversion factor from existing (source) to new (target)
        let factor = parsed_existing
            .conversion_factor(&parsed_new)
            .map_err(|e| RSCMError::UnitParseError {
                variable: variable_name.to_string(),
                unit_string: format!("{existing_unit} -> {new_unit}"),
                details: e.to_string(),
            })?;

        Ok(UnitValidationResult::Compatible { factor })
    } else {
        // Get dimensions for error message
        let dim1 = parsed_existing
            .dimension()
            .map(|d| d.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        let dim2 = parsed_new
            .dimension()
            .map(|d| d.to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        Err(RSCMError::IncompatibleUnits {
            variable: variable_name.to_string(),
            unit1: existing_unit.to_string(),
            unit2: new_unit.to_string(),
            dim1,
            dim2,
        })
    }
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
