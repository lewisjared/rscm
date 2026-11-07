//! Validation functions for model building.

use crate::component::{TimeseriesType, VariableDefinition, VariableRole};
use crate::errors::{RscmError, RscmResult};
use crate::units::Unit;
use std::collections::HashMap;

use super::types::{ResolvedVariable, UnitConversionInfo};

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
    definitions: &mut HashMap<String, ResolvedVariable>,
    definition: &VariableDefinition,
    component_name: &str,
    existing_component_name: Option<&str>,
    has_schema: bool,
) -> RscmResult<Option<UnitConversionInfo>> {
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
                if !has_schema && existing.grid != definition.grid {
                    return Err(RscmError::GridTypeMismatch {
                        variable: definition.name.clone(),
                        producer_component: existing_component_name
                            .unwrap_or("unknown")
                            .to_string(),
                        consumer_component: component_name.to_string(),
                        producer_grid: existing.grid.to_string(),
                        consumer_grid: definition.grid.to_string(),
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
                if !has_schema && existing.grid != definition.grid {
                    return Err(RscmError::GridTypeMismatch {
                        variable: definition.name.clone(),
                        producer_component: existing_component_name
                            .unwrap_or("unknown")
                            .to_string(),
                        consumer_component: component_name.to_string(),
                        producer_grid: existing.grid.to_string(),
                        consumer_grid: definition.grid.to_string(),
                    });
                }
            }
        }
        None => {
            definitions.insert(
                definition.name.clone(),
                ResolvedVariable::from_definition(definition),
            );
        }
    }
    Ok(None)
}

/// Resolve each variable's timeseries type and check that every declaration agrees with it.
///
/// A produced variable takes its producer's type.
/// Otherwise the builder's declaration, then a consumer's, then the unit default decides.
pub(crate) fn resolve_timeseries_types(
    definitions: &mut HashMap<String, ResolvedVariable>,
    names: &[String],
    component_defs: &[(Vec<VariableDefinition>, Vec<VariableDefinition>)],
    declared_types: &HashMap<String, TimeseriesType>,
    state_type_check: bool,
) -> RscmResult<()> {
    let mut sources: HashMap<&str, (TimeseriesType, String)> = HashMap::new();
    for (name, (_, provides)) in names.iter().zip(component_defs) {
        for r in provides {
            sources.insert(&r.name, (r.resolved_timeseries_type(), name.clone()));
        }
    }
    for (variable, &declared) in declared_types {
        match sources.get(variable.as_str()) {
            Some((resolved, producer)) if *resolved != declared => {
                return Err(RscmError::TimeseriesTypeMismatch {
                    variable: variable.clone(),
                    producer: producer.clone(),
                    producer_type: resolved.to_string(),
                    consumer: "ModelBuilder::with_timeseries_type".to_string(),
                    consumer_type: declared.to_string(),
                });
            }
            Some(_) => {}
            None => {
                sources.insert(
                    variable,
                    (declared, "ModelBuilder::with_timeseries_type".to_string()),
                );
            }
        }
    }
    let declarations = names
        .iter()
        .zip(component_defs)
        .flat_map(|(name, (requires, _))| {
            requires
                .iter()
                .filter_map(move |r| r.timeseries_type.map(|t| (name, r, t)))
        });
    for (consumer, def, declared) in declarations {
        let (resolved, producer) = sources.entry(def.name.as_str()).or_insert_with(|| {
            (
                declared,
                format!("prescribed data as declared by {consumer}"),
            )
        });
        if *resolved != declared {
            return Err(RscmError::TimeseriesTypeMismatch {
                variable: def.name.clone(),
                producer: producer.clone(),
                producer_type: resolved.to_string(),
                consumer: consumer.clone(),
                consumer_type: declared.to_string(),
            });
        }
    }
    for (name, definition) in definitions.iter_mut() {
        if let Some((resolved, _)) = sources.get(name.as_str()) {
            definition.timeseries_type = *resolved;
        }
    }

    if state_type_check {
        for (component, (requires, _)) in names.iter().zip(component_defs) {
            for r in requires {
                if r.role == VariableRole::State
                    && definitions[&r.name].timeseries_type == TimeseriesType::Average
                {
                    return Err(RscmError::AverageState {
                        variable: r.name.clone(),
                        component: component.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Validates that two unit strings are dimensionally compatible.
///
/// Returns `UnitValidationResult::Identical` if the normalised units are the same,
/// `UnitValidationResult::Compatible` with the conversion factor if they have the same dimension,
/// or an error if they are incompatible.
fn validate_unit_compatibility(
    existing_unit: &str,
    new_unit: &str,
    variable_name: &str,
) -> RscmResult<UnitValidationResult> {
    // Parse both units
    let parsed_existing = Unit::parse(existing_unit).map_err(|e| RscmError::UnitParseError {
        variable: variable_name.to_string(),
        unit_string: existing_unit.to_string(),
        details: e.to_string(),
    })?;

    let parsed_new = Unit::parse(new_unit).map_err(|e| RscmError::UnitParseError {
        variable: variable_name.to_string(),
        unit_string: new_unit.to_string(),
        details: e.to_string(),
    })?;

    // Check if they're the same after normalisation
    if parsed_existing == parsed_new {
        return Ok(UnitValidationResult::Identical);
    }

    // Check dimensional compatibility and get conversion factor
    if parsed_existing.is_compatible(&parsed_new) {
        // Calculate conversion factor from existing (source) to new (target)
        let factor = parsed_existing
            .conversion_factor(&parsed_new)
            .map_err(|e| RscmError::UnitParseError {
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

        Err(RscmError::IncompatibleUnits {
            variable: variable_name.to_string(),
            unit1: existing_unit.to_string(),
            unit2: new_unit.to_string(),
            dim1,
            dim2,
        })
    }
}
