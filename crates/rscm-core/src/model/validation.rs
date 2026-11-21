//! Validation functions for model building.

use crate::component::{TimeseriesType, VariableDefinition, VariableRole};
use crate::errors::{RSCMError, RSCMResult};
use crate::schema::VariableSchema;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

use super::types::ResolvedVariable;

/// Check that every component output is in the schema,
/// and every input is in the schema or produced by a component.
pub(crate) fn check_schema_covers(
    schema: &VariableSchema,
    labels: &[String],
    component_defs: &[(Vec<VariableDefinition>, Vec<VariableDefinition>)],
    produced: &HashMap<String, NodeIndex>,
) -> RSCMResult<()> {
    for (label, (requires, provides)) in labels.iter().zip(component_defs) {
        if let Some(output) = provides.iter().find(|d| !schema.contains(&d.name)) {
            return Err(RSCMError::SchemaUndefinedOutput {
                component: label.clone(),
                variable: output.name.clone(),
                unit: output.unit.clone(),
            });
        }
        let undefined =
            |d: &&VariableDefinition| !schema.contains(&d.name) && !produced.contains_key(&d.name);
        if let Some(input) = requires.iter().find(undefined) {
            return Err(RSCMError::SchemaUndefinedInput {
                component: label.clone(),
                variable: input.name.clone(),
                unit: input.unit.clone(),
            });
        }
    }
    Ok(())
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
) -> RSCMResult<()> {
    let mut sources: HashMap<&str, (TimeseriesType, String)> = HashMap::new();
    for (name, (_, provides)) in names.iter().zip(component_defs) {
        for r in provides {
            sources.insert(&r.name, (r.resolved_timeseries_type(), name.clone()));
        }
    }
    for (variable, &declared) in declared_types {
        match sources.get(variable.as_str()) {
            Some((resolved, producer)) if *resolved != declared => {
                return Err(RSCMError::TimeseriesTypeMismatch {
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
            return Err(RSCMError::TimeseriesTypeMismatch {
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
                    return Err(RSCMError::AverageState {
                        variable: r.name.clone(),
                        component: component.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}
