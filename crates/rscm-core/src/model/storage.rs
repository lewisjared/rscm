//! The storage rule: the unit and grid each variable is stored in, and how components reach it.
//!
//! A variable is stored as its schema entry says.
//! With no entry it is stored as its producer declares it.
//! A variable no component produces keeps the unit and grid of its prescribed data,
//! or its first consumer's declaration when it has none.
//! Each read converts stored to declared and each write declared to stored.
//! A finer grid aggregates to a coarser one, and the reverse is a build error.

use std::collections::HashMap;

use crate::component::{GridType, TimeseriesType, VariableDefinition, VariableRole};
use crate::errors::{RSCMError, RSCMResult};
use crate::schema::VariableSchema;
use crate::timeseries_collection::TimeseriesCollection;
use crate::units::{self, Unit};

use super::types::{
    RequiredTransformation, ResolvedVariable, TransformDirection, UnitConversionInfo,
};

/// Where every variable is stored and how each component reaches it.
pub(crate) struct Storage {
    /// Stored unit, grid and role of each variable.
    pub definitions: HashMap<String, ResolvedVariable>,
    /// Factor from the stored unit to each reader's declared unit, where they differ.
    pub read_conversions: Vec<UnitConversionInfo>,
    /// Grid aggregations on reads and writes.
    pub transformations: Vec<RequiredTransformation>,
}

/// Apply the storage rule to every variable a component, schema or aggregate declares.
///
/// `labels` and `component_defs` are aligned, with each component's inputs then outputs.
/// Aggregates match their schema entries exactly, so they need no conversion.
pub(crate) fn resolve_storage(
    schema: Option<&VariableSchema>,
    labels: &[String],
    component_defs: &[(Vec<VariableDefinition>, Vec<VariableDefinition>)],
    prescribed: &TimeseriesCollection,
) -> RSCMResult<Storage> {
    let mut definitions: HashMap<String, ResolvedVariable> = HashMap::new();
    let mut stored_by: HashMap<String, String> = HashMap::new();

    for (label, (_, provides)) in labels.iter().zip(component_defs) {
        for def in provides {
            definitions.insert(def.name.clone(), ResolvedVariable::from_definition(def));
            stored_by.insert(def.name.clone(), format!("its producer {label} declares"));
        }
    }
    for (label, (requires, _)) in labels.iter().zip(component_defs) {
        for def in requires {
            if definitions.contains_key(&def.name) {
                continue;
            }
            let mut resolved = ResolvedVariable::from_definition(def);
            let source = match prescribed.get_data(&def.name) {
                Some(data) => {
                    set_unit(&mut resolved, data.unit());
                    resolved.grid = data.grid();
                    "the prescribed data is".to_string()
                }
                None => format!("its first consumer {label} declares"),
            };
            definitions.insert(def.name.clone(), resolved);
            stored_by.insert(def.name.clone(), source);
        }
    }

    if let Some(schema) = schema {
        for (name, entry) in &schema.variables {
            let resolved = definitions
                .entry(name.clone())
                .or_insert_with(|| ResolvedVariable {
                    name: name.clone(),
                    unit: entry.unit.clone(),
                    parsed_unit: None,
                    grid: entry.grid,
                    role: VariableRole::Input,
                    timeseries_type: TimeseriesType::from_unit(&entry.unit),
                });
            set_unit(resolved, &entry.unit);
            resolved.grid = entry.grid;
            stored_by.insert(name.clone(), "the schema declares".to_string());
        }
        for (name, aggregate) in &schema.aggregates {
            let mut resolved = ResolvedVariable {
                name: name.clone(),
                unit: String::new(),
                parsed_unit: None,
                grid: aggregate.grid,
                role: VariableRole::Output,
                timeseries_type: TimeseriesType::from_unit(&aggregate.unit),
            };
            set_unit(&mut resolved, &aggregate.unit);
            definitions.insert(name.clone(), resolved);
            stored_by.insert(name.clone(), "the schema declares".to_string());
        }
    }

    let mut read_conversions = Vec::new();
    let mut transformations = Vec::new();
    for (label, (requires, provides)) in labels.iter().zip(component_defs) {
        for def in requires {
            let stored = &definitions[&def.name];
            if let Some(factor) = units::factor(&def.name, &stored.unit, &def.unit)? {
                read_conversions.push(UnitConversionInfo {
                    variable: def.name.clone(),
                    component: label.clone(),
                    factor,
                    source_unit: stored.unit.clone(),
                    target_unit: def.unit.clone(),
                });
            }
            let transform = grid_transform(
                def,
                label,
                stored.grid,
                &stored_by[&def.name],
                TransformDirection::Read,
            )?;
            transformations.extend(transform);
        }
        for def in provides {
            let stored = &definitions[&def.name];
            units::factor(&def.name, &def.unit, &stored.unit)?;
            let transform = grid_transform(
                def,
                label,
                stored.grid,
                &stored_by[&def.name],
                TransformDirection::Write,
            )?;
            transformations.extend(transform);
        }
    }

    Ok(Storage {
        definitions,
        read_conversions,
        transformations,
    })
}

fn set_unit(resolved: &mut ResolvedVariable, unit: &str) {
    resolved.unit = unit.to_string();
    resolved.parsed_unit = Unit::parse(unit).ok();
}

/// The aggregation from the finer to the coarser of the declared and stored grids, if they differ.
///
/// Fails if the side that must be finer is coarser.
fn grid_transform(
    def: &VariableDefinition,
    component: &str,
    stored_grid: GridType,
    stored_by: &str,
    direction: TransformDirection,
) -> RSCMResult<Option<RequiredTransformation>> {
    if def.grid == stored_grid {
        return Ok(None);
    }
    let (source_grid, target_grid) = match direction {
        TransformDirection::Read => (stored_grid, def.grid),
        TransformDirection::Write => (def.grid, stored_grid),
    };
    if !source_grid.can_aggregate_to(target_grid) {
        return Err(RSCMError::GridTypeMismatch {
            variable: def.name.clone(),
            component: component.to_string(),
            access: format!("{direction}s"),
            declared_grid: def.grid.to_string(),
            stored_grid: stored_grid.to_string(),
            stored_by: stored_by.to_string(),
        });
    }
    Ok(Some(RequiredTransformation {
        variable: def.name.clone(),
        unit: def.unit.clone(),
        source_grid,
        target_grid,
        direction,
    }))
}
