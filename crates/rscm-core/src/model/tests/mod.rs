//! Integration tests for the model module.
//!
//! These tests verify the complete model building and execution workflow,
//! including component graphs, schema validation, and grid transformations.

#[cfg(test)]
mod aggregate_execution;
#[cfg(test)]
mod average_storage;
#[cfg(test)]
mod basic;
#[cfg(test)]
mod coupling;
#[cfg(test)]
mod grid_validation;
#[cfg(test)]
mod grid_weights;
#[cfg(test)]
mod labels;
#[cfg(test)]
mod lags;
#[cfg(test)]
mod missing_prescribed;
#[cfg(test)]
mod output_staging;
#[cfg(test)]
mod parallel;
#[cfg(test)]
mod read_side_integration;
#[cfg(test)]
mod record;
#[cfg(test)]
mod relaxed_grid_validation;
#[cfg(test)]
mod runtime_plan;
#[cfg(test)]
mod schema_validation;
#[cfg(test)]
mod step_plan;
#[cfg(test)]
mod storage_rule;
#[cfg(test)]
mod timeseries_types;
#[cfg(test)]
mod unit_storage;
#[cfg(test)]
mod unit_validation;
#[cfg(test)]
mod variable_source_disambiguation;
#[cfg(test)]
mod write_side_integration;

use crate::model::{Model, Read, Write};
use crate::state::VariableSource;
use std::collections::BTreeMap;

/// The source of every read in `model`'s plan, keyed by variable and reader label.
fn read_sources(model: &Model) -> BTreeMap<(String, String), VariableSource> {
    model
        .plan()
        .steps
        .iter()
        .flat_map(|step| {
            step.reads.iter().map(|read| {
                (
                    (read.definition.name.clone(), step.label.clone()),
                    read.source,
                )
            })
        })
        .collect()
}

/// How the component labelled `label` reads `variable`.
fn read<'a>(model: &'a Model, variable: &str, label: &str) -> &'a Read {
    model
        .plan()
        .steps
        .iter()
        .filter(|step| step.label == label)
        .flat_map(|step| &step.reads)
        .find(|read| read.definition.name == variable)
        .unwrap_or_else(|| panic!("{label} does not read {variable}"))
}

/// How the component labelled `label` writes `variable`.
fn write<'a>(model: &'a Model, variable: &str, label: &str) -> &'a Write {
    model
        .plan()
        .steps
        .iter()
        .filter(|step| step.label == label)
        .flat_map(|step| &step.writes)
        .find(|write| write.definition.name == variable)
        .unwrap_or_else(|| panic!("{label} does not write {variable}"))
}
