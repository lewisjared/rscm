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
mod lags;
#[cfg(test)]
mod missing_exogenous;
#[cfg(test)]
mod parallel;
#[cfg(test)]
mod read_side_integration;
#[cfg(test)]
mod relaxed_grid_validation;
#[cfg(test)]
mod schema_validation;
#[cfg(test)]
mod timeseries_types;
#[cfg(test)]
mod unit_validation;
#[cfg(test)]
mod variable_source_disambiguation;
#[cfg(test)]
mod write_side_integration;
