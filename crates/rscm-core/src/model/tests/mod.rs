//! Integration tests for the model module.
//!
//! These tests verify the complete model building and execution workflow,
//! including component graphs, schema validation, and grid transformations.

#[cfg(test)]
mod aggregate_execution;
#[cfg(test)]
mod basic;
#[cfg(test)]
mod grid_validation;
#[cfg(test)]
mod grid_weights;
#[cfg(test)]
mod read_side_integration;
#[cfg(test)]
mod relaxed_grid_validation;
#[cfg(test)]
mod schema_validation;
#[cfg(test)]
mod unit_validation;
#[cfg(test)]
mod write_side_integration;
