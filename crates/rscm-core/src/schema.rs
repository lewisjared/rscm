//! Variable schema for declaring model variables and aggregates.
//!
//! This module provides types for declaring variables and aggregation relationships
//! at the model level. Components declare which variables they read/write, and
//! the [`ModelBuilder`](crate::model::ModelBuilder) validates consistency.
//!
//! # Overview
//!
//! Variables are first-class entities declared separately from components:
//! - Regular variables hold timeseries data produced by components
//! - Aggregate variables compute derived values from multiple contributors
//!
//! # Example
//!
//! ```
//! use rscm_core::schema::{AggregateOp, VariableSchema};
//!
//! let schema = VariableSchema::new()
//!     // Declare regular variables
//!     .variable("Effective Radiative Forcing|CO2", "W/m^2")
//!     .variable("Effective Radiative Forcing|CH4", "W/m^2")
//!     // Declare aggregate (sum of contributors)
//!     .aggregate("Effective Radiative Forcing", "W/m^2", AggregateOp::Sum)
//!         .from("Effective Radiative Forcing|CO2")
//!         .from("Effective Radiative Forcing|CH4")
//!         .build();
//! ```
//!
//! # Aggregation Operations
//!
//! Three operations are supported:
//!
//! - [`AggregateOp::Sum`]: Sum all contributor values
//! - [`AggregateOp::Mean`]: Arithmetic mean (divides by count of valid values)
//! - [`AggregateOp::Weighted`]: Weighted sum with provided weights
//!
//! NaN values are excluded from computations (treated as missing data).

use crate::component::GridType;
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Operation for computing aggregate values from contributors.
///
/// All operations handle NaN values by excluding them from computation.
/// If all contributors are NaN, the aggregate result is NaN.
///
/// Note: This enum cannot use `#[pyclass]` directly because PyO3 doesn't support
/// complex enums with data variants. Python bindings are provided via wrapper types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregateOp {
    /// Sum all contributor values.
    ///
    /// $$ \text{result} = \sum_{i} x_i $$
    Sum,

    /// Arithmetic mean of contributor values.
    ///
    /// Divides by the count of valid (non-NaN) values, not the total contributor count.
    ///
    /// $$ \text{result} = \frac{1}{n_{\text{valid}}} \sum_{i} x_i $$
    Mean,

    /// Weighted sum with provided weights per contributor.
    ///
    /// Weights must be provided in the same order as contributors.
    /// When a contributor is NaN, both the value and its weight are excluded.
    ///
    /// $$ \text{result} = \sum_{i} w_i \cdot x_i $$
    Weighted(Vec<f64>),
}

impl AggregateOp {
    /// Returns a display name for this operation
    pub fn name(&self) -> &'static str {
        match self {
            AggregateOp::Sum => "Sum",
            AggregateOp::Mean => "Mean",
            AggregateOp::Weighted(_) => "Weighted",
        }
    }

    /// Returns the weights if this is a Weighted operation, None otherwise
    pub fn weights(&self) -> Option<&[f64]> {
        match self {
            AggregateOp::Weighted(w) => Some(w),
            _ => None,
        }
    }
}

/// Definition of a single variable in the schema.
///
/// Variables represent timeseries data that can be produced by components
/// or provided as exogenous input.
#[pyclass]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaVariableDefinition {
    /// Variable identifier (e.g., "Atmospheric Concentration|CO2")
    #[pyo3(get)]
    pub name: String,

    /// Physical units (e.g., "ppm", "W/m^2")
    #[pyo3(get)]
    pub unit: String,

    /// Spatial resolution
    #[pyo3(get)]
    pub grid_type: GridType,
}

impl SchemaVariableDefinition {
    /// Create a new scalar variable definition
    pub fn new(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            grid_type: GridType::Scalar,
        }
    }

    /// Create a new variable definition with explicit grid type
    pub fn with_grid(
        name: impl Into<String>,
        unit: impl Into<String>,
        grid_type: GridType,
    ) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            grid_type,
        }
    }
}

#[pymethods]
impl SchemaVariableDefinition {
    #[new]
    #[pyo3(signature = (name, unit, grid_type=None))]
    fn py_new(name: String, unit: String, grid_type: Option<GridType>) -> Self {
        Self {
            name,
            unit,
            grid_type: grid_type.unwrap_or(GridType::Scalar),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SchemaVariableDefinition(name={:?}, unit={:?}, grid_type={:?})",
            self.name, self.unit, self.grid_type
        )
    }
}

/// Definition of an aggregate variable.
///
/// Aggregates compute derived values from multiple contributor variables
/// using a specified operation.
#[pyclass]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateDefinition {
    /// Variable identifier for the aggregate result
    #[pyo3(get)]
    pub name: String,

    /// Physical units (must match contributors)
    #[pyo3(get)]
    pub unit: String,

    /// Spatial resolution (must match contributors)
    #[pyo3(get)]
    pub grid_type: GridType,

    /// Operation to apply to contributors
    pub operation: AggregateOp,

    /// Names of variables that contribute to this aggregate.
    ///
    /// Contributors can be regular variables or other aggregates.
    #[pyo3(get)]
    pub contributors: Vec<String>,
}

impl AggregateDefinition {
    /// Create a new aggregate definition
    pub fn new(name: impl Into<String>, unit: impl Into<String>, operation: AggregateOp) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            grid_type: GridType::Scalar,
            operation,
            contributors: Vec::new(),
        }
    }

    /// Create a new aggregate definition with explicit grid type
    pub fn with_grid(
        name: impl Into<String>,
        unit: impl Into<String>,
        grid_type: GridType,
        operation: AggregateOp,
    ) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            grid_type,
            operation,
            contributors: Vec::new(),
        }
    }
}

#[pymethods]
impl AggregateDefinition {
    /// Get the operation type as a string ("Sum", "Mean", or "Weighted")
    #[getter]
    fn operation_type(&self) -> &'static str {
        self.operation.name()
    }

    /// Get the weights for a Weighted operation, or None for Sum/Mean
    #[getter]
    fn weights(&self) -> Option<Vec<f64>> {
        self.operation.weights().map(|w| w.to_vec())
    }

    fn __repr__(&self) -> String {
        format!(
            "AggregateDefinition(name={:?}, unit={:?}, grid_type={:?}, operation={:?}, contributors={:?})",
            self.name, self.unit, self.grid_type, self.operation, self.contributors
        )
    }
}

/// Complete variable schema for a model.
///
/// The schema declares all variables (regular and aggregates) for a model.
/// Components declare which variables they read/write, and the
/// [`ModelBuilder`](crate::model::ModelBuilder) validates consistency.
///
/// # Example
///
/// ```
/// use rscm_core::schema::{AggregateOp, VariableSchema};
///
/// let schema = VariableSchema::new()
///     .variable("Emissions|CO2", "GtCO2/yr")
///     .variable("Concentration|CO2", "ppm")
///     .aggregate("Total Emissions", "GtCO2/yr", AggregateOp::Sum)
///         .from("Emissions|CO2")
///         .build();
/// ```
#[pyclass]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VariableSchema {
    /// Regular variable definitions indexed by name
    #[pyo3(get)]
    pub variables: HashMap<String, SchemaVariableDefinition>,

    /// Aggregate definitions indexed by name
    #[pyo3(get)]
    pub aggregates: HashMap<String, AggregateDefinition>,
}

impl VariableSchema {
    /// Create a new empty variable schema
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a scalar variable to the schema
    ///
    /// Returns self for method chaining.
    pub fn variable(mut self, name: impl Into<String>, unit: impl Into<String>) -> Self {
        let name = name.into();
        let def = SchemaVariableDefinition::new(name.clone(), unit);
        self.variables.insert(name, def);
        self
    }

    /// Add a variable with explicit grid type to the schema
    ///
    /// Returns self for method chaining.
    pub fn variable_with_grid(
        mut self,
        name: impl Into<String>,
        unit: impl Into<String>,
        grid_type: GridType,
    ) -> Self {
        let name = name.into();
        let def = SchemaVariableDefinition::with_grid(name.clone(), unit, grid_type);
        self.variables.insert(name, def);
        self
    }

    /// Begin defining an aggregate variable
    ///
    /// Returns an [`AggregateBuilder`] for adding contributors.
    pub fn aggregate(
        self,
        name: impl Into<String>,
        unit: impl Into<String>,
        operation: AggregateOp,
    ) -> AggregateBuilder {
        let def = AggregateDefinition::new(name, unit, operation);
        AggregateBuilder {
            schema: self,
            aggregate: def,
        }
    }

    /// Begin defining an aggregate variable with explicit grid type
    ///
    /// Returns an [`AggregateBuilder`] for adding contributors.
    pub fn aggregate_with_grid(
        self,
        name: impl Into<String>,
        unit: impl Into<String>,
        grid_type: GridType,
        operation: AggregateOp,
    ) -> AggregateBuilder {
        let def = AggregateDefinition::with_grid(name, unit, grid_type, operation);
        AggregateBuilder {
            schema: self,
            aggregate: def,
        }
    }

    /// Check if a name exists in the schema (as variable or aggregate)
    pub fn contains(&self, name: &str) -> bool {
        self.variables.contains_key(name) || self.aggregates.contains_key(name)
    }

    /// Get a variable definition by name
    pub fn get_variable(&self, name: &str) -> Option<&SchemaVariableDefinition> {
        self.variables.get(name)
    }

    /// Get an aggregate definition by name
    pub fn get_aggregate(&self, name: &str) -> Option<&AggregateDefinition> {
        self.aggregates.get(name)
    }

    /// Get the unit for a name (variable or aggregate)
    pub fn get_unit(&self, name: &str) -> Option<&str> {
        self.variables
            .get(name)
            .map(|v| v.unit.as_str())
            .or_else(|| self.aggregates.get(name).map(|a| a.unit.as_str()))
    }

    /// Get the grid type for a name (variable or aggregate)
    pub fn get_grid_type(&self, name: &str) -> Option<GridType> {
        self.variables
            .get(name)
            .map(|v| v.grid_type)
            .or_else(|| self.aggregates.get(name).map(|a| a.grid_type))
    }
}

#[pymethods]
impl VariableSchema {
    #[new]
    fn py_new() -> Self {
        Self::new()
    }

    /// Add a scalar variable to the schema (Python API)
    #[pyo3(name = "add_variable")]
    #[pyo3(signature = (name, unit, grid_type=None))]
    fn py_add_variable(&mut self, name: String, unit: String, grid_type: Option<GridType>) {
        let def = SchemaVariableDefinition {
            name: name.clone(),
            unit,
            grid_type: grid_type.unwrap_or(GridType::Scalar),
        };
        self.variables.insert(name, def);
    }

    /// Add an aggregate to the schema (Python API)
    ///
    /// # Arguments
    /// * `name` - Variable identifier for the aggregate result
    /// * `unit` - Physical units (must match contributors)
    /// * `operation` - Operation type: "Sum", "Mean", or "Weighted"
    /// * `contributors` - Names of variables that contribute to this aggregate
    /// * `weights` - Weights for Weighted operation (required if operation="Weighted")
    /// * `grid_type` - Spatial resolution (defaults to Scalar)
    #[pyo3(name = "add_aggregate")]
    #[pyo3(signature = (name, unit, operation, contributors, weights=None, grid_type=None))]
    fn py_add_aggregate(
        &mut self,
        name: String,
        unit: String,
        operation: &str,
        contributors: Vec<String>,
        weights: Option<Vec<f64>>,
        grid_type: Option<GridType>,
    ) -> pyo3::PyResult<()> {
        let op = match operation {
            "Sum" => AggregateOp::Sum,
            "Mean" => AggregateOp::Mean,
            "Weighted" => {
                let w = weights.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "weights must be provided for Weighted operation",
                    )
                })?;
                AggregateOp::Weighted(w)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown operation '{}'. Valid operations: Sum, Mean, Weighted",
                    operation
                )))
            }
        };

        let def = AggregateDefinition {
            name: name.clone(),
            unit,
            grid_type: grid_type.unwrap_or(GridType::Scalar),
            operation: op,
            contributors,
        };
        self.aggregates.insert(name, def);
        Ok(())
    }

    /// Check if a name exists in the schema
    #[pyo3(name = "contains")]
    fn py_contains(&self, name: &str) -> bool {
        self.contains(name)
    }

    fn __repr__(&self) -> String {
        format!(
            "VariableSchema(variables={}, aggregates={})",
            self.variables.len(),
            self.aggregates.len()
        )
    }
}

/// Builder for aggregate definitions.
///
/// Created by [`VariableSchema::aggregate`], this builder allows
/// adding contributors before finalising the aggregate.
pub struct AggregateBuilder {
    schema: VariableSchema,
    aggregate: AggregateDefinition,
}

impl AggregateBuilder {
    /// Add a contributor to the aggregate
    ///
    /// Contributors can be regular variables or other aggregates defined in the schema.
    pub fn from(mut self, contributor: impl Into<String>) -> Self {
        self.aggregate.contributors.push(contributor.into());
        self
    }

    /// Finalise the aggregate and return the updated schema
    pub fn build(mut self) -> VariableSchema {
        let name = self.aggregate.name.clone();
        self.schema.aggregates.insert(name, self.aggregate);
        self.schema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_op_variants() {
        let sum = AggregateOp::Sum;
        let mean = AggregateOp::Mean;
        let weighted = AggregateOp::Weighted(vec![0.5, 0.3, 0.2]);

        assert_eq!(sum, AggregateOp::Sum);
        assert_eq!(mean, AggregateOp::Mean);
        assert!(matches!(weighted, AggregateOp::Weighted(_)));
    }

    #[test]
    fn test_aggregate_op_serialization() {
        let sum = AggregateOp::Sum;
        let json = serde_json::to_string(&sum).unwrap();
        let deserialized: AggregateOp = serde_json::from_str(&json).unwrap();
        assert_eq!(sum, deserialized);

        let weighted = AggregateOp::Weighted(vec![0.5, 0.5]);
        let json = serde_json::to_string(&weighted).unwrap();
        let deserialized: AggregateOp = serde_json::from_str(&json).unwrap();
        assert_eq!(weighted, deserialized);
    }

    #[test]
    fn test_variable_definition_new() {
        let def = SchemaVariableDefinition::new("Emissions|CO2", "GtCO2/yr");
        assert_eq!(def.name, "Emissions|CO2");
        assert_eq!(def.unit, "GtCO2/yr");
        assert_eq!(def.grid_type, GridType::Scalar);
    }

    #[test]
    fn test_variable_definition_with_grid() {
        let def = SchemaVariableDefinition::with_grid("Temperature", "K", GridType::FourBox);
        assert_eq!(def.name, "Temperature");
        assert_eq!(def.unit, "K");
        assert_eq!(def.grid_type, GridType::FourBox);
    }

    #[test]
    fn test_aggregate_definition_new() {
        let def = AggregateDefinition::new("Total ERF", "W/m^2", AggregateOp::Sum);
        assert_eq!(def.name, "Total ERF");
        assert_eq!(def.unit, "W/m^2");
        assert_eq!(def.grid_type, GridType::Scalar);
        assert_eq!(def.operation, AggregateOp::Sum);
        assert!(def.contributors.is_empty());
    }

    #[test]
    fn test_variable_schema_builder() {
        let schema = VariableSchema::new()
            .variable("Emissions|CO2", "GtCO2/yr")
            .variable("Concentration|CO2", "ppm")
            .variable_with_grid("Regional Temperature", "K", GridType::FourBox);

        assert_eq!(schema.variables.len(), 3);
        assert!(schema.contains("Emissions|CO2"));
        assert!(schema.contains("Concentration|CO2"));
        assert!(schema.contains("Regional Temperature"));
        assert!(!schema.contains("Nonexistent"));

        let co2 = schema.get_variable("Emissions|CO2").unwrap();
        assert_eq!(co2.unit, "GtCO2/yr");
        assert_eq!(co2.grid_type, GridType::Scalar);

        let temp = schema.get_variable("Regional Temperature").unwrap();
        assert_eq!(temp.grid_type, GridType::FourBox);
    }

    #[test]
    fn test_variable_schema_with_aggregate() {
        let schema = VariableSchema::new()
            .variable("ERF|CO2", "W/m^2")
            .variable("ERF|CH4", "W/m^2")
            .aggregate("Total ERF", "W/m^2", AggregateOp::Sum)
            .from("ERF|CO2")
            .from("ERF|CH4")
            .build();

        assert_eq!(schema.variables.len(), 2);
        assert_eq!(schema.aggregates.len(), 1);
        assert!(schema.contains("ERF|CO2"));
        assert!(schema.contains("Total ERF"));

        let agg = schema.get_aggregate("Total ERF").unwrap();
        assert_eq!(agg.unit, "W/m^2");
        assert_eq!(agg.operation, AggregateOp::Sum);
        assert_eq!(agg.contributors, vec!["ERF|CO2", "ERF|CH4"]);
    }

    #[test]
    fn test_variable_schema_weighted_aggregate() {
        let schema = VariableSchema::new()
            .variable("Source A", "units")
            .variable("Source B", "units")
            .aggregate(
                "Weighted Total",
                "units",
                AggregateOp::Weighted(vec![0.7, 0.3]),
            )
            .from("Source A")
            .from("Source B")
            .build();

        let agg = schema.get_aggregate("Weighted Total").unwrap();
        assert!(matches!(agg.operation, AggregateOp::Weighted(ref w) if w == &vec![0.7, 0.3]));
    }

    #[test]
    fn test_variable_schema_chained_aggregates() {
        // Test that aggregates can reference other aggregates
        let schema = VariableSchema::new()
            .variable("ERF|CO2", "W/m^2")
            .variable("ERF|CH4", "W/m^2")
            .variable("ERF|Other", "W/m^2")
            .aggregate("ERF|GHG", "W/m^2", AggregateOp::Sum)
            .from("ERF|CO2")
            .from("ERF|CH4")
            .build()
            .aggregate("Total ERF", "W/m^2", AggregateOp::Sum)
            .from("ERF|GHG")
            .from("ERF|Other")
            .build();

        assert_eq!(schema.aggregates.len(), 2);

        let ghg = schema.get_aggregate("ERF|GHG").unwrap();
        assert_eq!(ghg.contributors, vec!["ERF|CO2", "ERF|CH4"]);

        let total = schema.get_aggregate("Total ERF").unwrap();
        assert_eq!(total.contributors, vec!["ERF|GHG", "ERF|Other"]);
    }

    #[test]
    fn test_variable_schema_get_unit() {
        let schema = VariableSchema::new()
            .variable("Emissions|CO2", "GtCO2/yr")
            .aggregate("Total", "GtCO2/yr", AggregateOp::Sum)
            .from("Emissions|CO2")
            .build();

        assert_eq!(schema.get_unit("Emissions|CO2"), Some("GtCO2/yr"));
        assert_eq!(schema.get_unit("Total"), Some("GtCO2/yr"));
        assert_eq!(schema.get_unit("Nonexistent"), None);
    }

    #[test]
    fn test_variable_schema_get_grid_type() {
        let schema = VariableSchema::new()
            .variable("Global", "K")
            .variable_with_grid("Regional", "K", GridType::FourBox)
            .aggregate_with_grid("Regional Total", "K", GridType::FourBox, AggregateOp::Sum)
            .from("Regional")
            .build();

        assert_eq!(schema.get_grid_type("Global"), Some(GridType::Scalar));
        assert_eq!(schema.get_grid_type("Regional"), Some(GridType::FourBox));
        assert_eq!(
            schema.get_grid_type("Regional Total"),
            Some(GridType::FourBox)
        );
        assert_eq!(schema.get_grid_type("Nonexistent"), None);
    }

    #[test]
    fn test_variable_schema_serialization_roundtrip() {
        let schema = VariableSchema::new()
            .variable("ERF|CO2", "W/m^2")
            .variable("ERF|CH4", "W/m^2")
            .aggregate("Total ERF", "W/m^2", AggregateOp::Sum)
            .from("ERF|CO2")
            .from("ERF|CH4")
            .build();

        let json = serde_json::to_string(&schema).unwrap();
        let deserialized: VariableSchema = serde_json::from_str(&json).unwrap();

        assert_eq!(schema.variables.len(), deserialized.variables.len());
        assert_eq!(schema.aggregates.len(), deserialized.aggregates.len());
        assert_eq!(
            schema.get_aggregate("Total ERF"),
            deserialized.get_aggregate("Total ERF")
        );
    }

    #[test]
    fn test_variable_schema_toml_serialization() {
        let schema = VariableSchema::new()
            .variable("ERF|CO2", "W/m^2")
            .aggregate("Total ERF", "W/m^2", AggregateOp::Sum)
            .from("ERF|CO2")
            .build();

        let toml = toml::to_string(&schema).unwrap();
        let deserialized: VariableSchema = toml::from_str(&toml).unwrap();

        assert_eq!(schema, deserialized);
    }

    #[test]
    fn test_empty_schema() {
        let schema = VariableSchema::new();
        assert!(schema.variables.is_empty());
        assert!(schema.aggregates.is_empty());
        assert!(!schema.contains("anything"));
    }
}
