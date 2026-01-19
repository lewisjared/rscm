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
//!
//! # Aggregate Execution
//!
//! When a model has a schema with aggregates, virtual aggregator components are
//! inserted into the component graph during `ModelBuilder::build()`. These
//! aggregators read from their contributor variables and write to the aggregate
//! variable after all contributors have been solved.

use crate::component::GridType;
use crate::errors::{RSCMError, RSCMResult};
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct VariableSchema {
    /// Regular variable definitions indexed by name
    #[pyo3(get)]
    pub variables: HashMap<String, SchemaVariableDefinition>,

    /// Aggregate definitions indexed by name
    #[pyo3(get)]
    pub aggregates: HashMap<String, AggregateDefinition>,
}

/// Helper struct for deserializing VariableSchema without validation
/// Used internally to avoid infinite recursion in custom Deserialize impl
#[derive(Deserialize)]
struct VariableSchemaRaw {
    variables: HashMap<String, SchemaVariableDefinition>,
    aggregates: HashMap<String, AggregateDefinition>,
}

impl<'de> Deserialize<'de> for VariableSchema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = VariableSchemaRaw::deserialize(deserializer)?;
        let schema = VariableSchema {
            variables: raw.variables,
            aggregates: raw.aggregates,
        };
        schema.validate().map_err(serde::de::Error::custom)?;
        Ok(schema)
    }
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

    /// Validate the schema for consistency.
    ///
    /// Performs the following checks:
    /// - All aggregate contributors exist in the schema (as variables or aggregates)
    /// - Unit consistency between contributors and their aggregates
    /// - Grid type consistency between contributors and their aggregates
    /// - Weighted aggregate weight counts match contributor counts
    /// - No circular dependencies between aggregates
    ///
    /// # Errors
    ///
    /// Returns an error describing the first validation failure encountered.
    ///
    /// # Example
    ///
    /// ```
    /// use rscm_core::schema::{AggregateOp, VariableSchema};
    ///
    /// let schema = VariableSchema::new()
    ///     .variable("ERF|CO2", "W/m^2")
    ///     .aggregate("Total ERF", "W/m^2", AggregateOp::Sum)
    ///         .from("ERF|CO2")
    ///         .build();
    ///
    /// assert!(schema.validate().is_ok());
    /// ```
    pub fn validate(&self) -> RSCMResult<()> {
        // Check each aggregate
        for (agg_name, agg_def) in &self.aggregates {
            // 3.2: Validate contributor references exist
            for contributor in &agg_def.contributors {
                if !self.contains(contributor) {
                    return Err(RSCMError::UndefinedContributor {
                        contributor: contributor.clone(),
                        aggregate: agg_name.clone(),
                    });
                }

                // 3.3: Validate unit consistency
                if let Some(contributor_unit) = self.get_unit(contributor) {
                    if contributor_unit != agg_def.unit {
                        return Err(RSCMError::SchemaUnitMismatch {
                            aggregate: agg_name.clone(),
                            contributor: contributor.clone(),
                            contributor_unit: contributor_unit.to_string(),
                            aggregate_unit: agg_def.unit.clone(),
                        });
                    }
                }

                // 3.4: Validate grid type consistency
                if let Some(contributor_grid) = self.get_grid_type(contributor) {
                    if contributor_grid != agg_def.grid_type {
                        return Err(RSCMError::SchemaGridTypeMismatch {
                            aggregate: agg_name.clone(),
                            contributor: contributor.clone(),
                            contributor_grid: format!("{:?}", contributor_grid),
                            aggregate_grid: format!("{:?}", agg_def.grid_type),
                        });
                    }
                }
            }

            // 3.5: Validate weighted aggregate weight count
            if let AggregateOp::Weighted(weights) = &agg_def.operation {
                if weights.len() != agg_def.contributors.len() {
                    return Err(RSCMError::WeightCountMismatch {
                        aggregate: agg_name.clone(),
                        weight_count: weights.len(),
                        contributor_count: agg_def.contributors.len(),
                    });
                }
            }
        }

        // 3.6: Detect circular dependencies
        self.check_circular_dependencies()?;

        Ok(())
    }

    /// Return aggregates in topological order (dependencies before dependents).
    ///
    /// This ensures that when processing aggregates, any aggregate that references
    /// another aggregate will be processed after its dependencies.
    ///
    /// # Returns
    ///
    /// A vector of aggregate names in topological order.
    pub fn topological_order_aggregates(&self) -> Vec<String> {
        // Build in-degree map and adjacency list
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

        for name in self.aggregates.keys() {
            in_degree.entry(name.clone()).or_insert(0);
            dependents.entry(name.clone()).or_default();
        }

        // Count in-degrees (how many aggregates each aggregate depends on)
        for (name, agg_def) in &self.aggregates {
            for contributor in &agg_def.contributors {
                // Only count dependencies on other aggregates
                if self.aggregates.contains_key(contributor) {
                    *in_degree.get_mut(name).unwrap() += 1;
                    dependents.get_mut(contributor).unwrap().push(name.clone());
                }
            }
        }

        // Kahn's algorithm for topological sort
        let mut result = Vec::new();
        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(name, _)| name.clone())
            .collect();

        // Sort for deterministic ordering
        queue.sort();

        while let Some(name) = queue.pop() {
            result.push(name.clone());

            if let Some(deps) = dependents.get(&name) {
                for dep in deps {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            // Insert in sorted order for determinism
                            let pos = queue.binary_search(dep).unwrap_or_else(|e| e);
                            queue.insert(pos, dep.clone());
                        }
                    }
                }
            }
        }

        result
    }

    /// Check for circular dependencies in aggregate definitions.
    ///
    /// Uses depth-first search to detect cycles in the aggregate dependency graph.
    fn check_circular_dependencies(&self) -> RSCMResult<()> {
        // Track visited nodes and current path for cycle detection
        let mut visited = HashSet::new();
        let mut path = Vec::new();
        let mut path_set = HashSet::new();

        // Check from each aggregate
        for agg_name in self.aggregates.keys() {
            if !visited.contains(agg_name) {
                self.dfs_cycle_check(agg_name, &mut visited, &mut path, &mut path_set)?;
            }
        }

        Ok(())
    }

    /// Depth-first search for cycle detection.
    fn dfs_cycle_check(
        &self,
        current: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
        path_set: &mut HashSet<String>,
    ) -> RSCMResult<()> {
        // If we've seen this node in the current path, we have a cycle
        if path_set.contains(current) {
            // Build cycle description
            let cycle_start = path.iter().position(|n| n == current).unwrap();
            let mut cycle_nodes: Vec<_> = path[cycle_start..].to_vec();
            cycle_nodes.push(current.to_string());
            let cycle = cycle_nodes.join(" -> ");
            return Err(RSCMError::AggregateCircularDependency { cycle });
        }

        // Skip if already fully processed
        if visited.contains(current) {
            return Ok(());
        }

        // Add to current path
        path.push(current.to_string());
        path_set.insert(current.to_string());

        // Check all contributors that are themselves aggregates
        if let Some(agg_def) = self.aggregates.get(current) {
            for contributor in &agg_def.contributors {
                // Only follow edges to other aggregates (variables are leaf nodes)
                if self.aggregates.contains_key(contributor) {
                    self.dfs_cycle_check(contributor, visited, path, path_set)?;
                }
            }
        }

        // Remove from current path and mark as visited
        path.pop();
        path_set.remove(current);
        visited.insert(current.to_string());

        Ok(())
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

    /// Validate the schema for consistency (Python API)
    ///
    /// Raises ValueError if validation fails.
    #[pyo3(name = "validate")]
    fn py_validate(&self) -> pyo3::PyResult<()> {
        self.validate()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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

// =============================================================================
// Aggregate Computation
// =============================================================================

/// Compute an aggregate value from contributor values.
///
/// NaN values are excluded from the computation (treated as missing data).
/// If all contributors are NaN, the result is NaN.
///
/// For `Weighted` aggregates, weights corresponding to NaN values are also excluded.
///
/// # Arguments
///
/// * `contributors` - The contributor values (may contain NaN)
/// * `op` - The aggregation operation to apply
///
/// # Returns
///
/// The computed aggregate value, or NaN if all contributors are NaN.
///
/// # Examples
///
/// ```
/// use rscm_core::schema::{compute_aggregate, AggregateOp};
///
/// // Sum operation
/// let values = vec![1.0, 2.0, 3.0];
/// assert_eq!(compute_aggregate(&values, &AggregateOp::Sum), 6.0);
///
/// // Sum with NaN excluded
/// let values_nan = vec![1.0, f64::NAN, 3.0];
/// assert_eq!(compute_aggregate(&values_nan, &AggregateOp::Sum), 4.0);
///
/// // Mean divides by count of valid values
/// let values = vec![1.0, 2.0, 3.0];
/// assert_eq!(compute_aggregate(&values, &AggregateOp::Mean), 2.0);
///
/// // Weighted sum
/// let values = vec![10.0, 20.0];
/// let op = AggregateOp::Weighted(vec![0.3, 0.7]);
/// assert_eq!(compute_aggregate(&values, &op), 17.0); // 10*0.3 + 20*0.7
/// ```
pub fn compute_aggregate(contributors: &[f64], op: &AggregateOp) -> f64 {
    match op {
        AggregateOp::Sum => {
            let valid: Vec<f64> = contributors
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect();
            if valid.is_empty() {
                f64::NAN
            } else {
                valid.iter().sum()
            }
        }
        AggregateOp::Mean => {
            let valid: Vec<f64> = contributors
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect();
            if valid.is_empty() {
                f64::NAN
            } else {
                valid.iter().sum::<f64>() / valid.len() as f64
            }
        }
        AggregateOp::Weighted(weights) => {
            // Filter out NaN values and their corresponding weights
            let valid_pairs: Vec<(f64, f64)> = contributors
                .iter()
                .zip(weights.iter())
                .filter(|(v, _)| !v.is_nan())
                .map(|(v, w)| (*v, *w))
                .collect();

            if valid_pairs.is_empty() {
                f64::NAN
            } else {
                valid_pairs.iter().map(|(v, w)| v * w).sum()
            }
        }
    }
}

// =============================================================================
// Virtual Components (Aggregator and Grid Transformer)
// =============================================================================

use crate::component::{
    Component, InputState, OutputState, RequirementDefinition, RequirementType,
};
use crate::spatial::{FourBoxGrid, HemisphericGrid, SpatialGrid};
use crate::state::StateValue;
use crate::timeseries::Time;

/// A virtual component that computes aggregate values from contributors.
///
/// `AggregatorComponent` is automatically created by `ModelBuilder` when a
/// schema with aggregates is provided. It reads values from all contributors
/// and writes the computed aggregate to its output variable.
///
/// This component is internal and should not be created directly by users.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorComponent {
    /// The name of the aggregate variable this component produces
    pub aggregate_name: String,
    /// The unit of the aggregate variable
    pub unit: String,
    /// The grid type of the aggregate
    pub grid_type: GridType,
    /// The aggregation operation to apply
    pub operation: AggregateOp,
    /// The names of the contributor variables to read
    pub contributors: Vec<String>,
}

impl AggregatorComponent {
    /// Create a new aggregator component from an aggregate definition.
    pub fn from_definition(def: &AggregateDefinition) -> Self {
        Self {
            aggregate_name: def.name.clone(),
            unit: def.unit.clone(),
            grid_type: def.grid_type,
            operation: def.operation.clone(),
            contributors: def.contributors.clone(),
        }
    }
}

#[typetag::serde]
impl Component for AggregatorComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        let mut defs = Vec::with_capacity(self.contributors.len() + 1);

        // Input: read from each contributor
        for contributor in &self.contributors {
            defs.push(RequirementDefinition::with_grid(
                contributor,
                &self.unit,
                RequirementType::Input,
                self.grid_type,
            ));
        }

        // Output: the aggregate variable
        defs.push(RequirementDefinition::with_grid(
            &self.aggregate_name,
            &self.unit,
            RequirementType::Output,
            self.grid_type,
        ));

        defs
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let mut output = OutputState::new();

        // Aggregators read values that upstream components just wrote in this timestep.
        // Use at_end() to read from timestep N+1 where outputs were written, falling
        // back to at_start() (index N) if at_end() returns None (e.g., last timestep).

        match self.grid_type {
            GridType::Scalar => {
                // Collect scalar values from all contributors at end of timestep
                let values: Vec<f64> = self
                    .contributors
                    .iter()
                    .map(|name| {
                        let window = input_state.get_scalar_window(name);
                        // Read from at_end() where upstream values were just written
                        window.at_end().unwrap_or_else(|| window.at_start())
                    })
                    .collect();

                let result = compute_aggregate(&values, &self.operation);
                output.insert(self.aggregate_name.clone(), StateValue::Scalar(result));
            }
            GridType::FourBox => {
                use crate::state::FourBoxSlice;

                // For FourBox, we need to aggregate per-region
                let mut region_values: [Vec<f64>; 4] = Default::default();

                for name in &self.contributors {
                    let window = input_state.get_four_box_window(name);
                    // Read from at_end() where upstream values were just written
                    let values = window
                        .current_all_at_end()
                        .unwrap_or_else(|| window.current_all_at_start());
                    for (i, val) in values.into_iter().enumerate() {
                        region_values[i].push(val);
                    }
                }

                let mut result = FourBoxSlice::new();
                for (i, vals) in region_values.iter().enumerate() {
                    let agg = compute_aggregate(vals, &self.operation);
                    result.0[i] = agg;
                }

                output.insert(self.aggregate_name.clone(), StateValue::FourBox(result));
            }
            GridType::Hemispheric => {
                use crate::state::HemisphericSlice;

                // For Hemispheric, we need to aggregate per-region
                let mut region_values: [Vec<f64>; 2] = Default::default();

                for name in &self.contributors {
                    let window = input_state.get_hemispheric_window(name);
                    // Read from at_end() where upstream values were just written
                    let values = window
                        .current_all_at_end()
                        .unwrap_or_else(|| window.current_all_at_start());
                    for (i, val) in values.into_iter().enumerate() {
                        region_values[i].push(val);
                    }
                }

                let mut result = HemisphericSlice::new();
                for (i, vals) in region_values.iter().enumerate() {
                    let agg = compute_aggregate(vals, &self.operation);
                    result.0[i] = agg;
                }

                output.insert(self.aggregate_name.clone(), StateValue::Hemispheric(result));
            }
        }

        Ok(output)
    }
}

// =============================================================================
// Grid Transformer Component
// =============================================================================

/// A virtual component that transforms values from a finer grid to a coarser grid.
///
/// `GridTransformerComponent` is automatically created by `ModelBuilder` when schema-driven
/// auto-aggregation detects that a component needs to read a variable at a coarser resolution
/// than the schema declares, or when a component produces output at a finer resolution than
/// the schema declares.
///
/// # Transformation Types
///
/// Supports the following aggregation transformations:
/// - FourBox → Scalar: weighted average of all four regions
/// - FourBox → Hemispheric: weighted average within each hemisphere
/// - Hemispheric → Scalar: weighted average of both hemispheres
///
/// Disaggregation (coarser to finer) is not supported and will result in a build error.
///
/// # Variable Naming
///
/// The transformer reads from the source variable (at the finer grid) and writes to an
/// intermediate variable with the naming convention: `"VarName|_to_<target_grid>"`.
///
/// # Example
///
/// ```ignore
/// // Schema declares Temperature at FourBox resolution
/// // Consumer component wants Scalar
/// // GridTransformerComponent is inserted:
/// //   - Reads: "Temperature" (FourBox)
/// //   - Writes: "Temperature|_to_scalar" (Scalar)
/// // Consumer then reads "Temperature|_to_scalar"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridTransformerComponent {
    /// The name of the source variable to read (at finer resolution)
    pub source_var: String,
    /// The unit of the variable
    pub unit: String,
    /// The source grid type (finer resolution)
    pub source_grid: GridType,
    /// The target grid type (coarser resolution)
    pub target_grid: GridType,
    /// Weights for FourBox grid aggregation (if source is FourBox)
    pub four_box_weights: Option<[f64; 4]>,
    /// Weights for Hemispheric grid aggregation (if source is Hemispheric)
    pub hemispheric_weights: Option<[f64; 2]>,
}

impl GridTransformerComponent {
    /// Create a new grid transformer component.
    ///
    /// # Arguments
    ///
    /// * `source_var` - The name of the source variable (at finer resolution)
    /// * `unit` - The physical unit of the variable
    /// * `source_grid` - The grid type of the source variable
    /// * `target_grid` - The desired target grid type (must be coarser)
    ///
    /// # Returns
    ///
    /// Returns `Ok(GridTransformerComponent)` if the transformation is valid,
    /// or `Err` if the transformation is not supported (e.g., disaggregation).
    pub fn new(
        source_var: impl Into<String>,
        unit: impl Into<String>,
        source_grid: GridType,
        target_grid: GridType,
    ) -> RSCMResult<Self> {
        // Validate that this is a valid aggregation (not disaggregation)
        if !source_grid.can_aggregate_to(target_grid) {
            return Err(RSCMError::GridTransformationNotSupported {
                source_grid: source_grid.to_string(),
                target_grid: target_grid.to_string(),
                variable: source_var.into(),
            });
        }

        Ok(Self {
            source_var: source_var.into(),
            unit: unit.into(),
            source_grid,
            target_grid,
            four_box_weights: None,
            hemispheric_weights: None,
        })
    }

    /// Create with custom FourBox weights.
    ///
    /// Used when the model has custom area-based weights configured.
    pub fn with_four_box_weights(mut self, weights: [f64; 4]) -> Self {
        self.four_box_weights = Some(weights);
        self
    }

    /// Create with custom Hemispheric weights.
    ///
    /// Used when the model has custom area-based weights configured.
    pub fn with_hemispheric_weights(mut self, weights: [f64; 2]) -> Self {
        self.hemispheric_weights = Some(weights);
        self
    }

    /// Get the output variable name for this transformer.
    ///
    /// Uses the naming convention: `"VarName|_to_<target_grid>"`.
    pub fn output_name(&self) -> String {
        let target_suffix = match self.target_grid {
            GridType::Scalar => "scalar",
            GridType::Hemispheric => "hemispheric",
            GridType::FourBox => "fourbox",
        };
        format!("{}|_to_{}", self.source_var, target_suffix)
    }

    /// Get the FourBox grid to use for aggregation.
    fn get_four_box_grid(&self) -> FourBoxGrid {
        match self.four_box_weights {
            Some(weights) => FourBoxGrid::with_weights(weights),
            None => FourBoxGrid::magicc_standard(),
        }
    }

    /// Get the Hemispheric grid to use for aggregation.
    fn get_hemispheric_grid(&self) -> HemisphericGrid {
        match self.hemispheric_weights {
            Some(weights) => HemisphericGrid::with_weights(weights),
            None => HemisphericGrid::equal_weights(),
        }
    }
}

#[typetag::serde]
impl Component for GridTransformerComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::with_grid(
                &self.source_var,
                &self.unit,
                RequirementType::Input,
                self.source_grid,
            ),
            RequirementDefinition::with_grid(
                &self.output_name(),
                &self.unit,
                RequirementType::Output,
                self.target_grid,
            ),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        use crate::state::{FourBoxSlice, HemisphericSlice};

        let mut output = OutputState::new();

        match (self.source_grid, self.target_grid) {
            // FourBox → Scalar
            (GridType::FourBox, GridType::Scalar) => {
                let window = input_state.get_four_box_window(&self.source_var);
                let values = window.current_all();
                let grid = self.get_four_box_grid();
                let scalar = grid.aggregate_global(&values);
                output.insert(self.output_name(), StateValue::Scalar(scalar));
            }

            // FourBox → Hemispheric
            (GridType::FourBox, GridType::Hemispheric) => {
                let window = input_state.get_four_box_window(&self.source_var);
                let values = window.current_all();
                let grid = self.get_four_box_grid();
                let target = self.get_hemispheric_grid();
                let hemispheric = grid.transform_to(&values, &target)?;
                output.insert(
                    self.output_name(),
                    StateValue::Hemispheric(HemisphericSlice::from_array([
                        hemispheric[0],
                        hemispheric[1],
                    ])),
                );
            }

            // Hemispheric → Scalar
            (GridType::Hemispheric, GridType::Scalar) => {
                let window = input_state.get_hemispheric_window(&self.source_var);
                let values = window.current_all();
                let grid = self.get_hemispheric_grid();
                let scalar = grid.aggregate_global(&values);
                output.insert(self.output_name(), StateValue::Scalar(scalar));
            }

            // Same grid → identity (shouldn't happen, but handle it)
            (s, t) if s == t => match s {
                GridType::Scalar => {
                    let window = input_state.get_scalar_window(&self.source_var);
                    let value = window.current();
                    output.insert(self.output_name(), StateValue::Scalar(value));
                }
                GridType::FourBox => {
                    let window = input_state.get_four_box_window(&self.source_var);
                    let values = window.current_all();
                    output.insert(
                        self.output_name(),
                        StateValue::FourBox(FourBoxSlice::from_array([
                            values[0], values[1], values[2], values[3],
                        ])),
                    );
                }
                GridType::Hemispheric => {
                    let window = input_state.get_hemispheric_window(&self.source_var);
                    let values = window.current_all();
                    output.insert(
                        self.output_name(),
                        StateValue::Hemispheric(HemisphericSlice::from_array([
                            values[0], values[1],
                        ])),
                    );
                }
            },

            // Unsupported transformation (disaggregation)
            _ => {
                return Err(RSCMError::GridTransformationNotSupported {
                    source_grid: self.source_grid.to_string(),
                    target_grid: self.target_grid.to_string(),
                    variable: self.source_var.clone(),
                });
            }
        }

        Ok(output)
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

    // Validation tests - happy path

    #[test]
    fn test_validate_valid_schema() {
        let schema = VariableSchema::new()
            .variable("ERF|CO2", "W/m^2")
            .variable("ERF|CH4", "W/m^2")
            .aggregate("Total ERF", "W/m^2", AggregateOp::Sum)
            .from("ERF|CO2")
            .from("ERF|CH4")
            .build();

        assert!(schema.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_schema() {
        let schema = VariableSchema::new();
        assert!(schema.validate().is_ok());
    }

    #[test]
    fn test_validate_chained_aggregates() {
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

        assert!(schema.validate().is_ok());
    }

    #[test]
    fn test_validate_weighted_aggregate_matching_weights() {
        let schema = VariableSchema::new()
            .variable("A", "units")
            .variable("B", "units")
            .aggregate("Total", "units", AggregateOp::Weighted(vec![0.6, 0.4]))
            .from("A")
            .from("B")
            .build();

        assert!(schema.validate().is_ok());
    }

    // Validation tests - error cases

    #[test]
    fn test_validate_undefined_contributor() {
        let schema = VariableSchema::new()
            .variable("ERF|CO2", "W/m^2")
            .aggregate("Total ERF", "W/m^2", AggregateOp::Sum)
            .from("ERF|CO2")
            .from("ERF|CH4") // Not defined
            .build();

        let err = schema.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Undefined contributor"),
            "Expected UndefinedContributor error, got: {}",
            msg
        );
        assert!(
            msg.contains("ERF|CH4"),
            "Error should mention the missing contributor: {}",
            msg
        );
    }

    #[test]
    fn test_validate_unit_mismatch() {
        let schema = VariableSchema::new()
            .variable("ERF|CO2", "W/m^2")
            .variable("Emissions|CO2", "GtCO2/yr") // Different unit
            .aggregate("Total", "W/m^2", AggregateOp::Sum)
            .from("ERF|CO2")
            .from("Emissions|CO2") // Unit mismatch
            .build();

        let err = schema.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Unit mismatch"),
            "Expected SchemaUnitMismatch error, got: {}",
            msg
        );
        assert!(
            msg.contains("GtCO2/yr"),
            "Error should mention mismatched unit: {}",
            msg
        );
    }

    #[test]
    fn test_validate_grid_type_mismatch() {
        let schema = VariableSchema::new()
            .variable("Global Temp", "K")
            .variable_with_grid("Regional Temp", "K", GridType::FourBox)
            .aggregate("Total Temp", "K", AggregateOp::Sum) // Scalar aggregate
            .from("Global Temp")
            .from("Regional Temp") // FourBox contributor
            .build();

        let err = schema.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Grid type mismatch"),
            "Expected SchemaGridTypeMismatch error, got: {}",
            msg
        );
        assert!(
            msg.contains("FourBox"),
            "Error should mention FourBox: {}",
            msg
        );
    }

    #[test]
    fn test_validate_weight_count_mismatch() {
        let schema = VariableSchema::new()
            .variable("A", "units")
            .variable("B", "units")
            .variable("C", "units")
            .aggregate(
                "Total",
                "units",
                AggregateOp::Weighted(vec![0.5, 0.5]), // Only 2 weights
            )
            .from("A")
            .from("B")
            .from("C") // 3 contributors
            .build();

        let err = schema.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Weight count mismatch"),
            "Expected WeightCountMismatch error, got: {}",
            msg
        );
        assert!(
            msg.contains("2 weights"),
            "Error should mention weight count: {}",
            msg
        );
        assert!(
            msg.contains("3 contributors"),
            "Error should mention contributor count: {}",
            msg
        );
    }

    #[test]
    fn test_validate_circular_dependency_direct() {
        // Create schema with direct circular reference: A -> B -> A
        let mut schema = VariableSchema::new();
        schema.aggregates.insert(
            "A".to_string(),
            AggregateDefinition {
                name: "A".to_string(),
                unit: "units".to_string(),
                grid_type: GridType::Scalar,
                operation: AggregateOp::Sum,
                contributors: vec!["B".to_string()],
            },
        );
        schema.aggregates.insert(
            "B".to_string(),
            AggregateDefinition {
                name: "B".to_string(),
                unit: "units".to_string(),
                grid_type: GridType::Scalar,
                operation: AggregateOp::Sum,
                contributors: vec!["A".to_string()],
            },
        );

        let err = schema.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Circular dependency"),
            "Expected AggregateCircularDependency error, got: {}",
            msg
        );
    }

    #[test]
    fn test_validate_circular_dependency_self_reference() {
        // Create schema with self-reference: A -> A
        let mut schema = VariableSchema::new();
        schema.aggregates.insert(
            "A".to_string(),
            AggregateDefinition {
                name: "A".to_string(),
                unit: "units".to_string(),
                grid_type: GridType::Scalar,
                operation: AggregateOp::Sum,
                contributors: vec!["A".to_string()],
            },
        );

        let err = schema.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Circular dependency"),
            "Expected AggregateCircularDependency error, got: {}",
            msg
        );
    }

    #[test]
    fn test_validate_circular_dependency_indirect() {
        // Create schema with indirect cycle: A -> B -> C -> A
        let mut schema = VariableSchema::new();
        schema.aggregates.insert(
            "A".to_string(),
            AggregateDefinition {
                name: "A".to_string(),
                unit: "units".to_string(),
                grid_type: GridType::Scalar,
                operation: AggregateOp::Sum,
                contributors: vec!["B".to_string()],
            },
        );
        schema.aggregates.insert(
            "B".to_string(),
            AggregateDefinition {
                name: "B".to_string(),
                unit: "units".to_string(),
                grid_type: GridType::Scalar,
                operation: AggregateOp::Sum,
                contributors: vec!["C".to_string()],
            },
        );
        schema.aggregates.insert(
            "C".to_string(),
            AggregateDefinition {
                name: "C".to_string(),
                unit: "units".to_string(),
                grid_type: GridType::Scalar,
                operation: AggregateOp::Sum,
                contributors: vec!["A".to_string()],
            },
        );

        let err = schema.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Circular dependency"),
            "Expected AggregateCircularDependency error, got: {}",
            msg
        );
    }

    #[test]
    fn test_validate_diamond_dependency_no_cycle() {
        // Diamond pattern is valid (no cycle): A -> B -> D, A -> C -> D
        let schema = VariableSchema::new()
            .variable("X", "units")
            .aggregate("B", "units", AggregateOp::Sum)
            .from("X")
            .build()
            .aggregate("C", "units", AggregateOp::Sum)
            .from("X")
            .build()
            .aggregate("A", "units", AggregateOp::Sum)
            .from("B")
            .from("C")
            .build();

        assert!(schema.validate().is_ok());
    }

    #[test]
    fn test_validate_aggregate_referencing_aggregate() {
        // Valid case: aggregate references another aggregate
        let schema = VariableSchema::new()
            .variable("X", "units")
            .aggregate("Inner", "units", AggregateOp::Sum)
            .from("X")
            .build()
            .aggregate("Outer", "units", AggregateOp::Sum)
            .from("Inner")
            .build();

        assert!(schema.validate().is_ok());
    }

    // compute_aggregate tests

    #[test]
    fn test_compute_aggregate_sum() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let result = compute_aggregate(&values, &AggregateOp::Sum);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_compute_aggregate_sum_with_nan() {
        let values = vec![1.0, f64::NAN, 3.0, 4.0];
        let result = compute_aggregate(&values, &AggregateOp::Sum);
        assert_eq!(result, 8.0); // 1 + 3 + 4, NaN excluded
    }

    #[test]
    fn test_compute_aggregate_sum_all_nan() {
        let values = vec![f64::NAN, f64::NAN];
        let result = compute_aggregate(&values, &AggregateOp::Sum);
        assert!(result.is_nan());
    }

    #[test]
    fn test_compute_aggregate_sum_empty() {
        let values: Vec<f64> = vec![];
        let result = compute_aggregate(&values, &AggregateOp::Sum);
        assert!(result.is_nan());
    }

    #[test]
    fn test_compute_aggregate_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let result = compute_aggregate(&values, &AggregateOp::Mean);
        assert_eq!(result, 2.5); // (1+2+3+4) / 4
    }

    #[test]
    fn test_compute_aggregate_mean_with_nan() {
        let values = vec![1.0, f64::NAN, 3.0];
        let result = compute_aggregate(&values, &AggregateOp::Mean);
        assert_eq!(result, 2.0); // (1 + 3) / 2 (NaN excluded from count)
    }

    #[test]
    fn test_compute_aggregate_mean_all_nan() {
        let values = vec![f64::NAN, f64::NAN];
        let result = compute_aggregate(&values, &AggregateOp::Mean);
        assert!(result.is_nan());
    }

    #[test]
    fn test_compute_aggregate_weighted() {
        let values = vec![10.0, 20.0, 30.0];
        let weights = vec![0.5, 0.3, 0.2];
        let result = compute_aggregate(&values, &AggregateOp::Weighted(weights));
        assert_eq!(result, 17.0); // 10*0.5 + 20*0.3 + 30*0.2 = 5 + 6 + 6 = 17
    }

    #[test]
    fn test_compute_aggregate_weighted_with_nan() {
        let values = vec![10.0, f64::NAN, 30.0];
        let weights = vec![0.5, 0.3, 0.2];
        let result = compute_aggregate(&values, &AggregateOp::Weighted(weights));
        assert_eq!(result, 11.0); // 10*0.5 + 30*0.2 = 5 + 6 = 11 (NaN value and its weight excluded)
    }

    #[test]
    fn test_compute_aggregate_weighted_all_nan() {
        let values = vec![f64::NAN, f64::NAN];
        let weights = vec![0.6, 0.4];
        let result = compute_aggregate(&values, &AggregateOp::Weighted(weights));
        assert!(result.is_nan());
    }

    // AggregatorComponent tests

    #[test]
    fn test_aggregator_component_from_definition() {
        let def = AggregateDefinition {
            name: "Total ERF".to_string(),
            unit: "W/m^2".to_string(),
            grid_type: GridType::Scalar,
            operation: AggregateOp::Sum,
            contributors: vec!["ERF|CO2".to_string(), "ERF|CH4".to_string()],
        };

        let agg = AggregatorComponent::from_definition(&def);

        assert_eq!(agg.aggregate_name, "Total ERF");
        assert_eq!(agg.unit, "W/m^2");
        assert_eq!(agg.grid_type, GridType::Scalar);
        assert_eq!(agg.operation, AggregateOp::Sum);
        assert_eq!(agg.contributors, vec!["ERF|CO2", "ERF|CH4"]);
    }

    #[test]
    fn test_aggregator_component_definitions() {
        let def = AggregateDefinition {
            name: "Total ERF".to_string(),
            unit: "W/m^2".to_string(),
            grid_type: GridType::Scalar,
            operation: AggregateOp::Sum,
            contributors: vec!["ERF|CO2".to_string(), "ERF|CH4".to_string()],
        };

        let agg = AggregatorComponent::from_definition(&def);
        let defs = agg.definitions();

        // Should have 2 inputs + 1 output
        assert_eq!(defs.len(), 3);

        // First two are inputs
        assert_eq!(defs[0].name, "ERF|CO2");
        assert_eq!(defs[0].requirement_type, RequirementType::Input);
        assert_eq!(defs[1].name, "ERF|CH4");
        assert_eq!(defs[1].requirement_type, RequirementType::Input);

        // Last one is output
        assert_eq!(defs[2].name, "Total ERF");
        assert_eq!(defs[2].requirement_type, RequirementType::Output);
    }

    #[test]
    fn test_aggregator_component_serialization() {
        let def = AggregateDefinition {
            name: "Total".to_string(),
            unit: "units".to_string(),
            grid_type: GridType::Scalar,
            operation: AggregateOp::Mean,
            contributors: vec!["A".to_string(), "B".to_string()],
        };

        let agg = AggregatorComponent::from_definition(&def);
        let json = serde_json::to_string(&agg).unwrap();
        let deserialized: AggregatorComponent = serde_json::from_str(&json).unwrap();

        assert_eq!(agg.aggregate_name, deserialized.aggregate_name);
        assert_eq!(agg.operation, deserialized.operation);
        assert_eq!(agg.contributors, deserialized.contributors);
    }

    // Deserialization validation tests

    #[test]
    fn test_deserialize_valid_schema_json() {
        let json = r#"{
            "variables": {
                "ERF|CO2": {"name": "ERF|CO2", "unit": "W/m^2", "grid_type": "Scalar"},
                "ERF|CH4": {"name": "ERF|CH4", "unit": "W/m^2", "grid_type": "Scalar"}
            },
            "aggregates": {
                "Total ERF": {
                    "name": "Total ERF",
                    "unit": "W/m^2",
                    "grid_type": "Scalar",
                    "operation": "Sum",
                    "contributors": ["ERF|CO2", "ERF|CH4"]
                }
            }
        }"#;

        let schema: VariableSchema = serde_json::from_str(json).unwrap();
        assert_eq!(schema.variables.len(), 2);
        assert_eq!(schema.aggregates.len(), 1);
    }

    #[test]
    fn test_deserialize_invalid_schema_undefined_contributor() {
        let json = r#"{
            "variables": {
                "ERF|CO2": {"name": "ERF|CO2", "unit": "W/m^2", "grid_type": "Scalar"}
            },
            "aggregates": {
                "Total ERF": {
                    "name": "Total ERF",
                    "unit": "W/m^2",
                    "grid_type": "Scalar",
                    "operation": "Sum",
                    "contributors": ["ERF|CO2", "ERF|CH4"]
                }
            }
        }"#;

        let result: Result<VariableSchema, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Undefined contributor"),
            "Expected undefined contributor error, got: {}",
            err
        );
        assert!(
            err.contains("ERF|CH4"),
            "Error should mention missing contributor: {}",
            err
        );
    }

    #[test]
    fn test_deserialize_invalid_schema_unit_mismatch() {
        let json = r#"{
            "variables": {
                "ERF|CO2": {"name": "ERF|CO2", "unit": "W/m^2", "grid_type": "Scalar"},
                "Emissions|CO2": {"name": "Emissions|CO2", "unit": "GtCO2/yr", "grid_type": "Scalar"}
            },
            "aggregates": {
                "Total": {
                    "name": "Total",
                    "unit": "W/m^2",
                    "grid_type": "Scalar",
                    "operation": "Sum",
                    "contributors": ["ERF|CO2", "Emissions|CO2"]
                }
            }
        }"#;

        let result: Result<VariableSchema, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Unit mismatch"),
            "Expected unit mismatch error, got: {}",
            err
        );
    }

    #[test]
    fn test_deserialize_invalid_schema_circular_dependency() {
        let json = r#"{
            "variables": {},
            "aggregates": {
                "A": {
                    "name": "A",
                    "unit": "units",
                    "grid_type": "Scalar",
                    "operation": "Sum",
                    "contributors": ["B"]
                },
                "B": {
                    "name": "B",
                    "unit": "units",
                    "grid_type": "Scalar",
                    "operation": "Sum",
                    "contributors": ["A"]
                }
            }
        }"#;

        let result: Result<VariableSchema, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Circular dependency"),
            "Expected circular dependency error, got: {}",
            err
        );
    }

    #[test]
    fn test_deserialize_invalid_schema_weight_count_mismatch() {
        let json = r#"{
            "variables": {
                "A": {"name": "A", "unit": "units", "grid_type": "Scalar"},
                "B": {"name": "B", "unit": "units", "grid_type": "Scalar"},
                "C": {"name": "C", "unit": "units", "grid_type": "Scalar"}
            },
            "aggregates": {
                "Total": {
                    "name": "Total",
                    "unit": "units",
                    "grid_type": "Scalar",
                    "operation": {"Weighted": [0.5, 0.5]},
                    "contributors": ["A", "B", "C"]
                }
            }
        }"#;

        let result: Result<VariableSchema, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Weight count mismatch"),
            "Expected weight count mismatch error, got: {}",
            err
        );
    }

    #[test]
    fn test_deserialize_valid_schema_toml() {
        // TOML keys must match the name in the contributor list
        // The HashMap key is used for lookup, so keys must match contributor references
        let toml = r#"
            [variables."ERF|CO2"]
            name = "ERF|CO2"
            unit = "W/m^2"
            grid_type = "Scalar"

            [aggregates.Total]
            name = "Total"
            unit = "W/m^2"
            grid_type = "Scalar"
            operation = "Sum"
            contributors = ["ERF|CO2"]
        "#;

        let schema: VariableSchema = toml::from_str(toml).unwrap();
        assert_eq!(schema.variables.len(), 1);
        assert_eq!(schema.aggregates.len(), 1);
    }

    #[test]
    fn test_deserialize_invalid_schema_toml_undefined_contributor() {
        // Need to include variables field (can be empty)
        let toml = r#"
            [variables]

            [aggregates.Total]
            name = "Total"
            unit = "W/m^2"
            grid_type = "Scalar"
            operation = "Sum"
            contributors = ["Missing"]
        "#;

        let result: Result<VariableSchema, _> = toml::from_str(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Undefined contributor"),
            "Expected undefined contributor error, got: {}",
            err
        );
    }

    // GridTransformerComponent tests

    #[test]
    fn test_grid_transformer_new_valid_fourbox_to_scalar() {
        let transformer =
            GridTransformerComponent::new("Temperature", "K", GridType::FourBox, GridType::Scalar)
                .unwrap();
        assert_eq!(transformer.source_var, "Temperature");
        assert_eq!(transformer.unit, "K");
        assert_eq!(transformer.source_grid, GridType::FourBox);
        assert_eq!(transformer.target_grid, GridType::Scalar);
        assert_eq!(transformer.output_name(), "Temperature|_to_scalar");
    }

    #[test]
    fn test_grid_transformer_new_valid_fourbox_to_hemispheric() {
        let transformer = GridTransformerComponent::new(
            "Temperature",
            "K",
            GridType::FourBox,
            GridType::Hemispheric,
        )
        .unwrap();
        assert_eq!(transformer.output_name(), "Temperature|_to_hemispheric");
    }

    #[test]
    fn test_grid_transformer_new_valid_hemispheric_to_scalar() {
        let transformer = GridTransformerComponent::new(
            "Temperature",
            "K",
            GridType::Hemispheric,
            GridType::Scalar,
        )
        .unwrap();
        assert_eq!(transformer.output_name(), "Temperature|_to_scalar");
    }

    #[test]
    fn test_grid_transformer_new_invalid_scalar_to_fourbox() {
        let result =
            GridTransformerComponent::new("Temperature", "K", GridType::Scalar, GridType::FourBox);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Grid transformation not supported"),
            "Expected transformation not supported error, got: {}",
            err
        );
    }

    #[test]
    fn test_grid_transformer_new_invalid_scalar_to_hemispheric() {
        let result = GridTransformerComponent::new(
            "Temperature",
            "K",
            GridType::Scalar,
            GridType::Hemispheric,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_grid_transformer_new_invalid_hemispheric_to_fourbox() {
        let result = GridTransformerComponent::new(
            "Temperature",
            "K",
            GridType::Hemispheric,
            GridType::FourBox,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_grid_transformer_same_grid_is_valid() {
        // Same-grid transformations are identity operations and are valid
        let transformer =
            GridTransformerComponent::new("Temperature", "K", GridType::Scalar, GridType::Scalar)
                .unwrap();
        assert_eq!(transformer.output_name(), "Temperature|_to_scalar");
    }

    #[test]
    fn test_grid_transformer_definitions() {
        let transformer =
            GridTransformerComponent::new("Temperature", "K", GridType::FourBox, GridType::Scalar)
                .unwrap();
        let defs = transformer.definitions();

        assert_eq!(defs.len(), 2);
        // Input: FourBox
        assert_eq!(defs[0].name, "Temperature");
        assert_eq!(defs[0].unit, "K");
        assert_eq!(defs[0].requirement_type, RequirementType::Input);
        assert_eq!(defs[0].grid_type, GridType::FourBox);
        // Output: Scalar
        assert_eq!(defs[1].name, "Temperature|_to_scalar");
        assert_eq!(defs[1].unit, "K");
        assert_eq!(defs[1].requirement_type, RequirementType::Output);
        assert_eq!(defs[1].grid_type, GridType::Scalar);
    }

    #[test]
    fn test_grid_transformer_with_custom_fourbox_weights() {
        let transformer =
            GridTransformerComponent::new("Temperature", "K", GridType::FourBox, GridType::Scalar)
                .unwrap()
                .with_four_box_weights([0.36, 0.14, 0.36, 0.14]);

        assert_eq!(transformer.four_box_weights, Some([0.36, 0.14, 0.36, 0.14]));
    }

    #[test]
    fn test_grid_transformer_with_custom_hemispheric_weights() {
        let transformer = GridTransformerComponent::new(
            "Temperature",
            "K",
            GridType::Hemispheric,
            GridType::Scalar,
        )
        .unwrap()
        .with_hemispheric_weights([0.7, 0.3]);

        assert_eq!(transformer.hemispheric_weights, Some([0.7, 0.3]));
    }

    #[test]
    fn test_grid_transformer_serialization() {
        let transformer =
            GridTransformerComponent::new("Temperature", "K", GridType::FourBox, GridType::Scalar)
                .unwrap();

        let json = serde_json::to_string(&transformer).unwrap();
        let deserialized: GridTransformerComponent = serde_json::from_str(&json).unwrap();

        assert_eq!(transformer.source_var, deserialized.source_var);
        assert_eq!(transformer.unit, deserialized.unit);
        assert_eq!(transformer.source_grid, deserialized.source_grid);
        assert_eq!(transformer.target_grid, deserialized.target_grid);
    }

    #[test]
    fn test_grid_transformer_serialization_with_weights() {
        let transformer =
            GridTransformerComponent::new("Temperature", "K", GridType::FourBox, GridType::Scalar)
                .unwrap()
                .with_four_box_weights([0.36, 0.14, 0.36, 0.14]);

        let json = serde_json::to_string(&transformer).unwrap();
        let deserialized: GridTransformerComponent = serde_json::from_str(&json).unwrap();

        assert_eq!(transformer.four_box_weights, deserialized.four_box_weights);
    }

    // GridTransformerComponent solve() integration tests
    mod grid_transformer_solve_tests {
        use super::*;
        use crate::component::Component;
        use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use crate::spatial::{FourBoxGrid, HemisphericGrid};
        use crate::state::StateValue;
        use crate::timeseries::{FloatValue, GridTimeseries, TimeAxis};
        use crate::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};
        use numpy::array;
        use numpy::ndarray::Array2;
        use std::sync::Arc;

        fn create_four_box_item(name: &str, values: [FloatValue; 4]) -> TimeseriesItem {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
            let data = Array2::from_shape_vec(
                (2, 4),
                vec![
                    values[0], values[1], values[2], values[3], // Time 0
                    values[0], values[1], values[2], values[3], // Time 1
                ],
            )
            .unwrap();

            let ts = GridTimeseries::new(
                data,
                time_axis,
                grid,
                "K".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            TimeseriesItem {
                data: TimeseriesData::FourBox(ts),
                name: name.to_string(),
                variable_type: VariableType::Endogenous,
            }
        }

        fn create_hemispheric_item(name: &str, values: [FloatValue; 2]) -> TimeseriesItem {
            let grid = HemisphericGrid::equal_weights();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
            let data = Array2::from_shape_vec(
                (2, 2),
                vec![
                    values[0], values[1], // Time 0
                    values[0], values[1], // Time 1
                ],
            )
            .unwrap();

            let ts = GridTimeseries::new(
                data,
                time_axis,
                grid,
                "K".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            TimeseriesItem {
                data: TimeseriesData::Hemispheric(ts),
                name: name.to_string(),
                variable_type: VariableType::Endogenous,
            }
        }

        #[test]
        fn test_grid_transformer_solve_fourbox_to_scalar() {
            let transformer = GridTransformerComponent::new(
                "Temperature",
                "K",
                GridType::FourBox,
                GridType::Scalar,
            )
            .unwrap();

            // Values: NO=16.0, NL=14.0, SO=12.0, SL=10.0
            // With equal weights (0.25 each), global = (16+14+12+10)/4 = 13.0
            let item = create_four_box_item("Temperature", [16.0, 14.0, 12.0, 10.0]);
            let input_state = InputState::build(vec![&item], 2000.0);

            let output = transformer.solve(2000.0, 2001.0, &input_state).unwrap();

            assert_eq!(
                output.get("Temperature|_to_scalar"),
                Some(&StateValue::Scalar(13.0))
            );
        }

        #[test]
        fn test_grid_transformer_solve_fourbox_to_scalar_with_custom_weights() {
            let transformer = GridTransformerComponent::new(
                "Temperature",
                "K",
                GridType::FourBox,
                GridType::Scalar,
            )
            .unwrap()
            .with_four_box_weights([0.4, 0.1, 0.4, 0.1]); // Ocean-heavy weights

            // Values: NO=16.0, NL=14.0, SO=12.0, SL=10.0
            // Weighted: 16*0.4 + 14*0.1 + 12*0.4 + 10*0.1 = 6.4 + 1.4 + 4.8 + 1.0 = 13.6
            let item = create_four_box_item("Temperature", [16.0, 14.0, 12.0, 10.0]);
            let input_state = InputState::build(vec![&item], 2000.0);

            let output = transformer.solve(2000.0, 2001.0, &input_state).unwrap();

            let expected = 16.0 * 0.4 + 14.0 * 0.1 + 12.0 * 0.4 + 10.0 * 0.1;
            let value = output.get("Temperature|_to_scalar").unwrap();
            if let StateValue::Scalar(v) = value {
                assert!(
                    (v - expected).abs() < 1e-10,
                    "Expected {}, got {}",
                    expected,
                    v
                );
            } else {
                panic!("Expected scalar output, got {:?}", value);
            }
        }

        #[test]
        fn test_grid_transformer_solve_fourbox_to_hemispheric() {
            let transformer = GridTransformerComponent::new(
                "Temperature",
                "K",
                GridType::FourBox,
                GridType::Hemispheric,
            )
            .unwrap();

            // Values: NO=16.0, NL=14.0, SO=12.0, SL=10.0
            // With equal weights: NH = (16+14)/2 = 15.0, SH = (12+10)/2 = 11.0
            let item = create_four_box_item("Temperature", [16.0, 14.0, 12.0, 10.0]);
            let input_state = InputState::build(vec![&item], 2000.0);

            let output = transformer.solve(2000.0, 2001.0, &input_state).unwrap();

            let value = output.get("Temperature|_to_hemispheric").unwrap();
            if let StateValue::Hemispheric(slice) = value {
                assert!(
                    (slice.0[0] - 15.0).abs() < 1e-10,
                    "NH expected 15.0, got {}",
                    slice.0[0]
                );
                assert!(
                    (slice.0[1] - 11.0).abs() < 1e-10,
                    "SH expected 11.0, got {}",
                    slice.0[1]
                );
            } else {
                panic!("Expected hemispheric output, got {:?}", value);
            }
        }

        #[test]
        fn test_grid_transformer_solve_hemispheric_to_scalar() {
            let transformer = GridTransformerComponent::new(
                "Temperature",
                "K",
                GridType::Hemispheric,
                GridType::Scalar,
            )
            .unwrap();

            // Values: NH=15.0, SH=11.0
            // With equal weights: global = (15+11)/2 = 13.0
            let item = create_hemispheric_item("Temperature", [15.0, 11.0]);
            let input_state = InputState::build(vec![&item], 2000.0);

            let output = transformer.solve(2000.0, 2001.0, &input_state).unwrap();

            assert_eq!(
                output.get("Temperature|_to_scalar"),
                Some(&StateValue::Scalar(13.0))
            );
        }

        #[test]
        fn test_grid_transformer_solve_hemispheric_to_scalar_with_custom_weights() {
            let transformer = GridTransformerComponent::new(
                "Temperature",
                "K",
                GridType::Hemispheric,
                GridType::Scalar,
            )
            .unwrap()
            .with_hemispheric_weights([0.7, 0.3]);

            // Values: NH=15.0, SH=11.0
            // Weighted: 15*0.7 + 11*0.3 = 10.5 + 3.3 = 13.8
            let item = create_hemispheric_item("Temperature", [15.0, 11.0]);
            let input_state = InputState::build(vec![&item], 2000.0);

            let output = transformer.solve(2000.0, 2001.0, &input_state).unwrap();

            let expected = 15.0 * 0.7 + 11.0 * 0.3;
            let value = output.get("Temperature|_to_scalar").unwrap();
            if let StateValue::Scalar(v) = value {
                assert!(
                    (v - expected).abs() < 1e-10,
                    "Expected {}, got {}",
                    expected,
                    v
                );
            } else {
                panic!("Expected scalar output, got {:?}", value);
            }
        }
    }
}
