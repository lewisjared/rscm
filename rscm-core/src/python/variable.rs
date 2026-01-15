//! Python bindings for the variable registration system.
//!
//! This module provides Python access to:
//! - [`TimeConvention`] enum for temporal alignment
//! - [`PyVariableDefinition`] for variable metadata
//! - [`PyPreindustrialValue`] for preindustrial reference values
//! - Registry functions for registering and looking up variables

use crate::variable::{PreindustrialValue, TimeConvention, VariableDefinition, VARIABLE_REGISTRY};
use pyo3::prelude::*;

/// Python wrapper for TimeConvention enum.
///
/// Specifies when during a time period a variable's value is valid.
#[pyclass(name = "TimeConvention", module = "rscm.core")]
#[derive(Clone)]
pub struct PyTimeConvention(pub TimeConvention);

#[pymethods]
impl PyTimeConvention {
    /// Create a StartOfYear time convention.
    ///
    /// Value applies at the start of the year (Jan 1).
    /// Used for stock variables like concentrations and temperatures.
    #[staticmethod]
    pub fn start_of_year() -> Self {
        PyTimeConvention(TimeConvention::StartOfYear)
    }

    /// Create a MidYear time convention.
    ///
    /// Value applies at mid-year (Jul 1).
    /// Used for flow variables like emissions.
    #[staticmethod]
    pub fn mid_year() -> Self {
        PyTimeConvention(TimeConvention::MidYear)
    }

    /// Create an Instantaneous time convention.
    ///
    /// Instantaneous value with no temporal averaging.
    /// Used for derived quantities.
    #[staticmethod]
    pub fn instantaneous() -> Self {
        PyTimeConvention(TimeConvention::Instantaneous)
    }

    fn __repr__(&self) -> String {
        format!("TimeConvention.{}", self.0)
    }

    fn __eq__(&self, other: &PyTimeConvention) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

/// Python wrapper for VariableDefinition.
///
/// Contains metadata about a variable including name, unit, time convention, and description.
#[pyclass(name = "VariableDefinition", module = "rscm.core")]
#[derive(Clone)]
pub struct PyVariableDefinition(pub VariableDefinition);

#[pymethods]
impl PyVariableDefinition {
    /// Create a new variable definition.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for the variable (e.g., "Atmospheric Concentration|CO2")
    /// * `unit` - Canonical unit for the variable (e.g., "ppm", "GtC / yr")
    /// * `time_convention` - Time convention for temporal alignment
    /// * `description` - Human-readable description
    #[new]
    pub fn new(
        name: String,
        unit: String,
        time_convention: PyTimeConvention,
        description: String,
    ) -> Self {
        PyVariableDefinition(VariableDefinition::new(
            name,
            unit,
            time_convention.0,
            description,
        ))
    }

    /// Get the variable name.
    #[getter]
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// Get the variable unit.
    #[getter]
    pub fn unit(&self) -> &str {
        &self.0.unit
    }

    /// Get the time convention.
    #[getter]
    pub fn time_convention(&self) -> PyTimeConvention {
        PyTimeConvention(self.0.time_convention)
    }

    /// Get the description.
    #[getter]
    pub fn description(&self) -> &str {
        &self.0.description
    }

    fn __repr__(&self) -> String {
        format!(
            "VariableDefinition(name='{}', unit='{}', time_convention={:?}, description='{}')",
            self.0.name, self.0.unit, self.0.time_convention, self.0.description
        )
    }
}

/// Python wrapper for PreindustrialValue.
///
/// Preindustrial reference values can be scalar, four-box regional, or hemispheric.
#[pyclass(name = "PreindustrialValue", module = "rscm.core")]
#[derive(Clone)]
pub struct PyPreindustrialValue(pub PreindustrialValue);

#[pymethods]
impl PyPreindustrialValue {
    /// Create a scalar preindustrial value.
    #[staticmethod]
    pub fn scalar(value: f64) -> Self {
        PyPreindustrialValue(PreindustrialValue::Scalar(value))
    }

    /// Create a four-box regional preindustrial value.
    ///
    /// # Arguments
    ///
    /// * `values` - Array of 4 values [NorthernOcean, NorthernLand, SouthernOcean, SouthernLand]
    #[staticmethod]
    pub fn four_box(values: [f64; 4]) -> Self {
        PyPreindustrialValue(PreindustrialValue::FourBox(values))
    }

    /// Create a hemispheric preindustrial value.
    ///
    /// # Arguments
    ///
    /// * `values` - Array of 2 values [Northern, Southern]
    #[staticmethod]
    pub fn hemispheric(values: [f64; 2]) -> Self {
        PyPreindustrialValue(PreindustrialValue::Hemispheric(values))
    }

    /// Get the value as a global scalar.
    ///
    /// For FourBox and Hemispheric variants, uses area-weighted averaging.
    pub fn to_scalar(&self) -> f64 {
        self.0.to_scalar()
    }

    /// Check if this is a scalar value.
    pub fn is_scalar(&self) -> bool {
        matches!(self.0, PreindustrialValue::Scalar(_))
    }

    /// Check if this is a four-box value.
    pub fn is_four_box(&self) -> bool {
        matches!(self.0, PreindustrialValue::FourBox(_))
    }

    /// Check if this is a hemispheric value.
    pub fn is_hemispheric(&self) -> bool {
        matches!(self.0, PreindustrialValue::Hemispheric(_))
    }

    /// Get the scalar value if this is a Scalar variant.
    pub fn as_scalar(&self) -> Option<f64> {
        self.0.as_scalar()
    }

    /// Get the four-box values if this is a FourBox variant.
    pub fn as_four_box(&self) -> Option<[f64; 4]> {
        self.0.as_four_box()
    }

    /// Get the hemispheric values if this is a Hemispheric variant.
    pub fn as_hemispheric(&self) -> Option<[f64; 2]> {
        self.0.as_hemispheric()
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            PreindustrialValue::Scalar(v) => format!("PreindustrialValue.scalar({})", v),
            PreindustrialValue::FourBox(v) => format!("PreindustrialValue.four_box({:?})", v),
            PreindustrialValue::Hemispheric(v) => {
                format!("PreindustrialValue.hemispheric({:?})", v)
            }
        }
    }
}

/// Register a variable definition at runtime.
///
/// Variables registered via Python are stored in the global registry and can be
/// looked up by name from both Rust and Python code.
///
/// # Arguments
///
/// * `var` - The variable definition to register
///
/// # Returns
///
/// Returns an error if a variable with the same name already exists.
///
/// # Example
///
/// ```python
/// from rscm.core import (
///     VariableDefinition,
///     TimeConvention,
///     register_variable,
/// )
///
/// co2_conc = VariableDefinition(
///     name="Atmospheric Concentration|CO2",
///     unit="ppm",
///     time_convention=TimeConvention.start_of_year(),
///     description="Atmospheric CO2 concentration",
/// )
/// register_variable(co2_conc)
/// ```
#[pyfunction]
pub fn register_variable(var: PyVariableDefinition) -> PyResult<()> {
    VARIABLE_REGISTRY
        .register(var.0)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Get a variable definition by name from the registry.
///
/// Searches both static (Rust-defined) and runtime (Python-defined) variables.
///
/// # Arguments
///
/// * `name` - The variable name to look up
///
/// # Returns
///
/// The variable definition if found, otherwise None.
///
/// # Example
///
/// ```python
/// from rscm.core import get_variable
///
/// var = get_variable("Atmospheric Concentration|CO2")
/// if var:
///     print(f"Unit: {var.unit}")
/// ```
#[pyfunction]
pub fn get_variable(name: &str) -> Option<PyVariableDefinition> {
    VARIABLE_REGISTRY
        .get_with_static(name)
        .map(PyVariableDefinition)
}

/// List all registered variables.
///
/// Returns both static (Rust-defined) and runtime (Python-defined) variables,
/// sorted by name.
///
/// # Returns
///
/// A list of all variable definitions.
///
/// # Example
///
/// ```python
/// from rscm.core import list_variables
///
/// for var in list_variables():
///     print(f"{var.name}: {var.unit}")
/// ```
#[pyfunction]
pub fn list_variables() -> Vec<PyVariableDefinition> {
    VARIABLE_REGISTRY
        .list_all()
        .into_iter()
        .map(PyVariableDefinition)
        .collect()
}

/// Check if a variable is registered.
///
/// # Arguments
///
/// * `name` - The variable name to check
///
/// # Returns
///
/// True if the variable is registered, False otherwise.
#[pyfunction]
pub fn is_variable_registered(name: &str) -> bool {
    VARIABLE_REGISTRY.is_registered_any(name)
}
