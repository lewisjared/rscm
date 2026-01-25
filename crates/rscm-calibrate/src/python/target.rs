//! Python bindings for Target and related types.

use crate::target::{Observation, Target, VariableTarget};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python wrapper for Observation.
///
/// A single observational data point with uncertainty.
///
/// Parameters
/// ----------
/// time : float
///     Time coordinate (typically decimal year)
/// value : float
///     Observed value
/// uncertainty : float
///     Uncertainty (1-sigma standard deviation, must be positive)
///
/// Raises
/// ------
/// ValueError
///     If uncertainty is not positive
#[pyclass(name = "Observation")]
#[derive(Clone)]
pub struct PyObservation(pub Observation);

#[pymethods]
impl PyObservation {
    #[new]
    fn new(time: f64, value: f64, uncertainty: f64) -> PyResult<Self> {
        Ok(Self(
            Observation::new(time, value, uncertainty)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        ))
    }

    #[getter]
    fn time(&self) -> f64 {
        self.0.time
    }

    #[getter]
    fn value(&self) -> f64 {
        self.0.value
    }

    #[getter]
    fn uncertainty(&self) -> f64 {
        self.0.uncertainty
    }

    fn __repr__(&self) -> String {
        format!(
            "Observation(time={}, value={}, uncertainty={})",
            self.0.time, self.0.value, self.0.uncertainty
        )
    }
}

/// Python wrapper for VariableTarget.
///
/// Target observations for a single variable.
///
/// Parameters
/// ----------
/// name : str
///     Variable name (e.g., "Temperature|Global")
#[pyclass(name = "VariableTarget")]
pub struct PyVariableTarget(pub VariableTarget);

#[pymethods]
impl PyVariableTarget {
    #[new]
    fn new(name: String) -> Self {
        Self(VariableTarget::new(name))
    }

    #[getter]
    fn name(&self) -> String {
        self.0.name.clone()
    }

    #[getter]
    fn observations(&self) -> Vec<PyObservation> {
        self.0
            .observations
            .iter()
            .map(|obs| PyObservation(obs.clone()))
            .collect()
    }

    #[getter]
    fn reference_period(&self) -> Option<(f64, f64)> {
        self.0
            .reference_period
            .as_ref()
            .map(|r| (*r.start(), *r.end()))
    }

    /// Add an observation.
    ///
    /// Parameters
    /// ----------
    /// time : float
    ///     Time coordinate
    /// value : float
    ///     Observed value
    /// uncertainty : float
    ///     1-sigma uncertainty (must be positive)
    ///
    /// Returns
    /// -------
    /// VariableTarget
    ///     Self for method chaining
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If uncertainty is not positive
    fn add(
        mut slf: PyRefMut<'_, Self>,
        time: f64,
        value: f64,
        uncertainty: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.0
            .add(time, value, uncertainty)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(slf)
    }

    /// Add an observation with relative uncertainty.
    ///
    /// Parameters
    /// ----------
    /// time : float
    ///     Time coordinate
    /// value : float
    ///     Observed value
    /// relative_uncertainty : float
    ///     Relative uncertainty as a fraction (e.g., 0.05 for 5%)
    ///
    /// Returns
    /// -------
    /// VariableTarget
    ///     Self for method chaining
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If computed uncertainty is not positive
    fn add_relative(
        mut slf: PyRefMut<'_, Self>,
        time: f64,
        value: f64,
        relative_uncertainty: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.0
            .add_relative(time, value, relative_uncertainty)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(slf)
    }

    /// Set the reference period for anomaly calculation.
    ///
    /// When set, observations will be interpreted as anomalies relative to the
    /// mean over this period.
    ///
    /// Parameters
    /// ----------
    /// start : float
    ///     Start of reference period
    /// end : float
    ///     End of reference period
    ///
    /// Returns
    /// -------
    /// VariableTarget
    ///     Self for method chaining
    fn with_reference_period(
        mut slf: PyRefMut<'_, Self>,
        start: f64,
        end: f64,
    ) -> PyRefMut<'_, Self> {
        slf.0.with_reference_period(start, end);
        slf
    }

    /// Get the time range covered by observations.
    ///
    /// Returns
    /// -------
    /// tuple[float, float] or None
    ///     (min_time, max_time) or None if no observations
    fn time_range(&self) -> Option<(f64, f64)> {
        self.0.time_range()
    }

    fn __repr__(&self) -> String {
        format!(
            "VariableTarget(name='{}', n_observations={})",
            self.0.name,
            self.0.observations.len()
        )
    }
}

/// Python wrapper for Target.
///
/// Collection of target observations for multiple variables.
///
/// Examples
/// --------
/// >>> target = Target()
/// >>> target.add_observation("Temperature|Global", 2020.0, 1.2, 0.1)
/// >>> target.add_observation("Temperature|Global", 2021.0, 1.3, 0.1)
/// >>> target.set_reference_period("Temperature|Global", 1850.0, 1900.0)
/// >>> target.add_observation_relative("OHC", 2020.0, 200.0, 0.05)
#[pyclass(name = "Target")]
#[derive(Clone)]
pub struct PyTarget(pub Target);

#[pymethods]
impl PyTarget {
    #[new]
    fn new() -> Self {
        Self(Target::new())
    }

    /// Add an observation to a variable.
    ///
    /// If the variable doesn't exist, it will be created.
    ///
    /// Parameters
    /// ----------
    /// variable : str
    ///     Variable name (e.g., "Temperature|Global")
    /// time : float
    ///     Time coordinate
    /// value : float
    ///     Observed value
    /// uncertainty : float
    ///     1-sigma uncertainty (must be positive)
    ///
    /// Returns
    /// -------
    /// Target
    ///     Self for method chaining
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If uncertainty is not positive
    fn add_observation(
        mut slf: PyRefMut<'_, Self>,
        variable: String,
        time: f64,
        value: f64,
        uncertainty: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.0
            .add_variable(variable)
            .add(time, value, uncertainty)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(slf)
    }

    /// Add an observation with relative uncertainty.
    ///
    /// If the variable doesn't exist, it will be created.
    ///
    /// Parameters
    /// ----------
    /// variable : str
    ///     Variable name
    /// time : float
    ///     Time coordinate
    /// value : float
    ///     Observed value
    /// relative_uncertainty : float
    ///     Relative uncertainty as a fraction (e.g., 0.05 for 5%)
    ///
    /// Returns
    /// -------
    /// Target
    ///     Self for method chaining
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If computed uncertainty is not positive
    fn add_observation_relative(
        mut slf: PyRefMut<'_, Self>,
        variable: String,
        time: f64,
        value: f64,
        relative_uncertainty: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.0
            .add_variable(variable)
            .add_relative(time, value, relative_uncertainty)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(slf)
    }

    /// Set the reference period for a variable.
    ///
    /// When set, observations will be interpreted as anomalies relative to the
    /// mean over this period.
    ///
    /// Parameters
    /// ----------
    /// variable : str
    ///     Variable name
    /// start : float
    ///     Start of reference period
    /// end : float
    ///     End of reference period
    ///
    /// Returns
    /// -------
    /// Target
    ///     Self for method chaining
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If variable doesn't exist
    fn set_reference_period(
        mut slf: PyRefMut<'_, Self>,
        variable: String,
        start: f64,
        end: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.0
            .get_variable_mut(&variable)
            .ok_or_else(|| PyValueError::new_err(format!("Variable '{}' not found", variable)))?
            .with_reference_period(start, end);
        Ok(slf)
    }

    /// Get a variable by name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Variable name
    ///
    /// Returns
    /// -------
    /// VariableTarget or None
    ///     The variable if it exists, None otherwise
    fn get_variable(&self, name: &str) -> Option<PyVariableTarget> {
        self.0
            .get_variable(name)
            .map(|v| PyVariableTarget(v.clone()))
    }

    /// Get the names of all variables.
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     Variable names in insertion order
    fn variable_names(&self) -> Vec<String> {
        self.0
            .variable_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the total number of observations across all variables.
    ///
    /// Returns
    /// -------
    /// int
    ///     Total observation count
    fn total_observations(&self) -> usize {
        self.0.total_observations()
    }

    /// Get the time range covered by all observations.
    ///
    /// Returns
    /// -------
    /// tuple[float, float] or None
    ///     (min_time, max_time) or None if no observations
    fn time_range(&self) -> Option<(f64, f64)> {
        self.0.time_range()
    }

    fn __repr__(&self) -> String {
        format!(
            "Target(n_variables={}, n_observations={})",
            self.0.variables().len(),
            self.0.total_observations()
        )
    }
}

impl PyTarget {
    /// Consume the wrapper and return the inner Target.
    pub fn into_inner(self) -> Target {
        self.0
    }
}
