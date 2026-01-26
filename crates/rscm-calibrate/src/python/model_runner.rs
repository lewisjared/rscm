//! Python bindings for ModelRunner.
//!
//! This module provides a Python-callable model runner that wraps a Python function.

use crate::likelihood::{ModelOutput, VariableOutput};
use crate::model_runner::ModelRunner;
use crate::{Error, Result};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};

/// Python wrapper for ModelRunner trait.
///
/// This class allows Python code to provide a model factory function that will be
/// called during calibration. The factory receives a dict of parameter values and
/// should return a dict of variable outputs.
///
/// # Example
///
/// ```python
/// def model_factory(params):
///     # params is a dict like {"climate_sensitivity": 3.0, "ocean_diff": 1200.0}
///     # Build and run model...
///     return {
///         "Temperature|Global": {2020.0: 1.2, 2021.0: 1.3, ...},
///         "OHC": {2020.0: 100.0, 2021.0: 105.0, ...}
///     }
///
/// runner = ModelRunner(
///     model_factory=model_factory,
///     param_names=["climate_sensitivity", "ocean_diff"],
///     output_variables=["Temperature|Global", "OHC"]
/// )
/// ```
#[pyclass(name = "ModelRunner")]
#[derive(Clone)]
pub struct PyModelRunner {
    /// Python callable that creates and runs models
    factory: Arc<Mutex<Py<PyAny>>>,
    /// Parameter names in indexed order
    param_names: Vec<String>,
    /// Output variable names to extract
    output_variables: Vec<String>,
}

#[pymethods]
impl PyModelRunner {
    /// Create a new model runner.
    ///
    /// Parameters
    /// ----------
    /// model_factory : callable
    ///     Python function that takes a dict of parameters and returns a dict of outputs.
    ///     Signature: `model_factory(params: dict[str, float]) -> dict[str, dict[float, float]]`
    ///     The outer dict keys are variable names, inner dict keys are times.
    /// param_names : list[str]
    ///     Names of parameters in the order they appear in parameter vectors
    /// output_variables : list[str]
    ///     Names of variables to extract from model output
    ///
    /// Returns
    /// -------
    /// ModelRunner
    ///     A new model runner instance
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If model_factory is not callable
    ///
    /// Example
    /// -------
    /// ```python
    /// def my_model(params):
    ///     temp = params["sensitivity"] * params["forcing"]
    ///     return {"Temperature": {2020.0: temp}}
    ///
    /// runner = ModelRunner(
    ///     model_factory=my_model,
    ///     param_names=["sensitivity", "forcing"],
    ///     output_variables=["Temperature"]
    /// )
    /// ```
    #[new]
    fn new(
        py: Python<'_>,
        model_factory: Py<PyAny>,
        param_names: Vec<String>,
        output_variables: Vec<String>,
    ) -> PyResult<Self> {
        // Verify callable
        if !model_factory.bind(py).is_callable() {
            return Err(PyValueError::new_err("model_factory must be callable"));
        }

        Ok(Self {
            factory: Arc::new(Mutex::new(model_factory)),
            param_names,
            output_variables,
        })
    }

    /// Get parameter names.
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     Names of parameters in indexed order
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.param_names.clone()
    }

    /// Get output variable names.
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     Names of variables to extract from model output
    #[getter]
    fn output_variables(&self) -> Vec<String> {
        self.output_variables.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelRunner(param_names={:?}, output_variables={:?})",
            self.param_names, self.output_variables
        )
    }
}

impl ModelRunner for PyModelRunner {
    fn param_names(&self) -> &[String] {
        &self.param_names
    }

    fn run(&self, params: &[f64]) -> Result<ModelOutput> {
        if params.len() != self.param_names.len() {
            return Err(Error::SamplingError(format!(
                "Expected {} parameters, got {}",
                self.param_names.len(),
                params.len()
            )));
        }

        // Call Python function with GIL
        Python::attach(|py| {
            // Build parameter dict
            let param_dict = PyDict::new(py);
            for (name, value) in self.param_names.iter().zip(params.iter()) {
                param_dict.set_item(name, value).map_err(|e| {
                    Error::SamplingError(format!("Failed to create param dict: {}", e))
                })?;
            }

            // Call factory
            let factory = self.factory.lock().unwrap();
            let result = factory
                .bind(py)
                .call1((param_dict,))
                .map_err(|e| Error::ModelError(format!("Model factory failed: {}", e)))?;

            // Parse result: dict[str, dict[float, float]]
            let result_dict = result
                .downcast::<PyDict>()
                .map_err(|_| Error::ModelError("Model factory must return a dict".to_string()))?;

            let mut output = ModelOutput::new();

            for (var_name, var_data) in result_dict.iter() {
                let var_name_str: String = var_name
                    .extract()
                    .map_err(|_| Error::ModelError("Variable names must be strings".to_string()))?;

                // Check if this is a requested output variable
                if !self.output_variables.contains(&var_name_str) {
                    continue;
                }

                let var_dict = var_data.downcast::<PyDict>().map_err(|_| {
                    Error::ModelError(format!("Variable '{}' data must be a dict", var_name_str))
                })?;

                let mut var_output = VariableOutput::new(var_name_str.clone());

                for (time, value) in var_dict.iter() {
                    let time_f64: f64 = time.extract().map_err(|_| {
                        Error::ModelError(format!(
                            "Time values must be floats for variable '{}'",
                            var_name_str
                        ))
                    })?;

                    let value_f64: f64 = value.extract().map_err(|_| {
                        Error::ModelError(format!(
                            "Output values must be floats for variable '{}'",
                            var_name_str
                        ))
                    })?;

                    if !value_f64.is_finite() {
                        return Err(Error::ModelError(format!(
                            "Model output contains NaN or infinity for variable '{}' at time {}",
                            var_name_str, time_f64
                        )));
                    }

                    var_output.add(time_f64, value_f64);
                }

                output.add_variable(var_output);
            }

            // Verify all requested variables are present
            for var_name in &self.output_variables {
                if !output.variables.contains_key(var_name) {
                    return Err(Error::ModelError(format!(
                        "Model output missing required variable: {}",
                        var_name
                    )));
                }
            }

            Ok(output)
        })
    }

    fn run_batch(&self, param_sets: &[Vec<f64>]) -> Vec<Result<ModelOutput>> {
        // Sequential execution for now
        // Python GIL prevents true parallelism with Python callables
        param_sets.iter().map(|params| self.run(params)).collect()
    }
}
