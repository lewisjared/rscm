//! Python bindings for point estimation.

use pyo3::prelude::*;

use crate::{
    optimizer::{OptimizationResult, Optimizer},
    point_estimator::PointEstimator,
};

use super::{
    likelihood::PyGaussianLikelihood, model_runner::PyModelRunner, parameter_set::PyParameterSet,
    target::PyTarget,
};

/// Optimization algorithm for point estimation.
///
/// This enum specifies which optimization algorithm to use when finding
/// the maximum likelihood or maximum a posteriori (MAP) parameter estimates.
#[pyclass(name = "Optimizer")]
#[derive(Clone, Copy)]
pub struct PyOptimizer {
    inner: Optimizer,
}

#[pymethods]
impl PyOptimizer {
    /// Random search baseline.
    ///
    /// Samples random parameter sets from within the parameter bounds
    /// and returns the best. Useful as a baseline or for initialization.
    #[staticmethod]
    fn random_search() -> Self {
        Self {
            inner: Optimizer::RandomSearch,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            Optimizer::RandomSearch => "Optimizer.random_search()".to_string(),
        }
    }
}

/// Result of a point estimation optimization.
///
/// Contains the best parameters found, log likelihood, log posterior,
/// and optimization diagnostics.
#[pyclass(name = "OptimizationResult")]
#[derive(Clone)]
pub struct PyOptimizationResult {
    inner: OptimizationResult,
}

#[pymethods]
impl PyOptimizationResult {
    /// Best parameter vector found.
    #[getter]
    fn best_params(&self) -> Vec<f64> {
        self.inner.best_params.clone()
    }

    /// Log likelihood at best parameters (not including prior).
    #[getter]
    fn best_log_likelihood(&self) -> f64 {
        self.inner.best_log_likelihood
    }

    /// Log posterior at best parameters (prior + likelihood).
    #[getter]
    fn best_log_posterior(&self) -> f64 {
        self.inner.best_log_posterior
    }

    /// Number of function evaluations performed.
    #[getter]
    fn n_evaluations(&self) -> usize {
        self.inner.n_evaluations
    }

    /// Whether the optimizer converged.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    fn __repr__(&self) -> String {
        format!(
            "OptimizationResult(best_log_likelihood={:.4}, n_evaluations={}, converged={})",
            self.inner.best_log_likelihood, self.inner.n_evaluations, self.inner.converged
        )
    }
}

/// Point estimator for finding optimal parameter sets.
///
/// This class wraps the calibration components (parameters, model runner,
/// likelihood, target) and provides methods for point estimation via
/// optimization algorithms.
///
/// The estimator tracks all evaluated parameter sets and their log likelihoods,
/// enabling analysis of the optimization trajectory.
///
/// # Examples
///
/// ```python
/// from rscm._lib.calibrate import (
///     ParameterSet, Uniform, ModelRunner, GaussianLikelihood,
///     Target, PointEstimator, Optimizer
/// )
///
/// # Set up calibration components
/// params = ParameterSet()
/// params.add("x", Uniform(0.0, 10.0))
///
/// # ... create model_runner, target ...
///
/// # Create point estimator
/// likelihood = GaussianLikelihood()
/// estimator = PointEstimator(params, model_runner, likelihood, target)
///
/// # Run optimization
/// result = estimator.optimize(Optimizer.random_search(), n_samples=1000)
/// print(f"Best parameters: {result.best_params}")
/// print(f"Log likelihood: {result.best_log_likelihood}")
/// ```
#[pyclass(name = "PointEstimator")]
pub struct PyPointEstimator {
    inner: PointEstimator<PyModelRunner, PyGaussianLikelihood>,
}

#[pymethods]
impl PyPointEstimator {
    /// Create a new point estimator.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameter set defining priors and bounds
    /// * `runner` - Model runner for evaluating parameter sets
    /// * `likelihood` - Likelihood function
    /// * `target` - Target observations
    ///
    /// # Returns
    ///
    /// A new PointEstimator instance.
    #[new]
    fn new(
        params: PyParameterSet,
        runner: PyModelRunner,
        likelihood: PyGaussianLikelihood,
        target: PyTarget,
    ) -> Self {
        Self {
            inner: PointEstimator::new(
                params.into_inner(),
                runner,
                likelihood,
                target.into_inner(),
            ),
        }
    }

    /// Get the number of parameters.
    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// Get the parameter names.
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.inner
            .param_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the number of evaluations performed.
    #[getter]
    fn n_evaluations(&self) -> usize {
        self.inner.n_evaluations()
    }

    /// Run point estimation optimization.
    ///
    /// Finds the parameter set that maximizes the log posterior (prior + likelihood).
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Optimization algorithm to use
    /// * `n_samples` - Number of samples to evaluate (interpretation depends on algorithm)
    ///
    /// # Returns
    ///
    /// An OptimizationResult with the best parameters and diagnostics.
    ///
    /// # Raises
    ///
    /// ValueError: If optimization fails or finds no valid samples.
    ///
    /// # Examples
    ///
    /// ```python
    /// result = estimator.optimize(Optimizer.random_search(), n_samples=1000)
    /// print(f"Best: {result.best_params}")
    /// print(f"Log likelihood: {result.best_log_likelihood}")
    /// ```
    fn optimize(
        &mut self,
        optimizer: PyOptimizer,
        n_samples: usize,
    ) -> PyResult<PyOptimizationResult> {
        let result = self
            .inner
            .optimize(optimizer.inner, n_samples)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(PyOptimizationResult { inner: result })
    }

    /// Clear the evaluation history.
    ///
    /// This can be useful when running multiple optimization attempts
    /// and you want to track each attempt separately.
    fn clear_history(&mut self) {
        self.inner.clear_history();
    }

    /// Get the best parameter set found so far.
    ///
    /// Returns a tuple of (best_params, best_log_likelihood), or None
    /// if no evaluations have been performed yet.
    ///
    /// # Returns
    ///
    /// * `(list[float], float)` - Best parameters and their log likelihood
    /// * `None` - If no evaluations performed yet
    fn best(&self) -> Option<(Vec<f64>, f64)> {
        self.inner.best().map(|(params, ll)| (params.to_vec(), ll))
    }

    /// Get all evaluated parameter vectors.
    ///
    /// Returns the history of all parameter sets evaluated during optimization.
    ///
    /// # Returns
    ///
    /// List of parameter vectors (each vector is a list of floats).
    fn evaluated_params(&self) -> Vec<Vec<f64>> {
        self.inner.evaluated_params().to_vec()
    }

    /// Get all evaluated log likelihood values.
    ///
    /// Returns the history of log likelihoods corresponding to evaluated_params().
    /// Note: These are log likelihoods, not log posteriors (prior not included).
    ///
    /// # Returns
    ///
    /// List of log likelihood values.
    fn evaluated_log_likelihoods(&self) -> Vec<f64> {
        self.inner.evaluated_log_likelihoods().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "PointEstimator(n_params={}, n_evaluations={})",
            self.inner.n_params(),
            self.inner.n_evaluations()
        )
    }
}
