//! Python bindings for point estimation.

use pyo3::prelude::*;

use crate::{
    optimiser::{OptimisationResult, Optimiser},
    point_estimator::PointEstimator,
};

use super::{
    into_py_err, likelihood::PyGaussianLikelihood, model_runner::PyModelRunner,
    parameter_set::PyParameterSet, target::PyTarget, warn_failures,
};

/// Optimisation algorithm for point estimation.
///
/// This enum specifies which optimisation algorithm to use when finding
/// the maximum likelihood or maximum a posteriori (MAP) parameter estimates.
#[pyclass(name = "Optimiser")]
#[derive(Clone, Copy)]
pub struct PyOptimiser {
    inner: Optimiser,
}

#[pymethods]
impl PyOptimiser {
    /// Random search baseline.
    ///
    /// Samples random parameter sets from the prior
    /// and returns the best. Useful as a baseline or for initialisation.
    #[staticmethod]
    fn random_search() -> Self {
        Self {
            inner: Optimiser::RandomSearch,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            Optimiser::RandomSearch => "Optimiser.random_search()".to_string(),
        }
    }
}

/// Result of a point estimation optimisation.
///
/// Contains the best parameters found, log likelihood, log posterior,
/// and optimisation diagnostics.
#[pyclass(name = "OptimisationResult")]
#[derive(Clone)]
pub struct PyOptimisationResult {
    inner: OptimisationResult,
}

#[pymethods]
impl PyOptimisationResult {
    /// Best parameter vector found.
    #[getter]
    fn best_parameters(&self) -> Vec<f64> {
        self.inner.best_parameters.clone()
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

    /// Whether the optimiser converged.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Number of evaluations rejected because the prior, model or likelihood failed.
    #[getter]
    fn n_failed_evaluations(&self) -> usize {
        self.inner.failures.count
    }

    /// Message of the first failed evaluation, or None if none failed.
    #[getter]
    fn first_failure(&self) -> Option<String> {
        self.inner.failures.first.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OptimisationResult(best_log_likelihood={:.4}, n_evaluations={}, converged={})",
            self.inner.best_log_likelihood, self.inner.n_evaluations, self.inner.converged
        )
    }
}

/// Point estimator for finding optimal parameter sets.
///
/// This class wraps the calibration components (parameters, model runner,
/// likelihood, target) and provides methods for point estimation via
/// optimisation algorithms.
///
/// The estimator tracks all evaluated parameter sets and their log likelihoods,
/// enabling analysis of the optimisation trajectory.
///
/// Raises ValueError if the runner's `parameter_names` do not name each parameter exactly once.
/// They may be in a different order: values are passed to the runner by name.
///
/// # Examples
///
/// ```python
/// from rscm._lib.calibrate import (
///     ParameterSet, Uniform, ModelRunner, GaussianLikelihood,
///     Target, PointEstimator, Optimiser
/// )
///
/// # Set up calibration components
/// parameters = ParameterSet()
/// parameters.add("x", Uniform(0.0, 10.0))
///
/// # ... create model_runner, target ...
///
/// # Create point estimator
/// likelihood = GaussianLikelihood()
/// estimator = PointEstimator(parameters, model_runner, likelihood, target)
///
/// # Run optimisation
/// result = estimator.optimise(Optimiser.random_search(), n_samples=1000)
/// print(f"Best parameters: {result.best_parameters}")
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
    /// * `parameters` - Parameter set defining priors and bounds
    /// * `runner` - Model runner for evaluating parameter sets
    /// * `likelihood` - Likelihood function
    /// * `target` - Target observations
    ///
    /// # Returns
    ///
    /// A new PointEstimator instance.
    #[new]
    fn new(
        parameters: PyParameterSet,
        runner: PyModelRunner,
        likelihood: PyGaussianLikelihood,
        target: PyTarget,
    ) -> PyResult<Self> {
        let inner = PointEstimator::new(
            parameters.into_inner(),
            runner,
            likelihood,
            target.into_inner(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get the number of parameters.
    #[getter]
    fn n_parameters(&self) -> usize {
        self.inner.n_parameters()
    }

    /// Get the parameter names.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner
            .parameter_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the number of evaluations performed.
    #[getter]
    fn n_evaluations(&self) -> usize {
        self.inner.n_evaluations()
    }

    /// Run point estimation optimisation.
    ///
    /// Finds the parameter set that maximises the log posterior (prior + likelihood).
    ///
    /// # Arguments
    ///
    /// * `optimiser` - Optimisation algorithm to use
    /// * `n_samples` - Number of samples to evaluate, at least 1
    /// * `seed` - Seed for sampling, so equal seeds give equal results
    ///
    /// # Returns
    ///
    /// An OptimisationResult with the best parameters and diagnostics.
    ///
    /// # Raises
    ///
    /// ValueError: If `n_samples` is 0, or optimisation fails or finds no valid samples.
    ///
    /// # Warns
    ///
    /// RuntimeWarning: If evaluations failed and were rejected.
    ///
    /// # Examples
    ///
    /// ```python
    /// result = estimator.optimise(Optimiser.random_search(), n_samples=1000)
    /// print(f"Best: {result.best_parameters}")
    /// print(f"Log likelihood: {result.best_log_likelihood}")
    /// ```
    #[pyo3(signature = (optimiser, n_samples, *, seed=None))]
    fn optimise(
        &mut self,
        py: Python<'_>,
        optimiser: PyOptimiser,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<PyOptimisationResult> {
        self.inner.runner().take_interrupt();
        let result = self
            .inner
            .optimise(optimiser.inner, n_samples, seed)
            .map_err(|e| into_py_err(e, self.inner.runner()))?;
        warn_failures(py, &result.failures)?;

        Ok(PyOptimisationResult { inner: result })
    }

    /// Clear the evaluation history.
    ///
    /// This can be useful when running multiple optimisation attempts
    /// and you want to track each attempt separately.
    fn clear_history(&mut self) {
        self.inner.clear_history();
    }

    /// Get the best parameter set found so far.
    ///
    /// Returns a tuple of (best_parameters, best_log_likelihood), or None
    /// if no evaluations have been performed yet.
    ///
    /// # Returns
    ///
    /// * `(list[float], float)` - Best parameters and their log likelihood
    /// * `None` - If no evaluations performed yet
    fn best(&self) -> Option<(Vec<f64>, f64)> {
        self.inner
            .best()
            .map(|(parameters, ll)| (parameters.to_vec(), ll))
    }

    /// Get all evaluated parameter vectors.
    ///
    /// Returns the history of all parameter sets evaluated during optimisation.
    ///
    /// # Returns
    ///
    /// List of parameter vectors (each vector is a list of floats).
    fn evaluated_parameters(&self) -> Vec<Vec<f64>> {
        self.inner.evaluated_parameters().to_vec()
    }

    /// Get all evaluated log likelihood values.
    ///
    /// Returns the history of log likelihoods corresponding to evaluated_parameters().
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
            "PointEstimator(n_parameters={}, n_evaluations={})",
            self.inner.n_parameters(),
            self.inner.n_evaluations()
        )
    }
}
