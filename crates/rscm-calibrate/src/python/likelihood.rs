//! Python bindings for likelihood functions.

use crate::likelihood::{GaussianLikelihood, LikelihoodFn, ModelOutput};
use crate::target::Target;
use crate::Result;
use pyo3::prelude::*;

/// Python wrapper for GaussianLikelihood.
///
/// Assumes independent Gaussian errors with known uncertainties.
///
/// For each observation, computes:
/// ```text
/// ln L = -0.5 * Σ[(y_obs - y_model)² / σ²]
/// ```
///
/// # Example
///
/// ```python
/// # Create likelihood function (default: no normalisation for MCMC)
/// likelihood = GaussianLikelihood()
///
/// # Create with normalisation constant (for MLE)
/// likelihood_normalised = GaussianLikelihood(normalise=True)
/// ```
#[pyclass(name = "GaussianLikelihood")]
#[derive(Clone)]
pub struct PyGaussianLikelihood(pub GaussianLikelihood);

#[pymethods]
impl PyGaussianLikelihood {
    /// Create a new Gaussian likelihood function.
    ///
    /// Parameters
    /// ----------
    /// normalise : bool, optional
    ///     Whether to include normalisation constant (default: False for MCMC).
    ///     Set to True for maximum likelihood estimation or model comparison.
    ///
    /// Returns
    /// -------
    /// GaussianLikelihood
    ///     A new Gaussian likelihood function
    ///
    /// Example
    /// -------
    /// ```python
    /// # For MCMC (normalisation not needed)
    /// likelihood = GaussianLikelihood()
    ///
    /// # For MLE or model comparison
    /// likelihood = GaussianLikelihood(normalise=True)
    /// ```
    #[new]
    #[pyo3(signature = (normalise=false))]
    fn new(normalise: bool) -> Self {
        if normalise {
            PyGaussianLikelihood(GaussianLikelihood::with_normalisation())
        } else {
            PyGaussianLikelihood(GaussianLikelihood::new())
        }
    }

    fn __repr__(&self) -> String {
        format!("GaussianLikelihood(normalise={})", self.0.normalise)
    }
}

// Implement LikelihoodFn trait for PyGaussianLikelihood
// This allows it to be used with EnsembleSampler
impl LikelihoodFn for PyGaussianLikelihood {
    fn log_likelihood(&self, model_output: &ModelOutput, target: &Target) -> Result<f64> {
        self.0.log_likelihood(model_output, target)
    }
}

// Note: The log_likelihood method is not exposed to Python directly
// Instead, it will be called internally by EnsembleSampler
// If direct access is needed, we can add a method that takes PyModelOutput and PyTarget
