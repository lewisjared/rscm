//! Python bindings for the calibration framework.
//!
//! This module provides PyO3 bindings for:
//! - Probability distributions for parameter priors
//! - Parameter set specification and sampling
//! - Target observations and likelihood computation
//! - MCMC ensemble sampling
//! - Point estimation via optimization
//! - Convergence diagnostics

use pyo3::prelude::*;

pub mod chain;
pub mod distribution;
pub mod likelihood;
pub mod model_runner;
pub mod parameter_set;
pub mod point_estimator;
pub mod sampler;
pub mod target;

pub use chain::PyChain;
pub use distribution::{PyBound, PyDistribution, PyLogNormal, PyNormal, PyUniform};
pub use likelihood::PyGaussianLikelihood;
pub use model_runner::PyModelRunner;
pub use parameter_set::PyParameterSet;
pub use point_estimator::{PyOptimizationResult, PyOptimizer, PyPointEstimator};
pub use sampler::{PyEnsembleSampler, PyProgressInfo, PyWalkerInit};
pub use target::{PyObservation, PyTarget, PyVariableTarget};

/// Register the calibrate module with Python.
#[pymodule]
pub fn calibrate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Distribution types
    m.add_class::<PyUniform>()?;
    m.add_class::<PyNormal>()?;
    m.add_class::<PyLogNormal>()?;
    m.add_class::<PyBound>()?;

    // Parameter set
    m.add_class::<PyParameterSet>()?;

    // Target and observations
    m.add_class::<PyObservation>()?;
    m.add_class::<PyVariableTarget>()?;
    m.add_class::<PyTarget>()?;

    // Chain for MCMC results
    m.add_class::<PyChain>()?;

    // Model runner
    m.add_class::<PyModelRunner>()?;

    // Likelihood
    m.add_class::<PyGaussianLikelihood>()?;

    // Ensemble sampler
    m.add_class::<PyEnsembleSampler>()?;
    m.add_class::<PyWalkerInit>()?;
    m.add_class::<PyProgressInfo>()?;

    // Point estimator
    m.add_class::<PyPointEstimator>()?;
    m.add_class::<PyOptimizer>()?;
    m.add_class::<PyOptimizationResult>()?;

    Ok(())
}
