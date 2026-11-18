//! Python bindings for the calibration framework.
//!
//! This module provides PyO3 bindings for:
//! - Probability distributions for parameter priors
//! - Parameter set specification and sampling
//! - Target observations and likelihood computation
//! - MCMC ensemble sampling
//! - Point estimation via optimisation
//! - Convergence diagnostics

use pyo3::exceptions::{PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;

use crate::{Error, Failures};

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
pub use point_estimator::{PyOptimisationResult, PyOptimiser, PyPointEstimator};
pub use sampler::{PyEnsembleSampler, PyProgressInfo, PyWalkerInit};
pub use target::{PyObservation, PyTarget, PyVariableTarget};

/// Convert a calibration error, re-raising the runner's KeyboardInterrupt or SystemExit unchanged.
pub(crate) fn into_py_err(error: Error, runner: &PyModelRunner) -> PyErr {
    runner
        .take_interrupt()
        .unwrap_or_else(|| PyValueError::new_err(error.to_string()))
}

/// Warn when evaluations failed and were rejected rather than stopping the run.
pub(crate) fn warn_failures(py: Python<'_>, failures: &Failures) -> PyResult<()> {
    let Some(first) = &failures.first else {
        return Ok(());
    };
    let message = format!(
        "{} evaluations failed and were rejected. First failure: {first}",
        failures.count
    );
    warn::<PyRuntimeWarning>(py, &message)
}

/// Emit a Python warning of category `W` at the caller's stack level.
pub(crate) fn warn<W: pyo3::PyTypeInfo>(py: Python<'_>, message: &str) -> PyResult<()> {
    let message = std::ffi::CString::new(message.replace('\0', "")).unwrap_or_default();
    PyErr::warn(py, &py.get_type::<W>(), &message, 1)
}

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
    m.add_class::<PyOptimiser>()?;
    m.add_class::<PyOptimisationResult>()?;

    Ok(())
}
