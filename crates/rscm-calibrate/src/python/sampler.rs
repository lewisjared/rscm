//! Python bindings for the ensemble sampler.

use crate::sampler::{EnsembleSampler, ProgressInfo, WalkerInit};
use ndarray::Array2;
use pyo3::prelude::*;

use super::chain::PyChain;
use super::likelihood::PyGaussianLikelihood;
use super::model_runner::PyModelRunner;
use super::parameter_set::PyParameterSet;
use super::target::PyTarget;

/// Walker initialization strategy.
///
/// Determines how walkers are positioned at the start of MCMC sampling.
/// Use the static methods to create instances.
#[pyclass(name = "WalkerInit")]
#[derive(Clone)]
pub struct PyWalkerInit {
    inner: WalkerInitInner,
}

#[derive(Clone)]
enum WalkerInitInner {
    FromPrior,
    Ball { center: Vec<f64>, radius: f64 },
    Explicit { positions: Vec<Vec<f64>> },
}

impl PyWalkerInit {
    /// Convert to Rust WalkerInit.
    fn to_rust(&self) -> WalkerInit {
        match &self.inner {
            WalkerInitInner::FromPrior => WalkerInit::FromPrior,
            WalkerInitInner::Ball { center, radius } => WalkerInit::Ball {
                center: center.clone(),
                radius: *radius,
            },
            WalkerInitInner::Explicit { positions } => {
                // Convert Vec<Vec<f64>> to Array2<f64>
                let n_walkers = positions.len();
                let n_params = positions.first().map_or(0, |row| row.len());
                let flat: Vec<f64> = positions
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect();
                let array = Array2::from_shape_vec((n_walkers, n_params), flat)
                    .expect("Invalid explicit positions shape");
                WalkerInit::Explicit(array)
            }
        }
    }
}

#[pymethods]
impl PyWalkerInit {
    /// Create a FromPrior initialization strategy.
    ///
    /// Walkers will be sampled from the prior distributions.
    #[staticmethod]
    fn from_prior() -> Self {
        PyWalkerInit {
            inner: WalkerInitInner::FromPrior,
        }
    }

    /// Create a Ball initialization strategy.
    ///
    /// Parameters
    /// ----------
    /// center : list[float]
    ///     Parameter values at the center of the ball
    /// radius : float
    ///     Radius of the ball (std deviation for each parameter)
    #[staticmethod]
    fn ball(center: Vec<f64>, radius: f64) -> Self {
        PyWalkerInit {
            inner: WalkerInitInner::Ball { center, radius },
        }
    }

    /// Create an Explicit initialization strategy.
    ///
    /// Parameters
    /// ----------
    /// positions : list[list[float]]
    ///     2D array of walker positions, shape (n_walkers, n_params)
    #[staticmethod]
    fn explicit(positions: Vec<Vec<f64>>) -> Self {
        PyWalkerInit {
            inner: WalkerInitInner::Explicit { positions },
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            WalkerInitInner::FromPrior => "WalkerInit.from_prior()".to_string(),
            WalkerInitInner::Ball { center, radius } => {
                format!("WalkerInit.ball({:?}, {})", center, radius)
            }
            WalkerInitInner::Explicit { positions } => {
                format!("WalkerInit.explicit({} walkers)", positions.len())
            }
        }
    }
}

/// Information about MCMC sampling progress.
///
/// Passed to progress callbacks during sampling.
#[pyclass(name = "ProgressInfo")]
#[derive(Clone)]
pub struct PyProgressInfo {
    #[pyo3(get)]
    pub iteration: usize,

    #[pyo3(get)]
    pub total: usize,

    #[pyo3(get)]
    pub acceptance_rate: f64,

    #[pyo3(get)]
    pub mean_log_prob: f64,
}

impl From<&ProgressInfo> for PyProgressInfo {
    fn from(info: &ProgressInfo) -> Self {
        PyProgressInfo {
            iteration: info.iteration,
            total: info.total,
            acceptance_rate: info.acceptance_rate,
            mean_log_prob: info.mean_log_prob,
        }
    }
}

#[pymethods]
impl PyProgressInfo {
    fn __repr__(&self) -> String {
        format!(
            "ProgressInfo(iteration={}/{}, acceptance_rate={:.2}, mean_log_prob={:.2})",
            self.iteration, self.total, self.acceptance_rate, self.mean_log_prob
        )
    }
}

/// Affine-invariant ensemble sampler for Bayesian parameter estimation.
///
/// Implements the Goodman & Weare (2010) stretch move algorithm for MCMC sampling.
/// This is a parallel MCMC method that uses an ensemble of "walkers" to explore
/// parameter space together, with each walker's proposal informed by the positions
/// of other walkers. The algorithm is affine-invariant, meaning it's robust to
/// parameter correlations and rescaling.
///
/// The sampler performs Bayesian inference to estimate posterior distributions
/// of model parameters given observational data:
///
/// ```text
/// p(θ | data) ∝ p(data | θ) × p(θ)
/// ```
///
/// where p(θ) is the prior (ParameterSet) and p(data | θ) is the likelihood
/// (GaussianLikelihood).
///
/// Parameters
/// ----------
/// params : ParameterSet
///     Parameter set defining prior distributions for all model parameters
/// runner : ModelRunner
///     Model runner for evaluating parameter sets in parallel
/// likelihood : GaussianLikelihood
///     Likelihood function for computing log probability of observations
/// target : Target
///     Target observations to calibrate against with uncertainties
///
/// Examples
/// --------
/// Complete calibration workflow:
///
/// ```python
/// from rscm.calibrate import (
///     ParameterSet, Uniform, Normal, Target,
///     ModelRunner, GaussianLikelihood, EnsembleSampler, WalkerInit
/// )
///
/// # Define prior distributions
/// params = ParameterSet()
/// params.add("sensitivity", Uniform(0.5, 1.5))
/// params.add("offset", Normal(0.0, 0.1))
///
/// # Define observations
/// target = Target()
/// target.add_variable("Temperature").add(2020.0, 1.2, 0.1).add(2021.0, 1.3, 0.1)
///
/// # Create model runner (user-defined callable)
/// def model_factory(params):
///     # Build and run model with parameters
///     return model  # Returns model instance
/// runner = ModelRunner(model_factory, ["sensitivity", "offset"], ["Temperature"])
///
/// # Create sampler
/// likelihood = GaussianLikelihood()
/// sampler = EnsembleSampler(params, runner, likelihood, target)
///
/// # Run MCMC (10,000 iterations, initialize from prior, store every 10th sample)
/// chain = sampler.run(10000, WalkerInit.from_prior(), thin=10)
///
/// # Check convergence (discard first 1000 samples as burn-in)
/// r_hat = chain.r_hat(discard=1000)
/// print(f"R-hat: {r_hat}")  # Should be < 1.1 for convergence
///
/// # Extract posterior samples
/// samples = chain.to_param_dict(discard=1000)
/// print(f"Sensitivity: {samples['sensitivity'].mean():.3f} ± {samples['sensitivity'].std():.3f}")
/// ```
///
/// References
/// ----------
/// Goodman, J., & Weare, J. (2010). Ensemble samplers with affine invariance.
/// Communications in Applied Mathematics and Computational Science, 5(1), 65-80.
///
/// See Also
/// --------
/// WalkerInit : Walker initialization strategies
/// Chain : MCMC chain with convergence diagnostics
#[pyclass(name = "EnsembleSampler")]
pub struct PyEnsembleSampler {
    sampler: EnsembleSampler<PyModelRunner, PyGaussianLikelihood>,
}

#[pymethods]
impl PyEnsembleSampler {
    #[new]
    fn new(
        params: PyParameterSet,
        runner: PyModelRunner,
        likelihood: PyGaussianLikelihood,
        target: PyTarget,
    ) -> Self {
        let sampler = EnsembleSampler::new(params.0, runner, likelihood, target.0);
        PyEnsembleSampler { sampler }
    }

    /// Get the default number of walkers for this sampler.
    ///
    /// Returns
    /// -------
    /// int
    ///     Default number of walkers (max(2 * n_params, 32))
    fn default_n_walkers(&self) -> usize {
        self.sampler.default_n_walkers()
    }

    /// Run the ensemble sampler.
    ///
    /// Parameters
    /// ----------
    /// n_iterations : int
    ///     Number of MCMC iterations to run
    /// init : WalkerInit
    ///     Walker initialization strategy
    /// thin : int, optional
    ///     Thinning interval (store every thin-th sample). Default is 1.
    ///
    /// Returns
    /// -------
    /// Chain
    ///     Chain containing the samples and diagnostics
    ///
    /// Examples
    /// --------
    /// >>> chain = sampler.run(1000, WalkerInit.from_prior(), thin=1)
    #[pyo3(signature = (n_iterations, init, thin=1))]
    fn run(&self, n_iterations: usize, init: PyWalkerInit, thin: usize) -> PyResult<PyChain> {
        let rust_init = init.to_rust();
        let chain = self
            .sampler
            .run(n_iterations, rust_init, thin)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyChain(chain))
    }

    /// Run the ensemble sampler with progress callback.
    ///
    /// Parameters
    /// ----------
    /// n_iterations : int
    ///     Number of MCMC iterations to run
    /// init : WalkerInit
    ///     Walker initialization strategy
    /// thin : int
    ///     Thinning interval (store every thin-th sample)
    /// progress_callback : callable
    ///     Callback function that receives ProgressInfo objects
    ///
    /// Returns
    /// -------
    /// Chain
    ///     Chain containing the samples and diagnostics
    ///
    /// Examples
    /// --------
    /// >>> def show_progress(info):
    /// ...     print(f"Iteration {info.iteration}/{info.total}, acceptance={info.acceptance_rate:.2f}")
    /// >>> chain = sampler.run_with_progress(1000, WalkerInit.from_prior(), 1, show_progress)
    #[pyo3(signature = (n_iterations, init, thin, progress_callback))]
    fn run_with_progress(
        &self,
        py: Python<'_>,
        n_iterations: usize,
        init: PyWalkerInit,
        thin: usize,
        progress_callback: Py<PyAny>,
    ) -> PyResult<PyChain> {
        let rust_init = init.to_rust();

        // Create a Rust closure that calls the Python callback
        let callback = move |info: &ProgressInfo| {
            Python::attach(|py| {
                let py_info = PyProgressInfo::from(info);
                if let Err(e) = progress_callback.call1(py, (py_info,)) {
                    eprintln!("Error in progress callback: {}", e);
                }
            });
        };

        // Release GIL during sampling (except for callback invocations)
        let chain = py.detach(|| {
            self.sampler
                .run_with_progress(n_iterations, rust_init, thin, callback)
        });

        let chain =
            chain.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyChain(chain))
    }

    /// Run the ensemble sampler with checkpointing.
    ///
    /// Saves the sampler state and chain to checkpoint files at regular intervals.
    /// If the run is interrupted, it can be resumed with `resume_from_checkpoint`.
    ///
    /// Parameters
    /// ----------
    /// n_iterations : int
    ///     Number of MCMC iterations to run
    /// init : WalkerInit
    ///     Walker initialization strategy
    /// thin : int
    ///     Thinning interval (store every thin-th sample)
    /// checkpoint_every : int
    ///     Save checkpoint every N iterations
    /// checkpoint_path : str
    ///     Base path for checkpoint files (will append .state and .chain)
    /// progress_callback : callable, optional
    ///     Optional callback function that receives ProgressInfo objects
    ///
    /// Returns
    /// -------
    /// Chain
    ///     Chain containing the samples and diagnostics
    ///
    /// Examples
    /// --------
    /// >>> chain = sampler.run_with_checkpoint(
    /// ...     1000, WalkerInit.from_prior(), 1, 100, "checkpoint"
    /// ... )
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (n_iterations, init, thin, checkpoint_every, checkpoint_path, progress_callback=None))]
    fn run_with_checkpoint(
        &self,
        py: Python<'_>,
        n_iterations: usize,
        init: PyWalkerInit,
        thin: usize,
        checkpoint_every: usize,
        checkpoint_path: String,
        progress_callback: Option<Py<PyAny>>,
    ) -> PyResult<PyChain> {
        let rust_init = init.to_rust();

        // Create optional callback closure
        let callback = progress_callback.map(|cb| {
            move |info: &ProgressInfo| {
                Python::attach(|py| {
                    let py_info = PyProgressInfo::from(info);
                    if let Err(e) = cb.call1(py, (py_info,)) {
                        eprintln!("Error in progress callback: {}", e);
                    }
                });
            }
        });

        // Release GIL during sampling
        let chain = py.detach(|| {
            self.sampler.run_with_checkpoint(
                n_iterations,
                rust_init,
                thin,
                checkpoint_every,
                checkpoint_path,
                callback,
            )
        });

        let chain =
            chain.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyChain(chain))
    }

    /// Resume a checkpointed run.
    ///
    /// Loads the state and chain from checkpoint files and continues sampling.
    ///
    /// Parameters
    /// ----------
    /// n_iterations : int
    ///     Total number of iterations to reach (including already completed)
    /// thin : int
    ///     Thinning interval (must match original run)
    /// checkpoint_every : int
    ///     Save checkpoint every N iterations
    /// checkpoint_path : str
    ///     Base path for checkpoint files
    /// progress_callback : callable, optional
    ///     Optional callback function that receives ProgressInfo objects
    ///
    /// Returns
    /// -------
    /// Chain
    ///     The complete chain including both resumed and new samples
    ///
    /// Examples
    /// --------
    /// >>> # Resume from checkpoint and run to 2000 total iterations
    /// >>> chain = sampler.resume_from_checkpoint(2000, 1, 100, "checkpoint")
    #[pyo3(signature = (n_iterations, thin, checkpoint_every, checkpoint_path, progress_callback=None))]
    fn resume_from_checkpoint(
        &self,
        py: Python<'_>,
        n_iterations: usize,
        thin: usize,
        checkpoint_every: usize,
        checkpoint_path: String,
        progress_callback: Option<Py<PyAny>>,
    ) -> PyResult<PyChain> {
        // Create optional callback closure
        let callback = progress_callback.map(|cb| {
            move |info: &ProgressInfo| {
                Python::attach(|py| {
                    let py_info = PyProgressInfo::from(info);
                    if let Err(e) = cb.call1(py, (py_info,)) {
                        eprintln!("Error in progress callback: {}", e);
                    }
                });
            }
        });

        // Release GIL during sampling
        let chain = py.detach(|| {
            self.sampler.resume_from_checkpoint(
                n_iterations,
                thin,
                checkpoint_every,
                checkpoint_path,
                callback,
            )
        });

        let chain =
            chain.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyChain(chain))
    }

    /// Create a sampler with custom stretch move parameter.
    ///
    /// Parameters
    /// ----------
    /// params : ParameterSet
    ///     Parameter set defining prior distributions
    /// runner : ModelRunner
    ///     Model runner for evaluating parameter sets
    /// likelihood : GaussianLikelihood
    ///     Likelihood function for computing log probability
    /// target : Target
    ///     Target observations to calibrate against
    /// a : float
    ///     Stretch move parameter (default is 2.0). Must be > 1.0.
    ///
    /// Returns
    /// -------
    /// EnsembleSampler
    ///     New sampler with custom stretch parameter
    ///
    /// Notes
    /// -----
    /// The stretch parameter controls the proposal distribution. Larger values
    /// lead to more aggressive proposals. The default of 2.0 is recommended
    /// for most applications.
    #[staticmethod]
    fn with_stretch_param(
        params: PyParameterSet,
        runner: PyModelRunner,
        likelihood: PyGaussianLikelihood,
        target: PyTarget,
        a: f64,
    ) -> PyResult<Self> {
        let sampler = EnsembleSampler::new(params.0, runner, likelihood, target.0)
            .with_stretch_param(a)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyEnsembleSampler { sampler })
    }

    fn __repr__(&self) -> String {
        format!(
            "EnsembleSampler(default_n_walkers={})",
            self.sampler.default_n_walkers()
        )
    }
}
