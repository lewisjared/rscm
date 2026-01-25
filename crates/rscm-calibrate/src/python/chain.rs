use crate::sampler::Chain;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python wrapper for Chain.
///
/// Stores MCMC samples from ensemble sampling with diagnostic methods.
#[pyclass(name = "Chain")]
pub struct PyChain(pub Chain);

#[pymethods]
impl PyChain {
    /// Get the number of stored samples.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of stored samples (after thinning)
    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Get the total number of iterations (including thinned samples).
    ///
    /// Returns
    /// -------
    /// int
    ///     Total number of MCMC iterations run
    #[getter]
    fn total_iterations(&self) -> usize {
        self.0.total_iterations()
    }

    /// Get the thinning interval.
    ///
    /// Returns
    /// -------
    /// int
    ///     Thinning interval (every thin-th sample is stored)
    #[getter]
    fn thin(&self) -> usize {
        self.0.thin()
    }

    /// Get parameter names in order.
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     Parameter names
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.0.param_names().to_vec()
    }

    /// Get flattened samples, optionally discarding initial burn-in samples.
    ///
    /// Parameters
    /// ----------
    /// discard : int, optional
    ///     Number of initial samples to discard from each walker (default: 0)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array of shape ((len - discard) * n_walkers, n_params) containing all
    ///     post-burn-in samples from all walkers, concatenated.
    ///
    /// Examples
    /// --------
    /// >>> samples = chain.flat_samples(discard=100)
    /// >>> print(samples.shape)  # (n_samples, n_params)
    #[pyo3(signature = (discard=0))]
    fn flat_samples<'py>(&self, py: Python<'py>, discard: usize) -> Bound<'py, PyArray2<f64>> {
        self.0.flat_samples(discard).to_pyarray(py)
    }

    /// Get flattened log probabilities, optionally discarding initial burn-in samples.
    ///
    /// Parameters
    /// ----------
    /// discard : int, optional
    ///     Number of initial samples to discard from each walker (default: 0)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array of shape ((len - discard) * n_walkers,) containing all
    ///     post-burn-in log probabilities from all walkers, concatenated.
    ///
    /// Examples
    /// --------
    /// >>> log_probs = chain.flat_log_probs(discard=100)
    /// >>> print(log_probs.shape)  # (n_samples,)
    #[pyo3(signature = (discard=0))]
    fn flat_log_probs<'py>(&self, py: Python<'py>, discard: usize) -> Bound<'py, PyArray1<f64>> {
        self.0.flat_log_probs(discard).to_pyarray(py)
    }

    /// Compute the Gelman-Rubin statistic (R-hat) for each parameter.
    ///
    /// R-hat measures convergence by comparing within-chain and between-chain variances.
    /// Values close to 1.0 indicate convergence. As a rule of thumb, R-hat < 1.1 for all
    /// parameters suggests the chains have converged.
    ///
    /// Parameters
    /// ----------
    /// discard : int, optional
    ///     Number of initial samples to discard as burn-in (default: 0)
    ///
    /// Returns
    /// -------
    /// dict[str, float]
    ///     Map from parameter name to R-hat value. Returns empty dict if insufficient samples.
    ///
    /// Examples
    /// --------
    /// >>> r_hat = chain.r_hat(discard=100)
    /// >>> for param, value in r_hat.items():
    /// ...     print(f"{param}: {value:.3f}")
    ///
    /// References
    /// ----------
    /// Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using
    /// multiple sequences. Statistical Science, 7(4), 457-472.
    #[pyo3(signature = (discard=0))]
    fn r_hat<'py>(&self, py: Python<'py>, discard: usize) -> Bound<'py, PyDict> {
        let result = self.0.r_hat(discard);
        let dict = PyDict::new(py);
        for (name, value) in result {
            dict.set_item(name, value).unwrap();
        }
        dict
    }

    /// Check if the chain has converged based on R-hat statistic.
    ///
    /// Parameters
    /// ----------
    /// discard : int, optional
    ///     Number of initial samples to discard as burn-in (default: 0)
    /// threshold : float, optional
    ///     R-hat threshold for convergence (default: 1.1)
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if all parameters have R-hat < threshold, False otherwise.
    ///     Returns False if insufficient samples to compute R-hat.
    ///
    /// Examples
    /// --------
    /// >>> if chain.is_converged(discard=100, threshold=1.1):
    /// ...     print("Chain has converged!")
    #[pyo3(signature = (discard=0, threshold=1.1))]
    fn is_converged(&self, discard: usize, threshold: f64) -> bool {
        self.0.is_converged(discard, threshold)
    }

    /// Compute the effective sample size (ESS) for each parameter.
    ///
    /// ESS estimates the number of independent samples in the chain, accounting
    /// for autocorrelation. Higher values indicate better mixing. As a rule of
    /// thumb, ESS > 100 per chain is often sufficient for posterior inference.
    ///
    /// Parameters
    /// ----------
    /// discard : int, optional
    ///     Number of initial samples to discard as burn-in (default: 0)
    ///
    /// Returns
    /// -------
    /// dict[str, float]
    ///     Map from parameter name to ESS value. Returns empty dict if insufficient samples.
    ///
    /// Examples
    /// --------
    /// >>> ess = chain.ess(discard=100)
    /// >>> for param, value in ess.items():
    /// ...     print(f"{param}: {value:.1f} effective samples")
    ///
    /// References
    /// ----------
    /// Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., &
    /// Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.
    #[pyo3(signature = (discard=0))]
    fn ess<'py>(&self, py: Python<'py>, discard: usize) -> Bound<'py, PyDict> {
        let result = self.0.ess(discard);
        let dict = PyDict::new(py);
        for (name, value) in result {
            dict.set_item(name, value).unwrap();
        }
        dict
    }

    /// Compute the integrated autocorrelation time for each parameter.
    ///
    /// The autocorrelation time (τ) measures how many steps it takes for samples
    /// to become approximately independent. It is computed as:
    ///
    /// τ = 1 + 2 * Σ ρ(k)
    ///
    /// where ρ(k) is the autocorrelation at lag k, summed over positive values.
    ///
    /// This is useful for:
    /// - Determining appropriate thinning interval (thin by ~τ to get nearly independent samples)
    /// - Estimating effective sample size: ESS ≈ N / τ
    /// - Assessing mixing quality (smaller τ = better mixing)
    ///
    /// Parameters
    /// ----------
    /// discard : int, optional
    ///     Number of initial samples to discard as burn-in (default: 0)
    ///
    /// Returns
    /// -------
    /// dict[str, float]
    ///     Map from parameter name to autocorrelation time. Returns empty dict if
    ///     insufficient samples (< 10 after discard).
    ///
    /// Examples
    /// --------
    /// >>> tau = chain.autocorr_time(discard=100)
    /// >>> for param, time in tau.items():
    /// ...     print(f"{param}: τ = {time:.1f} (thin by ~{time:.0f} for independence)")
    #[pyo3(signature = (discard=0))]
    fn autocorr_time<'py>(&self, py: Python<'py>, discard: usize) -> Bound<'py, PyDict> {
        let result = self.0.autocorr_time(discard);
        let dict = PyDict::new(py);
        for (name, value) in result {
            dict.set_item(name, value).unwrap();
        }
        dict
    }

    /// Convert chain samples to a dictionary mapping parameter names to sample arrays.
    ///
    /// Useful for per-parameter analysis and plotting.
    ///
    /// Parameters
    /// ----------
    /// discard : int, optional
    ///     Number of initial samples to discard as burn-in (default: 0)
    ///
    /// Returns
    /// -------
    /// dict[str, numpy.ndarray]
    ///     Map from parameter name to 1D array of all post-burn-in samples for that parameter.
    ///
    /// Examples
    /// --------
    /// >>> param_dict = chain.to_param_dict(discard=100)
    /// >>> import matplotlib.pyplot as plt
    /// >>> plt.hist(param_dict["x"], bins=50)
    #[pyo3(signature = (discard=0))]
    fn to_param_dict<'py>(&self, py: Python<'py>, discard: usize) -> Bound<'py, PyDict> {
        let result = self.0.to_param_map(discard);
        let dict = PyDict::new(py);
        for (name, values) in result {
            dict.set_item(name, values.to_pyarray(py)).unwrap();
        }
        dict
    }

    /// Save the chain to a file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the file to create
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If serialization or file writing fails
    ///
    /// Examples
    /// --------
    /// >>> chain.save("my_chain.bin")
    fn save(&self, path: String) -> PyResult<()> {
        self.0
            .save(&path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load a chain from a file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the file to read
    ///
    /// Returns
    /// -------
    /// Chain
    ///     The loaded chain
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If deserialization fails
    ///
    /// Examples
    /// --------
    /// >>> chain = Chain.load("my_chain.bin")
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        Chain::load(&path)
            .map(PyChain)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Merge another chain into this one.
    ///
    /// This is useful for combining chain segments from checkpointed runs.
    /// The chains must have the same parameter names and thinning interval.
    ///
    /// Parameters
    /// ----------
    /// other : Chain
    ///     The chain to merge into this one
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If chains are incompatible (different parameters or thinning)
    ///
    /// Examples
    /// --------
    /// >>> chain1.merge(chain2)
    fn merge(&mut self, other: &PyChain) -> PyResult<()> {
        self.0
            .merge(&other.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Chain(n_samples={}, n_params={}, thin={})",
            self.0.len(),
            self.0.param_names().len(),
            self.0.thin()
        )
    }
}
