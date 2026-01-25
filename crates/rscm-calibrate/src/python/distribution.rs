use crate::distribution::{Bound, Distribution, LogNormal, Normal, Uniform};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python wrapper for Uniform distribution.
#[pyclass(name = "Uniform")]
#[derive(Clone)]
pub struct PyUniform(pub Uniform);

#[pymethods]
impl PyUniform {
    /// Create a uniform distribution over [low, high].
    ///
    /// Parameters
    /// ----------
    /// low : float
    ///     Lower bound (inclusive)
    /// high : float
    ///     Upper bound (inclusive)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If low >= high
    #[new]
    fn new(low: f64, high: f64) -> PyResult<Self> {
        Uniform::new(low, high)
            .map(PyUniform)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Sample a value from the distribution.
    ///
    /// Returns
    /// -------
    /// float
    ///     A random value from the distribution
    fn sample(&self) -> f64 {
        <dyn Distribution>::sample(&self.0)
    }

    /// Compute the natural logarithm of the probability density at x.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///     Value at which to evaluate the PDF
    ///
    /// Returns
    /// -------
    /// float
    ///     ln(p(x))
    fn ln_pdf(&self, x: f64) -> f64 {
        Distribution::ln_pdf(&self.0, x)
    }

    /// Get the support bounds [min, max] of the distribution.
    ///
    /// Returns
    /// -------
    /// tuple[float, float]
    ///     (lower_bound, upper_bound)
    fn bounds(&self) -> (f64, f64) {
        Distribution::bounds(&self.0).unwrap()
    }

    /// Get the lower bound.
    #[getter]
    fn low(&self) -> f64 {
        self.0.low()
    }

    /// Get the upper bound.
    #[getter]
    fn high(&self) -> f64 {
        self.0.high()
    }

    fn __repr__(&self) -> String {
        format!("Uniform(low={}, high={})", self.0.low(), self.0.high())
    }
}

/// Python wrapper for Normal distribution.
#[pyclass(name = "Normal")]
#[derive(Clone)]
pub struct PyNormal(pub Normal);

#[pymethods]
impl PyNormal {
    /// Create a normal (Gaussian) distribution with mean μ and standard deviation σ.
    ///
    /// Parameters
    /// ----------
    /// mean : float
    ///     Mean of the distribution
    /// std_dev : float
    ///     Standard deviation (must be positive)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If std_dev <= 0
    #[new]
    fn new(mean: f64, std_dev: f64) -> PyResult<Self> {
        Normal::new(mean, std_dev)
            .map(PyNormal)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Sample a value from the distribution.
    ///
    /// Returns
    /// -------
    /// float
    ///     A random value from the distribution
    fn sample(&self) -> f64 {
        <dyn Distribution>::sample(&self.0)
    }

    /// Compute the natural logarithm of the probability density at x.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///     Value at which to evaluate the PDF
    ///
    /// Returns
    /// -------
    /// float
    ///     ln(p(x))
    fn ln_pdf(&self, x: f64) -> f64 {
        Distribution::ln_pdf(&self.0, x)
    }

    /// Get the support bounds [min, max] of the distribution (unbounded for Normal).
    ///
    /// Returns
    /// -------
    /// None
    ///     Normal distribution is unbounded
    fn bounds(&self) -> Option<(f64, f64)> {
        Distribution::bounds(&self.0)
    }

    /// Get the mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.0.mean()
    }

    /// Get the standard deviation.
    #[getter]
    fn std_dev(&self) -> f64 {
        self.0.std_dev()
    }

    fn __repr__(&self) -> String {
        format!(
            "Normal(mean={}, std_dev={})",
            self.0.mean(),
            self.0.std_dev()
        )
    }
}

/// Python wrapper for LogNormal distribution.
#[pyclass(name = "LogNormal")]
#[derive(Clone)]
pub struct PyLogNormal(pub LogNormal);

#[pymethods]
impl PyLogNormal {
    /// Create a log-normal distribution.
    ///
    /// Can be parameterised in two ways:
    /// 1. Using underlying normal parameters: LogNormal(mu=..., sigma=...)
    /// 2. Using mean and std of the log-normal: LogNormal(mean=..., std=...)
    ///
    /// Parameters
    /// ----------
    /// mu : float, optional
    ///     Mean of the underlying normal distribution
    /// sigma : float, optional
    ///     Standard deviation of the underlying normal distribution (must be positive)
    /// mean : float, optional
    ///     Mean of the log-normal distribution (must be positive)
    /// std : float, optional
    ///     Standard deviation of the log-normal distribution (must be positive)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If parameters are invalid or wrong combination provided
    #[new]
    #[pyo3(signature = (*, mu=None, sigma=None, mean=None, std=None))]
    fn new(
        mu: Option<f64>,
        sigma: Option<f64>,
        mean: Option<f64>,
        std: Option<f64>,
    ) -> PyResult<Self> {
        match (mu, sigma, mean, std) {
            (Some(mu), Some(sigma), None, None) => LogNormal::new(mu, sigma)
                .map(PyLogNormal)
                .map_err(|e| PyValueError::new_err(e.to_string())),
            (None, None, Some(mean), Some(std)) => LogNormal::from_mean_std(mean, std)
                .map(PyLogNormal)
                .map_err(|e| PyValueError::new_err(e.to_string())),
            _ => Err(PyValueError::new_err(
                "Must provide either (mu, sigma) or (mean, std), not both or neither",
            )),
        }
    }

    /// Sample a value from the distribution.
    ///
    /// Returns
    /// -------
    /// float
    ///     A random value from the distribution
    fn sample(&self) -> f64 {
        <dyn Distribution>::sample(&self.0)
    }

    /// Compute the natural logarithm of the probability density at x.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///     Value at which to evaluate the PDF
    ///
    /// Returns
    /// -------
    /// float
    ///     ln(p(x))
    fn ln_pdf(&self, x: f64) -> f64 {
        Distribution::ln_pdf(&self.0, x)
    }

    /// Get the support bounds [min, max] of the distribution.
    ///
    /// Returns
    /// -------
    /// tuple[float, float] | None
    ///     (0, inf) for unbounded log-normal, or finite bounds if constrained
    fn bounds(&self) -> Option<(f64, f64)> {
        Distribution::bounds(&self.0)
    }

    /// Get the mean of the underlying normal distribution.
    #[getter]
    fn mu(&self) -> f64 {
        self.0.mu()
    }

    /// Get the standard deviation of the underlying normal distribution.
    #[getter]
    fn sigma(&self) -> f64 {
        self.0.sigma()
    }

    fn __repr__(&self) -> String {
        format!("LogNormal(mu={}, sigma={})", self.0.mu(), self.0.sigma())
    }
}

/// Python wrapper for Bound distribution wrapper.
#[pyclass(name = "Bound")]
#[derive(Clone)]
pub struct PyBound(pub Bound);

#[pymethods]
impl PyBound {
    /// Constrain a distribution to bounds using rejection sampling.
    ///
    /// Parameters
    /// ----------
    /// distribution : Uniform | Normal | LogNormal
    ///     The distribution to constrain
    /// low : float
    ///     Lower bound (inclusive)
    /// high : float
    ///     Upper bound (inclusive)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If low >= high
    #[new]
    fn new(distribution: PyDistribution, low: f64, high: f64) -> PyResult<Self> {
        let dist_box: Box<dyn Distribution> = match distribution {
            PyDistribution::Uniform(u) => Box::new(u.0.clone()),
            PyDistribution::Normal(n) => Box::new(n.0.clone()),
            PyDistribution::LogNormal(ln) => Box::new(ln.0.clone()),
            PyDistribution::Bound(b) => Box::new(b.0.clone()),
        };

        Bound::new(dist_box, low, high)
            .map(PyBound)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Sample a value from the distribution.
    ///
    /// Returns
    /// -------
    /// float
    ///     A random value from the distribution
    fn sample(&self) -> f64 {
        <dyn Distribution>::sample(&self.0)
    }

    /// Compute the natural logarithm of the probability density at x.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///     Value at which to evaluate the PDF
    ///
    /// Returns
    /// -------
    /// float
    ///     ln(p(x))
    fn ln_pdf(&self, x: f64) -> f64 {
        Distribution::ln_pdf(&self.0, x)
    }

    /// Get the support bounds [min, max] of the distribution.
    ///
    /// Returns
    /// -------
    /// tuple[float, float]
    ///     (lower_bound, upper_bound)
    fn bounds(&self) -> (f64, f64) {
        Distribution::bounds(&self.0).unwrap()
    }

    fn __repr__(&self) -> String {
        format!("Bound(low={}, high={})", self.0.low(), self.0.high())
    }
}

/// Enum to handle different distribution types in Python.
///
/// This is used internally to allow PyBound and PyParameterSet to accept any distribution type.
#[derive(FromPyObject)]
pub enum PyDistribution {
    Uniform(PyUniform),
    Normal(PyNormal),
    LogNormal(PyLogNormal),
    Bound(PyBound),
}
