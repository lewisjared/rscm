use crate::parameter_set::ParameterSet;
use crate::python::distribution::PyDistribution;
use crate::Distribution;
use numpy::{PyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python wrapper for ParameterSet.
#[pyclass(name = "ParameterSet")]
#[derive(Clone)]
pub struct PyParameterSet(pub ParameterSet);

#[pymethods]
impl PyParameterSet {
    /// Create a parameter set from a dictionary of distributions.
    ///
    /// Parameters
    /// ----------
    /// params : dict[str, Distribution], optional
    ///     Dictionary mapping parameter names to distributions.
    ///     If not provided, creates an empty parameter set.
    ///
    /// Examples
    /// --------
    /// >>> from rscm._lib.calibrate import ParameterSet, Uniform, Normal
    /// >>> params = ParameterSet({
    /// ...     "x": Uniform(0.0, 1.0),
    /// ...     "y": Normal(0.0, 1.0)
    /// ... })
    ///
    /// Returns
    /// -------
    /// ParameterSet
    ///     The parameter set
    #[new]
    #[pyo3(signature = (params=None))]
    fn new(params: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut param_set = ParameterSet::new();

        if let Some(dict) = params {
            for (key, value) in dict.iter() {
                let name: String = key.extract()?;
                let dist: PyDistribution = value.extract()?;

                let dist_box: Box<dyn Distribution> = match dist {
                    PyDistribution::Uniform(u) => Box::new(u.0.clone()),
                    PyDistribution::Normal(n) => Box::new(n.0.clone()),
                    PyDistribution::LogNormal(ln) => Box::new(ln.0.clone()),
                    PyDistribution::Bound(b) => Box::new(b.0.clone()),
                };

                param_set.add(name, dist_box);
            }
        }

        Ok(PyParameterSet(param_set))
    }

    /// Add a parameter distribution.
    ///
    /// Returns self for fluent chaining.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Parameter name
    /// distribution : Distribution
    ///     Prior distribution for this parameter
    ///
    /// Returns
    /// -------
    /// ParameterSet
    ///     Self for chaining
    ///
    /// Examples
    /// --------
    /// >>> from rscm._lib.calibrate import ParameterSet, Uniform, Normal
    /// >>> params = ParameterSet().add("x", Uniform(0.0, 1.0)).add("y", Normal(0.0, 1.0))
    fn add(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        distribution: PyDistribution,
    ) -> PyRefMut<'_, Self> {
        let dist_box: Box<dyn Distribution> = match distribution {
            PyDistribution::Uniform(u) => Box::new(u.0.clone()),
            PyDistribution::Normal(n) => Box::new(n.0.clone()),
            PyDistribution::LogNormal(ln) => Box::new(ln.0.clone()),
            PyDistribution::Bound(b) => Box::new(b.0.clone()),
        };

        slf.0.add(name, dist_box);
        slf
    }

    /// Get parameter names in definition order.
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     Parameter names
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.0.param_names().iter().map(|s| s.to_string()).collect()
    }

    /// Get the number of parameters.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of parameters
    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Sample n parameter vectors randomly from the priors.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Number of parameter vectors to sample
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array of shape (n, n_params) where each row is a parameter vector
    ///
    /// Examples
    /// --------
    /// >>> from rscm._lib.calibrate import ParameterSet, Uniform
    /// >>> params = ParameterSet({"x": Uniform(0.0, 1.0), "y": Uniform(0.0, 1.0)})
    /// >>> samples = params.sample_random(100)
    /// >>> samples.shape
    /// (100, 2)
    fn sample_random<'py>(&self, py: Python<'py>, n: usize) -> Bound<'py, PyArray2<f64>> {
        let samples = self.0.sample_random(n);
        samples.to_pyarray(py)
    }

    /// Sample n parameter vectors using Latin Hypercube Sampling.
    ///
    /// LHS ensures better coverage of the parameter space than random sampling.
    /// Each parameter dimension is divided into n equal-probability intervals,
    /// and exactly one sample is placed in each interval.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Number of parameter vectors to sample
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array of shape (n, n_params) where each row is a parameter vector
    ///
    /// Examples
    /// --------
    /// >>> from rscm._lib.calibrate import ParameterSet, Uniform
    /// >>> params = ParameterSet({"x": Uniform(0.0, 1.0), "y": Uniform(0.0, 1.0)})
    /// >>> samples = params.sample_lhs(100)
    /// >>> samples.shape
    /// (100, 2)
    fn sample_lhs<'py>(&self, py: Python<'py>, n: usize) -> Bound<'py, PyArray2<f64>> {
        let samples = self.0.sample_lhs(n);
        samples.to_pyarray(py)
    }

    /// Compute the log prior probability of a parameter vector.
    ///
    /// The parameter vector must match the order from param_names.
    ///
    /// Parameters
    /// ----------
    /// params : list[float] | numpy.ndarray
    ///     Parameter vector
    ///
    /// Returns
    /// -------
    /// float
    ///     Log prior probability
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If parameter vector length doesn't match parameter set size
    ///
    /// Examples
    /// --------
    /// >>> from rscm._lib.calibrate import ParameterSet, Uniform
    /// >>> params = ParameterSet({"x": Uniform(0.0, 1.0), "y": Uniform(0.0, 1.0)})
    /// >>> log_p = params.log_prior([0.5, 0.5])
    fn log_prior(&self, params: Vec<f64>) -> PyResult<f64> {
        self.0
            .log_prior(&params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Extract bounds for all parameters.
    ///
    /// Returns
    /// -------
    /// tuple[list[float], list[float]]
    ///     (lower_bounds, upper_bounds) where each list has length n_params.
    ///     Unbounded parameters use -inf and inf.
    ///
    /// Examples
    /// --------
    /// >>> from rscm._lib.calibrate import ParameterSet, Uniform, Normal
    /// >>> params = ParameterSet({"x": Uniform(0.0, 1.0), "y": Normal(0.0, 1.0)})
    /// >>> lower, upper = params.bounds()
    /// >>> lower
    /// [0.0, -inf]
    /// >>> upper
    /// [1.0, inf]
    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        self.0.bounds()
    }

    fn __repr__(&self) -> String {
        format!("ParameterSet(n_params={})", self.0.len())
    }
}

impl PyParameterSet {
    /// Consume the wrapper and return the inner ParameterSet.
    pub fn into_inner(self) -> ParameterSet {
        self.0
    }
}
