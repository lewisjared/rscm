use crate::interpolate::strategies::{
    InterpolationStrategy, LinearSplineStrategy, NextStrategy, PreviousStrategy,
};
use crate::python::{check_unit, resolve_index};
use crate::spatial::{FourBoxGrid, HemisphericGrid, ScalarGrid, ScalarRegion};
use crate::timeseries::{FloatValue, GridTimeseries, Time, TimeAxis, Timeseries};
use numpy::ndarray::{Array, Axis, Dimension, Ix1, Ix2};
use numpy::{PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, ToPyArray as _};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::sync::Arc;

/// Read an array-like of reals as a float64 array with the dimensions of `D`.
fn float_array<D: Dimension>(values: &Bound<PyAny>, name: &str) -> PyResult<Array<f64, D>> {
    let array = values
        .py()
        .import("numpy")?
        .call_method1("asarray", (values, "float64"))?
        .cast_into::<PyArrayDyn<f64>>()?;
    array
        .to_owned_array()
        .into_dimensionality::<D>()
        .map_err(|_| {
            let shape = array
                .getattr("shape")
                .map(|s| s.to_string())
                .unwrap_or_default();
            PyValueError::new_err(format!(
                "{name} must be {}-dimensional, got shape {shape}",
                D::NDIM.unwrap_or_default()
            ))
        })
}

fn checked_unit(kind: &str, unit: String) -> PyResult<String> {
    check_unit(&unit).map_err(|e| {
        PyValueError::new_err(format!(
            "{kind} unit '{unit}' is not a known unit: {e}. Use '' for no unit"
        ))
    })?;
    Ok(unit)
}

fn length_mismatch(values: usize, time_axis: usize) -> PyErr {
    PyValueError::new_err(format!(
        "values has length {values} but time_axis has length {time_axis}"
    ))
}

/// A callable and its arguments, which `__reduce__` returns to pickle an object.
pub(crate) type Reduced<'py> = (Bound<'py, PyAny>, Bound<'py, PyTuple>);

pub(crate) fn time_axis_repr(axis: &TimeAxis) -> String {
    format!(
        "TimeAxis(len={}, start={:?}, end={:?})",
        axis.len(),
        axis.first(),
        axis.last()
    )
}

#[pyclass(module = "rscm._lib.core")]
#[pyo3(name = "TimeAxis")]
pub struct PyTimeAxis(pub Arc<TimeAxis>);

#[pymethods]
impl PyTimeAxis {
    fn __repr__(&self) -> String {
        time_axis_repr(&self.0)
    }

    fn __eq__(&self, other: PyRef<Self>) -> bool {
        self.0.bounds() == other.0.bounds()
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Reduced<'py>> {
        let from_bounds = py.get_type::<Self>().getattr("from_bounds")?;
        Ok((from_bounds, (self.bounds(py),).into_pyobject(py)?))
    }

    #[staticmethod]
    fn from_values(values: Bound<PyAny>) -> PyResult<Self> {
        let values = float_array::<Ix1>(&values, "values")?;
        TimeAxis::try_from_values(values)
            .map(|axis| Self(Arc::new(axis)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_bounds(bounds: Bound<PyAny>) -> PyResult<Self> {
        let bounds = float_array::<Ix1>(&bounds, "bounds")?;
        TimeAxis::try_from_bounds(bounds)
            .map(|axis| Self(Arc::new(axis)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Time>> {
        self.0.values().to_pyarray(py)
    }

    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Time>> {
        self.0.bounds().to_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn at(&self, time_index: isize) -> Option<Time> {
        self.0.at(resolve_index(time_index, self.0.len())?)
    }

    fn at_bounds(&self, time_index: isize) -> Option<(Time, Time)> {
        self.0.at_bounds(resolve_index(time_index, self.0.len())?)
    }
}

#[derive(Clone, PartialEq)]
#[pyclass(frozen, eq, module = "rscm._lib.core")]
#[pyo3(name = "InterpolationStrategy")]
pub enum PyInterpolationStrategy {
    Linear,
    Previous,
    Next,
}

#[pymethods]
impl PyInterpolationStrategy {
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Reduced<'py>> {
        let name = match self {
            Self::Linear => "Linear",
            Self::Previous => "Previous",
            Self::Next => "Next",
        };
        let getattr = py.import("builtins")?.getattr("getattr")?;
        Ok((getattr, (py.get_type::<Self>(), name).into_pyobject(py)?))
    }
}

impl From<&InterpolationStrategy> for PyInterpolationStrategy {
    fn from(value: &InterpolationStrategy) -> Self {
        match value {
            InterpolationStrategy::Linear(_) => PyInterpolationStrategy::Linear,
            InterpolationStrategy::Previous(_) => PyInterpolationStrategy::Previous,
            InterpolationStrategy::Next(_) => PyInterpolationStrategy::Next,
        }
    }
}

impl From<PyInterpolationStrategy> for InterpolationStrategy {
    fn from(value: PyInterpolationStrategy) -> Self {
        match value {
            PyInterpolationStrategy::Linear => {
                InterpolationStrategy::from(LinearSplineStrategy::new(true))
            }
            PyInterpolationStrategy::Previous => {
                InterpolationStrategy::from(PreviousStrategy::new(true))
            }
            PyInterpolationStrategy::Next => InterpolationStrategy::from(NextStrategy::new(true)),
        }
    }
}

#[pyclass(module = "rscm._lib.core")]
#[pyo3(name = "Timeseries")]
pub struct PyTimeseries(pub Timeseries<FloatValue>);

#[pymethods]
impl PyTimeseries {
    #[new]
    fn new(
        values: Bound<PyAny>,
        time_axis: Bound<PyTimeAxis>,
        unit: String,
        interpolation_strategy: PyInterpolationStrategy,
    ) -> PyResult<Self> {
        let interpolation_strategy: InterpolationStrategy = interpolation_strategy.into();
        let unit = checked_unit("Timeseries", unit)?;

        let values = float_array::<Ix1>(&values, "values")?;
        let time_axis = time_axis.borrow().0.clone();

        if values.len() != time_axis.len() {
            Err(length_mismatch(values.len(), time_axis.len()))
        } else {
            // Convert 1D array to 2D with shape (n, 1) for ScalarGrid
            let values_2d = values.insert_axis(Axis(1));
            Ok(Self(Timeseries::new(
                values_2d,
                time_axis,
                ScalarGrid,
                unit,
                interpolation_strategy,
            )))
        }
    }

    #[staticmethod]
    fn from_values(values: Bound<PyAny>, time_axis: Bound<PyTimeAxis>) -> PyResult<Self> {
        let values = float_array::<Ix1>(&values, "values")?;
        let time_axis = time_axis.borrow().0.clone();

        if values.len() != time_axis.len() {
            Err(length_mismatch(values.len(), time_axis.len()))
        } else {
            Ok(PyTimeseries(Timeseries::new(
                values.insert_axis(Axis(1)),
                time_axis,
                ScalarGrid,
                String::new(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            )))
        }
    }

    fn __repr__(&self) -> String {
        format!("<Timeseries len={}>", self.0.len())
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Reduced<'py>> {
        let args = (
            self.values(py),
            self.time_axis(),
            self.unit(),
            self.interpolation_strategy(),
        );
        Ok((py.get_type::<Self>().into_any(), args.into_pyobject(py)?))
    }

    #[getter]
    fn interpolation_strategy(&self) -> PyInterpolationStrategy {
        self.0.interpolation_strategy().into()
    }

    fn set(&mut self, time_index: isize, value: FloatValue) -> PyResult<()> {
        let len = self.0.len();
        let index = resolve_index(time_index, len).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "time index {time_index} is out of range for {len} steps"
            ))
        })?;
        self.0.set(index, ScalarRegion::Global, value);
        Ok(())
    }

    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<FloatValue>> {
        // Extract the first (and only) column for scalar timeseries
        self.0.values().column(0).to_owned().to_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    #[getter]
    fn latest(&self) -> usize {
        self.0.latest()
    }

    #[getter]
    fn unit(&self) -> String {
        self.0.unit().to_string()
    }

    #[getter]
    fn time_axis(&self) -> PyTimeAxis {
        PyTimeAxis(self.0.time_axis())
    }

    // TODO: Figure out how to return a mutable ref to self to enable chaining
    fn with_interpolation_strategy(&mut self, interpolation_strategy: PyInterpolationStrategy) {
        let interpolation_strategy: InterpolationStrategy = interpolation_strategy.into();

        self.0.with_interpolation_strategy(interpolation_strategy);
    }

    fn latest_value(&self) -> Option<FloatValue> {
        self.0.latest_value()
    }

    fn at(&self, time_index: isize) -> Option<FloatValue> {
        self.0.at(
            resolve_index(time_index, self.0.len())?,
            ScalarRegion::Global,
        )
    }

    fn at_time(&self, time: Time) -> PyResult<FloatValue> {
        if time.is_nan() {
            return Err(PyValueError::new_err("Cannot interpolate at a NaN time"));
        }
        Ok(self.0.at_time(time, ScalarRegion::Global)?)
    }
}

impl From<PyTimeseries> for Timeseries<FloatValue> {
    fn from(value: PyTimeseries) -> Self {
        value.0
    }
}

/// Python wrapper for four-box grid timeseries
#[pyclass]
#[pyo3(name = "FourBoxTimeseries")]
pub struct PyFourBoxTimeseries(pub GridTimeseries<FloatValue, FourBoxGrid>);

#[pymethods]
impl PyFourBoxTimeseries {
    /// Create from values with shape (n_times, 4) in grid order
    /// (NH ocean, NH land, SH ocean, SH land).
    #[new]
    fn new(
        values: Bound<PyAny>,
        time_axis: Bound<PyTimeAxis>,
        unit: String,
        interpolation_strategy: PyInterpolationStrategy,
    ) -> PyResult<Self> {
        let unit = checked_unit("FourBoxTimeseries", unit)?;
        let values = float_array::<Ix2>(&values, "values")?;
        let time_axis = time_axis.borrow().0.clone();
        if values.nrows() != time_axis.len() || values.ncols() != 4 {
            return Err(PyValueError::new_err(
                "values must have shape (len(time_axis), 4)",
            ));
        }
        Ok(Self(GridTimeseries::new(
            values,
            time_axis,
            FourBoxGrid::magicc_standard(),
            unit,
            interpolation_strategy.into(),
        )))
    }

    fn __repr__(&self) -> String {
        format!("<FourBoxTimeseries len={}>", self.0.len())
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Get all values as a 2D array with shape (n_times, 4)
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<FloatValue>> {
        self.0.values().to_pyarray(py)
    }

    #[getter]
    fn latest(&self) -> usize {
        self.0.latest()
    }

    #[getter]
    fn unit(&self) -> String {
        self.0.unit().to_string()
    }

    #[getter]
    fn time_axis(&self) -> PyTimeAxis {
        PyTimeAxis(self.0.time_axis())
    }
}

/// Python wrapper for Hemispheric grid timeseries
#[pyclass]
#[pyo3(name = "HemisphericTimeseries")]
pub struct PyHemisphericTimeseries(pub GridTimeseries<FloatValue, HemisphericGrid>);

#[pymethods]
impl PyHemisphericTimeseries {
    /// Create from values with shape (n_times, 2) in grid order (northern, southern).
    #[new]
    fn new(
        values: Bound<PyAny>,
        time_axis: Bound<PyTimeAxis>,
        unit: String,
        interpolation_strategy: PyInterpolationStrategy,
    ) -> PyResult<Self> {
        let unit = checked_unit("HemisphericTimeseries", unit)?;
        let values = float_array::<Ix2>(&values, "values")?;
        let time_axis = time_axis.borrow().0.clone();
        if values.nrows() != time_axis.len() || values.ncols() != 2 {
            return Err(PyValueError::new_err(
                "values must have shape (len(time_axis), 2)",
            ));
        }
        Ok(Self(GridTimeseries::new(
            values,
            time_axis,
            HemisphericGrid::equal_weights(),
            unit,
            interpolation_strategy.into(),
        )))
    }

    fn __repr__(&self) -> String {
        format!("<HemisphericTimeseries len={}>", self.0.len())
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Get all values as a 2D array with shape (n_times, 2)
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<FloatValue>> {
        self.0.values().to_pyarray(py)
    }

    #[getter]
    fn latest(&self) -> usize {
        self.0.latest()
    }

    #[getter]
    fn unit(&self) -> String {
        self.0.unit().to_string()
    }

    #[getter]
    fn time_axis(&self) -> PyTimeAxis {
        PyTimeAxis(self.0.time_axis())
    }
}
