use crate::errors::RSCMResult;
use crate::interpolate::strategies::{
    InterpolationStrategy, LinearSplineStrategy, NextStrategy, PreviousStrategy,
};
use crate::timeseries::{FloatValue, Time, TimeAxis, Timeseries};
use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
#[pyo3(name = "TimeAxis")]
pub struct PyTimeAxis(pub Arc<TimeAxis>);

#[pymethods]
impl PyTimeAxis {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    #[staticmethod]
    fn from_values(values: Bound<PyArray1<Time>>) -> Self {
        Self(Arc::new(TimeAxis::from_values(values.to_owned_array())))
    }

    #[staticmethod]
    fn from_bounds(bounds: Bound<PyArray1<Time>>) -> Self {
        Self(Arc::new(TimeAxis::from_bounds(bounds.to_owned_array())))
    }

    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Time>> {
        self.0.values().to_pyarray_bound(py)
    }

    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Time>> {
        self.0.bounds().to_pyarray_bound(py)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn at(&self, index: usize) -> Option<Time> {
        self.0.at(index)
    }

    fn at_bounds(&self, index: usize) -> Option<(Time, Time)> {
        self.0.at_bounds(index)
    }
}

#[derive(Clone)]
#[pyclass(frozen)]
#[pyo3(name = "InterpolationStrategy")]
pub enum PyInterpolationStrategy {
    Linear,
    Previous,
    Next,
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

#[pyclass]
#[pyo3(name = "Timeseries")]
pub struct PyTimeseries(pub Timeseries<FloatValue>);

#[pymethods]
impl PyTimeseries {
    #[new]
    fn new(
        values: Bound<PyArray1<FloatValue>>,
        time_axis: Bound<PyTimeAxis>,
        units: String,
        interpolation_strategy: PyInterpolationStrategy,
    ) -> PyResult<Self> {
        let interpolation_strategy: InterpolationStrategy = interpolation_strategy.into();

        let values = values.to_owned_array();
        let time_axis = time_axis.borrow().0.clone();

        if values.len() != time_axis.len() {
            Err(PyValueError::new_err("Lengths do not match"))
        } else {
            Ok(Self(Timeseries::new(
                values,
                time_axis,
                units,
                interpolation_strategy,
            )))
        }
    }

    #[staticmethod]
    fn from_values(
        values: Bound<PyArray1<FloatValue>>,
        time: Bound<PyArray1<FloatValue>>,
    ) -> PyResult<Self> {
        let values = values.to_owned_array();
        let time = time.to_owned_array();

        if values.len() != time.len() {
            Err(PyValueError::new_err("Lengths do not match"))
        } else {
            Ok(PyTimeseries(Timeseries::from_values(values, time)))
        }
    }

    fn set(&mut self, time_index: usize, value: FloatValue) {
        self.0.set(time_index, value)
    }

    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<FloatValue>> {
        self.0.values().to_pyarray_bound(py)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    #[getter]
    fn latest(&self) -> isize {
        *self.0.latest()
    }

    // TODO: Figure out how to return a mutable ref to self to enable chaining
    fn with_interpolation_strategy(&mut self, interpolation_strategy: PyInterpolationStrategy) {
        let interpolation_strategy: InterpolationStrategy = interpolation_strategy.into();

        self.0.with_interpolation_strategy(interpolation_strategy);
    }

    fn latest_value(&self) -> Option<FloatValue> {
        self.0.latest_value()
    }

    fn at_time(&self, time: Time) -> RSCMResult<FloatValue> {
        self.0.at_time(time)
    }
}

impl From<PyTimeseries> for Timeseries<FloatValue> {
    fn from(value: PyTimeseries) -> Self {
        value.0
    }
}
