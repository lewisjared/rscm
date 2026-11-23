use crate::component::GridType;
use crate::errors::{close_matches, format_close_matches};
use crate::python::timeseries::{PyFourBoxTimeseries, PyHemisphericTimeseries, PyTimeseries};
use crate::timeseries_collection::{TimeseriesCollection, TimeseriesData};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
pub use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Extract a `Timeseries`, `FourBoxTimeseries` or `HemisphericTimeseries` given for `name`.
pub fn extract_timeseries_data(
    name: &str,
    timeseries: &Bound<'_, PyAny>,
) -> PyResult<TimeseriesData> {
    if let Ok(ts) = timeseries.cast::<PyTimeseries>() {
        Ok(ts.borrow().0.clone().into())
    } else if let Ok(ts) = timeseries.cast::<PyFourBoxTimeseries>() {
        Ok(ts.borrow().0.clone().into())
    } else if let Ok(ts) = timeseries.cast::<PyHemisphericTimeseries>() {
        Ok(ts.borrow().0.clone().into())
    } else {
        Err(PyTypeError::new_err(format!(
            "Expected Timeseries, FourBoxTimeseries or HemisphericTimeseries for '{name}', got {}",
            timeseries.get_type().name()?
        )))
    }
}

#[pyclass]
#[pyo3(name = "TimeseriesCollection")]
#[derive(Clone)]
pub struct PyTimeseriesCollection(pub TimeseriesCollection);

impl PyTimeseriesCollection {
    /// The data named `name`, or a TypeError naming its grid when it is not on `grid`.
    fn on_grid(&self, name: &str, grid: GridType) -> PyResult<Option<&TimeseriesData>> {
        match self.0.get_data(name) {
            Some(data) if data.grid() != grid => Err(PyTypeError::new_err(format!(
                "'{name}' is on the {} grid, not {grid}. Index the collection to get it on any grid",
                data.grid()
            ))),
            data => Ok(data),
        }
    }
}

/// Call `rscm.interop.<function>` with `arg` and `kwargs`.
fn interop<'py>(
    function: &str,
    arg: Bound<'py, PyAny>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    arg.py()
        .import("rscm.interop")?
        .getattr(function)?
        .call((arg,), kwargs)
}

#[pymethods]
impl PyTimeseriesCollection {
    #[new]
    fn new() -> Self {
        Self(TimeseriesCollection::new())
    }

    fn __repr__(&self) -> String {
        let names: Vec<&str> = self.0.iter().map(|x| x.name.as_str()).collect();
        format!("<TimeseriesCollection names={:?}>", names)
    }

    /// Add a timeseries on any grid, raising ValueError if the name is taken.
    pub fn add_timeseries(&mut self, name: String, timeseries: Bound<PyAny>) -> PyResult<()> {
        let data = extract_timeseries_data(&name, &timeseries)?;
        if self.0.get_data(&name).is_some() {
            return Err(PyValueError::new_err(format!(
                "The collection already holds a timeseries named '{name}'"
            )));
        }
        self.0.add_data(name, data);
        Ok(())
    }

    /// Get a timeseries on any grid by name, raising KeyError if there is none.
    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let Some(data) = self.0.get_data(name) else {
            let names: Vec<&str> = self.0.iter().map(|x| x.name.as_str()).collect();
            let hint = format_close_matches(&close_matches(name, &names));
            return Err(PyKeyError::new_err(format!(
                "No timeseries named '{name}'.{hint}"
            )));
        };
        Ok(match data {
            TimeseriesData::Scalar(ts) => PyTimeseries(ts.clone()).into_pyobject(py)?.into_any(),
            TimeseriesData::FourBox(ts) => PyFourBoxTimeseries(ts.clone())
                .into_pyobject(py)?
                .into_any(),
            TimeseriesData::Hemispheric(ts) => PyHemisphericTimeseries(ts.clone())
                .into_pyobject(py)?
                .into_any(),
        }
        .unbind())
    }

    fn __contains__(&self, name: &str) -> bool {
        self.0.get_data(name).is_some()
    }

    fn __len__(&self) -> usize {
        self.0.iter().count()
    }

    /// Iterate over the names of the timeseries on every grid.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyList::new(py, self.names())?
            .try_iter()
            .map(Bound::into_any)
    }

    /// A wide DataFrame of every timeseries, see `rscm.interop.to_dataframe`.
    #[pyo3(signature = (**kwargs))]
    fn to_dataframe<'py>(
        slf: Bound<'py, Self>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        interop("to_dataframe", slf.into_any(), kwargs)
    }

    /// An xarray Dataset of every timeseries, see `rscm.interop.to_xarray`.
    #[pyo3(signature = (**kwargs))]
    fn to_xarray<'py>(
        slf: Bound<'py, Self>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        interop("to_xarray", slf.into_any(), kwargs)
    }

    /// A collection from a wide DataFrame, see `rscm.interop.from_dataframe`.
    #[staticmethod]
    #[pyo3(signature = (df, **kwargs))]
    fn from_dataframe<'py>(
        df: Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        interop("from_dataframe", df, kwargs)
    }

    /// A collection from an xarray Dataset, see `rscm.interop.from_xarray`.
    #[staticmethod]
    #[pyo3(signature = (ds, **kwargs))]
    fn from_xarray<'py>(
        ds: Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        interop("from_xarray", ds, kwargs)
    }

    /// Get a scalar timeseries by name
    ///
    /// Returns None if there is no timeseries by that name, and raises TypeError if it is on another grid.
    pub fn get_timeseries_by_name(&self, name: &str) -> PyResult<Option<PyTimeseries>> {
        Ok(self
            .on_grid(name, GridType::Scalar)?
            .and_then(|data| data.as_scalar())
            .map(|ts| PyTimeseries(ts.clone())))
    }

    /// Get a four-box grid timeseries by name
    ///
    /// Returns None if there is no timeseries by that name, and raises TypeError if it is on another grid.
    pub fn get_four_box_timeseries_by_name(
        &self,
        name: &str,
    ) -> PyResult<Option<PyFourBoxTimeseries>> {
        Ok(self
            .on_grid(name, GridType::FourBox)?
            .and_then(|data| data.as_four_box())
            .map(|ts| PyFourBoxTimeseries(ts.clone())))
    }

    /// Get a Hemispheric grid timeseries by name
    ///
    /// Returns None if there is no timeseries by that name, and raises TypeError if it is on another grid.
    pub fn get_hemispheric_timeseries_by_name(
        &self,
        name: &str,
    ) -> PyResult<Option<PyHemisphericTimeseries>> {
        Ok(self
            .on_grid(name, GridType::Hemispheric)?
            .and_then(|data| data.as_hemispheric())
            .map(|ts| PyHemisphericTimeseries(ts.clone())))
    }

    pub fn names(&self) -> Vec<String> {
        self.0.iter().map(|x| x.name.clone()).collect()
    }

    /// Get all scalar timeseries from the collection
    ///
    /// Note: This only returns scalar timeseries, not grid timeseries.
    /// Use get_four_box_timeseries_by_name() or get_hemispheric_timeseries_by_name()
    /// to retrieve grid timeseries.
    pub fn timeseries(&self) -> Vec<PyTimeseries> {
        self.0
            .iter()
            .filter_map(|x| x.data.as_scalar().map(|ts| PyTimeseries(ts.clone())))
            .collect()
    }
}
