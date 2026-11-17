use crate::errors::{close_matches, format_close_matches};
use crate::python::timeseries::{PyFourBoxTimeseries, PyHemisphericTimeseries, PyTimeseries};
use crate::timeseries_collection::{TimeseriesCollection, TimeseriesData};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
pub use pyo3::prelude::*;

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

    /// Get a scalar timeseries by name
    ///
    /// Returns None if the timeseries is not found or is not a scalar timeseries.
    pub fn get_timeseries_by_name(&self, name: &str) -> Option<PyTimeseries> {
        self.0
            .get_data(name)
            .and_then(|data| data.as_scalar())
            .map(|ts| PyTimeseries(ts.clone()))
    }

    /// Get a four-box grid timeseries by name
    ///
    /// Returns None if the timeseries is not found or is not a four-box timeseries.
    pub fn get_four_box_timeseries_by_name(&self, name: &str) -> Option<PyFourBoxTimeseries> {
        self.0
            .get_data(name)
            .and_then(|data| data.as_four_box())
            .map(|ts| PyFourBoxTimeseries(ts.clone()))
    }

    /// Get a Hemispheric grid timeseries by name
    ///
    /// Returns None if the timeseries is not found or is not a Hemispheric timeseries.
    pub fn get_hemispheric_timeseries_by_name(
        &self,
        name: &str,
    ) -> Option<PyHemisphericTimeseries> {
        self.0
            .get_data(name)
            .and_then(|data| data.as_hemispheric())
            .map(|ts| PyHemisphericTimeseries(ts.clone()))
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
