use crate::python::timeseries::{PyFourBoxTimeseries, PyHemisphericTimeseries, PyTimeseries};
use crate::timeseries_collection::TimeseriesCollection;
pub use crate::timeseries_collection::VariableType;
use pyo3::prelude::*;

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

    #[pyo3(signature = (name, timeseries, variable_type=VariableType::Exogenous))]
    pub fn add_timeseries(
        &mut self,
        name: String,
        timeseries: Bound<PyTimeseries>,
        variable_type: VariableType,
    ) {
        let timeseries = timeseries.borrow().0.clone();
        self.0.add_timeseries(name, timeseries, variable_type);
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

    /// Get a FourBox grid timeseries by name
    ///
    /// Returns None if the timeseries is not found or is not a FourBox timeseries.
    pub fn get_fourbox_timeseries_by_name(&self, name: &str) -> Option<PyFourBoxTimeseries> {
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
    /// Use get_fourbox_timeseries_by_name() or get_hemispheric_timeseries_by_name()
    /// to retrieve grid timeseries.
    pub fn timeseries(&self) -> Vec<PyTimeseries> {
        self.0
            .iter()
            .filter_map(|x| x.data.as_scalar().map(|ts| PyTimeseries(ts.clone())))
            .collect()
    }
}
