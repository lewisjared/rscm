use crate::python::timeseries::PyTimeseries;
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

    #[pyo3(signature = (name, child_name, weight=1.0))]
    pub fn add_nested_timeseries(&mut self, name: String, child_name: String, weight: f64) {
        self.0
            .add_nested_timeseries_with_weight(name, child_name, weight);
    }

    pub fn get_timeseries_by_name(&self, name: &str) -> Option<PyTimeseries> {
        match self.0.get_timeseries_by_name(name) {
            // We must clone the result because we cannot return references to rust owned data
            Some(ts) => Option::from(PyTimeseries(ts.clone())),
            None => Option::None,
        }
    }

    pub fn set_value(&mut self, name: &str, time_index: usize, value: f64) {
        self.0.set_value(name, time_index, value);
    }

    pub fn names(&self) -> Vec<String> {
        self.0.iter().map(|x| x.name.clone()).collect()
    }

    pub fn timeseries(&self) -> Vec<PyTimeseries> {
        self.0
            .iter()
            .map(|x| PyTimeseries(x.timeseries.clone()))
            .collect()
    }
}
