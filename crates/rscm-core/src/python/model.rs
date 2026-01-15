use crate::model::{Model, ModelBuilder};
use crate::python::component::PyPythonComponent;
use crate::python::timeseries::{PyTimeAxis, PyTimeseries};
use crate::python::timeseries_collection::PyTimeseriesCollection;
use crate::python::PyRustComponent;
use crate::timeseries::{FloatValue, Time};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[pyo3(name = "ModelBuilder")]
pub struct PyModelBuilder(pub ModelBuilder);

#[pymethods]
impl PyModelBuilder {
    #[new]
    fn new() -> Self {
        Self(ModelBuilder::new())
    }

    /// Add a component that is defined in rust
    fn with_rust_component<'py>(
        mut self_: PyRefMut<'py, Self>,
        component: Bound<'py, PyRustComponent>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        self_.0.with_component(component.borrow().0.clone());
        Ok(self_)
    }

    /// Pass a component that is defined in python (UserDerivedComponent)
    fn with_py_component<'py>(
        mut self_: PyRefMut<'py, Self>,
        component: Bound<'py, PyPythonComponent>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let user_derived_component = component.borrow().0.clone();
        self_.0.with_component(user_derived_component);
        Ok(self_)
    }

    fn with_time_axis<'py>(
        mut self_: PyRefMut<'py, Self>,
        time_axis: Bound<PyTimeAxis>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let time_axis = time_axis.borrow().0.clone();

        self_.0.time_axis = time_axis;
        Ok(self_)
    }

    fn with_initial_values(
        mut self_: PyRefMut<Self>,
        initial_values: HashMap<String, FloatValue>,
    ) -> PyRefMut<Self> {
        self_.0.with_initial_values(initial_values);
        self_
    }

    fn with_exogenous_variable<'py>(
        mut self_: PyRefMut<'py, Self>,
        name: &str,
        timeseries: Bound<'py, PyTimeseries>,
    ) -> PyRefMut<'py, Self> {
        self_
            .0
            .with_exogenous_variable(name, timeseries.borrow().0.clone());
        self_
    }

    fn with_exogenous_collection<'py>(
        mut self_: PyRefMut<'py, Self>,
        timeseries: Bound<'py, PyTimeseriesCollection>,
    ) -> PyRefMut<'py, Self> {
        self_
            .0
            .with_exogenous_collection(timeseries.borrow().0.clone());
        self_
    }

    fn build(&self) -> PyResult<PyModel> {
        self.0
            .build()
            .map(PyModel)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
#[pyo3(name = "Model")]
pub struct PyModel(pub Model);

#[pymethods]
impl PyModel {
    // Not exposing initialiser deliberately

    fn current_time(&self) -> Time {
        self.0.current_time()
    }

    fn current_time_bounds(&self) -> (Time, Time) {
        self.0.current_time_bounds()
    }

    fn step(mut self_: PyRefMut<Self>) {
        self_.0.step()
    }
    fn run(mut self_: PyRefMut<Self>) {
        self_.0.run()
    }

    fn as_dot(&self) -> String {
        let dot = self.0.as_dot();
        format!("{:?}", dot)
    }

    fn finished(&self) -> bool {
        self.0.finished()
    }

    fn timeseries(&self) -> PyTimeseriesCollection {
        PyTimeseriesCollection(self.0.timeseries().clone())
    }

    /// Generate a JSON representation of the model
    ///
    /// This includes the components, their internal state and the model's
    /// state.
    fn to_toml(&self) -> PyResult<String> {
        let serialised = toml::to_string(&self.0);
        match serialised {
            Ok(serialised) => Ok(serialised),
            Err(e) => Err(PyValueError::new_err(format!("{}", e))),
        }
    }

    /// Initialise a model from a TOML representation
    #[staticmethod]
    fn from_toml(string: String) -> PyResult<Self> {
        let deserialised = toml::from_str::<Model>(string.as_str());
        match deserialised {
            Ok(deserialised) => Ok(PyModel(deserialised)),
            Err(e) => Err(PyValueError::new_err(format!("{}", e))),
        }
    }
}
