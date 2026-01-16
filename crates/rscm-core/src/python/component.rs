/// Macros for exposing a component to Python and using python-defined modules in rust
use crate::component::{Component, InputState, OutputState};
use crate::errors::RSCMResult;
use crate::python::state::{
    PyFourBoxTimeseriesWindow, PyHemisphericTimeseriesWindow, PyTimeseriesWindow,
};
use crate::timeseries::{FloatValue, Time};
use crate::timeseries_collection::TimeseriesData;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;
// Reexport the Requirement Definition and GridType
pub use crate::component::{GridType, RequirementDefinition, RequirementType};

/// Create a component builder that can be used by python to instantiate components created Rust.
#[macro_export]
macro_rules! create_component_builder {
    ($builder_name:ident, $rust_component:ty, $component_parameters:ty) => {
        #[pyclass]
        pub struct $builder_name {
            parameters: $component_parameters,
        }

        #[pymethods]
        impl $builder_name {
            #[staticmethod]
            pub fn from_parameters(parameters: Bound<PyAny>) -> PyResult<Self> {
                use pyo3::exceptions::PyValueError;

                // todo: figure out how to use an attrs class as parameters instead of a dict
                let parameters = pythonize::depythonize::<$component_parameters>(&parameters);
                match parameters {
                    Ok(parameters) => Ok(Self { parameters }),
                    Err(e) => Err(PyValueError::new_err(format!("{}", e))),
                }
            }
            pub fn build(&self) -> PyRustComponent {
                PyRustComponent(std::sync::Arc::new(<$rust_component>::from_parameters(
                    self.parameters.clone(),
                )))
            }
        }
    };
}

/// Expose component-related functionality to python
#[macro_export]
macro_rules! impl_component {
    ($py_component:ty) => {
        #[pymethods]
        impl $py_component {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                self.0.definitions()
            }

            pub fn solve(
                &mut self,
                t_current: Time,
                t_next: Time,
                collection: crate::python::timeseries_collection::PyTimeseriesCollection,
            ) -> PyResult<HashMap<String, FloatValue>> {
                let input_state =
                    crate::model::extract_state(&collection.0, self.0.input_names(), t_current);

                let output_state = self.0.solve(t_current, t_next, &input_state)?;
                // Convert StateValue to scalar values for Python interoperability
                let scalar_output = output_state
                    .into_iter()
                    .map(|(key, state_value)| (key, state_value.to_scalar()))
                    .collect();
                Ok(scalar_output)
            }
        }
    };
}

#[pymethods]
impl RequirementDefinition {
    #[new]
    #[pyo3(signature = (name, unit, requirement_type, grid_type=GridType::Scalar))]
    pub fn new_python(
        name: String,
        unit: String,
        requirement_type: RequirementType,
        grid_type: GridType,
    ) -> Self {
        Self {
            name,
            unit,
            requirement_type,
            grid_type,
        }
    }
}

/// Python wrapper for a Component defined in Rust
///
/// Instances of ['PyRustComponent'] are created via an associated ComponentBuilder for each
/// component of interest.
#[derive(Debug, Clone)]
#[pyclass]
#[pyo3{name = "RustComponent"}]
pub struct PyRustComponent(pub Arc<dyn Component + Send + Sync>);

impl_component!(PyRustComponent);

/// Wrapper to convert a Py<PyAny> (Python Class) into a Component
#[derive(Debug)]
pub struct PythonComponent {
    pub component: Py<PyAny>,
}

#[typetag::serde]
impl Component for PythonComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Python::attach(|py| {
            let py_result = self
                .component
                .bind(py)
                .call_method("definitions", (), None)
                .unwrap();
            let py_result: Vec<RequirementDefinition> = py_result.extract().unwrap();
            py_result
        })
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        Python::attach(|py| {
            let component = self.component.bind(py);

            // Check if this is a typed component (has _component_inputs attribute)
            let is_typed = component.hasattr("_component_inputs").unwrap_or(false);

            let py_result = if is_typed {
                // New typed component - pass TimeseriesWindow dict
                let windows = input_state_to_py_windows(py, input_state)
                    .expect("Failed to create Python window objects");

                // Get the Inputs class and construct typed inputs
                let inputs_class = component.getattr("Inputs").unwrap();
                let typed_inputs = inputs_class
                    .call_method("from_input_state", (windows,), None)
                    .unwrap();

                let result = component
                    .call_method("solve", (t_current, t_next, typed_inputs), None)
                    .unwrap();

                // Convert typed outputs to dict
                result.call_method("to_dict", (), None).unwrap()
            } else {
                // Legacy component - pass raw hashmap
                component
                    .call_method(
                        "solve",
                        (t_current, t_next, input_state.clone().to_hashmap()),
                        None,
                    )
                    .unwrap()
            };

            let scalar_output: HashMap<String, FloatValue> = py_result.extract().unwrap();
            // Convert scalar values from Python to StateValue
            let output_state = scalar_output
                .into_iter()
                .map(|(key, value)| (key, crate::state::StateValue::Scalar(value)))
                .collect();
            Ok(output_state)
        })
    }
}

impl Serialize for PythonComponent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Python::attach(|py| {
            let py_result = self
                .component
                .bind(py)
                .call_method("to_json", (), None)
                .unwrap();
            let py_result: String = py_result.extract().unwrap();
            serializer.serialize_str(&py_result)
        })
    }
}

impl<'de> Deserialize<'de> for PythonComponent {
    fn deserialize<D>(deserializer: D) -> Result<PythonComponent, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s: String = Deserialize::deserialize(deserializer)?;
        Python::attach(|py| {
            let code = CString::new(format!("Component.from_json('{}')", s)).unwrap();
            let component = py.eval(code.as_c_str(), None, None).unwrap().unbind();
            Ok(PythonComponent { component })
        })
    }
}

/// Interface for creating Components from Python
#[pyclass]
#[pyo3(name = "PythonComponent")]
pub struct PyPythonComponent(pub Arc<PythonComponent>);

#[pymethods]
impl PyPythonComponent {
    #[staticmethod]
    pub fn build(component: Py<PyAny>) -> Self {
        Self(Arc::new(PythonComponent { component }))
    }
}

impl_component!(PyPythonComponent);

/// Convert an InputState to a Python dict of TimeseriesWindow objects
///
/// This creates a dictionary mapping variable names to typed TimeseriesWindow
/// objects that provide access to current, previous, and historical values.
fn input_state_to_py_windows(py: Python<'_>, input_state: &InputState) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);

    for item in input_state.clone().into_iter() {
        let name = &item.name;
        let current_index = item.data.latest();

        let window: Py<PyAny> = match &item.data {
            TimeseriesData::Scalar(ts) => {
                // Extract scalar values up to and including current_index
                let values: Vec<FloatValue> = (0..=current_index)
                    .filter_map(|i| ts.at_scalar(i))
                    .collect();
                PyTimeseriesWindow::new(values, current_index)
                    .map(|w| w.into_pyobject(py).unwrap().into_any().unbind())?
            }
            TimeseriesData::FourBox(ts) => {
                // Extract all regional values for each timestep
                let values: Vec<[FloatValue; 4]> = (0..=current_index)
                    .filter_map(|i| {
                        ts.at_time_index(i)
                            .map(|vals| [vals[0], vals[1], vals[2], vals[3]])
                    })
                    .collect();
                PyFourBoxTimeseriesWindow::new(values, current_index)
                    .map(|w| w.into_pyobject(py).unwrap().into_any().unbind())?
            }
            TimeseriesData::Hemispheric(ts) => {
                // Extract all regional values for each timestep
                let values: Vec<[FloatValue; 2]> = (0..=current_index)
                    .filter_map(|i| ts.at_time_index(i).map(|vals| [vals[0], vals[1]]))
                    .collect();
                PyHemisphericTimeseriesWindow::new(values, current_index)
                    .map(|w| w.into_pyobject(py).unwrap().into_any().unbind())?
            }
        };

        dict.set_item(name, window)?;
    }

    Ok(dict.into_any().unbind())
}
