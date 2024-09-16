/// Macros for exposing a component to Python and using python-defined modules in rust
use crate::component::{Component, InputState, OutputState};
use crate::errors::RSCMResult;
use crate::timeseries::{FloatValue, Time};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
// Reexport the Requirement Definition
pub use crate::component::{RequirementDefinition, RequirementType};

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
                input_state: HashMap<String, FloatValue>,
            ) -> PyResult<HashMap<String, FloatValue>> {
                let state = InputState::from_hashmap(input_state);
                let output_state = self.0.solve(t_current, t_next, &state)?;
                Ok(output_state.to_hashmap())
            }
        }
    };
}

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
                let parameters = pythonize::depythonize_bound::<$component_parameters>(parameters);
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

/// Python wrapper for a Component defined in Rust
///
/// Instances of ['PyRustComponent'] are created via an associated ComponentBuilder for each
/// component of interest.
#[derive(Debug, Clone)]
#[pyclass]
#[pyo3{name = "RustComponent"}]
pub struct PyRustComponent(pub Arc<dyn Component + Send + Sync>);

impl_component!(PyRustComponent);

/// Wrapper to convert a PyObject (Python Class) into a Component
#[derive(Debug)]
pub struct PythonComponent {
    pub component: PyObject,
}

impl Component for PythonComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![]
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        Python::with_gil(|py| {
            let py_result = self
                .component
                .bind(py)
                .call_method(
                    "solve",
                    (t_current, t_next, input_state.clone().to_hashmap()),
                    None,
                )
                .unwrap();

            let state = OutputState::from_hashmap(py_result.extract().unwrap());
            Ok(state)
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