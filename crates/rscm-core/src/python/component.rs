/// Macros for exposing a component to Python and using python-defined modules in rust
use crate::component::{Component, InputState, OutputState};
use crate::errors::RSCMResult;
use crate::python::rscm_error_from_py;
use crate::python::state::{
    PyFourBoxTimeseriesWindow, PyHemisphericTimeseriesWindow, PyStateValue, PyTimeseriesWindow,
};
use crate::timeseries::{FloatValue, Time, TimeAxis};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
// Reexport the Requirement Definition and GridType
pub use crate::component::{GridType, RequirementDefinition, RequirementType, TimeseriesType};

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

/// Like [`create_component_builder`], for a component whose `from_parameters` returns a
/// `Result`, so `build` raises `ValueError` for invalid parameters.
#[macro_export]
macro_rules! create_fallible_component_builder {
    ($builder_name:ident, $rust_component:ty, $component_parameters:ty) => {
        #[pyclass]
        pub struct $builder_name {
            parameters: $component_parameters,
        }

        #[pymethods]
        impl $builder_name {
            #[staticmethod]
            pub fn from_parameters(parameters: Bound<PyAny>) -> PyResult<Self> {
                pythonize::depythonize::<$component_parameters>(&parameters)
                    .map(|parameters| Self { parameters })
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
            }

            pub fn build(&self) -> PyResult<PyRustComponent> {
                let component = <$rust_component>::from_parameters(self.parameters.clone())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(PyRustComponent(std::sync::Arc::new(component)))
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

            fn input_names(&self) -> Vec<String> {
                self.0.input_names()
            }

            fn output_names(&self) -> Vec<String> {
                self.0.output_names()
            }

            pub fn solve(
                &mut self,
                t_current: Time,
                t_next: Time,
                collection: $crate::python::timeseries_collection::PyTimeseriesCollection,
            ) -> PyResult<HashMap<String, $crate::python::state::PyStateValue>> {
                let input_state =
                    $crate::model::extract_state(&collection.0, self.0.input_names(), t_current);

                let output_state = self.0.solve(t_current, t_next, &input_state)?;
                // Return StateValue wrapped in PyStateValue for Python interoperability
                let py_output = output_state
                    .into_iter()
                    .map(|(key, state_value)| {
                        (key, $crate::python::state::PyStateValue(state_value))
                    })
                    .collect();
                Ok(py_output)
            }
        }
    };
}

#[pymethods]
impl RequirementDefinition {
    #[new]
    #[pyo3(signature = (name, unit, requirement_type, grid_type=GridType::Scalar, timeseries_type=None))]
    pub fn new_python(
        name: String,
        unit: String,
        requirement_type: RequirementType,
        grid_type: GridType,
        timeseries_type: Option<TimeseriesType>,
    ) -> Self {
        Self {
            name,
            unit,
            requirement_type,
            grid_type,
            timeseries_type,
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
pub struct PythonComponent {
    pub component: Py<PyAny>,
    name: String,
    definitions: Vec<RequirementDefinition>,
}

impl std::fmt::Debug for PythonComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&self.name).finish_non_exhaustive()
    }
}

impl PythonComponent {
    /// Wrap a Python component, reading its definitions once.
    pub fn new(py: Python<'_>, component: Py<PyAny>) -> PyResult<Self> {
        let bound = component.bind(py);
        let name = bound.get_type().qualname()?.to_string();
        let definitions = bound.call_method0("definitions")?.extract()?;
        Ok(Self {
            component,
            name,
            definitions,
        })
    }

    fn solve_py(
        &self,
        py: Python<'_>,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> PyResult<OutputState> {
        let component = self.component.bind(py);

        let py_result = if component.hasattr("_component_inputs")? {
            let windows = self.input_windows(py, input_state)?;
            let typed_inputs = component
                .getattr("Inputs")?
                .call_method1("from_input_state", (windows,))?;
            let outputs = component.call_method1("solve", (t_current, t_next, typed_inputs))?;
            if !outputs.hasattr("to_dict")? {
                return Err(PyTypeError::new_err(format!(
                    "{}.solve() must return self.Outputs(...), got {}",
                    self.name,
                    outputs.get_type().qualname()?
                )));
            }
            outputs.call_method0("to_dict")?
        } else {
            // Untyped component: raw input mapping in, dict out
            component.call_method1(
                "solve",
                (t_current, t_next, input_state.clone().to_hashmap()),
            )?
        };

        let py_dict = py_result.cast::<PyDict>().map_err(|_| {
            PyTypeError::new_err(format!("{}.solve() must return a dict", self.name))
        })?;

        let mut output_state = OutputState::new();
        for (key, value) in py_dict.iter() {
            let key: String = key.extract()?;
            let state_value = if let Ok(py_state_value) = value.extract::<PyStateValue>() {
                py_state_value.0
            } else if let Ok(float_value) = value.extract::<FloatValue>() {
                crate::state::StateValue::Scalar(float_value)
            } else {
                return Err(PyTypeError::new_err(format!(
                    "{} output '{}' must be a StateValue or float, got {}",
                    self.name,
                    key,
                    value.get_type().qualname()?
                )));
            };
            output_state.insert(key, state_value);
        }
        Ok(output_state)
    }

    /// Python windows over the inputs, built from the same Rust windows Rust components read.
    ///
    /// Values run from the first timestep to the end of the current one (index N+1),
    /// with unit conversion and grid aggregation applied.
    fn input_windows<'py>(
        &self,
        py: Python<'py>,
        input_state: &InputState,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for definition in &self.definitions {
            if !matches!(
                definition.requirement_type,
                RequirementType::Input | RequirementType::State
            ) {
                continue;
            }
            let name = definition.name.as_str();
            let window = match definition.grid_type {
                GridType::Scalar => {
                    let w = input_state.get_scalar_window(name);
                    let values = through_next(w.index(), w.len(), |o| w.at_offset(o));
                    let times = Some(times_of(w.time_axis(), values.len()));
                    PyTimeseriesWindow::new(
                        values,
                        w.index(),
                        times,
                        w.variable_source(),
                        w.timeseries_type(),
                    )?
                    .into_pyobject(py)?
                    .into_any()
                }
                GridType::FourBox => {
                    let w = input_state.get_four_box_window(name);
                    let values =
                        through_next(w.index(), w.len(), |o| w.at_offset_all(o)?.try_into().ok());
                    let times = Some(times_of(w.time_axis(), values.len()));
                    PyFourBoxTimeseriesWindow::new(
                        values,
                        w.index(),
                        times,
                        w.variable_source(),
                        w.timeseries_type(),
                    )?
                    .into_pyobject(py)?
                    .into_any()
                }
                GridType::Hemispheric => {
                    let w = input_state.get_hemispheric_window(name);
                    let values =
                        through_next(w.index(), w.len(), |o| w.at_offset_all(o)?.try_into().ok());
                    let times = Some(times_of(w.time_axis(), values.len()));
                    PyHemisphericTimeseriesWindow::new(
                        values,
                        w.index(),
                        times,
                        w.variable_source(),
                        w.timeseries_type(),
                    )?
                    .into_pyobject(py)?
                    .into_any()
                }
            };
            dict.set_item(name, window)?;
        }
        Ok(dict)
    }
}

/// The first `len` times of `axis`.
fn times_of(axis: &TimeAxis, len: usize) -> Vec<Time> {
    axis.values().iter().take(len).copied().collect()
}

/// Values from index 0 through `index + 1`, clipped to the series length.
fn through_next<T>(index: usize, len: usize, at_offset: impl Fn(isize) -> Option<T>) -> Vec<T> {
    let end = (index + 2).min(len);
    let mut values = Vec::with_capacity(end);
    values.extend((0..end).filter_map(|i| at_offset(i as isize - index as isize)));
    values
}

#[typetag::serde]
impl Component for PythonComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        self.definitions.clone()
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        Python::attach(|py| {
            self.solve_py(py, t_current, t_next, input_state)
                .map_err(|err| rscm_error_from_py(py, err))
        })
    }
}

/// Serialised form of a Python component: its class path and the state from `to_json()`.
#[derive(Serialize, Deserialize)]
struct PythonComponentRecord {
    module: String,
    class: String,
    state: String,
}

impl PythonComponentRecord {
    fn from_component(component: &PythonComponent, py: Python<'_>) -> PyResult<Self> {
        let name = &component.name;
        if name.contains("<locals>") {
            return Err(PyTypeError::new_err(format!(
                "{name} is defined inside a function and cannot be restored, define it at module level"
            )));
        }
        let bound = component.component.bind(py);
        Ok(Self {
            module: bound.get_type().module()?.to_string(),
            class: name.clone(),
            state: bound.call_method0("to_json")?.extract()?,
        })
    }

    fn restore<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut class = py.import(self.module.as_str())?.into_any();
        for part in self.class.split('.') {
            class = class.getattr(part)?;
        }
        class.call_method1("from_json", (self.state.as_str(),))
    }
}

impl Serialize for PythonComponent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let record = Python::attach(|py| {
            PythonComponentRecord::from_component(self, py)
                .map_err(|err| serde::ser::Error::custom(rscm_error_from_py(py, err)))
        })?;
        record.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PythonComponent {
    fn deserialize<D>(deserializer: D) -> Result<PythonComponent, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let record = PythonComponentRecord::deserialize(deserializer)?;
        Python::attach(|py| {
            record
                .restore(py)
                .and_then(|component| PythonComponent::new(py, component.unbind()))
                .map_err(|err| serde::de::Error::custom(rscm_error_from_py(py, err)))
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
    pub fn build(py: Python<'_>, component: Py<PyAny>) -> PyResult<Self> {
        Ok(Self(Arc::new(PythonComponent::new(py, component)?)))
    }
}

impl_component!(PyPythonComponent);
