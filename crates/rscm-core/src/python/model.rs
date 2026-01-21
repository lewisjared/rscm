use crate::model::{Model, ModelBuilder};
use crate::python::component::PyPythonComponent;
use crate::python::timeseries::{PyTimeAxis, PyTimeseries};
use crate::python::timeseries_collection::PyTimeseriesCollection;
use crate::python::PyRustComponent;
use crate::schema::VariableSchema;
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

    /// Add a variable schema to the model for validation and aggregation.
    ///
    /// The schema defines the variables the model expects and any aggregates
    /// that should be computed. Component inputs/outputs are validated against
    /// the schema at build time.
    fn with_schema<'py>(
        mut self_: PyRefMut<'py, Self>,
        schema: Bound<'py, VariableSchema>,
    ) -> PyRefMut<'py, Self> {
        self_.0.with_schema(schema.borrow().clone());
        self_
    }

    /// Set custom weights for a grid type.
    ///
    /// These weights override the default grid weights used when:
    /// - Creating timeseries for grid-based variables
    /// - Performing automatic grid transformations (aggregation)
    ///
    /// Args:
    ///     grid_type: The grid type to set weights for (FourBox or Hemispheric)
    ///     weights: The weights for each region. Must sum to 1.0.
    ///         - FourBox: [NorthernOcean, NorthernLand, SouthernOcean, SouthernLand]
    ///         - Hemispheric: [Northern, Southern]
    ///
    /// Raises:
    ///     ValueError: If grid_type is Scalar, weights have wrong length, or don't sum to 1.0
    ///
    /// Example:
    ///     >>> builder = ModelBuilder()
    ///     >>> builder.with_grid_weights(GridType.FourBox, [0.36, 0.14, 0.36, 0.14])
    fn with_grid_weights<'py>(
        mut self_: PyRefMut<'py, Self>,
        grid_type: crate::component::GridType,
        weights: Vec<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        use pyo3::exceptions::PyValueError;

        // Validate grid type
        let expected_size =
            match grid_type {
                crate::component::GridType::Scalar => return Err(PyValueError::new_err(
                    "Cannot set weights for Scalar grid type (scalars have no regional weights)",
                )),
                crate::component::GridType::FourBox => 4,
                crate::component::GridType::Hemispheric => 2,
            };

        // Validate weights length
        if weights.len() != expected_size {
            return Err(PyValueError::new_err(format!(
                "Weights length ({}) does not match grid type {} (expected {})",
                weights.len(),
                grid_type,
                expected_size
            )));
        }

        // Validate weights sum to 1.0
        let sum: f64 = weights.iter().sum();
        if (sum - 1.0).abs() >= 1e-6 {
            return Err(PyValueError::new_err(format!(
                "Weights must sum to 1.0, got {}",
                sum
            )));
        }

        self_.0.with_grid_weights(grid_type, weights);
        Ok(self_)
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
