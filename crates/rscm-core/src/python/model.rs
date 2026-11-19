use crate::component::TimeseriesType;
use crate::errors::{RSCMError, RSCMResult};
use crate::model::{self, Model, ModelBuilder, RegionValues};
use crate::python::component::{PyPythonComponent, PythonComponent};
use crate::python::pool::{resolve_pool, PyThreadPool};
use crate::python::timeseries::{time_axis_repr, PyTimeAxis, Reduced};
use crate::python::timeseries_collection::{extract_timeseries_data, PyTimeseriesCollection};
use crate::python::PyRustComponent;
use crate::schema::VariableSchema;
use crate::timeseries::{FloatValue, Time};
use pyo3::exceptions::{PyException, PyRuntimeError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;

#[pyclass]
#[pyo3(name = "ModelBuilder")]
pub struct PyModelBuilder(pub ModelBuilder);

#[pymethods]
impl PyModelBuilder {
    #[new]
    fn new() -> Self {
        Self(ModelBuilder::new())
    }

    fn __repr__(&self) -> String {
        match self.0.time_axis() {
            Some(time_axis) => format!("ModelBuilder(time_axis={})", time_axis_repr(time_axis)),
            None => "ModelBuilder(time_axis=None)".to_string(),
        }
    }

    /// Add a Rust component, a `PythonComponent` or an `rscm.component.Component` instance.
    fn with_component<'py>(
        mut self_: PyRefMut<'py, Self>,
        component: Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if let Ok(rust) = component.cast::<PyRustComponent>() {
            self_.0.with_component(rust.borrow().0.clone());
        } else if let Ok(wrapped) = component.cast::<PyPythonComponent>() {
            self_.0.with_component(wrapped.borrow().0.clone());
        } else {
            let wrapped = PythonComponent::new(component.py(), component.unbind())?;
            self_.0.with_component(Arc::new(wrapped));
        }
        Ok(self_)
    }

    fn with_time_axis<'py>(
        mut self_: PyRefMut<'py, Self>,
        time_axis: Bound<PyTimeAxis>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        self_.0.with_time_axis(time_axis.borrow().0.clone());
        Ok(self_)
    }

    fn with_initial_values(
        mut self_: PyRefMut<Self>,
        initial_values: HashMap<String, MaybeWithUnit<FloatValue>>,
    ) -> PyRefMut<Self> {
        for (name, entry) in initial_values {
            let (value, unit) = entry.into_parts();
            self_
                .0
                .with_initial_value(name, RegionValues::Uniform(value), unit);
        }
        self_
    }

    fn with_initial_grid_values(
        mut self_: PyRefMut<Self>,
        initial_values: HashMap<String, MaybeWithUnit<Vec<FloatValue>>>,
    ) -> PyRefMut<Self> {
        for (name, entry) in initial_values {
            let (values, unit) = entry.into_parts();
            self_
                .0
                .with_initial_value(name, RegionValues::PerRegion(values), unit);
        }
        self_
    }

    /// Prescribe a `Timeseries`, `FourBoxTimeseries` or `HemisphericTimeseries` for `name`.
    fn with_prescribed_variable<'py>(
        mut self_: PyRefMut<'py, Self>,
        name: &str,
        timeseries: Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_timeseries_data(name, &timeseries)?;
        self_.0.with_prescribed_variable(name, data);
        Ok(self_)
    }

    fn with_prescribed_collection<'py>(
        mut self_: PyRefMut<'py, Self>,
        timeseries: Bound<'py, PyTimeseriesCollection>,
    ) -> PyRefMut<'py, Self> {
        self_
            .0
            .with_prescribed_collection(timeseries.borrow().0.clone());
        self_
    }

    /// Lag every edge from `variable`, so each consumer reads the previous step's value.
    fn with_lag<'py>(mut self_: PyRefMut<'py, Self>, variable: &str) -> PyRefMut<'py, Self> {
        self_.0.with_lag(variable);
        self_
    }

    /// Lag the edge from `variable` into the component named `consumer` alone.
    fn with_lag_for<'py>(
        mut self_: PyRefMut<'py, Self>,
        variable: &str,
        consumer: &str,
    ) -> PyRefMut<'py, Self> {
        self_.0.with_lag_for(variable, consumer);
        self_
    }

    /// Declare whether a prescribed variable is a point or an average, when its unit default is wrong.
    fn with_timeseries_type<'py>(
        mut self_: PyRefMut<'py, Self>,
        variable: &str,
        timeseries_type: TimeseriesType,
    ) -> PyRefMut<'py, Self> {
        self_.0.with_timeseries_type(variable, timeseries_type);
        self_
    }

    /// Turn the check that every state variable is a point on or off.
    fn with_state_type_check<'py>(
        mut self_: PyRefMut<'py, Self>,
        enabled: bool,
    ) -> PyRefMut<'py, Self> {
        self_.0.with_state_type_check(enabled);
        self_
    }

    /// Allow inputs with no producer, prescribed data or initial value, filling them with NaN.
    fn with_missing_prescribed_allowed(mut self_: PyRefMut<Self>) -> PyRefMut<Self> {
        self_.0.with_missing_prescribed_allowed();
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

    /// Set custom weights for a grid.
    ///
    /// These weights override the default grid weights used when:
    /// - Creating timeseries for grid-based variables
    /// - Performing automatic grid transformations (aggregation)
    ///
    /// Args:
    ///     grid: The grid to set weights for (four-box or hemispheric)
    ///     weights: The weights for each region. Must sum to 1.0.
    ///         - FourBox: [NorthernOcean, NorthernLand, SouthernOcean, SouthernLand]
    ///         - Hemispheric: [Northern, Southern]
    ///
    /// Raises:
    ///     ValueError: If grid is Scalar, or the weights have the wrong length,
    ///         are not finite, are negative or don't sum to 1.0
    ///
    /// Example:
    ///     >>> builder = ModelBuilder()
    ///     >>> builder.with_grid_weights(GridType.FourBox, [0.36, 0.14, 0.36, 0.14])
    fn with_grid_weights<'py>(
        mut self_: PyRefMut<'py, Self>,
        grid: crate::component::GridType,
        weights: Vec<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        model::check_grid_weights(grid, &weights).map_err(PyValueError::new_err)?;
        self_.0.with_grid_weights(grid, weights);
        Ok(self_)
    }

    /// Every input that no component or schema aggregate produces, sorted by name.
    ///
    /// Each is marked with whether prescribed data or an initial value supplies it.
    /// `build` fails if any is not supplied.
    fn prescribed_inputs(&self) -> Vec<model::PrescribedInput> {
        self.0.prescribed_inputs()
    }

    /// Build the model, raising ValueError if it is inconsistent and warning about extrapolated data.
    fn build(&self, py: Python<'_>) -> PyResult<PyModel> {
        let (model, warnings) = self
            .0
            .build_with_warnings()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        warn_all(py, &warnings)?;
        Ok(PyModel(model))
    }
}

#[pyclass(module = "rscm._lib.core")]
#[pyo3(name = "Model")]
pub struct PyModel(pub Model);

#[pymethods]
impl PyModel {
    // Not exposing initialiser deliberately

    fn __repr__(&self) -> String {
        let finished = if self.0.finished() { "True" } else { "False" };
        format!(
            "Model(current_time={:?}, finished={finished})",
            self.0.current_time()
        )
    }

    fn current_time(&self) -> Time {
        self.0.current_time()
    }

    fn current_time_bounds(&self) -> (Time, Time) {
        self.0.current_time_bounds()
    }

    fn step(mut self_: PyRefMut<Self>) -> PyResult<()> {
        self_.0.try_step().map_err(PyErr::from)
    }

    /// Run to the end, releasing the GIL so other threads can run models meanwhile.
    fn run(mut self_: PyRefMut<Self>) -> PyResult<()> {
        let py = self_.py();
        let model = &mut self_.0;
        py.detach(|| model.try_run()).map_err(PyErr::from)
    }

    fn as_dot(&self) -> String {
        let dot = self.0.as_dot();
        format!("{:?}", dot)
    }

    fn finished(&self) -> bool {
        self.0.finished()
    }

    /// Whether `variable` is a point or an average, or `None` if the model has no such variable.
    fn timeseries_type(&self, variable: &str) -> Option<TimeseriesType> {
        self.0.timeseries_type(variable)
    }

    fn timeseries(&self) -> PyTimeseriesCollection {
        PyTimeseriesCollection(self.0.timeseries().clone())
    }

    /// Report the model's coupling.
    ///
    /// Lists components in the order the runtime steps them, every edge between them
    /// with whether it lags, and each variable's unit, grid, source and timeseries type.
    ///
    /// Parameters
    /// ----------
    /// format : str, optional
    ///     Output format: "rich" (coloured terminal, default), "plain" (no colours),
    ///     or "json" (structured JSON).
    #[pyo3(signature = (format = "rich"))]
    fn coupling(&self, format: &str) -> PyResult<String> {
        let render = match format {
            "rich" => model::Coupling::to_rich,
            "plain" => |c: &model::Coupling| c.to_string(),
            "json" => model::Coupling::to_json,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown format '{format}'. Expected 'rich', 'plain', or 'json'."
                )))
            }
        };
        let coupling = self
            .0
            .coupling()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(render(&coupling))
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

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Reduced<'py>> {
        let from_toml = py.get_type::<Self>().getattr("from_toml")?;
        Ok((from_toml, (self.to_toml()?,).into_pyobject(py)?))
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

/// Raise each message as a `UserWarning` pointing at the caller.
fn warn_all(py: Python<'_>, messages: &[String]) -> PyResult<()> {
    let category = py.get_type::<PyUserWarning>();
    for message in messages {
        PyErr::warn(py, &category, &CString::new(message.as_str())?, 1)?;
    }
    Ok(())
}

/// Unwrap per-model results, or raise `error` with the failure count and the first failure.
///
/// A Python interrupt raised in any model is re-raised unchanged.
fn all_ok<T>(
    mut results: Vec<RSCMResult<T>>,
    action: &str,
    error: fn(String) -> PyErr,
) -> PyResult<Vec<T>> {
    if let Some(i) = results
        .iter()
        .position(|r| matches!(r, Err(RSCMError::Interrupt(_))))
    {
        if let Err(err) = results.swap_remove(i) {
            return Err(err.into());
        }
    }
    let failed = results.iter().filter(|r| r.is_err()).count();
    let total = results.len();
    results
        .into_iter()
        .enumerate()
        .map(|(i, result)| {
            result.map_err(|e| {
                error(format!(
                    "{failed} of {total} models failed to {action}. Model {i}: {e}"
                ))
            })
        })
        .collect()
}

/// Build every builder in parallel, raising ValueError for the first that fails.
#[pyfunction]
#[pyo3(name = "build_models", signature = (builders, *, n_threads = None, pool = None))]
pub fn py_build_models(
    py: Python<'_>,
    builders: Vec<PyRef<'_, PyModelBuilder>>,
    n_threads: Option<usize>,
    pool: Option<PyRef<'_, PyThreadPool>>,
) -> PyResult<Vec<PyModel>> {
    let pool = resolve_pool(n_threads, pool.as_deref())?;
    let builders: Vec<&ModelBuilder> = builders.iter().map(|b| &b.0).collect();
    let results = py.detach(|| {
        pool.install(|| {
            builders
                .par_iter()
                .map(|b| b.build_with_warnings())
                .collect::<Vec<_>>()
        })
    });
    let built = all_ok(results, "build", PyValueError::new_err::<String>)?;
    let mut models = Vec::with_capacity(built.len());
    for (model, warnings) in built {
        warn_all(py, &warnings)?;
        models.push(PyModel(model));
    }
    Ok(models)
}

/// Run every model in parallel, raising RuntimeError for the first that fails.
#[pyfunction]
#[pyo3(name = "run_models", signature = (models, *, n_threads = None, pool = None))]
pub fn py_run_models(
    py: Python<'_>,
    mut models: Vec<PyRefMut<'_, PyModel>>,
    n_threads: Option<usize>,
    pool: Option<PyRef<'_, PyThreadPool>>,
) -> PyResult<()> {
    let pool = resolve_pool(n_threads, pool.as_deref())?;
    let mut refs: Vec<&mut Model> = models.iter_mut().map(|m| &mut m.0).collect();
    let results = py.detach(|| pool.install(|| model::run_models(&mut refs)));
    all_ok(results, "run", PyRuntimeError::new_err::<String>).map(|_| ())
}

/// An initial value, optionally paired with the unit it is given in.
#[derive(FromPyObject)]
enum MaybeWithUnit<T> {
    WithUnit((T, String)),
    Bare(T),
}

impl<T> MaybeWithUnit<T> {
    fn into_parts(self) -> (T, Option<String>) {
        match self {
            Self::WithUnit((value, unit)) => (value, Some(unit)),
            Self::Bare(value) => (value, None),
        }
    }
}

/// Build and run each builder from an iterable in parallel, starting each run as its builder arrives.
///
/// Raises the iterable's own error, then ValueError if a builder fails, then RuntimeError
/// if a run fails.
#[pyfunction]
#[pyo3(name = "run_builders", signature = (builders, *, n_threads = None, pool = None))]
pub fn py_run_builders(
    py: Python<'_>,
    builders: &Bound<'_, PyAny>,
    n_threads: Option<usize>,
    pool: Option<PyRef<'_, PyThreadPool>>,
) -> PyResult<Vec<PyModel>> {
    let pool = resolve_pool(n_threads, pool.as_deref())?;
    let iterator = builders.try_iter()?.unbind();
    let mut source_error = None;
    // The iterator takes the GIL per item, so Python components in running models can take it
    // in between, and the wait for the last runs happens without it.
    let results = py.detach(|| {
        let source = std::iter::from_fn(|| {
            Python::attach(|py| {
                let item = iterator.bind(py).clone().next()?;
                let builder =
                    item.and_then(|item| Ok(item.cast::<PyModelBuilder>()?.borrow().0.clone()));
                builder.map_err(|e| source_error = Some(e)).ok()
            })
        });
        pool.map_each(source, |_, builder: ModelBuilder| {
            model::build_and_run(&builder)
        })
    });
    // Interrupts come first, from the iterable and then from any model, ahead of ordinary errors.
    if let Some(err) = source_error.take_if(|e| !e.is_instance_of::<PyException>(py)) {
        return Err(err);
    }
    let mut results = results;
    if let Some(i) = results.iter().position(|r| {
        matches!(
            r,
            Err(RSCMError::Interrupt(_)) | Ok((_, _, Err(RSCMError::Interrupt(_))))
        )
    }) {
        if let Err(err) | Ok((_, _, Err(err))) = results.swap_remove(i) {
            return Err(err.into());
        }
    }
    if let Some(err) = source_error {
        return Err(err);
    }
    let built = all_ok(results, "build", PyValueError::new_err::<String>)?;
    let mut runs = Vec::with_capacity(built.len());
    let mut models = Vec::with_capacity(built.len());
    for (model, warnings, run) in built {
        warn_all(py, &warnings)?;
        runs.push(run);
        models.push(PyModel(model));
    }
    all_ok(runs, "run", PyRuntimeError::new_err::<String>)?;
    Ok(models)
}
