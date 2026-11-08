/// Python API to access core aspects of the RSCM library.
///
/// This module wraps the core functionality of the RSCM library for use in Python.
/// This doesn't create an extension module that can be imported into python,
/// this is done in the rscm crate, but uses the core pymodule that is exposed here.
///
/// ## Background
///
/// pyo3 provides an interface to be able to interact with the Python interpreter and PyObjects
/// which are allocated on the Python heap.
/// This allows us to write Rust code that can be called from Python,
/// and also Python code that can be called from Rust.
///
/// Rust and Python differ in how they access memory.
/// Rust has strong guarantees about reference checks with only one immutable reference
/// or many mutable reference at a time.
/// Python instead uses reference checks and the GIL to ensure that only one thread can
/// access a PyObject at any time.
///
/// These differences can cause some difficulties when trying to use Rust-owned data in Python.
/// One particular issue is that we can't return references (mutable or not) to Rust objects into
/// Python because Python can't guarantee that Rust's memory safety rules are followed.
/// Generally we return clones of the underlying Rust object which mean that changes to these
/// objects in Python won't be reflected in Rust.
/// Instead, we need to provide methods to update the Rust object from Python.
///
/// An example is the ['TimeseriesCollection'] which owns some ['Timeseries'] structs.
/// In Rust we can return mutable references so we have more control about where data is modified.
///
/// ```rust
/// use numpy::array;
/// use rscm_core::spatial::ScalarRegion;
/// use rscm_core::timeseries::Timeseries;
/// use rscm_core::timeseries_collection::TimeseriesCollection;
/// let mut collection = TimeseriesCollection::new();
/// let timeseries = Timeseries::from_values(array![1.0, 2.0, 3.0], array![1.0, 2.0, 3.0]);
///
/// collection.add_timeseries("Surface Air Temperature Change".to_string(), timeseries);
/// // We can't access timeseries anymore because the collection has taken ownership
/// // The line below generates a compiler error if uncommented
/// // timeseries.at(0, ScalarRegion::Global)
///
/// {
///     // Get mutable data which enables modifying the timeseries
///     if let Some(data) = collection.get_data_mut("Surface Air Temperature Change") {
///         if let Some(ts) = data.as_scalar_mut() {
///             ts.set(0, ScalarRegion::Global, 2.0);
///         }
///     }
/// }
/// {
///     // Get an immutable reference to the timeseries data
///     let timeseries = collection.get_data("Surface Air Temperature Change")
///         .and_then(|data| data.as_scalar())
///         .unwrap();
///     assert_eq!(timeseries.at(0, ScalarRegion::Global).unwrap(), 2.0);
/// }
/// ```
///
/// We don't have quite the same options for the Python interface which means that we need to
/// add methods to enable modification.
/// This results in a less flexible API, but is better than nothing!
///
/// ```py
/// import numpy as np
/// from rscm.core import TimeseriesCollection, Timeseries, InterpolationStrategy
///
/// collection = TimeseriesCollection()
/// timeseries = Timeseries(np.arange(2000.0, 2010.0), np.arange(2000.0, 2010.0), "units", InterpolationStrategy.Linear)
/// collection.add_timeseries("Surface Air Temperature Change", timeseries) # A clone is performed so that Rust can control ownership
///
/// # timeseries can still be used as it doesn't follow Rust's ownership model
/// # Modifications to timeseries aren't reflected in the collection so we don't expose any methods to modify it
///
/// # Setting values in the collection requires a method call instead of being able to modify a timeseries
/// collection.set_value("Surface Air Temperature Change", 0, 2.0)
/// assert timeseries.at(0) != 2
/// timeseries_2 = collection.get_timeseries_by_name("Surface Air Temperature Change")
/// assert timeseries_2.at(0) == 2
/// ```
///
/// There might be ways around this, but it would likely require deep integration of pyo3 throughout
/// the library which results in code that is less readable.
/// This tradeoff might be worth it if we need to expose a lot of functionality to Python,
/// and it would also reduce the amount of boilerplate code.
use crate::errors::RSCMError;
use crate::schema::{AggregateDefinition, SchemaVariableDefinition, VariableSchema};
use crate::units::python::PyUnit;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::{pymodule, wrap_pymodule, Bound, PyResult};

mod component;
mod example_component;
mod model;
pub mod names;
pub mod spatial;
pub mod state;
pub mod timeseries;
mod timeseries_collection;

pub use component::PyRustComponent;
pub use state::{
    PyFourBoxSlice, PyFourBoxTimeseriesWindow, PyHemisphericSlice, PyHemisphericTimeseriesWindow,
    PyTimeseriesWindow,
};

#[pymodule]
pub fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Time types
    m.add_class::<timeseries::PyTimeAxis>()?;
    m.add_class::<timeseries::PyTimeseries>()?;
    m.add_class::<timeseries::PyFourBoxTimeseries>()?;
    m.add_class::<timeseries::PyHemisphericTimeseries>()?;
    m.add_class::<timeseries::PyInterpolationStrategy>()?;

    // State management
    m.add_class::<timeseries_collection::PyTimeseriesCollection>()?;

    // Component definitions
    m.add_class::<component::PyPythonComponent>()?;
    m.add_class::<component::VariableDefinition>()?;
    m.add_class::<component::VariableRole>()?;
    m.add_class::<component::GridType>()?;
    m.add_class::<component::TimeseriesType>()?;

    // Model orchestration
    m.add_class::<model::PyModelBuilder>()?;
    m.add_class::<crate::model::PrescribedInput>()?;
    m.add_class::<model::PyModel>()?;
    m.add_function(wrap_pyfunction!(model::py_build_models, m)?)?;
    m.add_function(wrap_pyfunction!(model::py_run_models, m)?)?;

    // Schema types (for variable definitions and aggregates)
    m.add_class::<VariableSchema>()?;
    m.add_class::<SchemaVariableDefinition>()?;
    m.add_class::<AggregateDefinition>()?;

    // Example component
    m.add_class::<example_component::TestComponentBuilder>()?;

    // Unit types
    m.add_class::<PyUnit>()?;

    // Register submodules
    m.add_wrapped(wrap_pymodule!(names::names))?;
    m.add_wrapped(wrap_pymodule!(spatial::spatial))?;
    m.add_wrapped(wrap_pymodule!(state::state))?;

    Ok(())
}

impl From<RSCMError> for PyErr {
    fn from(e: RSCMError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}

/// Convert a Python exception into an error that keeps its traceback.
pub(crate) fn rscm_error_from_py(py: Python<'_>, err: PyErr) -> RSCMError {
    let traceback = err
        .traceback(py)
        .and_then(|tb| tb.format().ok())
        .unwrap_or_default();
    RSCMError::Error(format!("{err}\n{traceback}"))
}
