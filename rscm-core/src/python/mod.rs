/// Python API to access core aspects of the RSCM library.
///
/// This module wraps the core functionality of the RSCM library for use in Python.
/// This doesn't create an extension module that can be imported into python,
/// this is done in the root crate, but uses the core pymodule that is exposed here.
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
/// use rscm_core::timeseries::Timeseries;
/// use rscm_core::timeseries_collection::{TimeseriesCollection, VariableType};
/// let mut collection = TimeseriesCollection::new();
/// let timeseries = Timeseries::from_values(array![1.0, 2.0, 3.0], array![1.0, 2.0, 3.0]);
///
/// collection.add_timeseries("Surface Temperature".to_string(), timeseries, VariableType::Exogenous);
/// // We can't access timeseries anymore because the collection has taken ownership
/// // The line below generates a compiler error if uncommented
/// // timeseries.at(0)
///
/// {
///     // Get a mutable reference which enables modifying the timeseries
///     let mut timeseries = collection.get_timeseries_by_name_mut("Surface Temperature").unwrap();
///     timeseries.set(0, 2.0);
/// }
/// {
///     // Get a mutable reference which enables modifying the timeseries
///     let timeseries = collection.get_timeseries_by_name("Surface Temperature").unwrap();
///     // We can't modify this timeseries because we don't have a mutable reference
///     // timeseries.set(0, 2.0);
///     assert_eq!(timeseries.at(0).unwrap(), 2.0);
/// }
/// ```
///
/// We don't have quite the same options for the Python interface which means that we need to
/// add methods to enable modification.
/// This results in a less flexible API, but is better than nothing!
///
/// ```py
/// import numpy as np
/// from two_layer_model.core import TimeseriesCollection, Timeseries, InterpolationStrategy, VariableType
///
/// collection = TimeseriesCollection()
/// timeseries = Timeseries(np.arange(2000.0, 2010.0), np.arange(2000.0, 2010.0), "units", InterpolationStrategy.Linear)
/// collection.add_timeseries("Surface Temperature", timeseries, VariableType.Exogenous) # A clone is performed so that Rust can control ownership
///
/// # timeseries can still be used as it doesn't follow Rust's ownership model
/// # Modifications to timeseries aren't reflected in the collection so we don't expose any methods to modify it
///
/// # Setting values in the collection requires a method call instead of being able to modify a timeseries
/// collection.set_value("Surface Temperature", 0, 2.0)
/// assert timeseries.at(0) != 2
/// timeseries_2 = collection.get_timeseries_by_name("Surface Temperature")
/// assert timeseries_2.at(0) == 2
/// ```
///
/// There might be ways around this, but it would likely require deep integration of pyo3 throughout
/// the library which results in code that is less readable.
/// This tradeoff might be worth it if we need to expose a lot of functionality to Python,
/// and it would also reduce the amount of boilerplate code.
use crate::errors::RSCMError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};

mod component;
mod example_component;
mod model;
pub mod timeseries;
mod timeseries_collection;

#[pymodule]
pub fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<timeseries::PyTimeAxis>()?;
    m.add_class::<timeseries::PyTimeseries>()?;
    m.add_class::<timeseries::PyInterpolationStrategy>()?;
    m.add_class::<timeseries_collection::PyTimeseriesCollection>()?;
    m.add_class::<timeseries_collection::VariableType>()?;
    m.add_class::<component::PyUserDerivedComponent>()?;
    m.add_class::<component::RequirementDefinition>()?;
    m.add_class::<component::RequirementType>()?;
    m.add_class::<example_component::PyTestComponent>()?;
    Ok(())
}

impl From<RSCMError> for PyErr {
    fn from(e: RSCMError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}
