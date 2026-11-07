use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};

/// Variable names shared between components, from [`crate::names`].
#[pymodule]
pub fn names(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for (id, name) in crate::names::ALL {
        m.add(*id, *name)?;
    }
    Ok(())
}
