use pyo3::prelude::*;

#[pymodule]
pub fn magicc(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
