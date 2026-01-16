use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use rscm_components::python::components;
use rscm_core::python::core;
use rscm_magicc::python::magicc;
use rscm_two_layer::python::two_layer;
use std::ffi::CString;

#[pymodule]
#[pyo3(name = "_lib")]
fn rscm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_wrapped(wrap_pymodule!(core))?;
    m.add_wrapped(wrap_pymodule!(components))?;
    m.add_wrapped(wrap_pymodule!(two_layer))?;
    m.add_wrapped(wrap_pymodule!(magicc))?;

    set_path(m, "rscm._lib.core", "core")?;
    set_path(m, "rscm._lib.components", "components")?;
    set_path(m, "rscm._lib.two_layer", "two_layer")?;
    set_path(m, "rscm._lib.magicc", "magicc")?;
    set_submodule_path(m, "rscm._lib.core.spatial", "core", "spatial")?;
    set_submodule_path(m, "rscm._lib.core.state", "core", "state")?;

    Ok(())
}

fn set_path(m: &Bound<'_, PyModule>, path: &str, module: &str) -> PyResult<()> {
    let code = CString::new(format!(
        "\
import sys
sys.modules['{path}'] = {module}
    "
    ))
    .unwrap();
    m.py().run(code.as_c_str(), None, Some(&m.dict()))
}

fn set_submodule_path(
    m: &Bound<'_, PyModule>,
    path: &str,
    parent: &str,
    submodule: &str,
) -> PyResult<()> {
    let code = CString::new(format!(
        "\
import sys
sys.modules['{path}'] = {parent}.{submodule}
    "
    ))
    .unwrap();
    m.py().run(code.as_c_str(), None, Some(&m.dict()))
}
