//! Python module registration and submodule wiring.
//!
//! This module defines the PyO3 module structure exposed to Python. It registers
//! the root `_lib` module and all submodules, setting up the correct `sys.modules`
//! paths for Python's import system.
//!
//! # Module Hierarchy
//!
//! The bindings are structured as follows:
//!
//! ```text
//! rscm._lib                    # Root module (this file)
//! ├── core                     # Core abstractions from rscm-core
//! │   ├── spatial              # Spatial grid types
//! │   └── state                # State value types
//! ├── components               # Generic components from rscm-components
//! ├── two_layer                # Two-layer model from rscm-two-layer
//! └── magicc                   # MAGICC components from rscm-magicc
//! ```
//!
//! # Implementation Details
//!
//! PyO3 submodules require explicit registration in Python's `sys.modules` dictionary
//! to enable `from rscm._lib.submodule import X` syntax. The `set_path` and
//! `set_submodule_path` helper functions handle this registration.
//!
//! Each submodule is defined in its respective crate:
//!
//! - `core` module: [`rscm_core::python::core`]
//! - `components` module: [`rscm_components::python::components`]
//! - `two_layer` module: [`rscm_two_layer::python::two_layer`]
//! - `magicc` module: [`rscm_magicc::python::magicc`]

use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use rscm_calibrate::python::calibrate;
use rscm_components::python::components;
use rscm_core::python::core;
use rscm_magicc::python::magicc;
use rscm_two_layer::python::two_layer;
use std::ffi::CString;

/// Root Python module for RSCM bindings.
///
/// This function is called by PyO3 when the module is imported. It registers
/// all submodules and sets up the Python import paths.
#[pymodule]
#[pyo3(name = "_lib")]
fn rscm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_wrapped(wrap_pymodule!(core))?;
    m.add_wrapped(wrap_pymodule!(components))?;
    m.add_wrapped(wrap_pymodule!(two_layer))?;
    m.add_wrapped(wrap_pymodule!(magicc))?;
    m.add_wrapped(wrap_pymodule!(calibrate))?;

    set_path(m, "rscm._lib.core", "core")?;
    set_path(m, "rscm._lib.components", "components")?;
    set_path(m, "rscm._lib.two_layer", "two_layer")?;
    set_path(m, "rscm._lib.magicc", "magicc")?;
    set_path(m, "rscm._lib.calibrate", "calibrate")?;
    set_submodule_path(m, "rscm._lib.core.spatial", "core", "spatial")?;
    set_submodule_path(m, "rscm._lib.core.state", "core", "state")?;

    Ok(())
}

/// Register a submodule in Python's `sys.modules` dictionary.
///
/// This enables `from rscm._lib.{module} import X` syntax by making the
/// submodule discoverable through Python's import machinery.
///
/// # Arguments
///
/// * `m` - The parent module bound to Python
/// * `path` - The full dotted path to register (e.g., "rscm._lib.core")
/// * `module` - The local module name (e.g., "core")
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

/// Register a nested submodule in Python's `sys.modules` dictionary.
///
/// Similar to [`set_path`], but for submodules nested within another submodule
/// (e.g., `rscm._lib.core.spatial`).
///
/// # Arguments
///
/// * `m` - The root module bound to Python
/// * `path` - The full dotted path to register (e.g., "rscm._lib.core.spatial")
/// * `parent` - The parent module name (e.g., "core")
/// * `submodule` - The submodule name (e.g., "spatial")
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
