use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};
use rscm_core::create_component_builder;
use rscm_core::python::PyRustComponent;

use crate::component::{TwoLayer, TwoLayerParameters};

create_component_builder!(TwoLayerBuilder, TwoLayer, TwoLayerParameters);

#[pymodule]
pub fn two_layer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TwoLayerBuilder>()?;
    Ok(())
}
