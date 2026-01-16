use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};
use rscm_core::create_component_builder;
use rscm_core::python::PyRustComponent;

use crate::component::{TwoLayerComponent, TwoLayerComponentParameters};

create_component_builder!(
    TwoLayerComponentBuilder,
    TwoLayerComponent,
    TwoLayerComponentParameters
);

#[pymodule]
pub fn two_layer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TwoLayerComponentBuilder>()?;
    Ok(())
}
