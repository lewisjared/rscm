use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};

use rscm_core::create_component_builder;
use rscm_core::python::PyRustComponent;

use crate::components::*;

create_component_builder!(CO2ERFBuilder, CO2ERF, CO2ERFParameters);
create_component_builder!(
    CarbonCycleBuilder,
    CarbonCycleComponent,
    CarbonCycleParameters
);

#[pymodule]
pub fn components(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CO2ERFBuilder>()?;
    m.add_class::<CarbonCycleBuilder>()?;
    Ok(())
}
